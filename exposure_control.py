"""Modular option exposure control primitives and Parquet persistence."""
from dataclasses import dataclass, asdict
from datetime import date, datetime, time
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol
from zoneinfo import ZoneInfo

import polars as pl

import config
import greeks


REQUIRED = ("underlying", "expiry", "strike", "right", "quantity")
CONTROLLED_METRICS = ("delta_notional", "vega", "theta", "gross_notional", "net_notional", "long_premium", "short_premium")


class PositionSource(Protocol):
    def fetch_positions(self) -> pl.DataFrame: ...


class MarketDataSource(Protocol):
    def fetch_market_data(self, positions: pl.DataFrame) -> pl.DataFrame: ...


class LiveMarketDataSource(Protocol):
    def fetch_live(self, positions: pl.DataFrame) -> pl.DataFrame: ...


class HistoricalMarketDataSource(Protocol):
    def fetch_historical(self, positions: pl.DataFrame, as_of: datetime) -> pl.DataFrame: ...


class GreekSource(Protocol):
    def fetch_greeks(self, positions: pl.DataFrame) -> pl.DataFrame: ...


@dataclass(frozen=True)
class MarketSession:
    timezone: str = "America/New_York"
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)

    def is_open(self, at: datetime) -> bool:
        local = at.astimezone(ZoneInfo(self.timezone))
        return local.weekday() < 5 and self.open_time <= local.time().replace(tzinfo=None) < self.close_time


@dataclass(frozen=True)
class RefreshResult:
    path: Path
    market_data_mode: str
    rows: int
    captured_at: datetime
    degraded_rows: int


class DataFrameSource:
    def __init__(self, frame: pl.DataFrame): self.frame = frame
    def fetch_positions(self) -> pl.DataFrame: return self.frame.clone()


class ParquetPositionSource:
    def __init__(self, path: str | Path): self.path = Path(path)
    def fetch_positions(self) -> pl.DataFrame: return pl.read_parquet(self.path)


@dataclass(frozen=True)
class ExposureLimit:
    lower: float
    target: float
    upper: float
    unit: str = ""
    enabled: bool = True


def _stable_id(row: Mapping[str, Any]) -> str:
    key = "|".join(str(row.get(k, "")) for k in ("underlying", "expiry", "strike", "right", "contract_id", "conId", "strategy", "book"))
    return sha256(key.encode()).hexdigest()[:24]


def normalize_positions(frame: pl.DataFrame, as_of: Optional[date] = None) -> pl.DataFrame:
    aliases = {}
    if "symbol" in frame.columns and "underlying" not in frame.columns: aliases["symbol"] = "underlying"
    if "position" in frame.columns and "quantity" not in frame.columns: aliases["position"] = "quantity"
    if "lastTradeDateOrContractMonth" in frame.columns and "expiry" not in frame.columns: aliases["lastTradeDateOrContractMonth"] = "expiry"
    df = frame.rename(aliases)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Missing required position fields: {', '.join(missing)}")
    defaults = {"multiplier": 100.0, "strategy": "UNASSIGNED", "book": "UNASSIGNED", "contract_id": "", "option_price": None, "underlying_price": None, "implied_volatility": None}
    for col, value in defaults.items():
        if col not in df.columns: df = df.with_columns(pl.lit(value).alias(col))
    day = as_of or date.today()
    df = df.with_columns(
        pl.col("quantity").cast(pl.Float64, strict=False).alias("quantity"),
        pl.col("strike").cast(pl.Float64, strict=False).alias("strike"),
        pl.col("multiplier").cast(pl.Float64, strict=False).fill_null(100.0).alias("multiplier"),
        pl.col("right").str.to_uppercase().alias("right"),
        pl.col("expiry").cast(pl.String).str.slice(0, 8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("expiry_date"),
    ).with_columns(
        (pl.col("expiry_date") - pl.lit(day)).dt.total_days().cast(pl.Int64).alias("dte"),
        pl.when(pl.col("quantity") > 0).then(pl.lit("LONG")).otherwise(pl.lit("SHORT")).alias("position_side"),
    )
    # Do not pass broker objects (contract/combo legs) into a Polars struct:
    # Polars cannot serialize nested Python objects for a UDF expression.
    identity_columns = [
        c for c in ("underlying", "expiry", "strike", "right", "contract_id", "conId", "strategy", "book")
        if c in df.columns
    ]
    return df.with_columns(
        pl.struct(identity_columns).map_elements(_stable_id, return_dtype=pl.String).alias("position_id")
    )


def _model_greek(row: dict, fn, default_iv: float = .25):
    spot, strike, dte = row.get("underlying_price"), row.get("strike"), row.get("dte")
    vol = row.get("implied_volatility") or default_iv
    if spot is None or strike is None or dte is None: return None
    return fn(spot=float(spot), strike=float(strike), dte_days=float(dte), rate=.03, vol=float(vol))


def calculate_position_exposures(positions: pl.DataFrame) -> pl.DataFrame:
    rows = []
    suffix = "_live" if "underlying_price_live" in positions.columns else "_historical"
    for row in positions.iter_rows(named=True):
        delta = row.get("ib_delta") if row.get("ib_delta") is not None else _model_greek(row, lambda **x: greeks.black_scholes_delta(option_right=row["right"], **x))
        vega = row.get("ib_vega") if row.get("ib_vega") is not None else _model_greek(row, greeks.black_scholes_vega)
        theta = row.get("ib_theta") if row.get("ib_theta") is not None else _model_greek(row, lambda **x: greeks.black_scholes_theta(option_right=row["right"], **x))
        gamma = row.get("ib_gamma") if row.get("ib_gamma") is not None else _model_greek(row, lambda **x: greeks.black_scholes_gamma(option_right=row["right"], **x))
        qty, mult = row["quantity"], row["multiplier"]
        spot = row.get(f"underlying_price{suffix}")
        if spot is None:
            spot = row.get("underlying_price")
        price = row.get(f"option_price{suffix}")
        if price is None:
            price = row.get("option_price")
        abs_contracts = abs(qty) * mult
        dte = row.get("dte")
        normalized_vega = (
            vega * (config.VEGA_DTE_NORMALIZATION_TARGET / float(dte)) ** 0.5
            if vega is not None and dte is not None and dte > 0
            else vega
        )
        row.update({"delta": delta, "vega_per_contract": vega,
                    "dte_normalized_vega_per_contract": normalized_vega,
                    "net_delta": None if delta is None else delta * qty * mult,
                    "delta_notional": None if delta is None or spot is None else delta * qty * mult * spot,
                    "vega": None if normalized_vega is None else normalized_vega * qty * mult, "theta": None if theta is None else theta * qty * mult, "gamma": None if gamma is None else gamma * qty * mult,
                    "long_premium": abs_contracts * price if price is not None and qty > 0 else 0.0,
                    "short_premium": abs_contracts * price if price is not None and qty < 0 else 0.0,
                    "gross_premium": abs_contracts * price if price is not None else None,
                    "net_premium": qty * mult * price if price is not None else None,
                    "gross_notional": abs_contracts * spot if spot is not None else None,
                    "net_notional": qty * mult * spot if spot is not None else None})
        rows.append(row)
    # print(rows)
    # print(rows.columns)
    return pl.DataFrame(rows)


def apply_historical_market_data(positions: pl.DataFrame, historical: pl.DataFrame) -> pl.DataFrame:
    """Join completed bars; option trades/last prices are never used."""
    required = {"position_id", "bar_timestamp", "underlying_price", "implied_volatility"}
    missing = required - set(historical.columns)
    if missing:
        raise ValueError(f"Historical market data missing: {', '.join(sorted(missing))}")
    hist = historical.sort("bar_timestamp").unique("position_id", keep="last")
    if "midpoint" not in hist.columns:
        hist = hist.with_columns(pl.lit(None).cast(pl.Float64).alias("midpoint"))
    if "bid" not in hist.columns:
        hist = hist.with_columns(pl.lit(None).cast(pl.Float64).alias("bid"))
    if "ask" not in hist.columns:
        hist = hist.with_columns(pl.lit(None).cast(pl.Float64).alias("ask"))
    return positions.join(hist, on="position_id", how="left", suffix="_historical").with_columns(
        pl.when(pl.col("midpoint").is_not_null() & (pl.col("midpoint") >= 0))
        .then(pl.col("midpoint"))
        .when((pl.col("bid") >= 0) & (pl.col("ask") >= pl.col("bid")) & pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
        .then((pl.col("bid") + pl.col("ask")) / 2.0)
        .otherwise(None).alias("option_price"),
        pl.lit("HISTORICAL").alias("market_data_mode"),
        pl.lit("HISTORICAL_MIDPOINT").alias("price_source"),
    )


def apply_live_market_data(positions: pl.DataFrame, live: pl.DataFrame) -> pl.DataFrame:
    required = {"position_id", "bar_timestamp", "underlying_price", "option_price", "implied_volatility"}
    print(live)
    print(live.columns)
    missing = required - set(live.columns)
    if missing:
        raise ValueError(f"Live market data missing: {', '.join(sorted(missing))}")
    return positions.join(live, on="position_id", how="left", suffix="_live").with_columns(
        pl.lit("LIVE").alias("market_data_mode"), pl.lit("IBKR").alias("price_source")
    )


def refresh_exposure_snapshot(
    position_source: PositionSource,
    live_source: LiveMarketDataSource,
    historical_source: HistoricalMarketDataSource,
    output_path: str | Path,
    session: Optional[MarketSession] = None,
    captured_at: Optional[datetime] = None,
) -> RefreshResult:
    captured = captured_at or datetime.now().astimezone()
    positions = normalize_positions(position_source.fetch_positions(), captured.date())
    is_live = (session or MarketSession()).is_open(captured)
    if is_live:
        prepared = apply_live_market_data(positions, live_source.fetch_live(positions))
        greek_source = "IBKR"
    else:
        prepared = apply_historical_market_data(positions, historical_source.fetch_historical(positions, captured))
        greek_source = "LOCAL_QUANTLIB"
    exposures = calculate_position_exposures(prepared).with_columns(
        pl.lit(greek_source).alias("greek_source"), pl.lit(captured).alias("greek_calculated_at"),
        pl.when(pl.col("option_price").is_null() | pl.col("underlying_price").is_null() | pl.col("implied_volatility").is_null())
        .then(pl.lit("DEGRADED")).otherwise(pl.lit("OK")).alias("data_quality"),
        pl.lit(captured).alias("captured_at"),
    )
    print(prepared.select("underlying", "right", "option_price", "underlying_price", "implied_volatility", "underlying_price_live", "option_price_live", "implied_volatility_live"))
    print(prepared.columns)
    # print(exposures)
    # print(exposures.columns)
    # Broker contracts and combo objects are useful during retrieval but are
    # Python objects and cannot be serialized to Parquet.
    parquet_columns = [name for name, dtype in exposures.schema.items() if dtype != pl.Object]
    persisted = exposures.select(parquet_columns)
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{captured.strftime('%Y%m%d%H%M%S')}.tmp")
    try:
        persisted.write_parquet(temporary)
        temporary.replace(target)
    except Exception:
        if temporary.exists(): temporary.unlink()
        raise
    return RefreshResult(target, "LIVE" if is_live else "HISTORICAL", persisted.height, captured, persisted.filter(pl.col("data_quality") != "OK").height)


def aggregate_exposures(df: pl.DataFrame, dimension: str = "total") -> pl.DataFrame:
    column = {"underlying": "underlying", "strategy": "strategy", "book": "book", "dte": "dte", "expiry": "expiry_date", "side": "position_side"}.get(dimension)
    sums = [pl.col(m).sum().alias(m) for m in ("net_delta", "delta_notional", "vega", "theta", "gross_premium", "net_premium", "gross_notional", "net_notional", "long_premium", "short_premium")]
    if column: return df.group_by(column).agg(sums).rename({column: "group"})
    return df.select([pl.lit("TOTAL").alias("group"), *[pl.col(m).sum().alias(m) for m in ("net_delta", "delta_notional", "vega", "theta", "gross_premium", "net_premium", "gross_notional", "net_notional", "long_premium", "short_premium")]])


def evaluate_limits(aggregated: pl.DataFrame, limits: Mapping[str, ExposureLimit]) -> pl.DataFrame:
    out = aggregated
    for metric, limit in limits.items():
        value = pl.col(metric)
        out = out.with_columns(
            pl.when(value.is_null()).then(pl.lit("NO_DATA")).when(value < limit.lower).then(pl.lit("BELOW")).when(value > limit.upper).then(pl.lit("ABOVE")).otherwise(pl.lit("WITHIN")).alias(f"{metric}_status"),
            pl.when(value.is_null()).then(None).otherwise(pl.min_horizontal((value - limit.lower).abs(), (value - limit.upper).abs())).alias(f"{metric}_distance_to_nearest_limit"),
            pl.lit(limit.lower).alias(f"{metric}_lower"), pl.lit(limit.target).alias(f"{metric}_target"), pl.lit(limit.upper).alias(f"{metric}_upper"),
        )
    return out


def write_parquet_idempotent(frame: pl.DataFrame, path: str | Path, keys: list[str]) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): frame = pl.concat([pl.read_parquet(target), frame], how="diagonal_relaxed")
    frame.unique(subset=keys, keep="last", maintain_order=True).write_parquet(target)
    return target


def audit_frame(frame: pl.DataFrame, keys: list[str], required: Optional[list[str]] = None) -> dict[str, Any]:
    required = required or []
    return {"rows": frame.height, "duplicate_count": frame.height - frame.unique(subset=keys).height, "invalid_count": sum(frame.get_column(c).null_count() for c in required if c in frame.columns), "missing_columns": [c for c in required if c not in frame.columns]}
