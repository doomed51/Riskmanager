from datetime import date, datetime
from pathlib import Path
from typing import Optional
import numpy as np

import polars as pl

import config
import greeks


SHORT_SLEEVE_MAX_DTE = int(getattr(config, "SHORT_SLEEVE_MAX_DTE", 50))
LONG_SLEEVE_MIN_DTE = int(getattr(config, "LONG_SLEEVE_MIN_DTE", 90))
SHORT_ROLL_ALERT_DTE = int(getattr(config, "SHORT_ROLL_ALERT_DTE", 14))
LONG_ROLL_ALERT_DTE = int(getattr(config, "LONG_ROLL_ALERT_DTE", 45))
GREEKS_MODE = str(getattr(config, "GREEKS_MODE", "hybrid"))
DEFAULT_RISK_FREE_RATE = float(getattr(config, "DEFAULT_RISK_FREE_RATE", 0.03))
DEFAULT_IMPLIED_VOL = float(getattr(config, "DEFAULT_IMPLIED_VOL", 0.25))
DEFAULT_MULTIPLIER = float(getattr(config, "DEFAULT_MULTIPLIER", 100.0))
ENRICHED_PORTFOLIO_CSV_PATH = str(
    getattr(config, "ENRICHED_PORTFOLIO_CSV_PATH", "data/enriched_portfolio_history.csv")
)
SCENARIO_PRICE_SHOCK_MIN_PCT = int(getattr(config, "SCENARIO_PRICE_SHOCK_MIN_PCT", -20))
SCENARIO_PRICE_SHOCK_MAX_PCT = int(getattr(config, "SCENARIO_PRICE_SHOCK_MAX_PCT", 20))
SCENARIO_PRICE_SHOCK_STEP_PCT = int(getattr(config, "SCENARIO_PRICE_SHOCK_STEP_PCT", 1))
SCENARIO_VOL_SHOCK_MIN_PTS = int(getattr(config, "SCENARIO_VOL_SHOCK_MIN_PTS", -20))
SCENARIO_VOL_SHOCK_MAX_PTS = int(getattr(config, "SCENARIO_VOL_SHOCK_MAX_PTS", 20))
SCENARIO_VOL_SHOCK_STEP_PTS = int(getattr(config, "SCENARIO_VOL_SHOCK_STEP_PTS", 2))


def _parse_expiry(raw_expiry: str) -> Optional[date]:
    if raw_expiry is None:
        return None

    text = str(raw_expiry).strip()
    if not text:
        return None

    if len(text) >= 8:
        fmt = "%Y%m%d"
        text = text[:8]
    elif len(text) == 6:
        fmt = "%Y%m"
    else:
        return None

    try:
        parsed = datetime.strptime(text, fmt).date()
    except ValueError:
        return None

    # If expiry is only month granularity, place it at month-end proxy by keeping day=1 then adding a month bucket in reporting.
    return parsed


def _calc_dte(raw_expiry: str, as_of: date) -> Optional[int]:
    expiry = _parse_expiry(raw_expiry)
    if expiry is None:
        return None
    return (expiry - as_of).days


def _sleeve_from_dte(dte: Optional[int], short_max_dte: int, long_min_dte: int) -> str:
    if dte is None:
        return "UNKNOWN"
    if dte <= short_max_dte:
        return "SHORT_VRP"
    if dte >= long_min_dte:
        return "LONG_CRASH"
    return "MIDDLE"


def _roll_bucket(dte: Optional[int]) -> str:
    if dte is None:
        return "UNKNOWN"
    if dte <= 7:
        return "<=7"
    if dte <= 21:
        return "8-21"
    if dte <= 50:
        return "22-50"
    if dte <= 90:
        return "51-90"
    return ">90"


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_text_or_empty(value) -> str:
    if value is None:
        return ""
    return str(value)


def _format_multiplier_for_contract(value) -> str:
    num = _to_float_or_none(value)
    if num is None:
        return ""
    if float(num).is_integer():
        return str(int(num))
    return str(num)


def _build_ib_contract_from_row(row: dict):
    sec_type = _to_text_or_empty(row.get("secType")).upper()
    if sec_type not in {"OPT", "FOP"}:
        return None

    # Lazy import keeps analytics utilities usable in offline contexts.
    import ib_async as ib

    contract_cls = ib.Option if sec_type == "OPT" else ib.FuturesOption
    contract = contract_cls(
        symbol=_to_text_or_empty(row.get("symbol")),
        lastTradeDateOrContractMonth=_to_text_or_empty(row.get("lastTradeDateOrContractMonth")),
        strike=_to_float_or_none(row.get("strike")) or 0.0,
        right=_to_text_or_empty(row.get("right")),
        exchange=_to_text_or_empty(row.get("exchange")),
        multiplier=_format_multiplier_for_contract(row.get("multiplier")),
        currency=_to_text_or_empty(row.get("currency")),
    )

    con_id = _to_int_or_none(row.get("conId"))
    if con_id is not None:
        contract.conId = con_id

    local_symbol = row.get("localSymbol")
    if local_symbol is not None:
        contract.localSymbol = str(local_symbol)

    trading_class = row.get("tradingClass")
    if trading_class is not None:
        contract.tradingClass = str(trading_class)

    return contract


def _calc_fallback_vega(row: dict, risk_free_rate: float, default_iv: float) -> Optional[float]:
    right = row.get("right")
    strike = _to_float_or_none(row.get("strike"))
    dte = _to_float_or_none(row.get("dte"))
    spot = _to_float_or_none(row.get("ib_underlying_price"))
    vol = _to_float_or_none(row.get("ib_iv"))

    if vol is None:
        vol = default_iv

    if spot is None and strike is not None:
        # Offline approximation when underlying price is unavailable.
        spot = strike

    if spot is None or strike is None or dte is None:
        return None

    return greeks.black_scholes_vega(spot=spot, strike=strike, dte_days=dte, rate=risk_free_rate, vol=vol)


def _calc_fallback_theta(row: dict, risk_free_rate: float, default_iv: float) -> Optional[float]:
    right = row.get("right")
    strike = _to_float_or_none(row.get("strike"))
    dte = _to_float_or_none(row.get("dte"))
    spot = _to_float_or_none(row.get("ib_underlying_price"))
    vol = _to_float_or_none(row.get("ib_iv"))

    if vol is None:
        vol = default_iv

    if spot is None and strike is not None:
        # Offline approximation when underlying price is unavailable.
        spot = strike

    if spot is None or strike is None or dte is None:
        return None

    return greeks.black_scholes_theta(
        option_right=right,
        spot=spot,
        strike=strike,
        dte_days=dte,
        rate=risk_free_rate,
        vol=vol,
    )

def _calc_fallback_delta(row:dict, risk_free_rate: float, default_iv: float) -> Optional[float]:
    right = row.get("right")
    strike = _to_float_or_none(row.get("strike"))
    dte = _to_float_or_none(row.get("dte"))
    spot = _to_float_or_none(row.get("ib_underlying_price"))
    vol = _to_float_or_none(row.get("ib_iv"))

    if vol is None:
        vol = default_iv

    if spot is None and strike is not None:
        # Offline approximation when underlying price is unavailable.
        spot = strike

    if spot is None or strike is None or dte is None:
        return None

    return greeks.black_scholes_delta(
        option_right=right,
        spot=spot,
        strike=strike,
        dte_days=dte,
        rate=risk_free_rate,
        vol=vol,
    )

def _calc_fallback_gamma(row:dict, risk_free_rate: float, default_iv: float) -> Optional[float]:
    right = row.get("right")
    strike = _to_float_or_none(row.get("strike"))
    dte = _to_float_or_none(row.get("dte"))
    spot = _to_float_or_none(row.get("ib_underlying_price"))
    vol = _to_float_or_none(row.get("ib_iv"))

    if vol is None:
        vol = default_iv

    if spot is None and strike is not None:
        # Offline approximation when underlying price is unavailable.
        spot = strike

    if spot is None or strike is None or dte is None:
        return None

    return greeks.black_scholes_gamma(
        option_right=right,
        spot=spot,
        strike=strike,
        dte_days=dte,
        rate=risk_free_rate,
        vol=vol,
    )

def build_enriched_portfolio(
    positions_df: pl.DataFrame,
    greek_snapshots: Optional[pl.DataFrame] = None,
    as_of_date: Optional[date] = None,
    short_max_dte: int = SHORT_SLEEVE_MAX_DTE,
    long_min_dte: int = LONG_SLEEVE_MIN_DTE,
    short_roll_alert_dte: int = SHORT_ROLL_ALERT_DTE,
    long_roll_alert_dte: int = LONG_ROLL_ALERT_DTE,
    greeks_mode: str = GREEKS_MODE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    default_iv: float = DEFAULT_IMPLIED_VOL,
    default_multiplier: float = DEFAULT_MULTIPLIER,
) -> pl.DataFrame:
    """
    Build an enriched portfolio view by joining IB positions with greek snapshots, calculating fallback greeks as needed, and populating sleeves, roll buckets, and roll-readiness flags.
    """
    if positions_df.is_empty():
        return positions_df

    as_of = as_of_date or date.today()

    # populate multipliers
    df = positions_df
    if "multiplier" not in df.columns:
        df = df.with_columns(pl.lit(default_multiplier).alias("multiplier"))

    # handle null multipliers and positions 
    df = df.with_columns(
        pl.col("multiplier").cast(pl.Float64, strict=False).fill_null(default_multiplier).alias("multiplier"),
        pl.col("position").cast(pl.Float64, strict=False).fill_null(0.0).alias("position"),
    )

    # join greeks
    if greek_snapshots is not None and not greek_snapshots.is_empty():
        df = df.join(greek_snapshots, on="conId", how="left")
    else:
        df = df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("ib_delta"),
            pl.lit(None).cast(pl.Float64).alias("ib_gamma"),
            pl.lit(None).cast(pl.Float64).alias("ib_vega"),
            pl.lit(None).cast(pl.Float64).alias("ib_theta"),
            pl.lit(None).cast(pl.Float64).alias("ib_iv"),
            pl.lit(None).cast(pl.Float64).alias("ib_underlying_price"),
        )

    # calculate  position side 
    df = df.with_columns(
        # pl.col("lastTradeDateOrContractMonth")
        # .map_elements(lambda x: _calc_dte(x, as_of), return_dtype=pl.Int64)
        # .alias("dte"),
        pl.when(pl.col("position") > 0).then(pl.lit("LONG")).otherwise(pl.lit("SHORT")).alias("position_side"),
    )

    # populate sleeves and roll buckets
    df = df.with_columns(
        pl.col("dte")
        .map_elements(lambda x: _sleeve_from_dte(x, short_max_dte, long_min_dte), return_dtype=pl.Utf8)
        .alias("sleeve"),
        pl.col("dte").map_elements(_roll_bucket, return_dtype=pl.Utf8).alias("dte_bucket"),
    )

    # calcylate fallback greeks if needed 
    df = df.with_columns(
        pl.struct(["right", "strike", "dte", "ib_underlying_price", "ib_iv"])
        .map_elements(lambda row: _calc_fallback_vega(row, risk_free_rate, default_iv), return_dtype=pl.Float64)
        .alias("fallback_vega"),
        pl.struct(["right", "strike", "dte", "ib_underlying_price", "ib_iv"])
        .map_elements(lambda row: _calc_fallback_theta(row, risk_free_rate, default_iv), return_dtype=pl.Float64)
        .alias("fallback_theta"),
        pl.struct(["right", "strike", "dte", "ib_underlying_price", "ib_iv"])
        .map_elements(lambda row: _calc_fallback_delta(row, risk_free_rate, default_iv), return_dtype=pl.Float64)
        .alias("fallback_delta"),
        pl.struct(["right", "strike", "dte", "ib_underlying_price", "ib_iv"])
        .map_elements(lambda row: _calc_fallback_gamma(row, risk_free_rate, default_iv), return_dtype=pl.Float64)
        .alias("fallback_gamma"),
    )

    mode_lower = (greeks_mode or "hybrid").lower()

    df = df.with_columns(
        pl.struct(["ib_vega", "fallback_vega"]).map_elements(
            lambda row: greeks.choose_preferred_greek(row.get("ib_vega"), row.get("fallback_vega"), mode_lower),
            return_dtype=pl.Float64,
        ).alias("vega_per_contract"),
        pl.struct(["ib_theta", "fallback_theta"]).map_elements(
            lambda row: greeks.choose_preferred_greek(row.get("ib_theta"), row.get("fallback_theta"), mode_lower),
            return_dtype=pl.Float64,
        ).alias("theta_per_contract"),
        pl.struct(["ib_delta", "fallback_delta"]).map_elements(
            lambda row: greeks.choose_preferred_greek(row.get("ib_delta"), row.get("fallback_delta"), mode_lower),
            return_dtype=pl.Float64,
        ).alias("delta_per_contract"),
        pl.struct(["ib_gamma", "fallback_gamma"]).map_elements(
            lambda row: greeks.choose_preferred_greek(row.get("ib_gamma"), row.get("fallback_gamma"), mode_lower),
            return_dtype=pl.Float64,
        ).alias("gamma_per_contract"),
    )

    # populate greek source metadata 
    if mode_lower == "ib":
        df = df.with_columns(
            pl.when(pl.col("ib_vega").is_not_null()).then(pl.lit("IB")).otherwise(pl.lit("MISSING")).alias("greek_source_vega"),
            pl.when(pl.col("ib_theta").is_not_null()).then(pl.lit("IB")).otherwise(pl.lit("MISSING")).alias("greek_source_theta"),
            pl.when(pl.col("ib_delta").is_not_null()).then(pl.lit("IB")).otherwise(pl.lit("MISSING")).alias("greek_source_delta"),
            pl.when(pl.col("ib_gamma").is_not_null()).then(pl.lit("IB")).otherwise(pl.lit("MISSING")).alias("greek_source_gamma"),
        )
    elif mode_lower == "black_scholes":
        df = df.with_columns(
            pl.when(pl.col("fallback_vega").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_vega"),
            pl.when(pl.col("fallback_theta").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_theta"),
            pl.when(pl.col("fallback_delta").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_delta"),
            pl.when(pl.col("fallback_gamma").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_gamma"),
        )
    else:
        df = df.with_columns(
            pl.when(pl.col("ib_vega").is_not_null())
            .then(pl.lit("IB"))
            .when(pl.col("fallback_vega").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_vega"),
            pl.when(pl.col("ib_theta").is_not_null())
            .then(pl.lit("IB"))
            .when(pl.col("fallback_theta").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_theta"),
            pl.when(pl.col("ib_delta").is_not_null())
            .then(pl.lit("IB"))
            .when(pl.col("fallback_delta").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_delta"),
            pl.when(pl.col("ib_gamma").is_not_null())
            .then(pl.lit("IB"))
            .when(pl.col("fallback_gamma").is_not_null())
            .then(pl.lit("FALLBACK"))
            .otherwise(pl.lit("MISSING"))
            .alias("greek_source_gamma"),
        )

    df = df.with_columns(
        pl.when((pl.col("greek_source_vega") == pl.lit("IB")) & (pl.col("greek_source_theta") == pl.lit("IB")) & (pl.col("greek_source_delta") == pl.lit("IB")) & (pl.col("greek_source_gamma") == pl.lit("IB")))
        .then(pl.lit("HIGH"))
        .when((pl.col("greek_source_vega") == pl.lit("MISSING")) | (pl.col("greek_source_theta") == pl.lit("MISSING")) | (pl.col("greek_source_delta") == pl.lit("MISSING")) | (pl.col("greek_source_gamma") == pl.lit("MISSING")))
        .then(pl.lit("LOW"))
        .otherwise(pl.lit("MEDIUM"))
        .alias("greek_quality")
    )

    # normalize vega by dte
    df = df.with_columns(
        pl.when(pl.col("dte") > 0).then(pl.col("vega_per_contract") * np.sqrt(config.VEGA_DTE_NORMALIZATION_TARGET / pl.col("dte"))).otherwise(pl.col("vega_per_contract")).alias("dte_normalized_vega_per_contract")
        # pl.lit(0.0).alias("dte_normalized_vega_per_contract")
    )


    # calculate position size and multiplier adjusted greeks  
    df = df.with_columns(
        (pl.col("vega_per_contract") * pl.col("position") * pl.col("multiplier")).fill_null(0.0).alias("net_vega"),
        (pl.col("dte_normalized_vega_per_contract") * pl.col("position") * pl.col("multiplier")).fill_null(0.0).alias("net_dte_normalized_vega"),
        (pl.col("theta_per_contract") * pl.col("position") * pl.col("multiplier")).fill_null(0.0).alias("net_theta"),
        (pl.col("delta_per_contract") * pl.col("position") * pl.col("multiplier")).fill_null(0.0).alias("net_delta"),
        (pl.col("gamma_per_contract") * pl.col("position") * pl.col("multiplier")).fill_null(0.0).alias("net_gamma"),
    )


    # identify dte based roll candidates 
    df = df.with_columns(
        (
            (pl.col("sleeve") == pl.lit("SHORT_VRP"))
            & (pl.col("dte").is_not_null())
            & (pl.col("dte") <= pl.lit(short_roll_alert_dte))
        ).alias("roll_candidate_short"),
        (
            (pl.col("sleeve") == pl.lit("LONG_CRASH"))
            & (pl.col("dte").is_not_null())
            & (pl.col("dte") <= pl.lit(long_roll_alert_dte))
        ).alias("roll_candidate_long"),
    )

    return df



def summarize_net_greeks(enriched_portfolio: pl.DataFrame) -> pl.DataFrame:
    if enriched_portfolio.is_empty():
        return pl.DataFrame(
            {
                "segment": ["TOTAL"],
                "positions": [0],
                "net_vega": [0.0],
                "net_dte_normalized_vega": [0.0],
                "net_theta": [0.0],
                "net_delta": [0.0],
                "net_gamma": [0.0],
            }
        )

    total = enriched_portfolio.select(
        pl.lit("TOTAL").alias("segment"),
        pl.len().alias("positions"),
        pl.col("net_vega").sum().fill_null(0.0).alias("net_vega"),
        pl.col("net_dte_normalized_vega").sum().fill_null(0.0).alias("net_dte_normalized_vega"),
        pl.col("net_theta").sum().fill_null(0.0).alias("net_theta"),
        pl.col("net_delta").sum().fill_null(0.0).alias("net_delta"),
        pl.col("net_gamma").sum().fill_null(0.0).alias("net_gamma"),

        # normalized vega to 30dte, and then net them out 
    )

    side = (
        enriched_portfolio.group_by("position_side")
        .agg(
            pl.len().alias("positions"),
            pl.col("net_vega").sum().fill_null(0.0).alias("net_vega"),
            pl.col("net_dte_normalized_vega").sum().fill_null(0.0).alias("net_dte_normalized_vega"),
            pl.col("net_theta").sum().fill_null(0.0).alias("net_theta"),
            pl.col("net_delta").sum().fill_null(0.0).alias("net_delta"),
            pl.col("net_gamma").sum().fill_null(0.0).alias("net_gamma"),
        )
        .with_columns((pl.lit("SIDE_") + pl.col("position_side")).alias("segment"))
        .select(["segment", "positions", "net_vega", "net_dte_normalized_vega", "net_theta", "net_delta", "net_gamma"])
    )

    sleeve = (
        enriched_portfolio.group_by("sleeve")
        .agg(
            pl.len().alias("positions"),
            pl.col("net_vega").sum().fill_null(0.0).alias("net_vega"),
            pl.col("net_dte_normalized_vega").sum().fill_null(0.0).alias("net_dte_normalized_vega"),
            pl.col("net_theta").sum().fill_null(0.0).alias("net_theta"),
            pl.col("net_delta").sum().fill_null(0.0).alias("net_delta"),
            pl.col("net_gamma").sum().fill_null(0.0).alias("net_gamma"),
        )
        .with_columns((pl.lit("SLEEVE_") + pl.col("sleeve")).alias("segment"))
        .select(["segment", "positions", "net_vega", "net_dte_normalized_vega", "net_theta", "net_delta", "net_gamma"])
    )

    return pl.concat([total, side, sleeve], how="vertical")


def summarize_consolidated_long_portfolio(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("sleeve") == "LONG_CRASH")
        .group_by(["symbol", "dte_bucket"])
        .agg(
            pl.len().alias("positions"),
            pl.col("net_vega").sum().fill_null(0.0).alias("net_vega"),
            pl.col("net_dte_normalized_vega").sum().fill_null(0.0).alias("net_dte_normalized_vega"),
            pl.col("net_theta").sum().fill_null(0.0).alias("net_theta"),
            pl.col("net_delta").sum().fill_null(0.0).alias("net_delta"),
            pl.col("net_gamma").sum().fill_null(0.0).alias("net_gamma"),
        )
        .sort(["symbol", "dte_bucket"])
    )


def summarize_consolidated_short_portfolio(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("sleeve") == "SHORT_VRP")
        .group_by(["symbol", "dte_bucket"])
        .agg(
            pl.len().alias("positions"),
            pl.col("net_vega").sum().fill_null(0.0).alias("net_vega"),
            pl.col("net_dte_normalized_vega").sum().fill_null(0.0).alias("net_dte_normalized_vega"),
            pl.col("net_theta").sum().fill_null(0.0).alias("net_theta"),
            pl.col("net_delta").sum().fill_null(0.0).alias("net_delta"),
            pl.col("net_gamma").sum().fill_null(0.0).alias("net_gamma"),
        )
        .sort(["symbol", "dte_bucket"])
    )


def roll_readiness_view(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(
            [
                "symbol",
                "secType",
                "right",
                "strike",
                "lastTradeDateOrContractMonth",
                "dte",
                "dte_bucket",
                "position",
                "multiplier",
                "sleeve",
                "greek_source_vega",
                "greek_source_theta",
                "greek_quality",
                "net_vega",
                "net_dte_normalized_vega",
                "net_theta",
                "net_delta",
                "net_gamma",
                "roll_candidate_short",
                "roll_candidate_long",
            ]
        )
        .sort(["sleeve", "dte", "symbol"])
    )


def persist_enriched_portfolio_csv(
    enriched_portfolio: pl.DataFrame,
    output_path: Optional[str] = None,
    snapshot_time: Optional[datetime] = None,
) -> Path:
    """
    Append the enriched portfolio snapshot to a persistent CSV for later analysis.
    Adds snapshot_date and snapshot_timestamp columns to every saved row.
    """
    path = Path(output_path or ENRICHED_PORTFOLIO_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if enriched_portfolio.is_empty():
        return path

    captured_at = snapshot_time or datetime.now()
    snapshot_df = enriched_portfolio.with_columns(
        pl.lit(captured_at.date().isoformat()).alias("snapshot_date"),
        pl.lit(captured_at.strftime("%Y-%m-%d %H:%M")).alias("snapshot_timestamp"),
    )

    # drop the contract column 
    if "contract" in snapshot_df.columns:
        snapshot_df = snapshot_df.drop(["contract", "comboLegs"])

    if path.exists() and path.stat().st_size > 0:
        existing = pl.read_csv(path)
        pl.concat([existing, snapshot_df], how="diagonal_relaxed").write_csv(path)
    else:
        snapshot_df.write_csv(path)

    return path


def load_enriched_portfolio_csv_history(input_path: Optional[str] = None) -> pl.DataFrame:
    """
    Load previously persisted enriched portfolio snapshots.
    """
    path = Path(input_path or ENRICHED_PORTFOLIO_CSV_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()

    df = pl.read_csv(path)

    # convert snapshot_date and snapshot_timestamp back to datetime types
    if "snapshot_date" in df.columns:
        df = df.with_columns(pl.col("snapshot_date").str.strptime(pl.Date, "%Y-%m-%d").alias("snapshot_date"))
    if "snapshot_timestamp" in df.columns:
        df = df.with_columns(pl.col("snapshot_timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M").alias("snapshot_timestamp"))


    # Rebuild real IB contracts for downstream workflows that expect Option/FuturesOption objects.
    if "contract" not in df.columns and "secType" in df.columns:
        contract_fields = [
            col
            for col in [
                "conId",
                "symbol",
                "secType",
                "exchange",
                "currency",
                "lastTradeDateOrContractMonth",
                "right",
                "strike",
                "multiplier",
                "localSymbol",
                "tradingClass",
            ]
            if col in df.columns
        ]
        df = df.with_columns(
            pl.struct(contract_fields)
            .map_elements(_build_ib_contract_from_row, return_dtype=pl.Object)
            .alias("contract")
        )

    return df




####
# _________________________________________________________ SCENARIO ANALYTICS _________________________________________________________
####

def _default_price_shocks_pct() -> list[float]:
    min_pct = SCENARIO_PRICE_SHOCK_MIN_PCT
    max_pct = SCENARIO_PRICE_SHOCK_MAX_PCT
    step_pct = SCENARIO_PRICE_SHOCK_STEP_PCT
    if step_pct <= 0:
        step_pct = 1
    return [x / 100.0 for x in range(min_pct, max_pct + 1, step_pct)]


def _default_vol_shocks_pts() -> list[int]:
    min_pts = SCENARIO_VOL_SHOCK_MIN_PTS
    max_pts = SCENARIO_VOL_SHOCK_MAX_PTS
    step_pts = SCENARIO_VOL_SHOCK_STEP_PTS
    if step_pts <= 0:
        step_pts = 1
    return list(range(min_pts, max_pts + 1, step_pts))


def _calc_option_model_price(
    row: dict,
    spot: float,
    vol: float,
    risk_free_rate: float,
) -> Optional[float]:
    right = _to_text_or_empty(row.get("right"))
    strike = _to_float_or_none(row.get("strike"))
    dte = _to_float_or_none(row.get("dte"))
    if strike is None or dte is None:
        return None

    return greeks.quantlib_option_price(
        option_right=right,
        spot=spot,
        strike=strike,
        dte_days=dte,
        rate=risk_free_rate,
        vol=vol,
    )


def compute_scenario_pnl_surface(
    enriched_portfolio: pl.DataFrame,
    price_shocks_pct: Optional[list[float]] = None,
    vol_shocks_pts: Optional[list[int]] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    default_iv: float = DEFAULT_IMPLIED_VOL,
) -> pl.DataFrame:
    """
    Compute total portfolio absolute P&L over a 2D scenario grid.

    price_shocks_pct values are decimal percentages (e.g. -0.2 .. 0.2).
    vol_shocks_pts are implied-volatility points (e.g. -20 .. 20).
    """
    if enriched_portfolio.is_empty():
        return pl.DataFrame(
            {
                "price_shock_pct": [],
                "vol_shock_pts": [],
                "pnl_abs": [],
            }
        )

    price_grid = list(price_shocks_pct or _default_price_shocks_pct())
    vol_grid = list(vol_shocks_pts or _default_vol_shocks_pts())

    rows = []
    for row in enriched_portfolio.iter_rows(named=True):
        spot = _to_float_or_none(row.get("ib_underlying_price"))
        strike = _to_float_or_none(row.get("strike"))
        vol = _to_float_or_none(row.get("ib_iv"))
        position = _to_float_or_none(row.get("position"))
        multiplier = _to_float_or_none(row.get("multiplier"))
        base_price = _to_float_or_none(row.get("ib_option_price"))

        if position is None or multiplier is None:
            continue

        if vol is None:
            vol = default_iv

        if spot is None and strike is not None:
            # Keep existing offline approximation when spot is unavailable.
            spot = strike

        if spot is None or vol is None:
            continue

        if base_price is None:
            base_price = _calc_option_model_price(row=row, spot=spot, vol=vol, risk_free_rate=risk_free_rate)
            if base_price is None:
                continue

        rows.append(
            {
                "row": row,
                "base_price": base_price,
                "spot": spot,
                "vol": vol,
                "position": position,
                "multiplier": multiplier,
            }
        )

    results: list[dict] = []
    for price_shock in price_grid:
        for vol_shock_pts in vol_grid:
            total_pnl = 0.0
            vol_shock = vol_shock_pts / 100.0

            for row_info in rows:
                shocked_spot = row_info["spot"] * (1.0 + price_shock)
                shocked_vol = max(0.0001, row_info["vol"] + vol_shock)
                shocked_price = _calc_option_model_price(
                    row=row_info["row"],
                    spot=shocked_spot,
                    vol=shocked_vol,
                    risk_free_rate=risk_free_rate,
                )
                if shocked_price is None:
                    continue

                pnl = (shocked_price - row_info["base_price"]) * row_info["position"] * row_info["multiplier"]
                total_pnl += pnl

            results.append(
                {
                    "price_shock_pct": float(price_shock),
                    "vol_shock_pts": int(vol_shock_pts),
                    "pnl_abs": float(total_pnl),
                }
            )

    return pl.DataFrame(results).sort(["price_shock_pct", "vol_shock_pts"])


def scenario_pnl_matrix(surface: pl.DataFrame) -> pl.DataFrame:
    if surface.is_empty():
        return pl.DataFrame()

    return (
        surface.pivot(index="price_shock_pct", on="vol_shock_pts", values="pnl_abs", aggregate_function="first")
        .sort("price_shock_pct")
    )


def scenario_price_slices(surface: pl.DataFrame, selected_vol_shocks_pts: list[int]) -> pl.DataFrame:
    if surface.is_empty() or not selected_vol_shocks_pts:
        return pl.DataFrame()

    return surface.filter(pl.col("vol_shock_pts").is_in(selected_vol_shocks_pts)).sort(
        ["vol_shock_pts", "price_shock_pct"]
    )


def scenario_vol_slices(surface: pl.DataFrame, selected_price_shocks_pct: list[float]) -> pl.DataFrame:
    if surface.is_empty() or not selected_price_shocks_pct:
        return pl.DataFrame()

    return surface.filter(pl.col("price_shock_pct").is_in(selected_price_shocks_pct)).sort(
        ["price_shock_pct", "vol_shock_pts"]
    )


