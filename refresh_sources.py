"""Provider factory for dashboard exposure refreshes.

The adapter boundary is intentionally small so historical-bar retrieval can be
added without changing the dashboard or exposure calculations.
"""
import asyncio
from datetime import datetime, timezone

import polars as pl
import ib_async as ib
import config
from app_logging import get_logger, log

from helpers import get_option_greek_snapshots, get_positions_from_ib

LOGGER = get_logger(__name__)


class IBKRPositionSource:
    def fetch_positions(self) -> pl.DataFrame:
        log(LOGGER, 20, "ibkr_positions_start", "Fetching current IBKR positions", operation="positions")
        positions = asyncio.run(get_positions_from_ib())
        if "secType" in positions.columns:
            positions = positions.filter(pl.col("secType").str.to_uppercase().is_in(["OPT", "FOP"]))
        log(LOGGER, 20, "ibkr_positions_complete", "Filtered IBKR positions to options", operation="positions", rows=positions.height)
        return positions


class IBKRLiveSource:
    def fetch_live(self, positions: pl.DataFrame) -> pl.DataFrame:
        log(LOGGER, 20, "ibkr_live_start", "Fetching live option market data", operation="live_market_data", rows=positions.height)
        contracts = positions["contract"].to_list()
        snapshots = asyncio.run(get_option_greek_snapshots(contracts))
        log(LOGGER, 20, "ibkr_live_complete", "Fetched live option market data", operation="live_market_data", rows=snapshots.height)
        if snapshots.is_empty():
            return pl.DataFrame({"position_id": [], "bar_timestamp": [], "underlying_price": [], "option_price": [], "implied_volatility": []})
        ids = positions.select(["conId", "position_id"])
        return snapshots.join(ids, on="conId", how="left").select([
            "position_id", pl.lit(None).cast(pl.Datetime).alias("bar_timestamp"),
            pl.col("ib_underlying_price").alias("underlying_price"),
            pl.col("ib_option_price").alias("option_price"),
            pl.col("ib_iv").alias("implied_volatility"),
            "ib_delta", "ib_gamma", "ib_vega", "ib_theta",
        ])


class IBKRHistoricalSource:
    """Fetch the last completed IBKR bars used by the local Greek model."""

    async def _fetch(self, positions: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
        client = ib.IB()
        log(LOGGER, 20, "ibkr_historical_connect", "Connecting for historical option bars", operation="historical_market_data")
        await client.connectAsync(config.IB_HOST, config.IB_PORT, clientId=config.CLIENT_ID)
        rows = []
        try:
            for row in positions.iter_rows(named=True):
                contract = row.get("contract")
                if contract is None:
                    log(LOGGER, 30, "historical_contract_missing", "Skipping option without broker contract", operation="historical_market_data")
                    continue
                # IBKR may return BEST on the contract from the live ticker.
                # Historical bars must be requested through SMART here.
                contract = (await client.qualifyContractsAsync(contract))[0]
                log(LOGGER, 20, "ibkr_request", "Requesting historical option midpoint bars", operation="historical_market_data", endpoint="reqHistoricalDataAsync", sec_type=getattr(contract, "secType", None), con_id=getattr(contract, "conId", None), exchange=getattr(contract, "exchange", None), what_to_show="MIDPOINT", end_date_time=as_of.isoformat(), duration="2 D", bar_size="1 day", use_rth=True, format_date=2)
                midpoint_bars = await client.reqHistoricalDataAsync(
                    contract, endDateTime=as_of, durationStr="2 D", barSizeSetting="1 day",
                    # contract, durationStr="2 D", barSizeSetting="1 day",
                    whatToShow="MIDPOINT", useRTH=True, formatDate=2,
                )
                log(LOGGER, 20, "ibkr_request", "Requesting historical option IV bars", operation="historical_market_data", endpoint="reqHistoricalDataAsync", sec_type=getattr(contract, "secType", None), con_id=getattr(contract, "conId", None), exchange=getattr(contract, "exchange", None), what_to_show="OPTION_IMPLIED_VOLATILITY", end_date_time=as_of.isoformat(), duration="2 D", bar_size="1 day", use_rth=True, format_date=2)
                iv_bars = await client.reqHistoricalDataAsync(
                    contract, endDateTime=as_of, durationStr="2 D", barSizeSetting="1 day",
                    whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True, formatDate=2,
                )
                if not midpoint_bars:
                    log(LOGGER, 30, "historical_midpoint_missing", "No completed midpoint bar returned", operation="historical_market_data", con_id=getattr(contract, "conId", None))
                    continue
                midpoint = midpoint_bars[-1]
                iv = iv_bars[-1].close if iv_bars else None
                # Do not request historical bars for underlyings or any other
                # non-portfolio instrument. An underlying price can be supplied
                # by the position/market-data source when available.
                underlying_price = row.get("underlying_price")
                rows.append({
                    "position_id": row["position_id"],
                    "bar_timestamp": _bar_datetime(midpoint.date),
                    "midpoint": float(midpoint.close),
                    "bid": None, "ask": None,
                    "underlying_price": underlying_price,
                    "implied_volatility": float(iv) if iv is not None else None,
                })
        finally:
            client.disconnect()
            log(LOGGER, 20, "ibkr_historical_complete", "Historical option-bar retrieval complete", operation="historical_market_data", rows=len(rows))
        return pl.DataFrame(rows) if rows else pl.DataFrame({
            "position_id": [], "bar_timestamp": [], "midpoint": [], "bid": [], "ask": [],
            "underlying_price": [], "implied_volatility": [],
        })

    async def _underlying_close(self, client, option_contract, as_of):
        underlying = ib.Stock(option_contract.symbol, "SMART", "USD")
        details = await client.reqContractDetailsAsync(underlying)
        if details:
            underlying = details[0].contract
        underlying.exchange = "SMART"
        bars = await client.reqHistoricalDataAsync(
            underlying, endDateTime=as_of, durationStr="2 D", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=2,
        )
        return float(bars[-1].close) if bars else None

    def fetch_historical(self, positions: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
        return asyncio.run(self._fetch(positions, as_of))


def _bar_datetime(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def build_ibkr_sources():
    return IBKRPositionSource(), IBKRLiveSource(), IBKRHistoricalSource()
