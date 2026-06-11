import asyncio

import polars as pl
import ib_async as ib 
import config 
from bs4 import BeautifulSoup
import requests 

def get_sofr_from_fred() -> float:
    url = "https://fred.stlouisfed.org/series/SOFR"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch SOFR from FRED. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    sofr_value_str = soup.find("span", class_="series-meta-observation-value").text
    try:
        return float(sofr_value_str)/100.0  # Convert percentage to decimal
    except ValueError:
        raise ValueError(f"Failed to parse SOFR value: {sofr_value_str}")

def _ib_host() -> str:
    return getattr(config, "IB_HOST", "127.0.0.1")


def _ib_port() -> int:
    return int(getattr(config, "IB_PORT", 7496))


def _client_id() -> int:
    return int(getattr(config, "CLIENT_ID", 3))


async def get_positions_from_ib() -> pl.DataFrame:

    # get positions from ib 
    ib_client = ib.IB()
    await ib_client.connectAsync(_ib_host(), _ib_port(), clientId=_client_id())
    positions = await ib_client.reqPositionsAsync()
    ib_client.disconnect() 

    # conver to polars df
    positions_df = pl.DataFrame(positions)

    # break up the contract column dict 
    contract_dicts = [c.__dict__ for c in positions_df["contract"].to_list()]
    contract_df = pl.DataFrame(contract_dicts, infer_schema_length=len(contract_dicts))
    positions_df = positions_df.hstack(contract_df)

    return positions_df


async def get_contract_details(contract) -> pl.DataFrame:
    ib_client = ib.IB()
    await ib_client.connectAsync(_ib_host(), _ib_port(), clientId=_client_id())
    contracts = await ib_client.reqContractDetailsAsync(contract)
    ib_client.disconnect()
    return contracts


async def get_option_greek_snapshots(contracts: list) -> pl.DataFrame:
    if not contracts:
        return pl.DataFrame(
            {
                "conId": [],
                "ib_vega": [],
                "ib_theta": [],
                "ib_iv": [],
                "ib_underlying_price": [],
                "ib_option_price": [],
                "ib_bid_price": [],
                "ib_ask_price": [],
            }
        )

    ib_client = ib.IB()
    await ib_client.connectAsync(_ib_host(), _ib_port(), clientId=_client_id())
    try:
        qualified_contracts = await ib_client.qualifyContractsAsync(*contracts)
        tickers = await asyncio.wait_for(ib_client.reqTickersAsync(*qualified_contracts), timeout=30.0)
    except TimeoutError:
        ib_client.disconnect()
        return pl.DataFrame(
            {
                "conId": [],
                "ib_vega": [],
                "ib_theta": [],
                "ib_iv": [],
                "ib_underlying_price": [],
                "ib_option_price": [],
                "ib_bid_price": [],
                "ib_ask_price": [],
            }
        )
    ib_client.disconnect()

    rows = []
    for ticker in tickers:
        contract = getattr(ticker, "contract", None)
        model_greeks = getattr(ticker, "modelGreeks", None)
        if contract is None:
            continue

        rows.append(
            {
                "conId": getattr(contract, "conId", None),
                "ib_vega": getattr(model_greeks, "vega", None) if model_greeks else None,
                "ib_theta": getattr(model_greeks, "theta", None) if model_greeks else None,
                "ib_iv": getattr(model_greeks, "impliedVol", None) if model_greeks else None,
                "ib_underlying_price": getattr(model_greeks, "undPrice", None) if model_greeks else None,
                "ib_option_price": getattr(model_greeks, "optPrice", None) if model_greeks else None,
                "ib_bid_price": getattr(ticker, "bid", None),
                "ib_ask_price": getattr(ticker, "ask", None),
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "conId": [],
                "ib_vega": [],
                "ib_theta": [],
                "ib_iv": [],
                "ib_underlying_price": [],
                "ib_option_price": [],
                "ib_bid_price": [],
                "ib_ask_price": [],
            }
        )

    return pl.DataFrame(rows).unique(subset=["conId"], keep="first")


