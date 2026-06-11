from datetime import date, datetime
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

import portfolio_analytics as pa


def run_regression() -> None:
    positions = pl.DataFrame(
        {
            "conId": [101, 202, 303],
            "symbol": ["ABC", "XYZ", "ES"],
            "secType": ["OPT", "OPT", "FOP"],
            "position": [-2.0, 1.0, -1.0],
            "right": ["P", "C", "P"],
            "strike": [100.0, 50.0, 5000.0],
            "lastTradeDateOrContractMonth": ["20260620", "20261020", "20261218"],
            "multiplier": [100, 100, 50],
            "exchange": ["SMART", "SMART", "GLOBEX"],
            "currency": ["USD", "USD", "USD"],
            "localSymbol": ["ABC   260620P00100000", "XYZ   261020C00050000", "ESZ6 P5000"],
            "tradingClass": ["ABC", "XYZ", "ES"],
        }
    )

    greek_snapshots = pl.DataFrame(
        {
            "conId": [],
            "ib_vega": [],
            "ib_theta": [],
            "ib_iv": [],
            "ib_underlying_price": [],
        }
    )

    enriched = pa.build_enriched_portfolio(
        positions,
        greek_snapshots=greek_snapshots,
        as_of_date=date(2026, 6, 10),
        greeks_mode="hybrid",
    )

    assert "greek_source_vega" in enriched.columns
    assert "greek_source_theta" in enriched.columns
    assert "greek_quality" in enriched.columns

    sources = set(enriched["greek_source_vega"].to_list())
    assert sources == {"FALLBACK"}

    summary = pa.summarize_net_greeks(enriched)
    assert "TOTAL" in set(summary["segment"].to_list())

    surface = pa.compute_scenario_pnl_surface(enriched)
    assert not surface.is_empty()

    expected_price_points = 41
    expected_vol_points = 21
    assert surface.height == expected_price_points * expected_vol_points

    zero_cell = surface.filter((pl.col("price_shock_pct") == 0.0) & (pl.col("vol_shock_pts") == 0)).select("pnl_abs")
    assert zero_cell.height == 1
    assert abs(zero_cell.item()) < 1e-9

    finite_count = (
        surface.select(
            pl.col("pnl_abs")
            .is_nan()
            .not_()
            .and_(pl.col("pnl_abs").is_finite())
            .sum()
            .alias("finite_count")
        )
        .item()
    )
    assert finite_count == surface.height

    roll_view = pa.roll_readiness_view(enriched)
    assert "roll_candidate_short" in roll_view.columns
    assert "roll_candidate_long" in roll_view.columns

    # First position should be in short sleeve and close to expiry threshold.
    short_row = roll_view.filter(pl.col("symbol") == "ABC")
    assert short_row.height == 1
    assert short_row.select(pl.col("sleeve").first()).item() == "SHORT_VRP"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "enriched_history.csv"
        pa.persist_enriched_portfolio_csv(
            enriched,
            output_path=str(csv_path),
            snapshot_time=datetime(2026, 6, 10, 14, 35),
        )

        history = pa.load_enriched_portfolio_csv_history(str(csv_path))
        assert history.height == enriched.height
        assert "snapshot_date" in history.columns
        assert "snapshot_timestamp" in history.columns
        assert "contract" in history.columns

        date_sample = history.select(pl.col("snapshot_date").first()).item()
        assert date_sample.year == 2026
        assert date_sample.month == 6
        assert date_sample.day == 10

        ts_sample = history.select(pl.col("snapshot_timestamp").first()).item()
        assert ts_sample.year == 2026
        assert ts_sample.month == 6
        assert ts_sample.day == 10
        assert ts_sample.hour == 14
        assert ts_sample.minute == 35

        opt_contract = history.filter(pl.col("secType") == "OPT").select(pl.col("contract").first()).item()
        fop_contract = history.filter(pl.col("secType") == "FOP").select(pl.col("contract").first()).item()
        assert str(opt_contract).startswith("Option(")
        assert str(fop_contract).startswith("FuturesOption(")
        assert "multiplier='100'" in str(opt_contract)

    print("Regression checks passed.")


if __name__ == "__main__":
    run_regression()
