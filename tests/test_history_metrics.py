import polars as pl

from portfolio_history_metrics import BOOK_METRICS, METRICS, aggregate_history, aggregate_history_by_book, plot_history, plot_history_by_book


def sample_history():
    return pl.DataFrame(
        {
            "snapshot_timestamp": ["2026-01-01 09:30", "2026-01-01 09:30", "2026-01-02 09:30"],
            "position_side": ["LONG", "SHORT", "LONG"],
            "position": [2.0, -1.0, 2.0],
            "multiplier": [100.0, 100.0, 100.0],
            "ib_underlying_price": [10.0, 10.0, 12.0],
            "ib_option_price": [1.5, 2.0, 1.0],
            "net_delta": [0.5, -0.25, 0.75],
            "net_dte_normalized_vega": [4.0, -1.0, 5.0],
            "net_theta": [-2.0, 1.0, -3.0],
        }
    )


def test_aggregate_history_calculates_requested_metrics_by_timestamp():
    result = aggregate_history(sample_history())
    first = result.row(0, named=True)
    assert result["snapshot_timestamp"].to_list() == [
        "2026-01-01 09:30",
        "2026-01-02 09:30",
    ]
    assert first["delta_notional"] == 25.0
    assert first["vega"] == 3.0
    assert first["theta"] == -1.0
    assert first["gross_notional"] == 3000.0
    assert first["net_notional"] == 1000.0
    assert first["long_premium"] == 300.0
    assert first["short_premium"] == 200.0


def test_plot_history_returns_one_axis_per_metric():
    figure = plot_history(aggregate_history(sample_history()))
    assert len(figure.axes) == len(METRICS)


def test_aggregate_history_by_book_separates_long_and_short():
    result = aggregate_history_by_book(sample_history())
    long_row = result.filter(pl.col("position_side") == "LONG").row(0, named=True)
    short_row = result.filter(pl.col("position_side") == "SHORT").row(0, named=True)
    assert long_row["delta_notional"] == 10.0
    assert short_row["delta_notional"] == -2.5
    assert long_row["gross_notional"] == 2000.0
    assert short_row["gross_notional"] == 1000.0
    combined_row = result.filter((pl.col("snapshot_timestamp") == "2026-01-01 09:30") & (pl.col("position_side") == "COMBINED")).row(0, named=True)
    assert combined_row["delta_notional"] == 7.5


def test_plot_history_by_book_returns_one_axis_per_book_metric():
    figure = plot_history_by_book(aggregate_history_by_book(sample_history()))
    assert len(figure.axes) == len(BOOK_METRICS)
