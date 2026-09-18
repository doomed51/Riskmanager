from datetime import datetime

import polars as pl
from dashboard import (
    decimal_columns,
    format_exposure_value,
    display_range_bounds,
    range_marker_percent,
    status_color,
)

from exposure_control import (
    ExposureLimit,
    aggregate_exposures,
    calculate_position_exposures,
    evaluate_limits,
    normalize_positions,
    apply_historical_market_data,
)


def fixture_positions():
    return pl.DataFrame({
        "underlying": ["SPY", "SPY"], "expiry": ["20261218", "20270115"],
        "strike": [500.0, 520.0], "right": ["C", "P"], "quantity": [2, -1],
        "multiplier": [100, 100], "underlying_price": [510.0, 510.0],
        "option_price": [15.0, 20.0], "implied_volatility": [0.2, 0.25],
        "strategy": ["VRP", "CRASH"], "captured_at": [datetime(2026, 9, 16)] * 2,
    })


def test_normalize_and_position_exposure_conventions():
    normalized = normalize_positions(fixture_positions(), as_of=datetime(2026, 9, 16).date())
    assert {"position_id", "dte", "position_side", "strategy"} <= set(normalized.columns)
    enriched = calculate_position_exposures(normalized)
    long_row = enriched.filter(pl.col("quantity") > 0).row(0, named=True)
    short_row = enriched.filter(pl.col("quantity") < 0).row(0, named=True)
    assert long_row["long_premium"] == 3000.0
    assert long_row["short_premium"] == 0.0
    assert short_row["short_premium"] == 2000.0
    assert short_row["long_premium"] == 0.0
    assert long_row["gross_notional"] == 102000.0
    assert short_row["net_notional"] == -51000.0


def test_aggregate_by_required_dimensions_and_limit_states():
    exposures = calculate_position_exposures(normalize_positions(fixture_positions(), datetime(2026, 9, 16).date()))
    agg = aggregate_exposures(exposures, dimension="strategy")
    assert set(agg["group"]) == {"VRP", "CRASH"}
    total = aggregate_exposures(exposures, dimension="total")
    assert total.height == 1 and total["gross_notional"][0] == 153000.0
    limits = {"gross_notional": ExposureLimit(100000, 150000, 200000, "USD")}
    result = evaluate_limits(total, limits)
    assert result["gross_notional_status"][0] == "WITHIN"
    assert result["gross_notional_distance_to_nearest_limit"][0] == 30000.0
    assert "net_delta" in total.columns


def test_limit_boundaries_and_no_data():
    values = pl.DataFrame({"gross_notional": [100.0, 200.0, None]})
    limits = {"gross_notional": ExposureLimit(100, 150, 200, "USD")}
    result = evaluate_limits(values, limits)
    assert result["gross_notional_status"].to_list() == ["WITHIN", "WITHIN", "NO_DATA"]


def test_historical_local_pricing_ignores_trade_price():
    positions = normalize_positions(fixture_positions(), as_of=datetime(2026, 9, 16).date())
    historical = pl.DataFrame({
        "position_id": positions["position_id"],
        "bar_timestamp": [datetime(2026, 9, 15)] * 2,
        "underlying_price": [510.0] * 2,
        "bid": [14.0, 19.0], "ask": [16.0, 21.0],
        "trade_price": [99.0, 1.0],
        "implied_volatility": [.2, .25],
    })
    prepared = apply_historical_market_data(positions, historical)
    assert prepared["option_price"].to_list() == [15.0, 20.0]
    assert prepared["price_source"].unique().to_list() == ["HISTORICAL_MIDPOINT"]


def test_live_ibkr_greeks_are_used_downstream():
    positions = normalize_positions(fixture_positions(), as_of=datetime(2026, 9, 16).date())
    positions = positions.with_columns(pl.Series("dte", [30, 120]))
    live = positions.with_columns(
        pl.lit(0.123).alias("ib_delta"), pl.lit(4.5).alias("ib_vega"),
        pl.lit(-0.75).alias("ib_theta"), pl.lit(0.002).alias("ib_gamma"),
    )
    exposures = calculate_position_exposures(live)
    assert exposures["delta"].to_list() == [0.123, 0.123]
    assert exposures["vega"].to_list() == [900.0, -225.0]
    assert exposures["theta"].to_list() == [-150.0, 75.0]
    assert exposures["gamma"].to_list() == [0.4, -0.2]
    assert exposures["vega_per_contract"].to_list() == [4.5, 4.5]
    assert exposures["dte_normalized_vega_per_contract"].to_list() == [4.5, 2.25]


def test_vega_is_normalized_to_configured_target_dte():
    positions = normalize_positions(fixture_positions(), as_of=datetime(2026, 9, 16).date())
    positions = positions.with_columns(
        pl.Series("dte", [30, 120]),
        pl.lit(10.0).alias("ib_vega"),
    )
    exposures = calculate_position_exposures(positions)
    assert exposures["vega"].to_list() == [1000.0, 500.0]


def test_dashboard_decimal_columns_identifies_float_values():
    frame = pl.DataFrame({"group": ["SPY"], "quantity": [1], "delta": [1.23456], "vega": [2.0]})
    assert decimal_columns(frame) == ["delta", "vega"]


def test_dashboard_presentation_helpers_format_values_and_statuses():
    assert format_exposure_value(184000, "USD") == "+$184.0k"
    assert format_exposure_value(-1250000, "USD") == "-$1.25m"
    assert format_exposure_value(None, "USD") == "NO DATA"
    assert status_color("BELOW") == "#e58b57"
    assert status_color("WITHIN") == "#8fc9b0"
    assert status_color("ABOVE") == "#ef6f6c"
    assert status_color("NO_DATA") == "#8f9aaa"


def test_dashboard_range_marker_is_clamped_and_handles_missing_data():
    assert range_marker_percent(-100, -100, 100) == 0.0
    assert range_marker_percent(0, -100, 100) == 50.0
    assert range_marker_percent(100, -100, 100) == 100.0
    assert range_marker_percent(200, -100, 100) == 100.0
    assert range_marker_percent(None, -100, 100) is None


def test_dashboard_display_range_expands_user_limits_by_order_of_magnitude():
    assert display_range_bounds(5, 25) == (-85.0, 115.0)
    assert display_range_bounds(-25, -5) == (-115.0, 85.0)
    assert display_range_bounds(-100, 100) == (-1000.0, 1000.0)
    assert display_range_bounds(0, 100) == (-450.0, 550.0)
