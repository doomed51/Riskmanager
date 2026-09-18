"""Streamlit exposure-control dashboard (run with: streamlit run dashboard.py)."""
from pathlib import Path
import json
import math
from html import escape

import polars as pl

from exposure_control import ExposureLimit, aggregate_exposures, evaluate_limits, refresh_exposure_snapshot
from app_logging import get_logger, log

LOGGER = get_logger(__name__)

METRIC_LABELS = {
    "net_delta": "Net delta",
    "delta_notional": "Delta notional", "vega": "Vega", "theta": "Theta / day",
    "gross_notional": "Gross notional", "net_notional": "Net notional",
    "long_premium": "Long premium", "short_premium": "Short premium",
}


def format_exposure_value(value, unit: str = "USD") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NO DATA"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    amount = abs(float(value))
    suffix = ""
    if amount >= 1_000_000:
        amount, suffix = amount / 1_000_000, "m"
    elif amount >= 1_000:
        amount, suffix = amount / 1_000, "k"
    prefix = "$" if unit.upper().startswith("USD") else ""
    decimals = 2 if suffix == "m" else 1 if suffix == "k" else 2
    return f"{sign}{prefix}{amount:,.{decimals}f}{suffix}"


def status_color(status: str) -> str:
    return {"BELOW": "#e58b57", "WITHIN": "#8fc9b0", "ABOVE": "#ef6f6c", "NO_DATA": "#8f9aaa"}.get(status, "#8f9aaa")


def range_marker_percent(value, lower, upper):
    if value is None or lower is None or upper is None or upper <= lower:
        return None
    return max(0.0, min(100.0, (float(value) - float(lower)) / (float(upper) - float(lower)) * 100))


def display_range_bounds(lower, upper) -> tuple[float, float]:
    """Expand the visual domain tenfold while centering the permitted band."""
    lower, upper = float(lower), float(upper)
    if upper <= lower:
        raise ValueError("Range upper bound must exceed lower bound")
    span = upper - lower
    midpoint = (lower + upper) / 2
    display_lower = midpoint - (span * 5)
    display_upper = midpoint + (span * 5)
    return display_lower, display_upper


def _style(st):
    st.markdown("""<style>
    .stApp { background:#050505; color:#e8edf4; }
    [data-testid="stHeader"] { background:#050505; }
    [data-testid="stSidebar"] { background:#101113; border-right:1px solid #292b30; }
    .block-container { max-width:1500px; padding:2rem 2rem 3rem; }
    .hero { display:flex; justify-content:space-between; align-items:end; margin-bottom:1.4rem; }
    .hero h1 { font-size:2rem; margin:0; color:#f1f4f8; letter-spacing:-.03em; }
    .muted { color:#9ca9b8; font-size:.92rem; }
    .card { background:#202123; border:1px solid #2d2f33; border-radius:13px; padding:1rem 1.15rem; min-height:112px; }
    .card-label { color:#b7c6d6; font-size:.92rem; }
    .card-value { color:#f2f5f8; font-size:1.7rem; font-weight:650; margin:.45rem 0 .2rem; }
    .card-sub { color:#aab5c2; font-size:.83rem; }
    .section-title { color:#f2f5f8; font-size:1.25rem; font-weight:650; margin:1.8rem 0 .15rem; }
    .range-row { margin:1.05rem 0 1.35rem; }
    .range-head { display:flex; justify-content:flex-start; gap:.8rem; color:#dce4ed; font-size:.95rem; }
    .range-track { height:13px; background:#222528; border:1px solid #393c40; border-radius:9px; position:relative; margin:.45rem 0 .25rem; }
    .range-fill { position:absolute; height:100%; background:#b4673e; border-radius:8px; }
    .range-marker { position:absolute; top:-5px; height:23px; width:5px; background:#f3f0eb; border-radius:3px; transform:translateX(-50%); }
    .range-foot { display:flex; justify-content:space-between; color:#8793a0; font-size:.76rem; }
    .panel { background:#090a0b; border:1px solid #292b2f; border-radius:13px; padding:1rem 1.2rem; }
    div[data-testid="stDataFrame"] { border:1px solid #292b2f; border-radius:10px; overflow:hidden; }
    </style>""", unsafe_allow_html=True)


def load_limits(path: str | Path) -> dict[str, ExposureLimit]:
    data = json.loads(Path(path).read_text())
    return {name: ExposureLimit(**values) for name, values in data.items()}


def dashboard_frame(exposures: pl.DataFrame, limits: dict[str, ExposureLimit]) -> pl.DataFrame:
    return evaluate_limits(aggregate_exposures(exposures), limits)


def decimal_columns(frame: pl.DataFrame) -> list[str]:
    """Return columns containing decimal values for dashboard formatting."""
    return [
        name for name, dtype in zip(frame.columns, frame.dtypes)
        if dtype in (pl.Float32, pl.Float64)
    ]


def dashboard_column_config(frame: pl.DataFrame, st) -> dict:
    """Configure decimal table columns to render consistently to two decimals."""
    return {
        name: st.column_config.NumberColumn(format="%.2f")
        for name in decimal_columns(frame)
    }


def _range_html(metric: str, row: dict, limit: ExposureLimit) -> str:
    value, status = row.get(metric), row.get(metric + "_status", "NO_DATA")
    display_lower, display_upper = display_range_bounds(limit.lower, limit.upper)
    marker = range_marker_percent(value, display_lower, display_upper)
    marker_html = f'<span class="range-marker" style="left:{marker:.2f}%"></span>' if marker is not None else ""
    fill_left = range_marker_percent(limit.lower, display_lower, display_upper) or 0
    fill_width = (range_marker_percent(limit.upper, display_lower, display_upper) or 100) - fill_left
    return f'''<div class="range-row"><div class="range-head"><span>{escape(METRIC_LABELS.get(metric, metric))}</span>
    <span style="color:{status_color(status)}">{escape(format_exposure_value(value, limit.unit))} · {escape(status)}</span></div>
    <div class="range-track"><span class="range-fill" style="left:{fill_left:.2f}%;width:{fill_width:.2f}%"></span>{marker_html}</div>
    <div class="range-foot"><span>{escape(format_exposure_value(display_lower, limit.unit))}</span><span>permitted range · target {escape(format_exposure_value(limit.target, limit.unit))}</span><span>{escape(format_exposure_value(display_upper, limit.unit))}</span></div></div>'''


def refresh_dashboard_data(exposure_path: str | Path):
    """Invoke the configured provider factory and atomically refresh current exposures."""
    log(LOGGER, 20, "dashboard_refresh_start", "Starting exposure refresh", operation="refresh")
    from refresh_sources import build_ibkr_sources
    position_source, live_source, historical_source = build_ibkr_sources()
    return refresh_exposure_snapshot(position_source, live_source, historical_source, exposure_path)


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Option Exposure Control", layout="wide")
    _style(st)
    st.markdown('<div class="hero"><div><h1>Option exposure control</h1><div class="muted">Portfolio risk monitor · current exposures against permitted ranges</div></div></div>', unsafe_allow_html=True)
    exposure_path = st.sidebar.text_input("Exposure Parquet", "data/exposures.parquet")
    limits_path = st.sidebar.text_input("Limits JSON", "data/exposure_limits.json")
    if st.sidebar.button("Refresh exposures", type="primary"):
        with st.spinner("Retrieving latest IBKR data and recalculating exposures..."):
            try:
                result = refresh_dashboard_data(exposure_path)
                log(LOGGER, 20, "dashboard_refresh_complete", "Exposure refresh completed", operation="refresh", mode=result.market_data_mode, rows=result.rows)
                st.session_state["refresh_message"] = (
                    f"Refreshed {result.rows} positions using {result.market_data_mode} data "
                    f"at {result.captured_at.isoformat()} ({result.degraded_rows} degraded rows)."
                )
                st.rerun()
            except Exception as exc:
                log(LOGGER, 40, "dashboard_refresh_failed", "Exposure refresh failed; previous data preserved", operation="refresh", error_type=type(exc).__name__, error_detail=str(exc))
                st.error(f"Exposure refresh failed; previous data was preserved: {exc!s}")
    if st.session_state.get("refresh_message"):
        st.success(st.session_state["refresh_message"])
    if not Path(exposure_path).exists() or not Path(limits_path).exists():
        st.info("Provide an exposures Parquet file and exposure limits JSON to begin.")
        return
    exposures = pl.read_parquet(exposure_path)
    limits = load_limits(limits_path)
    state = dashboard_frame(exposures, limits)
    row = state.row(0, named=True)
    enabled_limits = [(metric, limit) for metric, limit in limits.items() if limit.enabled]
    card_metrics = ["net_delta", "vega", "theta", "delta_notional", "gross_notional", "net_notional", "long_premium", "short_premium"]
    cards = st.columns(4)
    for index, metric in enumerate(card_metrics):
        limit = limits.get(metric)
        with cards[index % len(cards)]:
            value, status = row.get(metric), row.get(metric + "_status", "NO_DATA")
            display_unit = limit.unit if limit is not None else "CONTRACTS"
            if limit is None:
                limit_text = "Signed contracts · not controlled"
            else:
                limit_text = f"Target {format_exposure_value(limit.target, limit.unit)} · {status}"
            st.markdown(f'''<div class="card"><div class="card-label">{escape(METRIC_LABELS.get(metric, metric))}</div>
            <div class="card-value" style="color:{status_color(status)}">{escape(format_exposure_value(value, display_unit))}</div>
            <div class="card-sub">{escape(limit_text)}</div></div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Range monitor</div><div class="muted">Current portfolio exposure and permitted control ranges</div>', unsafe_allow_html=True)
    st.markdown("".join(_range_html(metric, row, limit) for metric, limit in enabled_limits), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Portfolio components</div>', unsafe_allow_html=True)
    dimension = st.selectbox("Breakdown", ["underlying", "strategy", "book", "expiry", "dte", "side"])
    breakdown = evaluate_limits(aggregate_exposures(exposures, dimension), limits)
    left, right = st.columns(2)
    with left:
        st.markdown(f'<div class="panel"><div class="section-title">{escape(dimension.title())} breakdown</div></div>', unsafe_allow_html=True)
        st.dataframe(breakdown, column_config=dashboard_column_config(breakdown, st), use_container_width=True, hide_index=True)
    metric = st.selectbox("Position contribution metric", [name for name, _ in enabled_limits])
    contributions = exposures.sort(metric, descending=True)
    with right:
        st.markdown(f'<div class="panel"><div class="section-title">Position contribution</div><div class="muted">Sorted by {escape(METRIC_LABELS.get(metric, metric))}</div></div>', unsafe_allow_html=True)
        st.dataframe(contributions, column_config=dashboard_column_config(contributions, st), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
