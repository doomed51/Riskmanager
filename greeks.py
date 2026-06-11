from typing import Optional

import QuantLib as ql


def _option_type_from_right(option_right: str) -> Optional[int]:
    right = (option_right or "").upper()
    if right == "C":
        return ql.Option.Call
    if right == "P":
        return ql.Option.Put
    return None


def _build_european_option(option_right: str, strike: float, dte_days: float):
    option_type = _option_type_from_right(option_right)
    if option_type is None:
        return None

    eval_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = eval_date
    maturity_days = max(1, int(round(dte_days)))
    maturity = eval_date + maturity_days

    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(maturity)
    return ql.VanillaOption(payoff, exercise), eval_date


def _build_bsm_process(eval_date: ql.Date, spot: float, rate: float, vol: float):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    risk_free_curve = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, rate, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count))
    vol_surface = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, calendar, vol, day_count))

    return ql.BlackScholesMertonProcess(spot_handle, dividend_curve, risk_free_curve, vol_surface)


def quantlib_option_price(
    option_right: str,
    spot: float,
    strike: float,
    dte_days: float,
    rate: float,
    vol: float,
) -> Optional[float]:
    if spot <= 0 or strike <= 0 or dte_days <= 0 or vol <= 0:
        return None

    built = _build_european_option(option_right=option_right, strike=strike, dte_days=dte_days)
    if built is None:
        return None

    option, eval_date = built
    process = _build_bsm_process(eval_date=eval_date, spot=spot, rate=rate, vol=vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    try:
        return float(option.NPV())
    except RuntimeError:
        return None


def black_scholes_vega(
    spot: float,
    strike: float,
    dte_days: float,
    rate: float,
    vol: float,
) -> Optional[float]:
    if spot <= 0 or strike <= 0 or dte_days <= 0 or vol <= 0:
        return None

    built = _build_european_option(option_right="C", strike=strike, dte_days=dte_days)
    if built is None:
        return None

    option, eval_date = built
    process = _build_bsm_process(eval_date=eval_date, spot=spot, rate=rate, vol=vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    try:
        # Vega returned per 1.00 absolute vol change.
        return float(option.vega())
    except RuntimeError:
        return None


def black_scholes_theta(
    option_right: str,
    spot: float,
    strike: float,
    dte_days: float,
    rate: float,
    vol: float,
) -> Optional[float]:
    if spot <= 0 or strike <= 0 or dte_days <= 0 or vol <= 0:
        return None

    built = _build_european_option(option_right=option_right, strike=strike, dte_days=dte_days)
    if built is None:
        return None

    option, eval_date = built
    process = _build_bsm_process(eval_date=eval_date, spot=spot, rate=rate, vol=vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    try:
        annual_theta = float(option.theta())
    except RuntimeError:
        return None

    # Return theta per day to align with common trading convention used in this repo.
    return annual_theta / 365.0


def choose_preferred_greek(
    ib_value: Optional[float],
    fallback_value: Optional[float],
    mode: str,
) -> Optional[float]:
    mode_lower = (mode or "hybrid").lower()
    if mode_lower == "ib":
        return ib_value
    if mode_lower == "black_scholes":
        return fallback_value

    # Hybrid mode: prefer IB model greek, fallback to local value.
    if ib_value is not None:
        return ib_value
    return fallback_value
