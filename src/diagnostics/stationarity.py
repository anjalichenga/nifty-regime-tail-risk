"""Stationarity testing: ADF and KPSS decision matrix."""
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def evaluate_decision_matrix(adf_reject: bool, kpss_reject: bool) -> tuple[str, str]:
    """
    Evaluates unit root vs stationarity decision matrix:
    - ADF-reject (p < 0.05) & KPSS-not-reject (p >= 0.05) -> Stationary
    - ADF-not-reject (p >= 0.05) & KPSS-reject (p < 0.05) -> Unit root (Non-stationary)
    - Both reject -> Inconclusive (Structural breaks, long memory, or trend stationarity)
    - Neither reject -> Inconclusive (Low test power)
    """
    if adf_reject and not kpss_reject:
        return "Stationary", "Both tests agree on stationarity (ADF rejects unit root; KPSS fails to reject stationarity)."
    elif not adf_reject and kpss_reject:
        return "Unit Root", "Both tests agree on non-stationarity / unit root (ADF fails to reject; KPSS rejects stationarity)."
    elif adf_reject and kpss_reject:
        return "Inconclusive (Both Reject)", "Conflict: ADF indicates stationary while KPSS rejects. Often caused by structural breaks or heavy tails."
    else:
        return "Inconclusive (Neither Rejects)", "Conflict: ADF fails to reject unit root while KPSS fails to reject stationarity. Often caused by low power."

def run_stationarity_tests(series: pd.Series, name: str) -> dict[str, Any]:
    """
    Runs ADF (c and ct) and KPSS (c and ct) on a given time series.
    Returns test statistics, p-values, critical values, and joint interpretation.
    """
    clean_series = series.dropna()

    # ADF with constant ('c') and constant + trend ('ct')
    adf_c = adfuller(clean_series, regression="c", autolag="AIC")
    adf_ct = adfuller(clean_series, regression="ct", autolag="AIC")

    # KPSS with constant ('c') and trend ('ct')
    # Suppress potential InterpolationWarning from statsmodels when p-value is outside table
    import warnings

    from statsmodels.tools.sm_exceptions import InterpolationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_c = kpss(clean_series, regression="c", nlags="auto")
        kpss_ct = kpss(clean_series, regression="ct", nlags="auto")

    adf_c_reject = bool(adf_c[1] < 0.05)
    kpss_c_reject = bool(kpss_c[1] < 0.05)
    decision_c, desc_c = evaluate_decision_matrix(adf_c_reject, kpss_c_reject)

    adf_ct_reject = bool(adf_ct[1] < 0.05)
    kpss_ct_reject = bool(kpss_ct[1] < 0.05)
    decision_ct, desc_ct = evaluate_decision_matrix(adf_ct_reject, kpss_ct_reject)

    return {
        "series_name": name,
        "n_obs": len(clean_series),
        "adf_c": {
            "stat": float(adf_c[0]),
            "pvalue": float(adf_c[1]),
            "usedlag": int(adf_c[2]),
            "crit_values": {k: float(v) for k, v in adf_c[4].items()},
            "reject_5pct": adf_c_reject,
        },
        "adf_ct": {
            "stat": float(adf_ct[0]),
            "pvalue": float(adf_ct[1]),
            "usedlag": int(adf_ct[2]),
            "crit_values": {k: float(v) for k, v in adf_ct[4].items()},
            "reject_5pct": adf_ct_reject,
        },
        "kpss_c": {
            "stat": float(kpss_c[0]),
            "pvalue": float(kpss_c[1]),
            "lags": int(kpss_c[2]),
            "crit_values": {k: float(v) for k, v in kpss_c[3].items()},
            "reject_5pct": kpss_c_reject,
        },
        "kpss_ct": {
            "stat": float(kpss_ct[0]),
            "pvalue": float(kpss_ct[1]),
            "lags": int(kpss_ct[2]),
            "crit_values": {k: float(v) for k, v in kpss_ct[3].items()},
            "reject_5pct": kpss_ct_reject,
        },
        "decision_constant": decision_c,
        "interpretation_constant": desc_c,
        "decision_trend": decision_ct,
        "interpretation_trend": desc_ct,
    }
