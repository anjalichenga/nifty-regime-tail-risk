"""Autocorrelation and volatility clustering diagnostics."""
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf


def compute_acf_pacf_diagnostics(returns: pd.Series, max_lags: int = 25) -> dict[str, Any]:
    """
    Computes ACF and PACF for returns, squared returns, and absolute returns.
    Computes Ljung-Box Q statistics at lags 10 and 20.
    """
    r = returns.dropna()
    r_sq = r ** 2
    r_abs = np.abs(r)

    # 1. ACF & PACF with 95% confidence bands
    acf_r, _conf_r = acf(r, nlags=max_lags, alpha=0.05)
    pacf_r, _conf_pacf_r = pacf(r, nlags=max_lags, alpha=0.05)

    acf_r_sq, _conf_r_sq = acf(r_sq, nlags=max_lags, alpha=0.05)
    _pacf_r_sq, _ = pacf(r_sq, nlags=max_lags, alpha=0.05)

    acf_r_abs, _ = acf(r_abs, nlags=max_lags, alpha=0.05)

    # 2. Ljung-Box tests on r and r_sq at lags 10 and 20
    lb_r = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    lb_r_sq = acorr_ljungbox(r_sq, lags=[10, 20], return_df=True)

    # 3. Interpret clustering
    r_p10 = float(lb_r.loc[10, "lb_pvalue"])
    r_sq_p10 = float(lb_r_sq.loc[10, "lb_pvalue"])
    clustering_confirmed = r_sq_p10 < 0.01

    interpretation = (
        f"Ljung-Box p-values at lag 10: returns p={r_p10:.4f}, squared returns p={r_sq_p10:.4e}. "
    )
    if clustering_confirmed:
        interpretation += (
            "Returns exhibit weak linear autocorrelation, but squared/absolute returns demonstrate "
            "highly significant, persistent autocorrelation. This confirms the presence of volatility "
            "clustering (ARCH effects), justifying explicit conditional heteroskedasticity modeling (GARCH)."
        )
    else:
        interpretation += "Volatility clustering not statistically significant at 1% nominal level."

    return {
        "max_lags": max_lags,
        "acf_r": acf_r.tolist(),
        "pacf_r": pacf_r.tolist(),
        "acf_r_sq": acf_r_sq.tolist(),
        "acf_r_abs": acf_r_abs.tolist(),
        "conf_band_upper": float(1.96 / np.sqrt(len(r))),
        "conf_band_lower": float(-1.96 / np.sqrt(len(r))),
        "ljung_box_returns": {
            "lag_10": {"stat": float(lb_r.loc[10, "lb_stat"]), "pvalue": r_p10},
            "lag_20": {"stat": float(lb_r.loc[20, "lb_stat"]), "pvalue": float(lb_r.loc[20, "lb_pvalue"])},
        },
        "ljung_box_squared_returns": {
            "lag_10": {"stat": float(lb_r_sq.loc[10, "lb_stat"]), "pvalue": r_sq_p10},
            "lag_20": {"stat": float(lb_r_sq.loc[20, "lb_stat"]), "pvalue": float(lb_r_sq.loc[20, "lb_pvalue"])},
        },
        "clustering_confirmed": clustering_confirmed,
        "interpretation": interpretation,
    }

def run_arch_lm_test(residuals: pd.Series, lags: int = 12) -> dict[str, Any]:
    """
    Runs Engle's ARCH Lagrange Multiplier test for conditional heteroskedasticity.
    """
    res = residuals.dropna()
    lm_stat, pvalue, f_stat, f_pvalue = het_arch(res, nlags=lags)
    arch_present = bool(pvalue < 0.05)
    return {
        "lags": lags,
        "lm_stat": float(lm_stat),
        "pvalue": float(pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "arch_effects_detected": arch_present,
        "interpretation": (
            f"ARCH-LM test (p={pvalue:.4e}): "
            + ("Rejects null of homoskedasticity. Strong ARCH effects present."
               if arch_present else "Fails to reject homoskedasticity.")
        ),
    }
