"""GARCH and GJR-GARCH conditional volatility modeling and causal frozen-parameter filtering."""
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from src.utils.logging import get_logger

logger = get_logger(__name__)

ANNUALIZATION = 15.874507866387544 # sqrt(252)

def fit_garch_model(
    returns_pct: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 0, # o=1 for GJR-GARCH asymmetry
    dist: str = "t", # 'normal', 't', 'skewt'
    mean_type: str = "Constant",
):
    """
    Fits a univariate GARCH/GJR-GARCH model using the 'arch' package on percentage returns.
    """
    clean_r = returns_pct.dropna()
    am = arch_model(
        clean_r,
        mean=mean_type,  # type: ignore
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,  # type: ignore
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    return am, res

def extract_garch_parameters(fit_result, is_gjr: bool = False) -> dict[str, Any]:
    """
    Extracts GARCH coefficients, persistence, unconditional variance, and half-life.
    For GARCH(1,1): persistence = alpha + beta.
    For GJR-GARCH(1,1,1): persistence = alpha + beta + gamma / 2.
    """
    params = fit_result.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    gamma = float(params.get("gamma[1]", 0.0)) if is_gjr else 0.0

    if is_gjr:
        persistence = alpha + beta + (0.5 * gamma)
    else:
        persistence = alpha + beta

    # Unconditional variance: omega / (1 - persistence) in percentage squared
    if 0 < persistence < 1.0:
        uncond_var = omega / (1.0 - persistence)
        # Half life of volatility shocks: ln(0.5) / ln(persistence)
        half_life = np.log(0.5) / np.log(persistence)
    else:
        uncond_var = np.nan
        half_life = np.nan

    nu = float(params.get("nu", np.nan)) # degrees of freedom for Student-t

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "persistence": float(persistence),
        "unconditional_variance": float(uncond_var),
        "half_life_days": float(half_life),
        "degrees_of_freedom": nu,
        "bic": float(fit_result.bic),
        "aic": float(fit_result.aic),
        "log_likelihood": float(fit_result.loglikelihood),
    }

def filter_with_frozen_parameters(
    returns_pct: pd.Series,
    frozen_params: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "t",
) -> tuple[pd.Series, pd.Series, float, np.ndarray]:
    """
    CRITICAL CAUSAL FILTERING:
    Reconstructs the GARCH model on the extended history up to date t and evaluates
    conditional variances using FROZEN parameters fitted at the previous refit date.
    Zero future parameter lookahead.

    Returns:
    - cond_vol_daily: conditional standard deviation sigma_t (in %)
    - std_resid: standardized residuals z_t = (r_t - mu_t) / sigma_t
    - next_day_sigma: 1-step-ahead forecast sigma_{t+1|t} (in %)
    - forecast_10d_variance: 10-day forward variance forecasts
    """
    clean_r = returns_pct.dropna()
    am = arch_model(
        clean_r,
        mean="Constant",
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,  # type: ignore
        rescale=False,
    )
    # Filter with fixed parameters
    fixed_res = am.fix(frozen_params)
    
    cond_vol = fixed_res.conditional_volatility
    std_residuals = fixed_res.std_resid

    # Produce 1-step and 10-day variance forecasts
    fc = fixed_res.forecast(horizon=10, reindex=False)
    # The variance forecast for the next day
    next_day_var = float(fc.variance.iloc[-1, 0])
    next_day_sigma = float(np.sqrt(next_day_var))
    forecast_10d_var = fc.variance.iloc[-1].values

    return cond_vol, std_residuals, next_day_sigma, forecast_10d_var

def compute_ewma_volatility(returns_pct: pd.Series, decay: float = 0.94) -> pd.Series:
    """
    Computes RiskMetrics EWMA volatility (lambda = 0.94) as a benchmark.
    sigma_t^2 = (1 - lambda) * r_{t-1}^2 + lambda * sigma_{t-1}^2
    Strictly causal (using shift(1) for realized shock).
    """
    r_sq = (returns_pct.shift(1) ** 2).dropna()
    ewma_var = r_sq.ewm(alpha=(1.0 - decay), adjust=False).mean()
    # Align back to original index
    ewma_std = (ewma_var ** 0.5).reindex(returns_pct.index)
    return ewma_std
