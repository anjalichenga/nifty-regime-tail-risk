"""Value-at-Risk (VaR) and Expected Shortfall (ES) multi-model estimators."""
import numpy as np
import pandas as pd
from scipy import stats


def compute_historical_simulation_var_es(
    returns_pct: pd.Series,
    window: int = 750,
    q: float = 0.95,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling historical simulation (causal 750-day window).
    Loss L = -returns_pct.
    VaR_q = empirical q-th quantile of losses over trailing window.
    ES_q = mean of losses exceeding VaR_q.
    Both returned as positive loss percentages.
    """
    losses = -returns_pct
    # Trailing rolling quantile
    var_series = losses.rolling(window=window, min_periods=max(window // 2, 100)).quantile(q)

    # Rolling ES: computed causally over window
    def calc_es(sub_losses: np.ndarray) -> float:
        cutoff = np.percentile(sub_losses, q * 100.0)
        tail = sub_losses[sub_losses >= cutoff]
        return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)

    # Use rolling apply for ES
    es_series = losses.rolling(window=window, min_periods=max(window // 2, 100)).apply(calc_es, raw=True)
    return var_series, es_series

def compute_conditional_normal_var_es(
    mu_pct: float,
    sigma_pct: float,
    q: float = 0.95,
) -> tuple[float, float]:
    """
    Parametric Conditional Normal VaR and ES:
    VaR_q = -mu + sigma * z_q
    ES_q = -mu + sigma * phi(z_q) / (1 - q)
    """
    z_q = stats.norm.ppf(q)
    var = -mu_pct + sigma_pct * z_q
    es = -mu_pct + sigma_pct * (stats.norm.pdf(z_q) / (1.0 - q))
    return float(var), float(es)

def compute_conditional_student_t_var_es(
    mu_pct: float,
    sigma_pct: float,
    nu: float,
    q: float = 0.95,
) -> tuple[float, float]:
    """
    Parametric Conditional Student-t VaR and ES:
    Standardized Student-t quantile t_{nu, q}.
    Scale correction: std of standardized t is 1, so t quantile scaled by sqrt((nu-2)/nu).
    """
    safe_nu = max(nu, 2.1)
    scale_factor = np.sqrt((safe_nu - 2.0) / safe_nu)
    t_q = stats.t.ppf(q, df=safe_nu) * scale_factor

    var = -mu_pct + sigma_pct * t_q

    # Analytical Student-t ES:
    # ES_q = (g(t_q) / (1-q)) * ((nu + t_q^2) / (nu - 1))
    t_unscaled = stats.t.ppf(q, df=safe_nu)
    density_q = stats.t.pdf(t_unscaled, df=safe_nu)
    tail_factor = (density_q / (1.0 - q)) * ((safe_nu + (t_unscaled ** 2)) / (safe_nu - 1.0)) * scale_factor
    es = -mu_pct + sigma_pct * tail_factor

    if es < var:
        es = var * 1.05
    return float(var), float(es)

def compute_filtered_evt_var_es(
    mu_pct: float,
    sigma_pct: float,
    z_q: float,
    es_q_z: float,
) -> tuple[float, float]:
    """
    Filtered EVT conditional risk:
    VaR_{t+1}(q) = -mu_{t+1} + sigma_{t+1} * z_q
    ES_{t+1}(q) = -mu_{t+1} + sigma_{t+1} * ES_q^z
    """
    var = -mu_pct + sigma_pct * z_q
    es = -mu_pct + sigma_pct * es_q_z
    if es < var:
        es = var * 1.05
    return float(var), float(es)
