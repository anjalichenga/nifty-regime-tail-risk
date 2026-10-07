"""Regime statistics, stationary distributions, and forward transition projections."""
import numpy as np
import pandas as pd
from scipy import stats


def compute_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """
    Computes the stationary distribution pi such that pi * A = pi and sum(pi) = 1.
    Solves (A^T - I) pi = 0 subject to sum(pi) = 1.
    """
    K = transmat.shape[0]
    A = transmat.T - np.eye(K)
    # Add normalization constraint sum(pi) = 1
    A = np.vstack([A, np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0

    # Solve least squares
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, 1.0)
    return pi / np.sum(pi)

def compute_expected_durations(transmat: np.ndarray) -> np.ndarray:
    """
    Theoretical expected duration in each state: E[D_k] = 1 / (1 - A_kk).
    """
    K = transmat.shape[0]
    durations = np.zeros(K)
    for k in range(K):
        p_stay = transmat[k, k]
        durations[k] = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else np.nan
    return durations

def compute_empirical_regime_stats(
    filtered_map_regimes: np.ndarray,
    returns: np.ndarray,
    drawdowns: np.ndarray,
    regime_names: list[str],
    annualization_factor: float = 15.874507866387544,
) -> pd.DataFrame:
    """
    Calculates empirical state statistics conditioned on filtered MAP regime assignment:
    - Annualized mean return
    - Annualized volatility
    - Skewness
    - Kurtosis
    - Mean drawdown
    - Occupancy fraction
    - 1-day 95% Historical VaR within state (positive loss magnitude)
    """
    K = len(regime_names)
    total_n = len(filtered_map_regimes)
    records = []

    for k in range(K):
        mask = (filtered_map_regimes == k)
        n_k = int(np.sum(mask))
        occ = n_k / total_n if total_n > 0 else 0.0

        if n_k > 5:
            r_k = returns[mask]
            dd_k = drawdowns[mask]

            ann_mean = float(np.mean(r_k) * 252.0 * 100.0)
            ann_vol = float(np.std(r_k, ddof=1) * annualization_factor * 100.0)
            skew = float(stats.skew(r_k))
            kurt = float(stats.kurtosis(r_k))
            mean_dd = float(np.mean(dd_k) * 100.0)
            # Historical 95% VaR: 5th percentile of returns as positive loss %
            var_95 = float(-np.percentile(r_k, 5.0) * 100.0)
        else:
            ann_mean = np.nan
            ann_vol = np.nan
            skew = np.nan
            kurt = np.nan
            mean_dd = np.nan
            var_95 = np.nan

        records.append({
            "regime_index": k,
            "regime_name": regime_names[k],
            "occupancy_pct": float(round(occ * 100.0, 2)),
            "count_days": n_k,
            "annualized_mean_return_pct": float(round(ann_mean, 2)),
            "annualized_vol_pct": float(round(ann_vol, 2)),
            "skewness": float(round(skew, 2)),
            "excess_kurtosis": float(round(kurt, 2)),
            "mean_drawdown_pct": float(round(mean_dd, 2)),
            "hist_var_95_pct": float(round(var_95, 2)),
        })

    return pd.DataFrame(records)

def compute_forward_stress_probabilities(
    filtered_probs: np.ndarray,
    transmat: np.ndarray,
    stress_state_idx: int = -1,
    horizons: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Computes forward stress probabilities:
    P(S_{t+h} = Stress | x_{1..t}) = alpha_t * A^h
    """
    if horizons is None:
        horizons = [1, 5, 10]
    if stress_state_idx < 0:
        stress_state_idx = transmat.shape[0] - 1

    forward_probs = {}
    for h in horizons:
        # Matrix power A^h
        A_h = np.linalg.matrix_power(transmat, h)
        # alpha_t * A^h -> shape (T, K)
        projected = filtered_probs @ A_h
        forward_probs[f"p_stress_fwd_{h}d"] = projected[:, stress_state_idx]

    return forward_probs
