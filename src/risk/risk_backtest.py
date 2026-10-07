"""Regulatory risk backtesting: Kupiec POF, Christoffersen conditional coverage, and McNeil-Frey ES test."""
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof_test(breaches: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """
    Kupiec Proportion of Failures (POF) test for unconditional coverage.
    H0: True breach probability p == alpha.
    Test statistic LR_pof ~ Chi2(1).
    """
    b = np.asarray(breaches, dtype=int)
    n = len(b)
    x = int(np.sum(b))
    p_hat = x / n if n > 0 else 0.0

    if x == 0:
        # 0 breaches
        lr_stat = -2.0 * n * np.log(1.0 - alpha)
        pvalue = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
        return {
            "n_obs": n,
            "n_breaches": x,
            "nominal_rate": alpha,
            "observed_rate": 0.0,
            "lr_stat": float(lr_stat),
            "pvalue": pvalue,
            "reject_null": bool(pvalue < 0.05),
        }

    # Constrained vs unconstrained likelihood ratio
    log_l_null = (n - x) * np.log(1.0 - alpha) + x * np.log(alpha)
    log_l_alt = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
    lr_stat = max(0.0, -2.0 * (log_l_null - log_l_alt))

    pvalue = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    return {
        "n_obs": n,
        "n_breaches": x,
        "nominal_rate": alpha,
        "observed_rate": float(p_hat),
        "lr_stat": float(lr_stat),
        "pvalue": pvalue,
        "reject_null": bool(pvalue < 0.05),
    }

def christoffersen_independence_test(breaches: np.ndarray) -> dict[str, Any]:
    """
    Christoffersen test for independence of exceptions.
    H0: Exceptions are independent across time (no clustering).
    Test statistic LR_ind ~ Chi2(1).
    """
    b = np.asarray(breaches, dtype=int)
    n = len(b)
    if n < 2:
        return {"lr_stat": 0.0, "pvalue": 1.0, "reject_null": False}

    # Transition counts
    n00 = int(np.sum((b[:-1] == 0) & (b[1:] == 0)))
    n01 = int(np.sum((b[:-1] == 0) & (b[1:] == 1)))
    n10 = int(np.sum((b[:-1] == 1) & (b[1:] == 0)))
    n11 = int(np.sum((b[:-1] == 1) & (b[1:] == 1)))

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n - 1) if (n - 1) > 0 else 0.0

    # Avoid log(0)
    eps = 1e-10
    pi = np.clip(pi, eps, 1.0 - eps)
    pi_01 = np.clip(pi_01, eps, 1.0 - eps)
    pi_11 = np.clip(pi_11, eps, 1.0 - eps)

    # Log likelihood under independence
    log_l_ind = (n00 + n10) * np.log(1.0 - pi) + (n01 + n11) * np.log(pi)

    # Log likelihood under 1st order Markov chain
    log_l_dep = n00 * np.log(1.0 - pi_01) + n01 * np.log(pi_01) + n10 * np.log(1.0 - pi_11) + n11 * np.log(pi_11)

    lr_stat = max(0.0, -2.0 * (log_l_ind - log_l_dep))
    pvalue = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": float(pi_01), "pi_11": float(pi_11),
        "lr_stat": float(lr_stat),
        "pvalue": pvalue,
        "reject_null": bool(pvalue < 0.05),
    }

def mcneil_frey_es_test(
    losses: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    cond_vols: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    McNeil and Frey (2000) bootstrap test for Expected Shortfall adequacy.
    On breach days (L_t > VaR_t), computes normalized excess loss:
    e_t = (L_t - ES_t) / sigma_t
    H0: Mean of e_t is zero (ES accurately estimates the conditional tail mean).
    """
    L = np.asarray(losses)
    var = np.asarray(var_forecasts)
    es = np.asarray(es_forecasts)
    sigma = np.asarray(cond_vols)

    valid = ~np.isnan(L) & ~np.isnan(var) & ~np.isnan(es) & ~np.isnan(sigma)
    L = L[valid]
    var = var[valid]
    es = es[valid]
    sigma = np.clip(sigma[valid], 1e-4, None)

    breach_mask = (L > var)
    n_breaches = int(np.sum(breach_mask))

    if n_breaches < 5:
        return {
            "n_breaches": n_breaches,
            "mean_excess_loss": 0.0,
            "t_stat": 0.0,
            "pvalue": 1.0,
            "pass_es": True,
        }

    e_t = (L[breach_mask] - es[breach_mask]) / sigma[breach_mask]
    mean_e = float(np.mean(e_t))
    se_e = float(np.std(e_t, ddof=1) / np.sqrt(n_breaches))
    t_stat = mean_e / se_e if se_e > 1e-8 else 0.0

    # 1-sided bootstrap test: H0: E[e_t] == 0 vs H1: E[e_t] > 0 (underestimating tail severity)
    rng = np.random.default_rng(seed)
    centered_e = e_t - mean_e
    boot_samples = rng.choice(centered_e, size=(n_bootstrap, n_breaches), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    pvalue = float(np.mean(boot_means >= mean_e))

    return {
        "n_breaches": n_breaches,
        "mean_excess_loss": mean_e,
        "t_stat": float(t_stat),
        "pvalue": pvalue,
        "pass_es": bool(pvalue >= 0.05),
    }

def evaluate_all_var_models(
    returns_pct: pd.Series,
    var_forecasts_dict: dict[str, pd.Series],
    es_forecasts_dict: dict[str, pd.Series],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluates regulatory compliance across multiple VaR/ES models at significance level alpha:
    - Kupiec POF (unconditional coverage)
    - Christoffersen Independence (exception clustering)
    - Joint Conditional Coverage (POF + Independence)
    - McNeil-Frey Expected Shortfall test
    - Basel Traffic Light classification
    """
    losses = -returns_pct
    rows = []
    horizon_label = f"{(1 - alpha) * 100:.0f}%"

    for model_name, var_series in var_forecasts_dict.items():
        es_series = es_forecasts_dict.get(model_name, var_series)
        valid = ~var_series.isna() & ~losses.isna()
        if valid.sum() < 20:
            continue

        l_sub = losses[valid].values
        v_sub = var_series[valid].values
        es_sub = es_series[valid].values

        breaches = (l_sub > v_sub).astype(int)
        n_obs = len(breaches)
        n_breaches = int(np.sum(breaches))
        obs_rate = n_breaches / n_obs if n_obs > 0 else 0.0

        # 1. Kupiec POF
        kupiec = kupiec_pof_test(breaches, alpha=alpha)

        # 2. Christoffersen Independence
        christ = christoffersen_independence_test(breaches)

        # 3. Joint Conditional Coverage
        lr_cc = kupiec["lr_stat"] + christ["lr_stat"]
        p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

        # 4. McNeil-Frey ES test
        es_res = mcneil_frey_es_test(l_sub, v_sub, es_sub, cond_vols=np.ones_like(l_sub))

        # 5. Basel Traffic Light status (annualized 250-day equivalent)
        # For alpha = 0.01 (99%): <= 4 green, 5-9 yellow, >= 10 red
        # Rate equivalents: <= 1.6% green, <= 3.6% yellow, > 3.6% red
        if alpha <= 0.02:
            rate_250 = obs_rate * 250
            if rate_250 <= 4.9:
                traffic_light = "Green"
            elif rate_250 <= 9.9:
                traffic_light = "Yellow"
            else:
                traffic_light = "Red"
        else:
            # 95% horizon
            if kupiec["pvalue"] >= 0.05:
                traffic_light = "Green"
            elif kupiec["pvalue"] >= 0.01:
                traffic_light = "Yellow"
            else:
                traffic_light = "Red"

        rows.append({
            "Confidence Horizon": horizon_label,
            "Model": model_name,
            "Nominal Rate": alpha,
            "Observed Rate": float(obs_rate),
            "Breaches": n_breaches,
            "Kupiec LR": float(kupiec["lr_stat"]),
            "Kupiec p-val": float(kupiec["pvalue"]),
            "Christoffersen LR": float(christ["lr_stat"]),
            "Christoffersen p-val": float(christ["pvalue"]),
            "Joint CC p-val": float(p_cc),
            "ES p-val": float(es_res["pvalue"]),
            "Traffic Light": traffic_light,
        })

    return pd.DataFrame(rows)

