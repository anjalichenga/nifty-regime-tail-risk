"""Tests for EVT tail estimation, VaR/ES computation, and regulatory backtests."""
import numpy as np
from scipy import stats

from src.risk.evt import compute_gpd_quantiles_and_es, fit_gpd_tail
from src.risk.risk_backtest import (
    christoffersen_independence_test,
    kupiec_pof_test,
    mcneil_frey_es_test,
)


def test_evt_fit_and_quantiles():
    """Verify GPD fitting, quantile monotonicity, and ES >= VaR."""
    np.random.seed(42)
    # Generate GPD exceedances
    true_xi = 0.2
    true_beta = 1.5
    raw_losses = stats.genpareto.rvs(c=true_xi, scale=true_beta, size=500)

    fit_dict = fit_gpd_tail(raw_losses, q_u=0.85, min_exceedances=30)
    assert 0.0 < fit_dict["beta"] < 5.0
    assert not fit_dict["infinite_mean"]

    # Compute quantiles at 95% and 99%
    q_dict = compute_gpd_quantiles_and_es(fit_dict, q_levels=[0.95, 0.99])

    var_95, es_95 = q_dict[0.95]
    var_99, es_99 = q_dict[0.99]

    # Monotonicity checks
    assert var_99 > var_95
    assert es_95 >= var_95
    assert es_99 >= var_99
    assert es_99 > es_95

def test_evt_xi_zero_branch():
    """Verify xi=0 exponential tail branch executes properly."""
    fit_dict = {
        "u": 1.5,
        "xi": 0.0,
        "beta": 1.0,
        "n_total": 1000,
        "n_u": 100,
        "infinite_mean": False,
    }
    q_dict = compute_gpd_quantiles_and_es(fit_dict, q_levels=[0.95, 0.99])
    var_95, es_95 = q_dict[0.95]
    assert var_95 > 1.5
    assert es_95 >= var_95

def test_kupiec_pof_known_outcomes():
    """Verify Kupiec POF fails to reject nominal breach count and rejects extreme breaches."""
    # 50 breaches in 1000 observations -> exactly 5.0% observed
    breaches_nominal = np.zeros(1000)
    breaches_nominal[:50] = 1

    res_nominal = kupiec_pof_test(breaches_nominal, alpha=0.05)
    assert res_nominal["reject_null"] is False
    assert np.isclose(res_nominal["lr_stat"], 0.0, atol=1e-3)

    # 150 breaches in 1000 observations (15% vs 5% nominal) -> Should strongly reject
    breaches_excess = np.zeros(1000)
    breaches_excess[:150] = 1

    res_excess = kupiec_pof_test(breaches_excess, alpha=0.05)
    assert res_excess["reject_null"] is True
    assert res_excess["pvalue"] < 1e-4

def test_christoffersen_independence():
    """Verify Christoffersen independence test detects clustered breaches."""
    # Clustered breach sequence: 10 consecutive breaches
    clustered = np.zeros(500)
    clustered[100:110] = 1 # Clustered
    res_clustered = christoffersen_independence_test(clustered)
    # Consecutive breaches should yield low p-value (reject independence)
    assert res_clustered["n11"] > 0
    assert res_clustered["reject_null"] is True

def test_mcneil_frey_es():
    """Verify McNeil-Frey Expected Shortfall test."""
    np.random.seed(42)
    n = 200
    losses = np.random.normal(0, 1, size=n)
    var = np.ones(n) * 1.645
    # ES correctly set higher than VaR
    es = np.ones(n) * 2.06
    sigma = np.ones(n)

    res = mcneil_frey_es_test(losses, var, es, sigma, n_bootstrap=200, seed=42)
    assert "pvalue" in res
    assert "mean_excess_loss" in res
