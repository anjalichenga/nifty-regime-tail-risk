"""Tests for statistical diagnostics, ARIMA, and GARCH models."""
import numpy as np
import pandas as pd

from src.diagnostics.stationarity import (
    evaluate_decision_matrix,
    run_stationarity_tests,
)
from src.models.arima import fit_arma_model, select_arma_order
from src.models.garch import (
    extract_garch_parameters,
    filter_with_frozen_parameters,
    fit_garch_model,
)


def test_stationarity_decision_matrix():
    """Verify decision matrix mapping logic."""
    assert evaluate_decision_matrix(True, False)[0] == "Stationary"
    assert evaluate_decision_matrix(False, True)[0] == "Unit Root"
    assert "Both Reject" in evaluate_decision_matrix(True, True)[0]
    assert "Neither Rejects" in evaluate_decision_matrix(False, False)[0]

def test_stationarity_simulated_series():
    """
    Tests stationarity routines on:
    1. Stationary AR(1) with phi=0.5 -> Should indicate Stationary or reject ADF
    2. Random walk with unit root -> Should indicate Unit Root
    """
    np.random.seed(42)
    n = 500
    # Stationary AR(1)
    e = np.random.normal(0, 1, size=n)
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.5 * ar1[i - 1] + e[i]
    ar1_series = pd.Series(ar1)

    res_ar1 = run_stationarity_tests(ar1_series, "AR1")
    assert res_ar1["adf_c"]["reject_5pct"] is True

    # Random walk
    rw = np.cumsum(e)
    rw_series = pd.Series(rw)
    res_rw = run_stationarity_tests(rw_series, "RW")
    assert res_rw["adf_c"]["reject_5pct"] is False

def test_arima_train_window_selection():
    """Verify ARIMA order selection picks a bounded order and produces valid forecasts."""
    np.random.seed(42)
    # Generate weak AR(1)
    e = np.random.normal(0, 0.01, size=300)
    y = np.zeros(300)
    for i in range(1, 300):
        y[i] = 0.2 * y[i - 1] + e[i]
    s = pd.Series(y)

    order, has_const, summary = select_arma_order(s, p_max=2, q_max=2)
    assert len(order) == 3
    assert order[0] <= 2 and order[2] <= 2
    assert "best_bic" in summary

    fit_res = fit_arma_model(s, order=order, include_constant=has_const)
    assert len(fit_res.params) > 0

def test_garch_frozen_filtering():
    """
    Tests GARCH fitting and verifies that frozen-parameter filtering
    evaluates conditional volatilities causally without re-estimation.
    """
    np.random.seed(42)
    n = 500
    # Synthetic GARCH(1,1)
    r = np.random.normal(0, 1.5, size=n)
    r_pct = pd.Series(r, index=pd.date_range("2020-01-01", periods=n))

    # Fit on training portion (first 400 days)
    train_r = r_pct.iloc[:400]
    _am, fit_res = fit_garch_model(train_r, p=1, q=1, dist="normal")
    params = fit_res.params
    extracted = extract_garch_parameters(fit_res, is_gjr=False)
    assert 0.0 < extracted["persistence"] <= 1.05

    # Filter with frozen parameters on extended dataset (up to 450 days)
    test_r = r_pct.iloc[:450]
    cond_vol, std_resid, next_sigma, fcast_10d = filter_with_frozen_parameters(
        test_r, frozen_params=params, p=1, q=1, dist="normal"
    )

    assert len(cond_vol) == 450
    assert len(std_resid) == 450
    assert next_sigma > 0
    assert len(fcast_10d) == 10

    # Ensure conditional volatility of first 400 observations matches frozen filter
    assert np.isclose(cond_vol.iloc[399], fit_res.conditional_volatility.iloc[399], atol=1e-3)
