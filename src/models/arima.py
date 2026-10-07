"""Box-Jenkins ARIMA/ARMA modeling and causal walk-forward return forecasting."""
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.models.forecast_eval import compute_regression_metrics, diebold_mariano_test
from src.utils.logging import get_logger

logger = get_logger(__name__)

def select_arma_order(
    returns: pd.Series,
    p_max: int = 3,
    q_max: int = 3,
    include_constant: bool = True,
) -> tuple[tuple[int, int, int], bool, dict[str, Any]]:
    """
    Identifies optimal ARMA(p, 0, q) order over a bounded grid using BIC.
    Prefers parsimony within delta_BIC < 2.0.
    """
    clean_r = returns.dropna()
    best_bic = float("inf")
    best_aic = float("inf")
    best_order = (0, 0, 0)
    best_trend = "c" if include_constant else "n"
    grid_results: list[dict[str, Any]] = []

    trends = ["c", "n"] if include_constant else ["n"]

    for trend in trends:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    model = ARIMA(clean_r, order=(p, 0, q), trend=trend)
                    fit_res = model.fit()
                    bic = fit_res.bic
                    aic = fit_res.aic
                    grid_results.append({
                        "order": (p, 0, q),
                        "trend": trend,
                        "bic": float(bic),
                        "aic": float(aic),
                        "n_params": p + q + (1 if trend == "c" else 0),
                    })
                    if bic < best_bic:
                        best_bic = bic
                        best_aic = aic
                        best_order = (p, 0, q)
                        best_trend = trend
                except Exception:
                    continue

    # Apply parsimony rule: if a simpler model (fewer params) is within 2.0 BIC of the best, choose it
    candidate_best = [
        res for res in grid_results
        if res["bic"] <= best_bic + 2.0
    ]
    if candidate_best:
        candidate_best.sort(key=lambda x: (x["n_params"], x["bic"]))
        chosen = candidate_best[0]
        best_order = chosen["order"]
        best_trend = chosen["trend"]

    summary = {
        "best_order": best_order,
        "best_trend": best_trend,
        "best_bic": float(best_bic),
        "best_aic": float(best_aic),
        "grid_evaluations": grid_results,
    }
    return best_order, (best_trend == "c"), summary

def fit_arma_model(
    returns: pd.Series,
    order: tuple[int, int, int] = (0, 0, 0),
    include_constant: bool = True,
):
    """Fits an ARMA(p, 0, q) model to returns with given order."""
    trend = "c" if include_constant else "n"
    model = ARIMA(returns.dropna(), order=order, trend=trend)
    return model.fit()

def run_arma_walkforward(
    returns: pd.Series,
    initial_train_days: int = 1260,
    refit_cadence: int = 63,
    reselect_cadence: int = 252,
    p_max: int = 3,
    q_max: int = 3,
) -> dict[str, Any]:
    """
    Executes an expanding window walk-forward 1-step-ahead forecast for returns.
    Order is re-selected strictly within the training window every 252 days.
    Refit occurs every 63 days.
    """
    n = len(returns)
    forecast_dates = []
    actual_returns = []
    arma_forecasts = []
    zero_forecasts = []
    mean_forecasts = []

    current_order = (0, 0, 0)
    current_has_const = True
    last_reselect_idx = -999999
    current_model_fit = None
    last_refit_idx = -999999

    for t in range(initial_train_days, n):
        train_data = returns.iloc[:t].dropna()

        # Reselect order annually (252 days) strictly inside training data
        if t - last_reselect_idx >= reselect_cadence or current_model_fit is None:
            current_order, current_has_const, _ = select_arma_order(
                train_data, p_max=p_max, q_max=q_max
            )
            last_reselect_idx = t
            last_refit_idx = t
            current_model_fit = fit_arma_model(train_data, current_order, current_has_const)
        # Refit every 63 days
        elif t - last_refit_idx >= refit_cadence:
            last_refit_idx = t
            current_model_fit = fit_arma_model(train_data, current_order, current_has_const)

        # 1-step-ahead forecast for index t
        # In ARIMA, forecast(1) gives the out-of-sample prediction using data up to t-1
        try:
            # We can use the fitted parameters or 1-step forecast from the model
            fc = float(current_model_fit.forecast(steps=1).iloc[0])
        except Exception:
            fc = float(train_data.mean())

        actual_r = float(returns.iloc[t])
        mean_fc = float(train_data.mean())

        forecast_dates.append(returns.index[t])
        actual_returns.append(actual_r)
        arma_forecasts.append(fc)
        zero_forecasts.append(0.0)
        mean_forecasts.append(mean_fc)

    actuals = np.array(actual_returns)
    preds = np.array(arma_forecasts)
    zeros = np.array(zero_forecasts)
    means = np.array(mean_forecasts)

    # Evaluation metrics
    metrics_arma = compute_regression_metrics(actuals, preds)
    metrics_zero = compute_regression_metrics(actuals, zeros)
    metrics_mean = compute_regression_metrics(actuals, means)

    dm_vs_zero = diebold_mariano_test(actuals - preds, actuals - zeros, h=1)
    dm_vs_mean = diebold_mariano_test(actuals - preds, actuals - means, h=1)

    df_forecasts = pd.DataFrame(
        {
            "actual_return": actuals,
            "arma_forecast": preds,
            "zero_forecast": zeros,
            "mean_forecast": means,
        },
        index=pd.DatetimeIndex(forecast_dates),
    )

    return {
        "forecast_df": df_forecasts,
        "metrics_arma": metrics_arma,
        "metrics_zero": metrics_zero,
        "metrics_mean": metrics_mean,
        "dm_vs_zero": dm_vs_zero,
        "dm_vs_mean": dm_vs_mean,
        "last_selected_order": current_order,
        "has_constant": current_has_const,
    }
