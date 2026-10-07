"""Forecast evaluation metrics: Diebold-Mariano test and QLIKE loss."""

import numpy as np
from scipy import stats


def compute_qlike_loss(realized_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """
    Computes QLIKE loss for volatility forecasts:
    L(sigma_sq, h) = sigma_sq / h - ln(sigma_sq / h) - 1
    where sigma_sq is the realized proxy (e.g. r^2) and h is the forecasted variance.
    """
    rv = np.clip(realized_var, 1e-8, None)
    fv = np.clip(forecast_var, 1e-8, None)
    return (rv / fv) - np.log(rv / fv) - 1.0

def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_benchmark: np.ndarray,
    h: int = 1,
) -> dict[str, float]:
    """
    Computes the Diebold-Mariano test statistic with Harvey, Leybourne, and Newbold (1997)
    small-sample correction for h-step-ahead forecasts.

    Null hypothesis H0: Both forecasts have equal predictive accuracy (mean loss differential is 0).
    Negative DM stat means the model has lower loss than the benchmark.
    """
    e1 = np.asarray(errors_model)
    e2 = np.asarray(errors_benchmark)
    valid_mask = ~np.isnan(e1) & ~np.isnan(e2)
    e1 = e1[valid_mask]
    e2 = e2[valid_mask]

    n = len(e1)
    if n < 10:
        return {"dm_stat": 0.0, "pvalue": 1.0, "mean_diff": 0.0}

    # Loss differential (e.g. squared error difference)
    d = (e1 ** 2) - (e2 ** 2)
    d_mean = np.mean(d)

    # Autocovariance of d up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += 2.0 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 1e-12:
        return {"dm_stat": 0.0, "pvalue": 1.0, "mean_diff": float(d_mean)}

    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold small sample correction factor
    hln_factor = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    corrected_dm = dm_stat * hln_factor

    # Student-t distribution with n-1 degrees of freedom
    pvalue = 2.0 * (1.0 - stats.t.cdf(np.abs(corrected_dm), df=n - 1))

    return {
        "dm_stat": float(corrected_dm),
        "pvalue": float(pvalue),
        "mean_diff": float(d_mean),
    }

def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Computes RMSE, MAE, and directional accuracy."""
    y = np.asarray(actual)
    y_hat = np.asarray(predicted)
    valid = ~np.isnan(y) & ~np.isnan(y_hat)
    y = y[valid]
    y_hat = y_hat[valid]

    if len(y) == 0:
        return {"rmse": 0.0, "mae": 0.0, "dir_acc": 0.5}

    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y - y_hat)))
    # Directional accuracy: sign match
    dir_acc = float(np.mean(np.sign(y) == np.sign(y_hat)))

    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc}
