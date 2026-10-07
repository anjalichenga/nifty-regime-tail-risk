
import numpy as np
import pandas as pd


def compute_volatility_features(
    returns: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    valid_ohlc: pd.Series | None = None,
    annualization_factor: float = 15.874507866387544, # sqrt(252)
) -> pd.DataFrame:
    """
    Computes annualized realized volatilities, Parkinson range volatility,
    downside semi-deviation, vol z-score, and vol ratio.
    """
    df = pd.DataFrame(index=returns.index)

    # 2. Rolling realized volatilities
    for n in [5, 20, 60]:
        # ddof=1 sample standard deviation
        rv = returns.rolling(window=n, min_periods=n).std(ddof=1) * annualization_factor
        df[f"rv_{n}"] = rv

    # Log RV20
    df["ln_rv20"] = np.log(df["rv_20"].clip(lower=1e-6))

    # Volatility term structure ratio: ln(RV_5 / RV_20)
    df["vol_ratio_5_20"] = np.log((df["rv_5"] / df["rv_20"]).clip(lower=1e-6))

    # 3. Range-based Parkinson Volatility
    if high is not None and low is not None:
        # Check OHLC validity mask
        h = high.copy()
        l = low.copy()
        if valid_ohlc is not None:
            h = h.where(valid_ohlc, np.nan)
            l = l.where(valid_ohlc, np.nan)

        ratio = (h / l).clip(lower=1.0)
        daily_parkinson_var = (np.log(ratio) ** 2) / (4.0 * np.log(2.0))
        df["parkinson_daily"] = daily_parkinson_var

        # 20-day rolling mean of daily Parkinson variance, annualized
        rolling_park_var = daily_parkinson_var.rolling(window=20, min_periods=20).mean()
        df["parkinson_20"] = np.sqrt(rolling_park_var) * annualization_factor
        df["ln_parkinson"] = np.log(df["parkinson_20"].clip(lower=1e-6))
    else:
        df["parkinson_daily"] = np.nan
        df["parkinson_20"] = np.nan
        df["ln_parkinson"] = np.nan

    # 4. Downside semi-deviation (20d): sqrt(252) * sqrt(mean(min(r, 0)^2))
    downside_squared = np.minimum(returns, 0.0) ** 2
    downside_mean_20 = downside_squared.rolling(window=20, min_periods=20).mean()
    df["downside_semi_dev"] = np.sqrt(downside_mean_20) * annualization_factor

    # 8. Vol z-score over 252 days: (ln RV_20 - mean_252) / std_252
    ln_rv20_mean = df["ln_rv20"].rolling(window=252, min_periods=252).mean()
    ln_rv20_std = df["ln_rv20"].rolling(window=252, min_periods=252).std(ddof=1)
    df["vol_zscore_252"] = (df["ln_rv20"] - ln_rv20_mean) / ln_rv20_std.clip(lower=1e-6)

    # 11. Rolling skewness and kurtosis (60d) - DESCRIPTIVE ONLY
    df["skew_60_descriptive"] = returns.rolling(window=60, min_periods=60).skew()
    df["kurt_60_descriptive"] = returns.rolling(window=60, min_periods=60).kurt()

    return df
