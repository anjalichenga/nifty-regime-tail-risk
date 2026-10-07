"""Cross-asset correlation and implied volatility features."""

import numpy as np
import pandas as pd


def compute_cross_asset_features(
    primary_returns: pd.Series,
    secondary_dfs: dict[str, pd.DataFrame],
    corr_window: int = 60,
    vix_change_window: int = 20,
) -> pd.DataFrame:
    """
    Computes rolling correlations and VIX dynamics.
    Assumes secondary_dfs have ALREADY been aligned to the master calendar with availability lags.
    """
    df = pd.DataFrame(index=primary_returns.index)

    # 1. Bank NIFTY correlation
    if "^NSEBANK" in secondary_dfs and "Close" in secondary_dfs["^NSEBANK"]:
        sec_ret = np.log(secondary_dfs["^NSEBANK"]["Close"]).diff()
        df["corr_banknifty_60"] = primary_returns.rolling(window=corr_window, min_periods=corr_window).corr(sec_ret)
    else:
        df["corr_banknifty_60"] = np.nan

    # 2. S&P 500 correlation (already lagged by 1 day)
    if "^GSPC" in secondary_dfs and "Close" in secondary_dfs["^GSPC"]:
        sec_ret = np.log(secondary_dfs["^GSPC"]["Close"]).diff()
        df["corr_sp500_60"] = primary_returns.rolling(window=corr_window, min_periods=corr_window).corr(sec_ret)
    else:
        df["corr_sp500_60"] = np.nan

    # 3. USD/INR correlation (already lagged by 1 day)
    if "INR=X" in secondary_dfs and "Close" in secondary_dfs["INR=X"]:
        sec_ret = np.log(secondary_dfs["INR=X"]["Close"]).diff()
        df["corr_usdinr_60"] = primary_returns.rolling(window=corr_window, min_periods=corr_window).corr(sec_ret)
    else:
        df["corr_usdinr_60"] = np.nan

    # 4. India VIX features
    if "^INDIAVIX" in secondary_dfs and "Close" in secondary_dfs["^INDIAVIX"]:
        vix_close = secondary_dfs["^INDIAVIX"]["Close"].clip(lower=1e-3)
        df["vix_level"] = vix_close
        log_vix = np.log(vix_close)
        df["vix_change_20"] = log_vix - log_vix.shift(vix_change_window)
    else:
        df["vix_level"] = np.nan
        df["vix_change_20"] = np.nan

    return df
