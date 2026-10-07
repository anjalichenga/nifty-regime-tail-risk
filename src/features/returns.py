"""Causal return and momentum feature calculations."""
import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.DataFrame:
    """
    Computes log returns and multi-horizon rolling returns/momentum.
    Strictly causal: trailing rolling windows only, no backward shifts.
    """
    df = pd.DataFrame(index=close.index)
    
    # 1. Daily log return: r_t = ln(P_t / P_{t-1})
    log_p = np.log(close)
    df["return"] = log_p.diff()
    df["return_pct"] = df["return"] * 100.0

    # 6. Rolling returns: sum of r over 5, 20, 60 days
    df["rolling_ret_5"] = df["return"].rolling(window=5, min_periods=5).sum()
    df["momentum_20"] = df["return"].rolling(window=20, min_periods=20).sum()
    df["momentum_60"] = df["return"].rolling(window=60, min_periods=60).sum()

    return df
