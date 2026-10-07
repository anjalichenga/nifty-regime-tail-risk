"""Technical indicator features: Moving average relations and volume z-scores."""

import numpy as np
import pandas as pd


def compute_technical_features(
    close: pd.Series,
    volume: pd.Series | None = None,
    volume_enabled: bool = True,
    sma_short: int = 50,
    sma_long: int = 200,
    volume_window: int = 60,
) -> pd.DataFrame:
    """
    Computes:
    - ln(P / SMA_50)
    - ln(SMA_50 / SMA_200)
    - volume z-score over 60 days (if volume_enabled)
    """
    df = pd.DataFrame(index=close.index)

    # 7. Moving-average relations
    sma_50 = close.rolling(window=sma_short, min_periods=sma_short).mean()
    sma_200 = close.rolling(window=sma_long, min_periods=sma_long).mean()

    df["ln_p_sma50"] = np.log((close / sma_50).clip(lower=1e-6))
    df["ln_sma50_sma200"] = np.log((sma_50 / sma_200).clip(lower=1e-6))

    # 9. Volume z-score
    if volume_enabled and volume is not None and (volume > 0).sum() > 0:
        log_vol = np.log(volume.replace(0, np.nan))
        vol_mean = log_vol.rolling(window=volume_window, min_periods=volume_window).mean()
        vol_std = log_vol.rolling(window=volume_window, min_periods=volume_window).std(ddof=1)
        df["volume_z_60"] = (log_vol - vol_mean) / vol_std.clip(lower=1e-6)
    else:
        df["volume_z_60"] = np.nan

    return df
