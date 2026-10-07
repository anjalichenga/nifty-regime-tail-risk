"""Market Stress Index component generation with train-fit ECDF transforms."""
import numpy as np
import pandas as pd


class TrainFittedECDF:
    """
    Empirical Cumulative Distribution Function fitted on training data
    and applied causally forward.
    Values exceeding training max clip to 1.0; below min clip to 0.0.
    """
    def __init__(self):
        self.sorted_train_: np.ndarray = np.array([])

    def fit(self, x_train: np.ndarray) -> "TrainFittedECDF":
        clean = x_train[~np.isnan(x_train)]
        self.sorted_train_ = np.sort(clean)
        return self

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms values into percentiles in [0.0, 1.0].
        Returns: (percentiles, exceeds_range_flag)
        """
        if len(self.sorted_train_) == 0:
            return np.zeros_like(x), np.zeros_like(x, dtype=bool)

        n = len(self.sorted_train_)
        ranks = np.searchsorted(self.sorted_train_, x, side="right")
        pct = ranks / n
        pct_clipped = np.clip(pct, 0.0, 1.0)
        exceeds_flag = (x > self.sorted_train_[-1])
        return pct_clipped, exceeds_flag

def build_msi_components(
    hmm_stress_probs: np.ndarray,
    garch_vols: np.ndarray,
    drawdowns_252: np.ndarray,
    anomaly_percentiles: np.ndarray,
    cross_asset_corrs: np.ndarray | None = None,
    train_end_idx: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds [0, 1] normalized stress components c_k:
    1. HMM Stress Probability: (already [0, 1]; identity)
    2. Volatility Stress: ECDF of GARCH annualized conditional vol (train-fitted)
    3. Drawdown Stress: ECDF of |DD_252| (train-fitted)
    4. Anomaly Stress: Isolation Forest percentile (already [0, 1])
    5. Optional Cross-Asset Correlation Stress: ECDF of correlation (train-fitted)

    Returns:
    - components_df: DataFrame of columns c_k in [0, 1]
    - exceeds_flag_df: DataFrame of boolean flags where values exceed training max
    """
    N = len(hmm_stress_probs)
    t_end = train_end_idx if train_end_idx is not None else N

    # 1. HMM Stress Probability
    c_hmm = np.clip(hmm_stress_probs, 0.0, 1.0)

    # 2. Volatility ECDF
    ecdf_vol = TrainFittedECDF().fit(garch_vols[:t_end])
    c_vol, flag_vol = ecdf_vol.transform(garch_vols)

    # 3. Drawdown Severity ECDF: |DD_252|
    abs_dd = np.abs(drawdowns_252)
    ecdf_dd = TrainFittedECDF().fit(abs_dd[:t_end])
    c_dd, flag_dd = ecdf_dd.transform(abs_dd)

    # 4. Anomaly Percentile
    c_anom = np.clip(anomaly_percentiles, 0.0, 1.0)

    comp_dict = {
        "c_hmm": c_hmm,
        "c_vol": c_vol,
        "c_drawdown": c_dd,
        "c_anomaly": c_anom,
    }
    flag_dict = {
        "flag_vol_exceeds": flag_vol,
        "flag_dd_exceeds": flag_dd,
    }

    # 5. Optional cross-asset correlation
    if cross_asset_corrs is not None and not np.all(np.isnan(cross_asset_corrs[:t_end])):
        ecdf_corr = TrainFittedECDF().fit(cross_asset_corrs[:t_end])
        c_corr, flag_corr = ecdf_corr.transform(cross_asset_corrs)
        comp_dict["c_cross_asset"] = c_corr
        flag_dict["flag_corr_exceeds"] = flag_corr

    components_df = pd.DataFrame(comp_dict)
    flags_df = pd.DataFrame(flag_dict)

    return components_df, flags_df
