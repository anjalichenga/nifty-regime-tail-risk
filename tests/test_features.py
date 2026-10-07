"""Tests for feature engineering correctness and causal invariance."""
import numpy as np
import pandas as pd

from src.features.build import build_feature_dataset
from src.features.drawdown import compute_drawdowns
from src.features.returns import compute_returns
from src.features.volatility import compute_volatility_features


def test_returns_known_values():
    """Verify log returns calculation, first value NaN, percentage returns."""
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    ret_df = compute_returns(prices)

    assert pd.isna(ret_df["return"].iloc[0])
    expected_ret_1 = np.log(110.0 / 100.0)
    expected_ret_2 = np.log(99.0 / 110.0)

    assert np.isclose(ret_df["return"].iloc[1], expected_ret_1)
    assert np.isclose(ret_df["return"].iloc[2], expected_ret_2)
    assert np.isclose(ret_df["return_pct"].iloc[1], expected_ret_1 * 100.0)

def test_drawdowns_hand_computed():
    """Verify drawdown properties: DD <= 0, DD == 0 at all-time highs."""
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0, 65.0], index=pd.date_range("2020-01-01", periods=6))
    dd_df = compute_drawdowns(prices, rolling_window=3)

    # All-time high drawdowns
    # running_max: [100, 120, 120, 120, 130, 130]
    # dd:          [0,    0,   -0.25, -0.0833, 0, -0.5]
    assert np.isclose(dd_df["drawdown"].iloc[0], 0.0)
    assert np.isclose(dd_df["drawdown"].iloc[1], 0.0)
    assert np.isclose(dd_df["drawdown"].iloc[2], (90.0 / 120.0) - 1.0)
    assert np.isclose(dd_df["drawdown"].iloc[4], 0.0)
    assert np.isclose(dd_df["drawdown"].iloc[5], (65.0 / 130.0) - 1.0)
    assert (dd_df["drawdown"] <= 1e-12).all()

def test_rolling_volatility():
    """Verify rolling volatility against manual numpy/pandas standard deviation and annualization."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.01, size=100), index=pd.date_range("2020-01-01", periods=100))
    vol_df = compute_volatility_features(rets, annualization_factor=np.sqrt(252))

    # Min periods check
    assert vol_df["rv_5"].iloc[:4].isna().all()
    assert not pd.isna(vol_df["rv_5"].iloc[4])

    manual_5 = rets.iloc[0:5].std(ddof=1) * np.sqrt(252)
    assert np.isclose(vol_df["rv_5"].iloc[4], manual_5)

def test_features_truncation_invariance():
    """
    CRITICAL NO-LOOKAHEAD TEST:
    For any cut point T, features computed on data[:T] MUST strictly equal
    the values computed on the full dataset at indices < T.
    """
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    
    # Generate synthetic prices
    drift = 0.0005
    vol = 0.015
    ret_shocks = np.random.normal(drift, vol, size=n)
    prices = 10000.0 * np.exp(np.cumsum(ret_shocks))
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, size=n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, size=n)))
    volumes = np.random.uniform(1e6, 5e6, size=n)

    full_df = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )
    full_df["valid_ohlc"] = True

    # Compute features on full dataset
    feat_full = build_feature_dataset(full_df, volume_enabled=True)

    # Pick random cut points T
    cut_points = [100, 180, 250]
    cols_to_check = [
        "return", "rolling_ret_5", "momentum_20", "rv_5", "rv_20",
        "ln_rv20", "vol_ratio_5_20", "drawdown", "ln_p_sma50"
    ]

    for T in cut_points:
        sub_df = full_df.iloc[:T].copy()
        feat_sub = build_feature_dataset(sub_df, volume_enabled=True)

        for col in cols_to_check:
            sub_vals = feat_sub[col].dropna()
            full_vals = feat_full[col].iloc[:T].loc[sub_vals.index]
            diff = np.abs(sub_vals.values - full_vals.values)
            assert np.nanmax(diff) < 1e-9, f"Lookahead leakage detected in column {col} at cut {T}!"
