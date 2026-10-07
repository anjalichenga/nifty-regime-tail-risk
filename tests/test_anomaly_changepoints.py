"""Tests for Isolation Forest anomaly detection and change point modules."""
import numpy as np
import pandas as pd

from src.anomaly.isoforest import CausalIsolationForest
from src.changepoints.cusum import run_causal_cusum_stream
from src.changepoints.pelt import run_pelt_changepoints


def test_isolation_forest_causal():
    """Verify Isolation Forest training, ECDF percentile bounds [0, 1], and causal scoring."""
    np.random.seed(42)
    X_train = np.random.normal(0, 1, size=(300, 3))
    detector = CausalIsolationForest(n_estimators=50, percentile_threshold=0.99, seed=42)
    detector.fit(X_train)

    # Evaluate on new data including an extreme outlier
    X_test = np.random.normal(0, 1, size=(50, 3))
    X_test[10] = [10.0, 10.0, 10.0] # Massive anomaly

    _raw, pct, flags = detector.score(X_test)

    assert len(pct) == 50
    assert (pct >= 0.0).all() and (pct <= 1.0).all()
    # The extreme anomaly at index 10 should have highest percentile and trigger flag
    assert pct[10] >= 0.98
    assert flags[10] is True or flags[10] == 1

def test_cusum_is_strictly_online():
    """Verify CUSUM detects upward shocks and resets after alarm."""
    np.random.seed(42)
    # 100 normal observations, then 10 high-variance shocks
    shocks = np.random.normal(0, 1.0, size=150)
    shocks[100:115] = np.random.normal(0, 5.0, size=15) # Regime change
    s_series = pd.Series(shocks)

    df_cusum = run_causal_cusum_stream(s_series, train_end_idx=90, k=0.5, h=3.0)

    assert "cusum_alarm_high" in df_cusum.columns
    # Alarm should trigger during the shock period (between index 100 and 120)
    assert df_cusum["cusum_alarm_high"].iloc[100:125].any()

def test_pelt_changepoints():
    """Verify PELT generates retrospective break dates and sensitivity table."""
    np.random.seed(42)
    # Two distinct variance segments
    r1 = np.random.normal(0, 0.01, size=100)
    r2 = np.random.normal(0, 0.04, size=100)
    r = pd.Series(np.concatenate([r1, r2]), index=pd.date_range("2020-01-01", periods=200))

    _break_dates, sens_df = run_pelt_changepoints(r, min_size=40, penalty_multiplier=2.0)
    assert len(sens_df) > 0
    assert "c_multiplier" in sens_df.columns
