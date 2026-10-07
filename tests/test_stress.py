"""Tests for Market Stress Index components, aggregation methods, and stress bands."""
import numpy as np
import pandas as pd
import pytest

from src.stress.aggregation import (
    aggregate_msi,
    classify_stress_band,
    compute_component_contributions,
    fit_pca_weights,
    generate_stress_alarms,
)
from src.stress.components import build_msi_components


def test_msi_components_and_bounds():
    """Verify MSI components are strictly in [0.0, 1.0]."""
    N = 100
    hmm_probs = np.random.uniform(0, 1, N)
    garch_vols = np.random.uniform(0.5, 3.0, N)
    drawdowns = -np.random.uniform(0, 0.25, N)
    anom_pcts = np.random.uniform(0, 1, N)

    comp_df, flags_df = build_msi_components(
        hmm_stress_probs=hmm_probs,
        garch_vols=garch_vols,
        drawdowns_252=drawdowns,
        anomaly_percentiles=anom_pcts,
        train_end_idx=80,
    )

    assert comp_df.shape == (N, 4)
    assert (comp_df >= 0.0).all().all()
    assert (comp_df <= 1.0).all().all()
    assert "flag_vol_exceeds" in flags_df.columns

def test_msi_aggregation_methods():
    """Verify W-EQ, W-PCA, and band classifications."""
    comp_df = pd.DataFrame(
        {
            "c_hmm": [0.1, 0.5, 0.9],
            "c_vol": [0.2, 0.6, 0.8],
            "c_drawdown": [0.0, 0.4, 0.7],
            "c_anomaly": [0.1, 0.5, 0.8],
        }
    )

    # 1. W-EQ
    msi_eq = aggregate_msi(comp_df, method="W-EQ")
    assert len(msi_eq) == 3
    assert (msi_eq >= 0.0).all() and (msi_eq <= 100.0).all()
    assert msi_eq.iloc[0] == pytest.approx(10.0, abs=0.1)

    # 2. W-PCA
    pca_w = fit_pca_weights(comp_df)
    msi_pca = aggregate_msi(comp_df, method="W-PCA", weights=pca_w)
    assert len(msi_pca) == 3
    assert (msi_pca >= 0.0).all() and (msi_pca <= 100.0).all()

    # 3. Stress Bands
    bands = classify_stress_band(msi_eq)
    assert bands.iloc[0] == "Normal"
    assert bands.iloc[1] in ("Elevated", "High Risk")
    assert bands.iloc[2] in ("High Risk", "Severe")

    # 4. Alarms
    alarms = generate_stress_alarms(msi_eq, threshold_primary=50.0)
    assert bool(alarms["alarm_primary"].iloc[0]) is False
    assert bool(alarms["alarm_primary"].iloc[2]) is True

    # 5. Component contributions sum to MSI
    contribs = compute_component_contributions(comp_df, weights=None)
    np.testing.assert_allclose(contribs.sum(axis=1), msi_eq, rtol=1e-5)
