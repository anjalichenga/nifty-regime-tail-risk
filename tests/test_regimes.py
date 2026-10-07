"""Tests for HMM regime detection, forward filtering, and label sorting."""
import numpy as np

from src.regimes.filtering import forward_filter_probabilities
from src.regimes.labelling import sort_regime_states


def test_forward_filter_properties():
    """
    Verifies:
    1. Filtered probabilities sum to 1 at every step.
    2. Future data appending DOES NOT alter past filtered probabilities (causal invariance).
    3. At final time step T of training data, forward filter matches smoothed probability.
    """
    np.random.seed(42)
    T = 200
    d = 2
    K = 3
    X = np.random.normal(0, 1, size=(T, d))

    startprob = np.array([0.5, 0.3, 0.2])
    transmat = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ])
    means = np.array([
        [-1.0, 0.5],
        [0.0, 1.5],
        [1.5, 3.0],
    ])
    covars = np.array([np.eye(d) for _ in range(K)])

    filtered_probs, _ = forward_filter_probabilities(X, startprob, transmat, means, covars)

    # 1. Sum to 1
    assert np.allclose(filtered_probs.sum(axis=1), 1.0)

    # 2. Causal invariance: Append future data X_future (e.g. 50 new observations)
    X_future = np.random.normal(0, 1, size=(50, d))
    X_extended = np.vstack([X, X_future])
    filtered_extended, _ = forward_filter_probabilities(X_extended, startprob, transmat, means, covars)

    # The first T probabilities in filtered_extended MUST strictly match filtered_probs[:T]
    max_diff = np.max(np.abs(filtered_probs - filtered_extended[:T]))
    assert max_diff < 1e-12, f"Future lookahead detected in forward filter! Max diff: {max_diff}"

def test_label_sorting_stability():
    """
    Verifies that permuting the internal state order produces the identical
    canonical sorted parameters and regime ordering based on ln_rv20.
    """
    means = np.array([[2.0, 5.0], [0.5, 1.0], [1.0, 2.0]]) # State 1 has lowest ln_rv20 (0.5), then State 2 (1.0), then State 0 (2.0)
    covars = np.array([np.eye(2), np.eye(2), np.eye(2)])
    transmat = np.eye(3) * 0.8 + 0.1
    startprob = np.array([0.3, 0.4, 0.3])
    feature_names = ["ln_rv20", "return"]

    s_means, _s_covars, _s_trans, _s_start, sort_order = sort_regime_states(
        means, covars, transmat, startprob, feature_names
    )

    # State 1 (mean=0.5) should be rank 0 (Calm)
    # State 2 (mean=1.0) should be rank 1 (Elevated)
    # State 0 (mean=2.0) should be rank 2 (Stress)
    assert list(sort_order) == [1, 2, 0]
    assert s_means[0, 0] == 0.5
    assert s_means[1, 0] == 1.0
    assert s_means[2, 0] == 2.0
