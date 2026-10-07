"""State re-ordering and canonical regime labeling to resolve label switching."""
import numpy as np

REGIME_NAMES_MAP = {
    2: ["Calm", "Stress"],
    3: ["Calm", "Elevated", "Stress"],
    4: ["Calm", "Moderate", "Elevated", "Stress"],
    5: ["Calm", "Moderate", "Elevated", "Stress", "Turbulent"],
}

def get_regime_names(k: int) -> list[str]:
    """Returns canonical regime names for a given number of states K."""
    if k in REGIME_NAMES_MAP:
        return REGIME_NAMES_MAP[k]
    return [f"State_{i}" for i in range(k)]

def sort_regime_states(
    means: np.ndarray,
    covars: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolves label-switching across refits:
    Reorders states strictly by ascending state-mean of ln(RV_20).
    Ties broken by the state variance of the 'return' feature.

    Returns:
    - sorted_means: shape (K, d)
    - sorted_covars: shape (K, d, d) or (K, d)
    - sorted_transmat: shape (K, K)
    - sorted_startprob: shape (K,)
    - sort_order: indices mapping original states to sorted states
    """
    K, _d = means.shape

    # Find index of ln_rv20
    if "ln_rv20" in feature_names:
        vol_col_idx = feature_names.index("ln_rv20")
    else:
        vol_col_idx = 0  # fallback

    vol_means = means[:, vol_col_idx]

    # Find index of return for tie breaking
    if "return" in feature_names:
        ret_col_idx = feature_names.index("return")
    else:
        ret_col_idx = 0

    if covars.ndim == 3:
        ret_vars = covars[:, ret_col_idx, ret_col_idx]
    else:
        ret_vars = covars[:, ret_col_idx]

    # Sort keys: primary = vol_means, secondary = ret_vars
    sort_keys = [
        (vol_means[i], ret_vars[i], i)
        for i in range(K)
    ]
    sort_keys.sort(key=lambda x: (x[0], x[1]))
    sort_order = np.array([x[2] for x in sort_keys])

    # Reorder parameters
    sorted_means = means[sort_order]
    sorted_covars = covars[sort_order]
    sorted_startprob = startprob[sort_order]

    # Transition matrix re-indexing: A_new[i, j] = A_old[order[i], order[j]]
    sorted_transmat = transmat[np.ix_(sort_order, sort_order)]

    return sorted_means, sorted_covars, sorted_transmat, sorted_startprob, sort_order
