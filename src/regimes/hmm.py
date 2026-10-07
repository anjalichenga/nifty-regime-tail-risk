"""Gaussian Hidden Markov Model estimation and causal forward filtering wrapper."""
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.regimes.filtering import forward_filter_probabilities
from src.regimes.labelling import get_regime_names, sort_regime_states
from src.utils.logging import get_logger

logger = get_logger(__name__)

def compute_hmm_n_params(k: int, d: int, covariance_type: str = "full") -> int:
    """Computes free parameter count for GaussianHMM."""
    n_startprob = k - 1
    n_transmat = k * (k - 1)
    n_means = k * d
    if covariance_type == "full":
        n_covars = k * (d * (d + 1)) // 2
    else:
        n_covars = k * d
    return n_startprob + n_transmat + n_means + n_covars

def compute_hmm_bic(log_likelihood: float, n_samples: int, k: int, d: int, covariance_type: str = "full") -> float:
    """Computes Bayesian Information Criterion for HMM."""
    m = compute_hmm_n_params(k, d, covariance_type)
    return -2.0 * log_likelihood + m * np.log(n_samples)

def fit_single_hmm(
    X: np.ndarray,
    k: int,
    covariance_type: str = "full",
    seed: int = 42,
    min_covar: float = 1e-3,
    warm_start_params: dict[str, np.ndarray] | None = None,
) -> GaussianHMM | None:
    """Fits a single GaussianHMM with given initialization."""
    try:
        model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            n_iter=200,
            tol=1e-4,
            random_state=seed,
            init_params="stmc" if warm_start_params is None else "",
        )
        if warm_start_params is not None:
            model.startprob_ = warm_start_params["startprob"].copy()
            model.transmat_ = warm_start_params["transmat"].copy()
            model.means_ = warm_start_params["means"].copy()
            model.covars_ = warm_start_params["covars"].copy()

        model.fit(X)
        if model.monitor_.converged:
            return model
        return model # Return even if hit max iter
    except Exception as e:
        logger.debug("HMM fit failed with seed %d: %s", seed, e)
        return None

def fit_robust_hmm(
    X: np.ndarray,
    k: int = 3,
    feature_names: list[str] | None = None,
    covariance_type: str = "full",
    base_seed: int = 42,
    n_init: int = 10,
    min_covar: float = 1e-3,
    min_occupancy: float = 0.02,
    warm_start_params: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Fits GaussianHMM over multiple restarts + warm start, keeping the best log-likelihood.
    Re-orders states canonically by ln_rv20 to guarantee consistent regime labels.

    Returns:
    - model_dict: sorted parameters, regime names, transition matrix, BIC, convergence flag
    - filtered_probs: causal forward filtered probabilities on X
    """
    T, d = X.shape
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(d)]

    best_score = -float("inf")
    best_model = None

    # 1. Warm start from previous refit if available
    if warm_start_params is not None:
        ws_model = fit_single_hmm(
            X, k, covariance_type, seed=base_seed, min_covar=min_covar,
            warm_start_params=warm_start_params
        )
        if ws_model is not None:
            score = ws_model.score(X)
            if score > best_score:
                best_score = score
                best_model = ws_model

    # 2. Multi-start random initializations
    for init_i in range(n_init):
        seed = base_seed + 100 * init_i
        candidate = fit_single_hmm(
            X, k, covariance_type, seed=seed, min_covar=min_covar
        )
        if candidate is not None:
            try:
                score = candidate.score(X)
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError(f"HMM fitting failed for K={k} across all {n_init} initializations.")

    # 3. Canonical state reordering by volatility (resolving label switching)
    means = best_model.means_
    covars = best_model.covars_
    transmat = best_model.transmat_
    startprob = best_model.startprob_

    sorted_means, sorted_covars, sorted_transmat, sorted_startprob, sort_order = sort_regime_states(
        means, covars, transmat, startprob, feature_names
    )

    # 4. Causal forward filtering
    filtered_probs, _log_lik_steps = forward_filter_probabilities(
        X,
        startprob=sorted_startprob,
        transmat=sorted_transmat,
        means=sorted_means,
        covars=sorted_covars,
        covariance_type=covariance_type,
    )

    # 5. Check for degenerate states
    empirical_occupancy = np.mean(filtered_probs, axis=0)
    degenerate = bool(np.any(empirical_occupancy < min_occupancy))
    if degenerate:
        logger.warning(
            "HMM K=%d produced degenerate state(s) with occupancy < %.1f%%: %s",
            k, min_occupancy * 100, empirical_occupancy
        )

    bic = compute_hmm_bic(best_score, T, k, d, covariance_type)
    names = get_regime_names(k)

    model_dict = {
        "k": k,
        "covariance_type": covariance_type,
        "regime_names": names,
        "means": sorted_means,
        "covars": sorted_covars,
        "transmat": sorted_transmat,
        "startprob": sorted_startprob,
        "log_likelihood": float(best_score),
        "bic": float(bic),
        "degenerate": degenerate,
        "occupancy": empirical_occupancy.tolist(),
        "sort_order": sort_order.tolist(),
    }

    return model_dict, filtered_probs
