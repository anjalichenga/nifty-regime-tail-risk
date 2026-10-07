"""Numerically stable forward-only causal HMM filtering recursion in log-space."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def compute_emission_log_densities(
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str = "full",
) -> np.ndarray:
    """
    Computes emission log-likelihoods log P(x_t | S_t = k) for each state k.
    X: shape (T, d)
    means: shape (K, d)
    covars: shape (K, d, d) if full, or (K, d) if diag
    Returns: shape (T, K)
    """
    T, d = X.shape
    K = means.shape[0]
    log_emissions = np.zeros((T, K))

    for k in range(K):
        mu_k = means[k]
        if covariance_type == "full":
            cov_k = covars[k]
            # Ensure symmetry and positive definiteness
            cov_k = 0.5 * (cov_k + cov_k.T) + 1e-6 * np.eye(d)
            dist = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
            log_emissions[:, k] = dist.logpdf(X)
        elif covariance_type == "diag":
            cov_diag = np.clip(covars[k], 1e-6, None)
            cov_k = np.diag(cov_diag)
            dist = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
            log_emissions[:, k] = dist.logpdf(X)
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return log_emissions

def forward_filter_probabilities(
    X: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """
    CRITICAL CAUSAL REGIME FILTER:
    Implements exact forward recursion in log space:
        alpha_t(k) = P(S_t = k | x_{1..t})

    Properties:
    - Purely forward-looking: alpha_t depends ONLY on x_{1..t}.
    - Appending future data x_{t+1..T} has ZERO effect on alpha_t.
    - Prevents the lookahead bias inherent in smoothed probabilities (predict_proba).

    Returns:
    - filtered_probs: array of shape (T, K) where each row sums to 1.0
    - log_likelihood_steps: array of shape (T,) of step log-likelihoods
    """
    T, _d = X.shape
    K = means.shape[0]

    # Precompute all emission log-densities: shape (T, K)
    log_emissions = compute_emission_log_densities(X, means, covars, covariance_type)

    filtered_probs = np.zeros((T, K))
    log_likelihood_steps = np.zeros(T)

    # Initial step t = 0
    safe_start = np.clip(startprob, 1e-12, None)
    log_prior_0 = np.log(safe_start / np.sum(safe_start))
    log_joint_0 = log_prior_0 + log_emissions[0]
    log_norm_0 = logsumexp(log_joint_0)
    filtered_probs[0] = np.exp(log_joint_0 - log_norm_0)
    log_likelihood_steps[0] = log_norm_0

    # Transition matrix in log space
    safe_trans = np.clip(transmat, 1e-12, None)
    safe_trans = safe_trans / safe_trans.sum(axis=1, keepdims=True)
    log_trans = np.log(safe_trans)

    # Recursive forward updates
    for t in range(1, T):
        # Predict: P(S_t = k | x_{1..t-1}) = sum_j alpha_{t-1}(j) * A_{jk}
        # In log space: logsumexp_j (log(alpha_{t-1}(j)) + log A_{jk})
        log_alpha_prev = np.log(np.clip(filtered_probs[t - 1], 1e-12, None))
        log_pred = logsumexp(log_alpha_prev[:, np.newaxis] + log_trans, axis=0)

        # Update: log alpha_t(k) = log_pred(k) + log b_k(x_t)
        log_joint = log_pred + log_emissions[t]
        log_norm = logsumexp(log_joint)

        filtered_probs[t] = np.exp(log_joint - log_norm)
        log_likelihood_steps[t] = log_norm

    # Re-normalize for exact numerical sum=1
    filtered_probs = filtered_probs / np.sum(filtered_probs, axis=1, keepdims=True)
    return filtered_probs, log_likelihood_steps
