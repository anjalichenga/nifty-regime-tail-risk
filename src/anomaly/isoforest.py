"""Isolation Forest anomaly detection with fit-on-train ECDF percentile scoring."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class CausalIsolationForest:
    """
    Causal anomaly detector using Isolation Forest.
    - Scaler and Isolation Forest are fit strictly on the training window.
    - Anomaly percentile is computed via the ECDF of training-window scores applied forward.
    - Out-of-sample observations scoring higher than all training samples clip to 1.0.
    """
    def __init__(
        self,
        n_estimators: int = 300,
        contamination: str = "auto",
        percentile_threshold: float = 0.99,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.percentile_threshold = percentile_threshold
        self.seed = seed

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            max_samples="auto",
        )
        self.train_scores_: np.ndarray = np.array([])

    def fit(self, X_train: np.ndarray) -> "CausalIsolationForest":
        """Fits scaler and IsolationForest on the training matrix."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        # Raw score: higher = more anomalous (score_samples returns negative offset)
        raw_train_scores = -self.model.score_samples(X_scaled)
        self.train_scores_ = np.sort(raw_train_scores)
        return self

    def score(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates anomaly scores and ECDF percentiles on X_eval.

        Returns:
        - raw_scores: shape (N,)
        - percentiles: shape (N,) in [0.0, 1.0]
        - flags: boolean shape (N,) where percentile >= threshold
        """
        X_scaled = self.scaler.transform(X_eval)
        raw_scores = -self.model.score_samples(X_scaled)

        # Apply training ECDF: np.searchsorted gives rank in training scores
        ranks = np.searchsorted(self.train_scores_, raw_scores, side="right")
        percentiles = ranks / len(self.train_scores_)
        percentiles = np.clip(percentiles, 0.0, 1.0)

        flags = (percentiles >= self.percentile_threshold)
        return raw_scores, percentiles, flags

def extract_top_anomalies(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    percentile_col: str = "anomaly_percentile",
    top_n: int = 20,
) -> pd.DataFrame:
    """Extracts top N most anomalous dates with contextual features."""
    if score_col not in df.columns:
        return pd.DataFrame()
    top_df = df.sort_values(by=score_col, ascending=False).head(top_n).copy()
    return top_df
