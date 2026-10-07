"""Market Stress Index aggregation methods: Equal Weight, PCA-weighted, and W-LOGIT with purge/embargo."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

STRESS_BANDS = {
    "Normal": (0.0, 25.0),
    "Elevated": (25.0, 50.0),
    "High Risk": (50.0, 75.0),
    "Severe": (75.0, 100.0),
}

def fit_pca_weights(train_components: pd.DataFrame) -> dict[str, float]:
    """
    Fits PCA on training component matrix to derive data-driven weights.
    Weights are proportional to the absolute loadings of the first principal component,
    normalized to sum to 1.0.
    """
    cols = train_components.columns.tolist()
    clean_df = train_components.dropna()
    if len(clean_df) < 10:
        # Fallback to equal weights
        return {col: 1.0 / len(cols) for col in cols}

    pca = PCA(n_components=1, random_state=42)
    pca.fit(clean_df.values)
    loadings = np.abs(pca.components_[0])
    total = np.sum(loadings)
    if total < 1e-8:
        return {col: 1.0 / len(cols) for col in cols}

    norm_weights = loadings / total
    return {col: float(norm_weights[i]) for i, col in enumerate(cols)}

def fit_logit_model(
    train_components: pd.DataFrame,
    event_labels: pd.Series,
    purge_days: int = 10,
    embargo_days: int = 5,
) -> LogisticRegression | None:
    """
    Fits a regularized Logistic Regression model for W-LOGIT.
    Enforces purge of trailing labels within purge_days of the training window boundary,
    plus embargo_days gap to prevent lookahead leakage into out-of-sample evaluation.
    """
    # Align on common index
    common_idx = train_components.dropna().index.intersection(event_labels.dropna().index)
    if len(common_idx) < 50:
        return None

    # Apply purge and embargo: drop the last (purge_days + embargo_days) rows from train
    cutoff_len = len(common_idx) - (purge_days + embargo_days)
    if cutoff_len < 30:
        return None

    train_idx = common_idx[:cutoff_len]
    X_train = train_components.loc[train_idx].values
    y_train = event_labels.loc[train_idx].values.astype(int)

    # Need both classes present
    if len(np.unique(y_train)) < 2:
        return None

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=500,
    )
    clf.fit(X_train, y_train)
    return clf

def aggregate_msi(
    components_df: pd.DataFrame,
    method: str = "W-EQ",
    weights: dict[str, float] | None = None,
    logit_model: LogisticRegression | None = None,
) -> pd.Series:
    """
    Aggregates [0, 1] stress components into a composite [0, 100] Market Stress Index.

    Methods:
    - W-EQ: Equal weighting across available components: 100 * mean(c_k)
    - W-PCA: Linear combination using PCA weights: 100 * sum(w_k * c_k)
    - W-LOGIT: Calibrated probability from logistic model: 100 * P(event = 1 | c)
    """
    cols = components_df.columns.tolist()
    K = len(cols)

    if method == "W-EQ":
        # Equal weights
        msi = components_df.mean(axis=1) * 100.0

    elif method == "W-PCA":
        if weights is None:
            weights = {c: 1.0 / K for c in cols}
        # Re-normalize weights for columns present
        w_arr = np.array([weights.get(c, 1.0 / K) for c in cols])
        w_sum = np.sum(w_arr)
        if w_sum > 0:
            w_arr = w_arr / w_sum
        msi = (components_df[cols].values @ w_arr) * 100.0
        msi = pd.Series(msi, index=components_df.index)

    elif method == "W-LOGIT":
        if logit_model is None:
            # Fallback to W-EQ if logit model is not calibrated
            msi = components_df.mean(axis=1) * 100.0
        else:
            probs = logit_model.predict_proba(components_df.values)[:, 1]
            msi = pd.Series(probs * 100.0, index=components_df.index)

    else:
        raise ValueError(f"Unknown aggregation method: {method}. Choose from ['W-EQ', 'W-PCA', 'W-LOGIT'].")

    return msi.clip(0.0, 100.0)

def classify_stress_band(
    msi_series: pd.Series,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.Series:
    """
    Classifies continuous MSI [0, 100] into categorical stress bands:
    - Normal: [0, 25)
    - Elevated: [25, 50)
    - High Risk: [50, 75)
    - Severe: [75, 100]
    """
    if bands is None:
        bands = STRESS_BANDS

    band_labels = pd.Series("Normal", index=msi_series.index)
    for band_name, (low, high) in bands.items():
        if band_name == "Severe":
            mask = (msi_series >= low) & (msi_series <= high)
        else:
            mask = (msi_series >= low) & (msi_series < high)
        band_labels[mask] = band_name

    return band_labels

def compute_component_contributions(
    components_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Computes additive contribution of each component to the headline MSI.
    For linear combinations (W-EQ, W-PCA):
    contrib_k = 100 * w_k * c_k
    sum_k contrib_k = MSI.
    """
    cols = components_df.columns.tolist()
    K = len(cols)
    if weights is None:
        weights = {c: 1.0 / K for c in cols}

    contrib_dict = {}
    for c in cols:
        w = weights.get(c, 1.0 / K)
        contrib_dict[f"contrib_{c}"] = components_df[c] * w * 100.0

    return pd.DataFrame(contrib_dict, index=components_df.index)

def generate_stress_alarms(
    msi_series: pd.Series,
    threshold_primary: float = 50.0,
    threshold_secondary: float = 75.0,
) -> pd.DataFrame:
    """
    Generates binary alarms from MSI:
    - alarm_primary: MSI >= 50.0 (High Risk or Severe)
    - alarm_secondary: MSI >= 75.0 (Severe)
    """
    return pd.DataFrame(
        {
            "alarm_primary": msi_series >= threshold_primary,
            "alarm_secondary": msi_series >= threshold_secondary,
        },
        index=msi_series.index,
    )
