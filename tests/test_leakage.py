"""Static analysis lint test and runtime leakage assertions."""
import re

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.stress.components import TrainFittedECDF
from src.utils.config import compute_config_hash, load_config
from src.utils.paths import PROJECT_ROOT


def test_static_lint_forbidden_patterns():
    """
    CRITICAL STATIC LINT:
    Checks all files in src/features/ to ensure zero instances of forbidden lookahead patterns:
    - center=True
    - shift(-
    - bfill
    - interpolate
    """
    features_dir = PROJECT_ROOT / "src" / "features"
    forbidden_patterns = [
        (re.compile(r"center\s*=\s*True"), "center=True"),
        (re.compile(r"shift\s*\(\s*-[1-9]"), "shift(-k) negative shift"),
        (re.compile(r"\.bfill\("), ".bfill()"),
        (re.compile(r"\.interpolate\("), ".interpolate()"),
    ]

    py_files = list(features_dir.glob("*.py"))
    assert len(py_files) > 0, "No python files found in src/features/"

    violations = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        # Remove triple quoted strings (docstrings)
        cleaned_code = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
        cleaned_code = re.sub(r"'''.*?'''", "", cleaned_code, flags=re.DOTALL)
        lines = cleaned_code.splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, name in forbidden_patterns:
                if pattern.search(line):
                    violations.append(f"{f.name}:{idx} -> {name}: {line.strip()}")

    assert not violations, "Forbidden leakage patterns detected in feature code:\n" + "\n".join(violations)

def test_scaler_no_future_leakage():
    """
    Leakage Risk 1:
    StandardScaler must be fit strictly on training window and evaluate forward
    without modifying mean or scale based on future data.
    """
    np.random.seed(42)
    train_data = np.random.normal(10.0, 2.0, size=(200, 3))
    future_data = np.random.normal(50.0, 10.0, size=(100, 3))  # Severe distribution shift

    scaler = StandardScaler()
    scaler.fit(train_data)
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    # Transform future data
    transformed_future = scaler.transform(future_data)

    # Assert scaler parameters were not modified
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scale_, scale_before)
    # The mean of future transformed data should reflect the true shift (far from 0)
    assert np.all(np.mean(transformed_future, axis=0) > 10.0)

def test_ecdf_fit_on_train_only():
    """
    Leakage Risk 2:
    Empirical CDFs must be computed strictly on the training set.
    Observations beyond historical extremes clip to 1.0 with an exceeds_historical_range flag.
    """
    train_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ecdf = TrainFittedECDF().fit(train_vals)

    test_vals = np.array([0.5, 3.0, 5.0, 10.0])
    pcts, flags = ecdf.transform(test_vals)

    assert pcts[0] == 0.0
    assert pcts[1] == 0.6  # 3 is 3rd of 5 -> rank 3 / 5 = 0.6
    assert pcts[2] == 1.0
    assert pcts[3] == 1.0  # Clipped to 1.0
    assert bool(flags[3]) is True  # Exceeds range flag triggered
    assert bool(flags[1]) is False

def test_frozen_config_hash():
    """
    Leakage Risk 13:
    Config SHA256 hash must be deterministic and verifiable.
    """
    config = load_config()
    hash_1 = compute_config_hash(config)
    hash_2 = compute_config_hash(config)
    assert hash_1 == hash_2
    assert len(hash_1) == 64
