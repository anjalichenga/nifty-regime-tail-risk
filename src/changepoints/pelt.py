"""Retrospective change-point detection using PELT (Descriptive Layer Only)."""
import numpy as np
import pandas as pd
import ruptures as rpt


def run_pelt_changepoints(
    returns: pd.Series,
    model: str = "normal",
    min_size: int = 40,
    penalty_multiplier: float = 3.0, # c * ln(n)
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """
    Runs retrospective PELT change point analysis on returns.
    STRICTLY IN-SAMPLE / DESCRIPTIVE: Cannot be used for operational signals.

    Returns:
    - break_dates: list of change point timestamps
    - sensitivity_table: DataFrame of # of breakpoints vs penalty multiplier c
    """
    clean_r = returns.dropna()
    n = len(clean_r)
    signal = clean_r.values.reshape(-1, 1)

    # 1. Primary fit with config penalty
    algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal)
    penalty_val = penalty_multiplier * np.log(n)
    break_indices = algo.predict(pen=penalty_val)

    # Convert indices (1-indexed endpoints from ruptures) to dates
    break_dates = []
    for idx in break_indices:
        if idx < n:
            break_dates.append(clean_r.index[idx])

    # 2. Penalty sensitivity analysis
    c_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    sensitivity_records = []
    for c in c_grid:
        pen = c * np.log(n)
        bps = algo.predict(pen=pen)
        # Exclude the final n marker
        n_breaks = len([b for b in bps if b < n])
        sensitivity_records.append({
            "c_multiplier": c,
            "penalty_value": float(pen),
            "num_breakpoints": n_breaks,
        })

    sensitivity_df = pd.DataFrame(sensitivity_records)
    return break_dates, sensitivity_df
