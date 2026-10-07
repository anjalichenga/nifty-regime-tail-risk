"""Drawdown features: path-dependent cumulative and rolling 252-day variants."""
import pandas as pd


def compute_drawdowns(close: pd.Series, rolling_window: int = 252) -> pd.DataFrame:
    """
    Computes:
    - Path-dependent drawdown from start of series: P_t / max_{s <= t} P_s - 1
    - Rolling 252-day drawdown: P_t / max_{s in [t-251, t]} P_s - 1
    Strictly causal. Returns values <= 0.
    """
    df = pd.DataFrame(index=close.index)
    
    # Cumulative running high from the beginning of available history
    running_max = close.cummax()
    df["drawdown"] = (close / running_max) - 1.0

    # Rolling window drawdown (default 252 trading days)
    rolling_max = close.rolling(window=rolling_window, min_periods=rolling_window).max()
    df[f"drawdown_{rolling_window}"] = (close / rolling_max) - 1.0

    return df
