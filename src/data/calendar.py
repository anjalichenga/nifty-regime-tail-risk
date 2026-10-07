"""Master calendar and cross-asset availability lag alignment."""
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

def align_to_master_calendar(
    primary_df: pd.DataFrame,
    secondary_dfs: dict[str, pd.DataFrame],
    availability_lags: dict[str, int],
    max_ffill_limit: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Aligns all secondary series to the primary series' master trading calendar.

    Rules:
    - Master calendar is the exact DatetimeIndex of the primary series.
    - Never creates rows on non-trading days.
    - Applies availability lag per series:
        - Lag 0: synchronous close on the same trading day (e.g. Bank NIFTY, India VIX).
        - Lag 1: observation at date t is the latest observation strictly BEFORE date t
          on the secondary asset's calendar (e.g. S&P 500, USD/INR, Gold).
    - After applying lag and reindexing to the master calendar, forward-fills with limit=3.
    - Never backfills (bfill) or interpolates.
    """
    master_index = primary_df.index
    aligned_dfs: dict[str, pd.DataFrame] = {primary_df.attrs.get("ticker", "PRIMARY"): primary_df}

    for ticker, df in secondary_dfs.items():
        if df is None or df.empty:
            logger.warning("Secondary series %s is missing or empty. Skipping.", ticker)
            continue

        lag = availability_lags.get(ticker, 0)
        df_sorted = df.sort_index()

        if lag > 0:
            # Shift by 'lag' trading days on its own calendar before aligning to master
            # To ensure observation at NSE date t strictly uses observation dated < t:
            # We can use pd.merge_asof or shift(lag) on its own business days.
            # Using shift(lag) on its own sorted calendar guarantees lag-k observations:
            shifted_df = df_sorted.shift(lag)
        else:
            shifted_df = df_sorted

        # Reindex to master calendar
        reindexed = shifted_df.reindex(master_index)
        
        # Forward fill with strict limit=3
        filled = reindexed.ffill(limit=max_ffill_limit)
        filled.attrs["ticker"] = ticker
        aligned_dfs[ticker] = filled

    return aligned_dfs
