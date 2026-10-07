"""Data ingestion via yfinance with caching, retries, and manual fallback."""
import time

import pandas as pd
import yfinance as yf

from src.data.cache import get_cached_raw, save_cached_raw, ticker_to_slug
from src.utils.logging import get_logger
from src.utils.paths import DATA_RAW_MANUAL_DIR

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

def clean_yfinance_dataframe(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flattens MultiIndex columns from yfinance and ensures required OHLCV columns."""
    df = raw_df.copy()

    # Handle MultiIndex columns (e.g. ('Close', '^NSEI') or ('^NSEI', 'Close'))
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            # Pick the string matching standard OHLCV names
            chosen = None
            for part in col:
                part_clean = str(part).strip()
                if part_clean in REQUIRED_COLUMNS or part_clean == "Adj Close":
                    chosen = part_clean
                    break
            flat_cols.append(chosen if chosen else str(col[0]))
        df.columns = flat_cols

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "Volume":
                df["Volume"] = 0.0
            else:
                raise ValueError(f"Missing required price column '{col}' for {ticker}")

    df = df[REQUIRED_COLUMNS].copy()

    # Ensure index is timezone-naive datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    # Drop duplicates keeping last
    df = df[~df.index.duplicated(keep="last")]
    df.attrs["ticker"] = ticker
    return df

def load_manual_csv(ticker: str) -> pd.DataFrame | None:
    """Attempts to load user-supplied manual CSV from data/raw/manual/{ticker}.csv."""
    slug = ticker_to_slug(ticker)
    candidate_paths = [
        DATA_RAW_MANUAL_DIR / f"{slug}.csv",
        DATA_RAW_MANUAL_DIR / f"{ticker}.csv",
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # Locate date column
                date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), None)
                if not date_col:
                    logger.warning("Manual CSV %s lacks a 'Date' column.", p)
                    continue
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                return clean_yfinance_dataframe(df, ticker)
            except Exception as e:
                logger.warning("Error reading manual CSV %s: %s", p, e)
    return None

def fetch_single_ticker(
    ticker: str,
    start_date: str = "2007-09-01",
    end_date: str | None = None,
    refresh: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """
    Fetches raw OHLCV data for a ticker with 3 retries, cache fallback, and manual CSV fallback.
    """
    if not refresh:
        cached = get_cached_raw(ticker)
        if cached is not None and not cached.empty:
            return clean_yfinance_dataframe(cached, ticker)

    # Attempt download via yfinance with retries
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (Attempt %d/%d)...", ticker, attempt, max_retries)
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )
            if df is not None and not df.empty:
                cleaned = clean_yfinance_dataframe(df, ticker)
                save_cached_raw(ticker, cleaned)
                return cleaned
            logger.warning("Empty data returned for %s on attempt %d", ticker, attempt)
        except Exception as e:
            logger.warning("yfinance download failed for %s (attempt %d): %s", ticker, attempt, e)

        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2.0

    # Fallback to cache if download failed
    logger.info("Download failed for %s. Checking cached data...", ticker)
    cached = get_cached_raw(ticker)
    if cached is not None and not cached.empty:
        return clean_yfinance_dataframe(cached, ticker)

    # Fallback to manual CSV
    logger.info("Checking manual CSV fallback in %s for %s...", DATA_RAW_MANUAL_DIR, ticker)
    manual = load_manual_csv(ticker)
    if manual is not None and not manual.empty:
        logger.info("Loaded manual CSV for %s", ticker)
        save_cached_raw(ticker, manual)
        return manual

    return None

def fetch_all_data(
    primary_ticker: str = "^NSEI",
    optional_tickers: list[str] | None = None,
    start_date: str = "2007-09-01",
    end_date: str | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetches primary and optional series. Raises RuntimeError if primary ticker fails.
    Optional series degrade gracefully by logging a warning and skipping.
    """
    if optional_tickers is None:
        optional_tickers = []

    primary_df = fetch_single_ticker(
        primary_ticker, start_date=start_date, end_date=end_date, refresh=refresh
    )
    if primary_df is None or primary_df.empty:
        raise RuntimeError(
            f"FATAL: Primary asset {primary_ticker} could not be downloaded and has no cache or manual CSV. "
            f"Please check your internet connection or place '{primary_ticker}.csv' in {DATA_RAW_MANUAL_DIR}."
        )

    all_data = {primary_ticker: primary_df}

    for ticker in optional_tickers:
        sec_df = fetch_single_ticker(
            ticker, start_date=start_date, end_date=end_date, refresh=refresh
        )
        if sec_df is not None and not sec_df.empty:
            all_data[ticker] = sec_df
        else:
            logger.warning("Optional asset %s unavailable. Pipeline will proceed without it.", ticker)

    return all_data
