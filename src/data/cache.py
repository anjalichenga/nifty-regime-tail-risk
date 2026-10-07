"""Raw data caching and metadata management."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import DATA_RAW_DIR

logger = get_logger(__name__)

def ticker_to_slug(ticker: str) -> str:
    """Converts a ticker symbol like '^NSEI' to a filesystem-safe slug 'NSEI'."""
    return ticker.replace("^", "").replace("=", "_").replace("/", "_")

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_metadata_path() -> Path:
    return DATA_RAW_DIR / "metadata.json"

def load_cache_metadata() -> dict[str, Any]:
    meta_path = get_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read metadata.json: %s", e)
    return {}

def save_cache_metadata(metadata: dict[str, Any]) -> None:
    meta_path = get_metadata_path()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def get_cached_raw(ticker: str) -> pd.DataFrame | None:
    slug = ticker_to_slug(ticker)
    parquet_path = DATA_RAW_DIR / f"{slug}.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            logger.info("Loaded cached raw data for %s (%d rows)", ticker, len(df))
            return df
        except Exception as e:
            logger.warning("Failed to read parquet cache for %s: %s", ticker, e)
    return None

def save_cached_raw(ticker: str, df: pd.DataFrame) -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = ticker_to_slug(ticker)
    parquet_path = DATA_RAW_DIR / f"{slug}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=True)
    
    sha256 = compute_file_sha256(parquet_path)
    metadata = load_cache_metadata()
    metadata[ticker] = {
        "slug": slug,
        "parquet_file": str(parquet_path.name),
        "sha256": sha256,
        "rows": len(df),
        "first_date": str(df.index.min()),
        "last_date": str(df.index.max()),
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }
    save_cache_metadata(metadata)
    logger.info("Saved cache for %s to %s with SHA256 %s", ticker, parquet_path.name, sha256[:8])
