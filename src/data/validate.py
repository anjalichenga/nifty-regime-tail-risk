"""Data quality validation, OHLC sanity checking, and report generation."""
import json
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import OUTPUTS_DIR, REPORTS_DIR

logger = get_logger(__name__)

def validate_ohlc_consistency(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Validates:
    - Close > 0
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Low > 0
    Rows violating these are flagged and excluded from OHLC-derived features (flag set to False).
    Never silently 'fixed'.
    """
    valid_mask = pd.Series(True, index=df.index)
    violations: list[str] = []

    non_positive = df["Close"] <= 0
    if non_positive.any():
        dates = df.index[non_positive].strftime("%Y-%m-%d").tolist()
        violations.append(f"{ticker}: {len(dates)} rows with Close <= 0: {dates[:5]}")
        valid_mask &= ~non_positive

    high_invalid = df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-6
    if high_invalid.any():
        dates = df.index[high_invalid].strftime("%Y-%m-%d").tolist()
        violations.append(f"{ticker}: {len(dates)} rows with High < max(Open, Close): {dates[:5]}")
        valid_mask &= ~high_invalid

    low_invalid = df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-6
    if low_invalid.any():
        dates = df.index[low_invalid].strftime("%Y-%m-%d").tolist()
        violations.append(f"{ticker}: {len(dates)} rows with Low > min(Open, Close): {dates[:5]}")
        valid_mask &= ~low_invalid

    low_non_positive = df["Low"] <= 0
    if low_non_positive.any():
        dates = df.index[low_non_positive].strftime("%Y-%m-%d").tolist()
        violations.append(f"{ticker}: {len(dates)} rows with Low <= 0: {dates[:5]}")
        valid_mask &= ~low_non_positive

    validated_df = df.copy()
    validated_df["valid_ohlc"] = valid_mask
    return validated_df, violations

def generate_data_quality_report(
    datasets: dict[str, pd.DataFrame],
    primary_ticker: str = "^NSEI",
    min_volume_coverage: float = 0.90,
) -> dict[str, Any]:
    """
    Analyzes datasets and writes reports to outputs/data_quality.json and reports/data_quality.md.
    """
    report: dict[str, Any] = {
        "primary_ticker": primary_ticker,
        "tickers": {},
        "summary": {},
    }

    large_moves_all: list[dict[str, Any]] = []
    primary_volume_ok = True

    for ticker, df in datasets.items():
        n_rows = len(df)
        first_date = str(df.index.min().date()) if n_rows > 0 else "N/A"
        last_date = str(df.index.max().date()) if n_rows > 0 else "N/A"

        # Check duplicates
        n_dups = df.index.duplicated().sum()

        # Check returns and large moves
        returns = np.log(df["Close"] / df["Close"].shift(1))
        large_moves = returns[returns.abs() > 0.10]
        large_move_records = [
            {"date": str(d.date()), "return_pct": float(r * 100)}
            for d, r in large_moves.items()
        ]
        if ticker == primary_ticker:
            large_moves_all = large_move_records

        # Check volume coverage (treat Volume == 0 as missing)
        valid_vol_count = (df["Volume"] > 0).sum() if "Volume" in df.columns else 0
        vol_coverage = valid_vol_count / n_rows if n_rows > 0 else 0.0

        if ticker == primary_ticker and vol_coverage < min_volume_coverage:
            primary_volume_ok = False
            logger.warning(
                "Primary asset %s valid volume coverage is %.1f%% (< %.1f%%). Disabling volume features.",
                ticker, vol_coverage * 100, min_volume_coverage * 100
            )

        # Check calendar gaps (greater than 4 calendar days)
        day_diffs = (df.index[1:] - df.index[:-1]).days
        n_large_gaps = int((day_diffs > 4).sum())

        report["tickers"][ticker] = {
            "rows": int(n_rows),
            "first_date": first_date,
            "last_date": last_date,
            "duplicates": int(n_dups),
            "volume_coverage_pct": float(round(vol_coverage * 100, 2)),
            "large_moves_count": len(large_move_records),
            "large_moves": large_move_records,
            "large_calendar_gaps": n_large_gaps,
        }

    report["summary"] = {
        "primary_volume_features_enabled": primary_volume_ok,
        "primary_large_moves_count": len(large_moves_all),
        "primary_total_observations": report["tickers"].get(primary_ticker, {}).get("rows", 0),
    }

    # Write outputs/data_quality.json
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUTS_DIR / "data_quality.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write reports/data_quality.md
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "data_quality.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality & Ingestion Audit Report\n\n")
        f.write(f"- **Primary Asset:** `{primary_ticker}`\n")
        f.write(f"- **Volume Features Enabled:** `{primary_volume_ok}`\n\n")
        f.write("## Series Summary\n\n")
        f.write("| Ticker | Start Date | End Date | Total Rows | Valid Volume % | Large Moves (|r| > 10%) | Gaps > 4d |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for ticker, info in report["tickers"].items():
            f.write(
                f"| `{ticker}` | {info['first_date']} | {info['last_date']} | {info['rows']:,} | "
                f"{info['volume_coverage_pct']}% | {info['large_moves_count']} | {info['large_calendar_gaps']} |\n"
            )

        f.write("\n## Large Moves (|Return| > 10%)\n\n")
        if large_moves_all:
            f.write("| Date | Return (%) | Note |\n|---|---|---|\n")
            f.writelines(f"| {lm['date']} | {lm['return_pct']:+.2f}% | Valid market shock (retained) |\n" for lm in large_moves_all)
        else:
            f.write("No moves exceeding 10% detected.\n")

    logger.info("Saved data quality report to %s and %s", json_path, md_path)
    return report
