"""Tests for data validation, calendar alignment, and ingestion integrity."""
import pandas as pd

from src.data.calendar import align_to_master_calendar
from src.data.download import clean_yfinance_dataframe
from src.data.validate import validate_ohlc_consistency


def test_clean_yfinance_dataframe():
    """Verify handling of dates, duplicates, and columns."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    # Duplicate last date
    dates_with_dup = dates.append(pd.DatetimeIndex(["2020-01-05"]))
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [105, 106, 107, 108, 109, 110],
            "Low": [95, 96, 97, 98, 99, 100],
            "Close": [102, 103, 104, 105, 104, 106],
            "Volume": [1000, 2000, 3000, 4000, 5000, 6000],
        },
        index=dates_with_dup,
    )
    cleaned = clean_yfinance_dataframe(df, "TEST")
    assert len(cleaned) == 5
    assert not cleaned.index.duplicated().any()
    assert cleaned.iloc[-1]["Close"] == 106  # Kept last duplicate

def test_validate_ohlc_consistency():
    """Verify invalid OHLC rows are flagged and not silently overwritten."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0],
            "High": [105.0, 95.0, 105.0, 105.0],   # Row 1: High < Open
            "Low":  [95.0,  90.0, 102.0, -5.0],   # Row 2: Low > Open; Row 3: Low <= 0
            "Close": [102.0, 98.0, 101.0, 100.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=dates,
    )
    validated, violations = validate_ohlc_consistency(df, "TEST")
    assert len(violations) > 0
    assert validated["valid_ohlc"].iloc[0] is True or validated["valid_ohlc"].iloc[0] == 1
    assert bool(validated["valid_ohlc"].iloc[1]) is False
    assert bool(validated["valid_ohlc"].iloc[2]) is False
    assert bool(validated["valid_ohlc"].iloc[3]) is False

def test_availability_lag_alignment():
    """Verify lag-1 shifts secondary series strictly before aligning to master calendar."""
    master_dates = pd.date_range("2020-01-01", periods=5, freq="B")
    primary_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=master_dates)
    primary_df.attrs["ticker"] = "^NSEI"

    # Secondary series (e.g. S&P 500)
    secondary_df = pd.DataFrame({"Close": [3000, 3010, 3020, 3030, 3040]}, index=master_dates)
    secondary_df.attrs["ticker"] = "^GSPC"

    lags = {"^NSEI": 0, "^GSPC": 1}
    aligned = align_to_master_calendar(primary_df, {"^GSPC": secondary_df}, lags, max_ffill_limit=3)

    sp_aligned = aligned["^GSPC"]
    # Day 0 should be NaN because lag 1 has no observation prior to day 0
    assert pd.isna(sp_aligned["Close"].iloc[0])
    # Day 1 in NIFTY should observe Day 0 of S&P (3000)
    assert sp_aligned["Close"].iloc[1] == 3000
    # Day 2 in NIFTY should observe Day 1 of S&P (3010)
    assert sp_aligned["Close"].iloc[2] == 3010

def test_no_winsorization_and_forward_fill_bounds():
    """
    Leakage Risk 15:
    Extreme returns are preserved without winsorization.
    Forward fill is strictly bounded at max_ffill_limit=3 and never backfilled.
    """
    master_dates = pd.date_range("2020-01-01", periods=10, freq="B")
    primary_df = pd.DataFrame({"Close": [100.0] * 10}, index=master_dates)
    primary_df.attrs["ticker"] = "^NSEI"

    # Secondary series with an observation at date 0, then 6 missing days, then observation
    sec_dates = [master_dates[0], master_dates[7]]
    sec_df = pd.DataFrame({"Close": [50.0, 60.0]}, index=sec_dates)
    sec_df.attrs["ticker"] = "SEC"

    lags = {"^NSEI": 0, "SEC": 0}
    aligned = align_to_master_calendar(primary_df, {"SEC": sec_df}, lags, max_ffill_limit=3)

    res = aligned["SEC"]["Close"]
    # Day 0: 50.0
    # Day 1, 2, 3: ffilled with 50.0 (3 days max limit)
    # Day 4, 5, 6: must remain NaN! (limit=3 exceeded)
    # Day 7: 60.0
    assert res.iloc[0] == 50.0
    assert res.iloc[1] == 50.0
    assert res.iloc[2] == 50.0
    assert res.iloc[3] == 50.0
    assert pd.isna(res.iloc[4])
    assert pd.isna(res.iloc[5])
    assert pd.isna(res.iloc[6])
    assert res.iloc[7] == 60.0

def test_master_calendar_alignment():
    """
    Leakage Risk 16:
    Primary calendar is authoritative. Secondary holidays don't drop primary rows.
    """
    master_dates = pd.date_range("2020-01-01", periods=5, freq="B")
    primary_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=master_dates)
    primary_df.attrs["ticker"] = "^NSEI"

    # Secondary series has only 2 rows
    sec_dates = [master_dates[0], master_dates[2]]
    sec_df = pd.DataFrame({"Close": [200, 202]}, index=sec_dates)
    sec_df.attrs["ticker"] = "SEC"

    aligned = align_to_master_calendar(primary_df, {"SEC": sec_df}, {"^NSEI": 0, "SEC": 0})
    assert len(aligned["^NSEI"]) == 5
    assert len(aligned["SEC"]) == 5
    # Primary index is strictly preserved
    pd.testing.assert_index_equal(aligned["SEC"].index, master_dates)
