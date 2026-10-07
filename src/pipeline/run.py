"""Master end-to-end pipeline runner for Market Regime Detection & Tail-Risk Forecasting."""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.backtest.walk_forward import run_walk_forward_backtest
from src.changepoints.pelt import run_pelt_changepoints
from src.data.calendar import align_to_master_calendar
from src.data.download import fetch_all_data
from src.data.validate import generate_data_quality_report, validate_ohlc_consistency
from src.diagnostics.acf_pacf import compute_acf_pacf_diagnostics
from src.diagnostics.residuals import (
    compute_residual_diagnostics,
    write_diagnostics_report,
)
from src.diagnostics.stationarity import run_stationarity_tests
from src.features.build import build_feature_dataset
from src.models.garch import fit_garch_model
from src.utils.config import compute_config_hash, load_config
from src.utils.logging import get_logger
from src.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    ensure_directories,
)
from src.utils.render_results import generate_all_reports_and_figures

logger = get_logger(__name__)

def run_pipeline(
    config_path: Path | None = None,
    refresh_data: bool = False,
    quick_test: bool = False,
) -> dict[str, Any]:
    """
    Executes the entire quantitative pipeline from data ingestion to report generation.
    """
    start_time = time.time()
    ensure_directories()

    # 1. Load Configuration
    config = load_config(config_path)
    config_hash = compute_config_hash(config)
    logger.info("Loaded configuration with SHA256: %s", config_hash)

    # 2. Ingest Data
    logger.info("=== STEP 1: Ingesting Market Data ===")
    raw_data = fetch_all_data(
        primary_ticker=config.data.primary_ticker,
        optional_tickers=config.data.optional_tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        refresh=refresh_data,
    )
    primary_ticker = config.data.primary_ticker
    primary_df = raw_data[primary_ticker]
    secondary_raw = {k: v for k, v in raw_data.items() if k != primary_ticker}

    # 3. Validate OHLC and Data Quality
    logger.info("=== STEP 2: Validating Data Quality ===")
    primary_df_validated, violations = validate_ohlc_consistency(primary_df, primary_ticker)
    if violations:
        for v in violations:
            logger.warning("Data quality violation: %s", v)
    raw_data[primary_ticker] = primary_df_validated

    data_quality_metrics = generate_data_quality_report(
        raw_data,
        primary_ticker=primary_ticker,
        min_volume_coverage=config.data.min_volume_coverage,
    )

    # 4. Calendar Alignment with Availability Lags
    logger.info("=== STEP 3: Aligning Calendar & Availability Lags ===")
    aligned_data = align_to_master_calendar(
        primary_df=primary_df_validated,
        secondary_dfs=secondary_raw,
        availability_lags=config.data.availability_lags,
        max_ffill_limit=config.data.max_ffill_limit,
    )
    aligned_secondary = {k: v for k, v in aligned_data.items() if k != primary_ticker}

    # 5. Build Master Features
    logger.info("=== STEP 4: Computing Feature Matrix ===")
    volume_enabled = data_quality_metrics.get("volume_feature_enabled", True)
    features_df = build_feature_dataset(
        primary_df=aligned_data[primary_ticker],
        secondary_dfs=aligned_secondary,
        config=config,
        volume_enabled=volume_enabled,
    )
    logger.info("Master feature dataset shape: %s", features_df.shape)

    # 6. In-Sample Descriptive Diagnostics
    logger.info("=== STEP 5: Running In-Sample Descriptive Diagnostics ===")
    ret_series = features_df["return"].dropna()
    price_series = features_df["Close"].dropna()

    # Stationarity
    stat_price = run_stationarity_tests(price_series, "NIFTY_Price")
    stat_ret = run_stationarity_tests(ret_series, "NIFTY_Log_Returns")

    # Autocorrelation & Volatility Clustering
    acf_results = compute_acf_pacf_diagnostics(ret_series, max_lags=25)

    # Descriptive Full-Sample GARCH
    _, desc_garch_res = fit_garch_model(
        features_df["return_pct"].dropna(),
        p=1, q=1, o=0, dist="t"
    )
    resid_results = compute_residual_diagnostics(desc_garch_res.std_resid)

    # Write diagnostics.md
    write_diagnostics_report(
        stationarity_p=stat_price,
        stationarity_r=stat_ret,
        acf_results=acf_results,
        arch_results=acf_results.get("ljung_box_squared_returns", {}),
        resid_results=resid_results,
    )

    # Retrospective PELT Change Points
    pelt_breaks, _pelt_sens = run_pelt_changepoints(
        ret_series,
        min_size=config.models.changepoints.pelt.min_size,
        penalty_multiplier=config.models.changepoints.pelt.penalty_multiplier,
    )

    # 7. Walk-Forward Backtesting Simulation
    logger.info("=== STEP 6: Running Temporal Walk-Forward Backtest ===")
    wf_results = run_walk_forward_backtest(
        features_df=features_df,
        config=config,
        verbose=True,
    )
    oos_df = wf_results.oos_df

    # 8. Save Outputs & Manifest
    logger.info("=== STEP 7: Persisting Artifacts & Results ===")
    oos_out_path = OUTPUTS_DIR / "oos_results.parquet"
    oos_df.to_parquet(oos_out_path, engine="pyarrow", index=True)

    metrics_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_hash": config_hash,
        "n_samples_total": len(features_df),
        "n_samples_oos": len(oos_df),
        "oos_start_date": oos_df.index[0].strftime("%Y-%m-%d"),
        "oos_end_date": oos_df.index[-1].strftime("%Y-%m-%d"),
        "n_refits": len(wf_results.refit_records),
        "early_warning_metrics": {
            k: v for k, v in wf_results.early_warning_eval.items() if not isinstance(v, list)
        },
        "regulatory_backtest": wf_results.regulatory_backtest.to_dict(orient="records"),
        "baseline_comparison": wf_results.baseline_comparison.to_dict(orient="records"),
        "pelt_changepoint_dates": [d.strftime("%Y-%m-%d") for d in pelt_breaks],
    }

    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    manifest_payload = {
        "run_id": f"run_{int(time.time())}",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution_duration_sec": time.time() - start_time,
        "config_hash": config_hash,
        "python_version": sys.version,
        "data_hashes": data_quality_metrics.get("hashes", {}),
        "artifacts": {
            "features_parquet": str(DATA_PROCESSED_DIR / "features.parquet"),
            "oos_results_parquet": str(oos_out_path),
            "metrics_json": str(metrics_path),
            "results_markdown": str(REPORTS_DIR / "RESULTS.md"),
            "diagnostics_markdown": str(REPORTS_DIR / "diagnostics.md"),
        },
    }
    manifest_path = OUTPUTS_DIR / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # 9. Render RESULTS.md and Figures
    logger.info("=== STEP 8: Rendering RESULTS.md & Publication Figures ===")
    report_path = generate_all_reports_and_figures(wf_results, config_hash=config_hash)

    elapsed = time.time() - start_time
    logger.info("Pipeline completed successfully in %.1f seconds. Report rendered at %s", elapsed, report_path)

    return {
        "status": "success",
        "duration_sec": elapsed,
        "config_hash": config_hash,
        "report_path": str(report_path),
        "oos_rows": len(oos_df),
    }

def main():
    parser = argparse.ArgumentParser(description="Run Market Regime & Tail-Risk Quantitative Pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of market data.")
    parser.add_argument("--quick", action="store_true", help="Run in quick smoke-test mode.")
    args = parser.parse_args()

    cfg_p = Path(args.config) if args.config else None
    run_pipeline(config_path=cfg_p, refresh_data=args.refresh, quick_test=args.quick)

if __name__ == "__main__":
    main()
