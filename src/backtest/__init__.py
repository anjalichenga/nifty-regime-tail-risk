"""Backtesting modules: event labeling, walk-forward engine, and evaluation."""
from src.backtest.events import (
    compare_alarms_to_baselines,
    evaluate_early_warning_alarms,
    extract_crisis_episodes,
    identify_alarm_spells,
    label_drawdown_events,
    moving_block_bootstrap_metric,
)
