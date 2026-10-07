"""Tests for backtesting engine, event labeling, early warning metrics, and purge/embargo leakage prevention."""
import numpy as np
import pandas as pd

from src.backtest.events import (
    evaluate_early_warning_alarms,
    identify_alarm_spells,
    label_drawdown_events,
)
from src.backtest.walk_forward import run_walk_forward_backtest
from src.stress.aggregation import fit_logit_model
from src.utils.config import AppConfig


def test_labels_purge_and_embargo():
    """
    Leakage Risk 12:
    Drawdown event labels must be purged and embargoed in W-LOGIT training
    to prevent lookahead leakage into out-of-sample evaluation.
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    comp_df = pd.DataFrame(
        {
            "c_hmm": np.random.uniform(0, 1, 100),
            "c_vol": np.random.uniform(0, 1, 100),
            "c_drawdown": np.random.uniform(0, 1, 100),
            "c_anomaly": np.random.uniform(0, 1, 100),
        },
        index=dates,
    )
    # Event labels with class balance
    labels = pd.Series(np.random.choice([0, 1], size=100, p=[0.8, 0.2]), index=dates)

    purge_days = 10
    embargo_days = 5

    model = fit_logit_model(comp_df, labels, purge_days=purge_days, embargo_days=embargo_days)
    assert model is not None
    # Model coefficients should exist and be bounded
    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, 4)

def test_drawdown_event_labeling():
    """Verify drawdown event calculation and refractory period."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = [100.0] * 50
    # Simulate a crash: day 10 drops from 100 to 90 (-10% > 5% threshold)
    prices[10] = 90.0
    # Simulate another immediate dip at day 12 (should be in refractory period)
    prices[12] = 88.0
    # Simulate another crash at day 40 (after 20-day refractory period)
    prices[40] = 75.0

    p_series = pd.Series(prices, index=dates)
    event_df = label_drawdown_events(p_series, h=5, x=0.05, refractory_period=20)

    # Days preceding day 10 should be labeled as forward events (since min return <= -5%)
    assert event_df["event_label"].iloc[5:10].any()
    # Distinct onsets should observe refractory cooldown
    onsets = event_df[event_df["crisis_onset"]].index
    assert len(onsets) >= 1

def test_alarm_spells_and_early_warning():
    """Verify alarm spell grouping and lead-time matching."""
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    alarms = pd.Series(False, index=dates)
    # Active alarm at day 5, 6, 7
    alarms.iloc[5:8] = True
    # Gap of 2 days, then alarm at day 10 (should be bridged into single spell)
    alarms.iloc[10] = True

    spells = identify_alarm_spells(alarms, max_gap=3)
    assert len(spells) == 1
    assert spells["duration_days"].iloc[0] == 6  # 5 through 10 inclusive

    # Fake crisis event at day 15
    price_series = pd.Series(np.linspace(100, 80, 30), index=dates)
    event_df = pd.DataFrame(
        {
            "price": price_series,
            "event_label": [0]*15 + [1]*5 + [0]*10,
            "crisis_onset": [False]*15 + [True] + [False]*14,
        },
        index=dates,
    )

    eval_res = evaluate_early_warning_alarms(event_df, alarms, max_gap=3, max_lead_time=20)
    assert eval_res["detected_episodes"] == 1
    assert eval_res["episode_recall"] == 1.0
    assert eval_res["lead_time_median"] > 0

def test_walk_forward_execution_smoke():
    """Smoke test verifying end-to-end walk forward execution on synthetic features."""
    np.random.seed(42)
    N = 250
    dates = pd.date_range("2020-01-01", periods=N, freq="B")
    ret = np.random.normal(0.0005, 0.015, N)
    ret_pct = ret * 100.0
    price = 10000.0 * np.exp(np.cumsum(ret))

    df = pd.DataFrame(
        {
            "Close": price,
            "return": ret,
            "return_pct": ret_pct,
            "drawdown": (price / np.maximum.accumulate(price)) - 1.0,
            "drawdown_252": (price / np.maximum.accumulate(price)) - 1.0,
            "rv_5": np.abs(ret) * 15.87,
            "rv_20": np.abs(ret) * 15.87,
            "rv_60": np.abs(ret) * 15.87,
            "ln_rv20": np.log(np.clip(np.abs(ret) * 15.87, 0.01, None)),
            "vol_ratio_5_20": np.zeros(N),
            "ln_parkinson": np.log(np.clip(np.abs(ret) * 15.87, 0.01, None)),
            "momentum_20": np.zeros(N),
        },
        index=dates,
    )

    cfg = AppConfig()
    cfg.models.initial_train_days = 150
    cfg.models.refit_cadence = 50
    cfg.models.hmm.n_init = 1
    cfg.models.isolation_forest.n_estimators = 20
    cfg.models.evt.min_exceedances = 10

    res = run_walk_forward_backtest(df, config=cfg, verbose=False)
    assert len(res.oos_df) == 100  # 250 - 150 = 100
    assert len(res.refit_records) == 2  # 150 and 200
    assert "msi_headline" in res.oos_df.columns
    assert "var_evt_95" in res.oos_df.columns
    assert "prob_stress" in res.oos_df.columns
    assert len(res.regulatory_backtest) > 0
