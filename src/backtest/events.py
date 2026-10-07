"""Drawdown event labeling, crisis episode extraction, alarm spell identification, and early warning evaluation."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def label_drawdown_events(
    price: pd.Series,
    h: int = 10,
    x: float = 0.05,
    refractory_period: int = 20,
) -> pd.DataFrame:
    """
    Labels forward drawdown events for post-hoc evaluation and W-LOGIT calibration.
    
    Mathematical definition:
    Forward minimum return: R_{t, h}^{min} = min_{1 <= s <= h} (P_{t+s} - P_t) / P_t
    Binary event label: y_t = 1 if R_{t, h}^{min} <= -x, else 0.
    
    Refractory period:
    Enforces a minimum spacing of `refractory_period` trading days between distinct
    crisis episode onsets so a single sustained downturn is not counted as multiple crises.
    """
    clean_p = price.dropna()
    N = len(clean_p)
    prices_arr = clean_p.values
    dates = clean_p.index

    # 1. Forward minimum return over h days
    fwd_min_ret = np.full(N, np.nan)
    for i in range(N - 1):
        end_idx = min(i + 1 + h, N)
        future_prices = prices_arr[i + 1 : end_idx]
        min_p = np.min(future_prices)
        fwd_min_ret[i] = (min_p - prices_arr[i]) / prices_arr[i]

    # Binary label: y_t = 1 if min return <= -x
    raw_labels = (fwd_min_ret <= -x).astype(int)
    # The last h rows cannot evaluate forward return
    raw_labels[N - h :] = 0

    # 2. Extract discrete crisis episode onsets enforcing refractory period
    crisis_onset = np.zeros(N, dtype=bool)
    episode_ids = np.full(N, -1, dtype=int)
    current_ep_id = 0
    cooldown = 0

    for i in range(N):
        if cooldown > 0:
            cooldown -= 1
        if raw_labels[i] == 1:
            if cooldown == 0:
                crisis_onset[i] = True
                episode_ids[i] = current_ep_id
                current_ep_id += 1
                cooldown = refractory_period
            else:
                # Part of ongoing crisis cooldown
                episode_ids[i] = current_ep_id - 1

    return pd.DataFrame(
        {
            "price": clean_p,
            "fwd_min_ret": fwd_min_ret,
            "event_label": raw_labels,
            "crisis_onset": crisis_onset,
            "episode_id": episode_ids,
        },
        index=dates,
    )

def extract_crisis_episodes(
    price: pd.Series,
    event_df: pd.DataFrame,
    search_window: int = 30,
) -> pd.DataFrame:
    """
    Extracts distinct crisis episodes with peak, trough, drawdown severity, and duration.
    """
    onsets = event_df[event_df["crisis_onset"]].index
    if len(onsets) == 0:
        return pd.DataFrame()

    episodes = []
    prices = price.reindex(event_df.index)

    for ep_id, onset_date in enumerate(onsets):
        idx_loc = event_df.index.get_loc(onset_date)
        start_idx = max(0, idx_loc - 5)
        end_idx = min(len(event_df), idx_loc + search_window)

        window_prices = prices.iloc[start_idx:end_idx]
        peak_date = window_prices.iloc[: idx_loc - start_idx + 1].idxmax()
        peak_price = prices.loc[peak_date]

        post_window = prices.iloc[idx_loc:end_idx]
        trough_date = post_window.idxmin()
        trough_price = prices.loc[trough_date]

        max_dd = (trough_price - peak_price) / peak_price

        episodes.append({
            "episode_id": ep_id,
            "onset_date": onset_date,
            "peak_date": peak_date,
            "trough_date": trough_date,
            "peak_price": float(peak_price),
            "trough_price": float(trough_price),
            "max_drawdown": float(max_dd),
            "duration_days": (trough_date - onset_date).days,
        })

    return pd.DataFrame(episodes)

def identify_alarm_spells(
    alarm_series: pd.Series,
    msi_series: pd.Series | None = None,
    max_gap: int = 5,
) -> pd.DataFrame:
    """
    Groups binary alarms into continuous spells.
    Bridges gaps of <= max_gap days between active alarms into the same spell.
    """
    clean_alarms = alarm_series.fillna(False).astype(bool)
    dates = clean_alarms.index
    alarm_indices = np.where(clean_alarms.values)[0]

    if len(alarm_indices) == 0:
        return pd.DataFrame()

    spells = []
    current_start = alarm_indices[0]
    current_end = alarm_indices[0]

    for idx in alarm_indices[1:]:
        gap = idx - current_end - 1
        if gap <= max_gap:
            # Bridge gap
            current_end = idx
        else:
            # Close previous spell
            spells.append((current_start, current_end))
            current_start = idx
            current_end = idx
    spells.append((current_start, current_end))

    spell_records = []
    for sp_id, (s_idx, e_idx) in enumerate(spells):
        start_date = dates[s_idx]
        end_date = dates[e_idx]
        spell_msi = (
            msi_series.iloc[s_idx : e_idx + 1].max()
            if msi_series is not None
            else np.nan
        )
        spell_records.append({
            "spell_id": sp_id,
            "start_date": start_date,
            "end_date": end_date,
            "start_idx": s_idx,
            "end_idx": e_idx,
            "duration_days": e_idx - s_idx + 1,
            "peak_msi": float(spell_msi),
        })

    return pd.DataFrame(spell_records)

def evaluate_early_warning_alarms(
    event_df: pd.DataFrame,
    alarm_series: pd.Series,
    msi_series: pd.Series | None = None,
    max_gap: int = 5,
    max_lead_time: int = 30,
    min_lead_time: int = 1,
) -> dict[str, Any]:
    """
    Evaluates early-warning detection performance both at the crisis-episode level
    and at the day level.
    
    Episode Matching:
    A crisis episode starting at T_crisis is detected (True Positive) if an alarm spell
    started within [T_crisis - max_lead_time, T_crisis - min_lead_time].
    Lead time = T_crisis - T_alarm_onset in trading days.
    
    Spell Matching:
    An alarm spell is a True Positive if a crisis onset occurs within [T_spell_start + 1, T_spell_start + max_lead_time].
    Otherwise, it is a False Positive (false alarm).
    """
    episodes = extract_crisis_episodes(event_df["price"], event_df)
    spells = identify_alarm_spells(alarm_series, msi_series, max_gap=max_gap)

    n_episodes = len(episodes)
    n_spells = len(spells)

    lead_times: list[int] = []
    detected_episodes = 0
    tp_spells = 0

    if n_episodes > 0 and n_spells > 0:
        episode_dates = event_df.index[event_df["crisis_onset"]].tolist()
        spell_start_dates = spells["start_date"].tolist()

        date_to_idx = {d: i for i, d in enumerate(event_df.index)}

        # 1. Episode matching
        for ep_date in episode_dates:
            ep_idx = date_to_idx[ep_date]
            matching_leads = []
            for sp_date in spell_start_dates:
                sp_idx = date_to_idx[sp_date]
                lead = ep_idx - sp_idx
                if min_lead_time <= lead <= max_lead_time:
                    matching_leads.append(lead)

            if matching_leads:
                detected_episodes += 1
                lead_times.append(int(min(matching_leads))) # Earliest alarm or closest

        # 2. Spell matching
        for sp_date in spell_start_dates:
            sp_idx = date_to_idx[sp_date]
            matched = False
            for ep_date in episode_dates:
                ep_idx = date_to_idx[ep_date]
                lead = ep_idx - sp_idx
                if min_lead_time <= lead <= max_lead_time:
                    matched = True
                    break
            if matched:
                tp_spells += 1

    fp_spells = n_spells - tp_spells
    fn_episodes = n_episodes - detected_episodes

    episode_recall = detected_episodes / n_episodes if n_episodes > 0 else 0.0
    spell_precision = tp_spells / n_spells if n_spells > 0 else 0.0
    f1 = (
        2.0 * spell_precision * episode_recall / (spell_precision + episode_recall)
        if (spell_precision + episode_recall) > 0
        else 0.0
    )

    # Lead time statistics
    if lead_times:
        lead_median = float(np.median(lead_times))
        lead_mean = float(np.mean(lead_times))
        lead_min = int(np.min(lead_times))
        lead_max = int(np.max(lead_times))
        q25, q75 = np.percentile(lead_times, [25, 75])
        lead_iqr = float(q75 - q25)
    else:
        lead_median = lead_mean = lead_iqr = 0.0
        lead_min = lead_max = 0

    # 3. Day-level classification metrics
    y_true = event_df["event_label"].values.astype(int)
    y_pred = alarm_series.reindex(event_df.index).fillna(False).values.astype(int)

    tp_days = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp_days = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn_days = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn_days = int(np.sum((y_pred == 0) & (y_true == 1)))

    day_precision = tp_days / (tp_days + fp_days) if (tp_days + fp_days) > 0 else 0.0
    day_recall = tp_days / (tp_days + fn_days) if (tp_days + fn_days) > 0 else 0.0
    day_far = fp_days / (fp_days + tn_days) if (fp_days + tn_days) > 0 else 0.0

    # Continuous AUC metrics if MSI series is available
    roc_auc = 0.5
    pr_auc = 0.0
    if msi_series is not None:
        clean_msi = msi_series.reindex(event_df.index).fillna(0.0).values
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, clean_msi))
                prec, rec, _ = precision_recall_curve(y_true, clean_msi)
                pr_auc = float(auc(rec, prec))
            except Exception:
                pass

    return {
        "n_crisis_episodes": n_episodes,
        "n_alarm_spells": n_spells,
        "detected_episodes": detected_episodes,
        "unwarned_episodes": fn_episodes,
        "tp_spells": tp_spells,
        "fp_spells": fp_spells,
        "episode_recall": float(episode_recall),
        "spell_precision": float(spell_precision),
        "f1_score": float(f1),
        "lead_times": lead_times,
        "lead_time_median": lead_median,
        "lead_time_mean": lead_mean,
        "lead_time_min": lead_min,
        "lead_time_max": lead_max,
        "lead_time_iqr": lead_iqr,
        "day_precision": float(day_precision),
        "day_recall": float(day_recall),
        "day_far": float(day_far),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

def compare_alarms_to_baselines(
    features_df: pd.DataFrame,
    msi_series: pd.Series,
    h: int = 10,
    x: float = 0.05,
    refractory_period: int = 20,
) -> pd.DataFrame:
    """
    Compares the MSI headline alarm against 3 standard market risk baselines:
    1. Volatility Baseline: trailing 20d volatility (rv_20) >= 90th percentile
    2. Drawdown Baseline: trailing 252d drawdown <= -5% (|DD| >= 5%)
    3. India VIX Baseline: VIX >= 24.0 (standard stress cutoff)
    """
    event_df = label_drawdown_events(features_df["Close"], h=h, x=x, refractory_period=refractory_period)

    # 1. MSI Primary Alarm (>= 50.0)
    msi_alarm = msi_series >= 50.0
    msi_eval = evaluate_early_warning_alarms(event_df, msi_alarm, msi_series)

    # 2. Volatility Baseline: rv_20 >= 90th percentile
    if "rv_20" in features_df.columns:
        vol_thresh = features_df["rv_20"].quantile(0.90)
        vol_alarm = features_df["rv_20"] >= vol_thresh
        vol_score = features_df["rv_20"]
    else:
        vol_alarm = pd.Series(False, index=features_df.index)
        vol_score = pd.Series(0.0, index=features_df.index)
    vol_eval = evaluate_early_warning_alarms(event_df, vol_alarm, vol_score)

    # 3. Drawdown Baseline: |drawdown_252| >= 0.05 (5%)
    if "drawdown_252" in features_df.columns:
        dd_alarm = features_df["drawdown_252"] <= -0.05
        dd_score = -features_df["drawdown_252"]
    else:
        dd_alarm = pd.Series(False, index=features_df.index)
        dd_score = pd.Series(0.0, index=features_df.index)
    dd_eval = evaluate_early_warning_alarms(event_df, dd_alarm, dd_score)

    # 4. India VIX Baseline: VIX >= 24.0
    if "vix_level" in features_df.columns and not features_df["vix_level"].dropna().empty:
        vix_alarm = features_df["vix_level"] >= 24.0
        vix_score = features_df["vix_level"]
    else:
        vix_alarm = pd.Series(False, index=features_df.index)
        vix_score = pd.Series(0.0, index=features_df.index)
    vix_eval = evaluate_early_warning_alarms(event_df, vix_alarm, vix_score)

    rows = [
        {
            "Model": "Market Stress Index (MSI >= 50)",
            "Spell Precision": msi_eval["spell_precision"],
            "Episode Recall": msi_eval["episode_recall"],
            "F1-Score": msi_eval["f1_score"],
            "PR-AUC": msi_eval["pr_auc"],
            "ROC-AUC": msi_eval["roc_auc"],
            "Median Lead Time (Days)": msi_eval["lead_time_median"],
            "IQR Lead Time": msi_eval["lead_time_iqr"],
            "False Alarm Rate": msi_eval["day_far"],
        },
        {
            "Model": "Baseline: Realized Vol (rv_20 >= 90th pct)",
            "Spell Precision": vol_eval["spell_precision"],
            "Episode Recall": vol_eval["episode_recall"],
            "F1-Score": vol_eval["f1_score"],
            "PR-AUC": vol_eval["pr_auc"],
            "ROC-AUC": vol_eval["roc_auc"],
            "Median Lead Time (Days)": vol_eval["lead_time_median"],
            "IQR Lead Time": vol_eval["lead_time_iqr"],
            "False Alarm Rate": vol_eval["day_far"],
        },
        {
            "Model": "Baseline: Trailing Drawdown (|DD_252| >= 5%)",
            "Spell Precision": dd_eval["spell_precision"],
            "Episode Recall": dd_eval["episode_recall"],
            "F1-Score": dd_eval["f1_score"],
            "PR-AUC": dd_eval["pr_auc"],
            "ROC-AUC": dd_eval["roc_auc"],
            "Median Lead Time (Days)": dd_eval["lead_time_median"],
            "IQR Lead Time": dd_eval["lead_time_iqr"],
            "False Alarm Rate": dd_eval["day_far"],
        },
        {
            "Model": "Baseline: India VIX (VIX >= 24)",
            "Spell Precision": vix_eval["spell_precision"],
            "Episode Recall": vix_eval["episode_recall"],
            "F1-Score": vix_eval["f1_score"],
            "PR-AUC": vix_eval["pr_auc"],
            "ROC-AUC": vix_eval["roc_auc"],
            "Median Lead Time (Days)": vix_eval["lead_time_median"],
            "IQR Lead Time": vix_eval["lead_time_iqr"],
            "False Alarm Rate": vix_eval["day_far"],
        },
    ]

    return pd.DataFrame(rows)

def moving_block_bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_type: str = "pr_auc",
    n_resamples: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes 95% confidence intervals via moving block bootstrap.
    Preserves temporal autocorrelation within blocks.
    
    Returns: (point_estimate, ci_lower_2.5, ci_upper_97.5)
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    n_blocks = int(np.ceil(N / block_size))

    # Point estimate
    if metric_type == "pr_auc":
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        point_est = float(auc(rec, prec))
    elif metric_type == "roc_auc":
        point_est = float(roc_auc_score(y_true, y_score))
    else:
        raise ValueError(f"Unknown metric_type {metric_type}")

    boot_estimates = []
    for _ in range(n_resamples):
        start_indices = rng.integers(0, N - block_size + 1, size=n_blocks)
        idx_list: list[int] = []
        for s in start_indices:
            idx_list.extend(range(s, s + block_size))
        sampled_indices = np.array(idx_list[:N])

        y_t_boot = y_true[sampled_indices]
        y_s_boot = y_score[sampled_indices]

        if len(np.unique(y_t_boot)) < 2:
            continue

        try:
            if metric_type == "pr_auc":
                p, r, _ = precision_recall_curve(y_t_boot, y_s_boot)
                val = auc(r, p)
            else:
                val = roc_auc_score(y_t_boot, y_s_boot)
            boot_estimates.append(val)
        except Exception:
            continue

    if len(boot_estimates) < 50:
        return point_est, point_est, point_est

    ci_lower = float(np.percentile(boot_estimates, 2.5))
    ci_upper = float(np.percentile(boot_estimates, 97.5))
    return point_est, ci_lower, ci_upper
