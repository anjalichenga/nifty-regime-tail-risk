"""Temporal walk-forward backtesting engine with expanding training windows and zero lookahead leakage."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from src.anomaly.isoforest import CausalIsolationForest
from src.backtest.events import (
    compare_alarms_to_baselines,
    evaluate_early_warning_alarms,
    label_drawdown_events,
)
from src.changepoints.cusum import OnlinePageCUSUM
from src.models.garch import extract_garch_parameters, fit_garch_model
from src.regimes.filtering import forward_filter_probabilities
from src.regimes.hmm import fit_robust_hmm
from src.risk.evt import compute_gpd_quantiles_and_es, fit_gpd_tail
from src.risk.risk_backtest import evaluate_all_var_models
from src.risk.var_es import (
    compute_conditional_normal_var_es,
    compute_conditional_student_t_var_es,
    compute_historical_simulation_var_es,
)
from src.stress.aggregation import (
    aggregate_msi,
    classify_stress_band,
    compute_component_contributions,
    fit_logit_model,
    fit_pca_weights,
)
from src.stress.components import TrainFittedECDF
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class WalkForwardResults:
    oos_df: pd.DataFrame
    refit_records: list[dict[str, Any]]
    regulatory_backtest: pd.DataFrame
    early_warning_eval: dict[str, Any]
    baseline_comparison: pd.DataFrame
    config_dict: dict[str, Any]

def run_walk_forward_backtest(
    features_df: pd.DataFrame,
    config: AppConfig | None = None,
    verbose: bool = True,
) -> WalkForwardResults:
    """
    Executes the temporal walk-forward backtesting simulation:
    - Expanding training window starting at initial_train_days (1,260 days ~ 5 years).
    - Refits models every refit_cadence (63 trading days ~ quarterly).
    - Daily out-of-sample evaluation uses frozen parameters:
        - GARCH: conditional variance filtered via frozen arch params.
        - HMM: forward probabilities via log-space recursion.
        - EVT: POT tail quantiles and conditional VaR/ES at 95% and 99%.
        - Isolation Forest: causal scoring and train-fit ECDF percentiles.
        - Page-CUSUM: causal sequential update of standardized shocks.
        - MSI: W-EQ, W-PCA, and W-LOGIT aggregation with train-fit ECDFs.
    - Zero future parameter or label lookahead across the entire out-of-sample horizon.
    """
    if config is None:
        config = load_config()

    m_cfg = config.models
    init_train = m_cfg.initial_train_days
    refit_cadence = m_cfg.refit_cadence

    # Ensure clean price and return series
    N = len(features_df)
    if N <= init_train + refit_cadence:
        raise ValueError(f"Insufficient data length {N} for initial training {init_train} and refits.")

    dates = features_df.index
    returns_pct = features_df["return_pct"]
    price = features_df["Close"]

    # Pre-compute labels for evaluation & W-LOGIT training
    event_df = label_drawdown_events(
        price,
        h=config.backtest.event_h,
        x=config.backtest.event_x,
        refractory_period=config.backtest.refractory_period,
    )
    event_labels = event_df["event_label"]

    # Rolling Historical Simulation VaR/ES benchmark (causal 750-day window)
    hs_var_95, hs_es_95 = compute_historical_simulation_var_es(returns_pct, window=750, q=0.95)
    hs_var_99, hs_es_99 = compute_historical_simulation_var_es(returns_pct, window=750, q=0.99)

    # Initialize tracking structures for out-of-sample evaluations
    oos_records: list[dict[str, Any]] = []
    refit_records: list[dict[str, Any]] = []

    # Features for models
    hmm_feats = m_cfg.hmm.features
    if_feats = m_cfg.isolation_forest.features

    # Setup refit steps
    refit_indices = list(range(init_train, N, refit_cadence))
    if refit_indices[-1] < N:
        refit_indices.append(N)

    # Online CUSUM detector preserved across daily steps
    cusum_tracker = OnlinePageCUSUM(k=m_cfg.changepoints.cusum.k, h=m_cfg.changepoints.cusum.h)

    # Warm-start container for HMM
    hmm_warm_params: dict[str, np.ndarray] | None = None

    for step_i in range(len(refit_indices) - 1):
        t_refit = refit_indices[step_i]
        t_next = refit_indices[step_i + 1]

        refit_date = dates[t_refit]
        if verbose:
            logger.info("Refit %d/%d at index %d (%s) -> OOS horizon [%d, %d)...",
                        step_i + 1, len(refit_indices) - 1, t_refit, refit_date.strftime("%Y-%m-%d"), t_refit, t_next)

        train_slice = features_df.iloc[:t_refit]

        # -------------------------------------------------------------
        # 1. FIT GARCH(1,1)-Student-t on training returns
        # -------------------------------------------------------------
        train_ret_clean = train_slice["return_pct"].dropna()
        _garch_model_obj, garch_res = fit_garch_model(
            train_ret_clean,
            p=m_cfg.garch.p,
            q=m_cfg.garch.q,
            o=m_cfg.garch.o,
            dist=m_cfg.garch.dist,
        )
        frozen_garch_params = garch_res.params
        garch_meta = extract_garch_parameters(garch_res, is_gjr=(m_cfg.garch.o > 0))

        # Training standardized residuals: z = (r - mu) / sigma
        train_cond_vol = garch_res.conditional_volatility
        train_mu = float(frozen_garch_params.get("mu", 0.0))
        train_std_resid = garch_res.std_resid.dropna().values
        train_losses = -train_std_resid

        # -------------------------------------------------------------
        # 2. FIT EVT (POT) on negative standardized residuals
        # -------------------------------------------------------------
        gpd_fit = fit_gpd_tail(train_losses, q_u=m_cfg.evt.q_u, min_exceedances=m_cfg.evt.min_exceedances)
        evt_quantiles = compute_gpd_quantiles_and_es(
            gpd_fit,
            q_levels=m_cfg.evt.risk_quantiles,
            empirical_losses=train_losses,
        )
        z_evt_95, es_evt_95 = evt_quantiles[0.95]
        z_evt_99, es_evt_99 = evt_quantiles[0.99]

        # Parametric Student-t degrees of freedom
        nu_t = garch_meta.get("degrees_of_freedom", 6.0)

        # -------------------------------------------------------------
        # 3. FIT HMM (K=3) on training features
        # -------------------------------------------------------------
        hmm_train_data = train_slice[hmm_feats].dropna()
        X_hmm_train = hmm_train_data.values

        hmm_model_dict, hmm_train_probs = fit_robust_hmm(
            X_hmm_train,
            k=m_cfg.hmm.default_k,
            feature_names=hmm_feats,
            covariance_type=m_cfg.hmm.covariance_type,
            base_seed=m_cfg.seed,
            n_init=m_cfg.hmm.n_init,
            warm_start_params=hmm_warm_params,
        )
        # Store for warm start on next cycle
        hmm_warm_params = {
            "startprob": hmm_model_dict["startprob"],
            "transmat": hmm_model_dict["transmat"],
            "means": hmm_model_dict["means"],
            "covars": hmm_model_dict["covars"],
        }

        # -------------------------------------------------------------
        # 4. FIT CAUSAL ISOLATION FOREST
        # -------------------------------------------------------------
        if_train_data = train_slice[if_feats].dropna()
        iso_detector = CausalIsolationForest(
            n_estimators=m_cfg.isolation_forest.n_estimators,
            percentile_threshold=m_cfg.isolation_forest.percentile_threshold,
            seed=m_cfg.seed,
        )
        iso_detector.fit(if_train_data.values)

        # -------------------------------------------------------------
        # 5. CALIBRATE PAGE-CUSUM (In-control statistics on train)
        # -------------------------------------------------------------
        cusum_tracker.calibrate(train_std_resid)

        # -------------------------------------------------------------
        # 6. FIT ECDFs FOR STRESS COMPONENTS
        # -------------------------------------------------------------
        ecdf_vol = TrainFittedECDF().fit(train_cond_vol.values)
        ecdf_dd = TrainFittedECDF().fit(np.abs(train_slice["drawdown_252"].dropna().values))

        # Optional cross-asset correlation ECDF
        ecdf_corr = None
        if "corr_banknifty_60" in train_slice.columns:
            corr_train = train_slice["corr_banknifty_60"].dropna().to_numpy()
            if len(corr_train) > 10:
                ecdf_corr = TrainFittedECDF().fit(corr_train)

        # -------------------------------------------------------------
        # 7. FIT MSI WEIGHTS (W-PCA & W-LOGIT)
        # -------------------------------------------------------------
        # Align training components for weights estimation
        # HMM train stress prob is state 2 (Stress, highest vol)
        stress_state_idx = m_cfg.hmm.default_k - 1
        c_hmm_train = hmm_train_probs[:, stress_state_idx]

        # Subsample to common index
        common_train_idx = hmm_train_data.index
        c_vol_train, _ = ecdf_vol.transform(train_cond_vol.reindex(common_train_idx).ffill().values)
        c_dd_train, _ = ecdf_dd.transform(np.abs(train_slice.loc[common_train_idx, "drawdown_252"].fillna(0.0).values))
        _, c_anom_train, _ = iso_detector.score(train_slice.loc[common_train_idx, if_feats].fillna(0.0).values)

        train_comp_df = pd.DataFrame(
            {
                "c_hmm": c_hmm_train,
                "c_vol": c_vol_train,
                "c_drawdown": c_dd_train,
                "c_anomaly": c_anom_train,
            },
            index=common_train_idx,
        )

        pca_weights = fit_pca_weights(train_comp_df)
        logit_model = fit_logit_model(
            train_comp_df,
            event_labels.reindex(common_train_idx),
            purge_days=config.backtest.event_h,
            embargo_days=5,
        )

        # Record refit metadata
        refit_records.append({
            "refit_idx": step_i,
            "refit_date": refit_date,
            "train_start": dates[0],
            "train_end": dates[t_refit - 1],
            "garch_meta": garch_meta,
            "gpd_fit": gpd_fit,
            "hmm_bic": hmm_model_dict["bic"],
            "hmm_occupancy": hmm_model_dict["occupancy"],
            "pca_weights": pca_weights,
            "has_logit_model": logit_model is not None,
        })

        # -------------------------------------------------------------
        # 8. OUT-OF-SAMPLE DAILY EVALUATION (t in [t_refit, t_next))
        # -------------------------------------------------------------
        # Evaluate GARCH on extended window up to t_next using FROZEN parameters
        extended_ret = features_df["return_pct"].iloc[:t_next].dropna()
        am_ext = arch_model(
            extended_ret,
            mean="Constant",
            vol="GARCH",
            p=m_cfg.garch.p,
            o=m_cfg.garch.o,
            q=m_cfg.garch.q,
            dist=m_cfg.garch.dist,  # type: ignore
            rescale=False,
        )
        frozen_res = am_ext.fix(frozen_garch_params)
        ext_cond_vol_arr = np.asarray(frozen_res.conditional_volatility)
        ext_std_resid_arr = np.asarray(frozen_res.std_resid)

        # Evaluate HMM on extended window up to t_next using FROZEN parameters
        ext_hmm_slice = features_df[hmm_feats].iloc[:t_next].ffill().values
        ext_hmm_probs, _ = forward_filter_probabilities(
            ext_hmm_slice,
            startprob=hmm_model_dict["startprob"],
            transmat=hmm_model_dict["transmat"],
            means=hmm_model_dict["means"],
            covars=hmm_model_dict["covars"],
            covariance_type=hmm_model_dict["covariance_type"],
        )

        # Loop daily across the out-of-sample block [t_refit, t_next)
        for t_day in range(t_refit, t_next):
            day_date = dates[t_day]
            r_day = returns_pct.iloc[t_day]
            p_day = price.iloc[t_day]
            loss_day = -r_day  # positive percentage loss

            # Daily GARCH metrics
            sigma_t = float(ext_cond_vol_arr[t_day]) if t_day < len(ext_cond_vol_arr) else float(ext_cond_vol_arr[-1])
            z_t = float(ext_std_resid_arr[t_day]) if t_day < len(ext_std_resid_arr) else 0.0

            # Daily HMM regime filtered probabilities
            prob_calm = float(ext_hmm_probs[t_day, 0])
            prob_elev = float(ext_hmm_probs[t_day, 1])
            prob_stress = float(ext_hmm_probs[t_day, 2])
            pred_regime = int(np.argmax(ext_hmm_probs[t_day]))

            # Daily EVT Tail Risk (conditional on sigma_t)
            # Loss = -r_t. VaR_q = -mu + sigma * z_q; ES_q = -mu + sigma * ES_q
            var_evt_95 = -train_mu + sigma_t * z_evt_95
            es_evt_95 = -train_mu + sigma_t * es_evt_95
            var_evt_99 = -train_mu + sigma_t * z_evt_99
            es_evt_99 = -train_mu + sigma_t * es_evt_99

            # Daily Benchmark VaR / ES
            var_norm_95, es_norm_95 = compute_conditional_normal_var_es(train_mu, sigma_t, q=0.95)
            var_norm_99, es_norm_99 = compute_conditional_normal_var_es(train_mu, sigma_t, q=0.99)

            var_t_95, es_t_95 = compute_conditional_student_t_var_es(train_mu, sigma_t, nu=nu_t, q=0.95)
            var_t_99, es_t_99 = compute_conditional_student_t_var_es(train_mu, sigma_t, nu=nu_t, q=0.99)

            # Daily Isolation Forest anomaly score
            x_if_day = features_df[if_feats].iloc[t_day : t_day + 1].fillna(0.0).values
            raw_anom, pct_anom, flag_anom = iso_detector.score(x_if_day)
            score_anom = float(raw_anom[0])
            percentile_anom = float(pct_anom[0])
            is_anom = bool(flag_anom[0])

            # Daily Page-CUSUM update
            c_high, c_low, s_pos, s_neg, days_alarm = cusum_tracker.step(z_t)

            # Daily MSI Components
            c_k_hmm = prob_stress
            c_k_vol, _ = ecdf_vol.transform(np.array([sigma_t]))
            dd_val = np.abs(features_df["drawdown_252"].iloc[t_day])
            c_k_dd, _ = ecdf_dd.transform(np.array([dd_val]))
            c_k_anom = percentile_anom

            day_comp_dict = {
                "c_hmm": float(c_k_hmm),
                "c_vol": float(c_k_vol[0]),
                "c_drawdown": float(c_k_dd[0]),
                "c_anomaly": float(c_k_anom),
            }

            if ecdf_corr is not None and "corr_banknifty_60" in features_df.columns:
                corr_val = features_df["corr_banknifty_60"].iloc[t_day]
                if not np.isnan(corr_val):
                    c_k_corr, _ = ecdf_corr.transform(np.array([corr_val]))
                    day_comp_dict["c_cross_asset"] = float(c_k_corr[0])

            day_comp_df = pd.DataFrame([day_comp_dict], index=[day_date])

            # Daily MSI Aggregations
            msi_eq = float(aggregate_msi(day_comp_df, method="W-EQ").iloc[0])
            msi_pca = float(aggregate_msi(day_comp_df, method="W-PCA", weights=pca_weights).iloc[0])
            msi_logit = float(aggregate_msi(day_comp_df, method="W-LOGIT", logit_model=logit_model).iloc[0])

            # Headline is W-EQ by default
            headline_msi = msi_eq if config.stress_index.headline_method == "W-EQ" else msi_pca
            stress_band = str(classify_stress_band(pd.Series([headline_msi])).iloc[0])

            # Contributions for headline
            contribs = compute_component_contributions(day_comp_df, weights=None).iloc[0].to_dict()

            # Record all variables
            oos_records.append({
                "date": day_date,
                "price": float(p_day),
                "return_pct": float(r_day),
                "loss_pct": float(loss_day),
                # GARCH
                "conditional_vol": sigma_t,
                "std_resid": z_t,
                # HMM Regimes
                "prob_calm": prob_calm,
                "prob_elevated": prob_elev,
                "prob_stress": prob_stress,
                "regime": pred_regime,
                # VaR / ES (EVT)
                "var_evt_95": float(var_evt_95),
                "es_evt_95": float(es_evt_95),
                "breach_evt_95": bool(loss_day > var_evt_95),
                "var_evt_99": float(var_evt_99),
                "es_evt_99": float(es_evt_99),
                "breach_evt_99": bool(loss_day > var_evt_99),
                # Benchmark VaR / ES
                "var_norm_95": float(var_norm_95),
                "es_norm_95": float(es_norm_95),
                "breach_norm_95": bool(loss_day > var_norm_95),
                "var_norm_99": float(var_norm_99),
                "es_norm_99": float(es_norm_99),
                "breach_norm_99": bool(loss_day > var_norm_99),
                "var_t_95": float(var_t_95),
                "es_t_95": float(es_t_95),
                "breach_t_95": bool(loss_day > var_t_95),
                "var_t_99": float(var_t_99),
                "es_t_99": float(es_t_99),
                "breach_t_99": bool(loss_day > var_t_99),
                "var_hs_95": float(hs_var_95.iloc[t_day]) if not np.isnan(hs_var_95.iloc[t_day]) else np.nan,
                "es_hs_95": float(hs_es_95.iloc[t_day]) if not np.isnan(hs_es_95.iloc[t_day]) else np.nan,
                "breach_hs_95": bool(loss_day > hs_var_95.iloc[t_day]) if not np.isnan(hs_var_95.iloc[t_day]) else False,
                "var_hs_99": float(hs_var_99.iloc[t_day]) if not np.isnan(hs_var_99.iloc[t_day]) else np.nan,
                "es_hs_99": float(hs_es_99.iloc[t_day]) if not np.isnan(hs_es_99.iloc[t_day]) else np.nan,
                "breach_hs_99": bool(loss_day > hs_var_99.iloc[t_day]) if not np.isnan(hs_var_99.iloc[t_day]) else False,
                # Isolation Forest
                "anomaly_score": score_anom,
                "anomaly_percentile": percentile_anom,
                "anomaly_flag": is_anom,
                # CUSUM
                "cusum_alarm_high": c_high,
                "cusum_alarm_low": c_low,
                "cusum_s_pos": s_pos,
                "cusum_s_neg": s_neg,
                "days_since_cusum": days_alarm,
                # MSI Components & Aggregations
                "c_hmm": float(c_k_hmm),
                "c_vol": float(c_k_vol[0]),
                "c_drawdown": float(c_k_dd[0]),
                "c_anomaly": float(c_k_anom),
                "msi_eq": msi_eq,
                "msi_pca": msi_pca,
                "msi_logit": msi_logit,
                "msi_headline": headline_msi,
                "stress_band": stress_band,
                "alarm_primary": headline_msi >= config.stress_index.alarm_threshold_primary,
                "alarm_secondary": headline_msi >= config.stress_index.alarm_threshold_secondary,
                # Component contributions
                **contribs,
                # Event label
                "event_label": int(event_labels.iloc[t_day]),
                "crisis_onset": bool(event_df["crisis_onset"].iloc[t_day]),
            })

    oos_df = pd.DataFrame(oos_records).set_index("date")

    # -------------------------------------------------------------
    # 9. REGULATORY TAIL-RISK BACKTESTING (Kupiec, Christoffersen, McNeil-Frey)
    # -------------------------------------------------------------
    returns_oos = oos_df["return_pct"]
    var_dict_95 = {
        "Historical Simulation": oos_df["var_hs_95"],
        "Conditional Normal": oos_df["var_norm_95"],
        "Conditional Student-t": oos_df["var_t_95"],
        "GARCH-EVT (POT)": oos_df["var_evt_95"],
    }
    es_dict_95 = {
        "Historical Simulation": oos_df["es_hs_95"],
        "Conditional Normal": oos_df["es_norm_95"],
        "Conditional Student-t": oos_df["es_t_95"],
        "GARCH-EVT (POT)": oos_df["es_evt_95"],
    }
    var_dict_99 = {
        "Historical Simulation": oos_df["var_hs_99"],
        "Conditional Normal": oos_df["var_norm_99"],
        "Conditional Student-t": oos_df["var_t_99"],
        "GARCH-EVT (POT)": oos_df["var_evt_99"],
    }
    es_dict_99 = {
        "Historical Simulation": oos_df["es_hs_99"],
        "Conditional Normal": oos_df["es_norm_99"],
        "Conditional Student-t": oos_df["es_t_99"],
        "GARCH-EVT (POT)": oos_df["es_evt_99"],
    }

    reg_95 = evaluate_all_var_models(returns_oos, var_dict_95, es_dict_95, alpha=0.05)
    reg_99 = evaluate_all_var_models(returns_oos, var_dict_99, es_dict_99, alpha=0.01)
    regulatory_summary = pd.concat([reg_95, reg_99], ignore_index=True)

    # -------------------------------------------------------------
    # 10. EARLY-WARNING CRISIS DETECTION EVALUATION & BASELINES
    # -------------------------------------------------------------
    sub_event_df = event_df.reindex(oos_df.index)
    early_warning_eval = evaluate_early_warning_alarms(
        sub_event_df,
        oos_df["alarm_primary"],
        oos_df["msi_headline"],
        max_gap=config.backtest.spell_max_gap,
        max_lead_time=30,
    )

    baseline_comparison = compare_alarms_to_baselines(
        features_df.reindex(oos_df.index),
        oos_df["msi_headline"],
        h=config.backtest.event_h,
        x=config.backtest.event_x,
        refractory_period=config.backtest.refractory_period,
    )

    return WalkForwardResults(
        oos_df=oos_df,
        refit_records=refit_records,
        regulatory_backtest=regulatory_summary,
        early_warning_eval=early_warning_eval,
        baseline_comparison=baseline_comparison,
        config_dict=config.model_dump(),
    )
