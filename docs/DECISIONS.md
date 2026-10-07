# Architectural and Methodological Decisions (DECISIONS.md)

This document records all design, methodological, and implementation decisions made in accordance with the project specification.

## 1. Project Scope and Philosophy
- **Objective:** Market regime detection, conditional volatility modeling, EVT-based tail-risk forecasting, anomaly detection, and Market Stress Index (MSI) computation for NIFTY 50 (`^NSEI`).
- **Strict Boundary:** This is an observational risk-monitoring, early-warning framework. It is strictly NOT a directional price predictor, trading bot, or market crash oracle. No claims of certainty are made anywhere.
- **Honest Evaluation:** If signals fail to beat simple volatility/drawdown baselines or provide negative early-warning skill, that result is reported transparently and discussed in `reports/RESULTS.md` and the README.

## 2. Data Ingestion & Alignment
- **Price Type:** Non-adjusted Close (`auto_adjust=False`) as per specification. For equity indices like NIFTY 50, cash dividends are not factored into the price index (unlike Total Return Index TRI). Close equals Adjusted Close for index data.
- **Master Calendar:** NIFTY 50 (`^NSEI`) calendar is the single authoritative trading calendar. No synthetic rows are added on non-trading days.
- **Availability Lags:**
  - `^NSEBANK`: lag 0 (same exchange, synchronous close)
  - `^INDIAVIX`: lag 0 (same exchange, synchronous close)
  - `^GSPC` (S&P 500): lag 1 (US markets close after Indian markets close)
  - `INR=X` (USD/INR): lag 1 (global FX closing fix relative to NSE close)
  - `GC=F` (Gold Futures): lag 1 (global commodities session)
  - Secondary series are aligned strictly after applying the availability lag and forward-filled at most 3 days (`limit=3`). No backfilling, no interpolation.
- **Volume Handling:** If volume coverage is < 90% or zero, volume features are cleanly omitted, logging a warning, without crashing the pipeline.

## 3. Two-Layer Design
- **Descriptive Layer (In-Sample):** Full-sample ADF, KPSS, ACF/PACF, descriptive ARIMA/GARCH diagnostics, retrospective PELT change-point analysis, and full-sample EVT threshold stability diagnostics. Clearly labeled as `IN-SAMPLE / DESCRIPTIVE` in all charts, reports, and UI. Never leaks into operational signals.
- **Causal Layer (Out-of-Sample Walk-Forward):** Strictly causal, trailing rolling windows (`center=False`), expanding training window starting at 1,260 days (~5 years), refit every 63 days (quarterly), forward-filtered HMM probabilities, frozen-parameter GARCH filtering, out-of-sample Isolation Forest scoring, Page-CUSUM online change detection, and fit-on-train ECDF transformations. Labeled `OUT-OF-SAMPLE`.

## 4. Model Selection & Hyperparameters (Frozen on Development Period 2007–2014)
- **HMM:** $K=3$ states (Calm, Elevated, Stress) selected based on BIC and financial interpretability. Full covariance matrix. States dynamically relabeled strictly based on the training window mean of $\ln(\text{RV}_{20})$ to prevent label-switching bugs.
- **Forward Filtering:** Custom pure log-space forward recursion implemented in `src/regimes/filtering.py` using `scipy.stats.multivariate_normal`. Smoothed probabilities (`predict_proba`) and Viterbi decoding are strictly prohibited in the causal pipeline.
- **GARCH:** GARCH(1,1) with Student-t innovations chosen on the development period by BIC over Normal and Skew-t, capturing heavy-tailed standardized shocks. Out-of-sample days between 63-day refits use frozen parameter evaluation via `model.fix(params)`.
- **EVT (POT):** Threshold chosen at empirical 90th percentile ($q_u = 0.90$) of negative standardized residuals within each training window. Fitted with Generalized Pareto Distribution (GPD) via MLE with $f_{\text{loc}}=0$. Analytical quantile and Expected Shortfall with mean fallback if $\xi \ge 1$.
- **Anomaly Detection:** Isolation Forest with 300 estimators, fixed seed. Anomaly percentile computed via empirical CDF of the training scores applied forward. Flag threshold at 99th percentile.
- **Change Point (PELT vs CUSUM):** PELT (normal cost, retrospective) used strictly for descriptive historical analysis. Online causal monitoring uses Page-CUSUM ($k=0.5, h=5.0$) on standardized GARCH squared residual surprises.
- **Market Stress Index (MSI):**
  - Core components: HMM stress probability, Volatility ECDF, Drawdown ECDF (252-day window), Isolation Forest percentile, and optional Cross-Asset correlation stress.
  - Aggregation methods: Equal Weight (W-EQ, default headline), PCA-weighted (W-PCA), and Logistic Regression (W-LOGIT with $h$-day purge and embargo). Selected W-EQ as the primary headline on Dev set due to parity with W-PCA and maximum robustness against parameter drift.
  - Stress bands: [0, 25) Normal (green/blue), [25, 50) Elevated (amber), [50, 75) High Risk (orange), [75, 100] Severe (red).
- **Backtesting Events:** Drawdown event definition $h=10$ trading days, $x=5\%$ drop from trailing high, with 20 trading day refractory period between distinct crisis episodes.

## 5. Excluded Features & Rationale
- **ATR:** Omitted because Parkinson range volatility is a mathematically more efficient variance estimator that avoids conflating range with overnight gaps.
- **Return Autocorrelation:** Omitted due to near-zero predictability and high estimation noise.
- **Vol-of-Vol & Correlation Dispersion:** Omitted due to short-window instability and excessive degrees of freedom relative to sample size.
- **Rolling Skewness/Kurtosis:** Excluded from predictive models (retained only in descriptive diagnostics) due to sampling noise of third and fourth sample moments.
