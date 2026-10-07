# NIFTY 50 Market Regime Detection & Tail-Risk Forecasting Framework

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Type Checking](https://img.shields.io/badge/type%20checking-pyright-purple.svg)](pyrightconfig.json)
[![Tests](https://img.shields.io/badge/tests-pytest%20%7C%20leakage%20verified-brightgreen.svg)](tests/)
[![Methodology](https://img.shields.io/badge/methodology-EVT%20%7C%20HMM%20%7C%20GARCH-orange.svg)](docs/DECISIONS.md)

An institutional-grade quantitative risk modeling framework engineered for the **NIFTY 50 Index (`^NSEI`)**. This system couples unsupervised regime classification, conditional heteroskedasticity modeling, Extreme Value Theory (EVT) tail-risk forecasting, and anomaly detection into a strictly causal, walk-forward **Market Stress Index (MSI)**.

Designed around a **zero-lookahead, leakage-immune architecture**, this framework replaces textbook Gaussian assumptions with empirical heavy-tailed dynamics to deliver dependable early-warning risk monitoring for systematic portfolio management and institutional risk desks.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Why Standard Risk Models Fail](#why-standard-risk-models-fail)
- [Quantitative Architecture](#quantitative-architecture)
  - [1. Two-Layer Design](#1-two-layer-design)
  - [2. Unsupervised Regime Detection (Hidden Markov Models)](#2-unsupervised-regime-detection-hidden-markov-models)
  - [3. Conditional Volatility Modeling (GARCH-Student-t)](#3-conditional-volatility-modeling-garch-student-t)
  - [4. Heavy-Tail Risk Forecasting (EVT / POT-GPD)](#4-heavy-tail-risk-forecasting-evt--pot-gpd)
  - [5. Anomaly & Change-Point Detection (Isolation Forest & Page-CUSUM)](#5-anomaly--change-point-detection-isolation-forest--page-cusum)
  - [6. Composite Market Stress Index (MSI)](#6-composite-market-stress-index-msi)
- [Institutional Leakage Prevention Protocol (17 Vectors)](#institutional-leakage-prevention-protocol-17-vectors)
- [Repository Structure](#repository-structure)
- [Quickstart & Installation](#quickstart--installation)
- [Running the Pipeline & Tests](#running-the-pipeline--tests)
- [Empirical Diagnostics & Findings](#empirical-diagnostics--findings)
- [Quantitative Interview & Resume Talking Points](#quantitative-interview--resume-talking-points)
- [License & Disclaimer](#license--disclaimer)

---

## Executive Summary

Traditional financial risk models notoriously break down during crisis episodes. Under the Gaussian distribution, a daily drawdown like the March 23, 2020 COVID shock ($-12.98\%$) represents a $> 8\sigma$ event—statistically impossible within the lifespan of the universe. In emerging equity markets like the NIFTY 50, volatility clusters persistently, return distributions exhibit fat tails, and latent market regimes switch abruptly between low-volatility drifts and turbulent liquidity panics.

This framework solves this problem with an end-to-end, reproducible quantitative engine that:
1. **Identifies latent volatility regimes** ($K=3$: Calm, Elevated, Stress) via pure **log-space forward-filtered HMMs** without lookahead.
2. **Models conditional heteroskedasticity** via **GARCH(1,1) with Student-$t$ innovations** ($\nu \approx 7.2$).
3. **Calibrates tail losses** via **Extreme Value Theory (POT-GPD)** on negative standardized innovations to compute 95% and 99% conditional Value-at-Risk (VaR) and Expected Shortfall (ES).
4. **Fuses multi-modal signals** into an intuitive **0–100 Market Stress Index (MSI)** calibrated to provide early warnings ahead of drawdown spells.

```
+---------------------------------------------------------------------------------------------------+
|                                MASTER NIFTY 50 CALENDAR PIPELINE                                  |
|   NIFTY 50 (^NSEI)  |  Bank NIFTY (^NSEBANK)  |  India VIX (^INDIAVIX)  |  S&P 500 (Lag-1)  | FX  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                            CAUSAL EXPANDING WALK-FORWARD ENGINE                                    |
|              Initial Training: 1,260 days (5 yrs)  │  Refit Cadence: 63 days (quarterly)          |
+---------------------------------------------------------------------------------------------------+
       │                                  │                                   │
       ▼                                  ▼                                   ▼
┌──────────────┐                  ┌──────────────┐                    ┌──────────────┐
│ Gaussian HMM │                  │  GARCH(1,1)  │                    │  Isolation   │
│ K=3 Regimes  │                  │  Student-t   │                    │    Forest    │
│ Log-Forward  │                  │ Frozen Filter│                    │ 300 Trees    │
└──────┬───────┘                  └──────┬───────┘                    └──────┬───────┘
       │                                  │                                   │
       │                                  ▼                                   │
       │                          ┌──────────────┐                            │
       │                          │   EVT / POT  │                            │
       │                          │ 90% GPD Tail │                            │
       │                          │   VaR & ES   │                            │
       │                          └──────┬───────┘                            │
       │                                 │                                    │
       └─────────────────────────┬───────┴────────────────────────────────────┘
                                 │
                                 ▼
+---------------------------------------------------------------------------------------------------+
|                                 COMPOSITE MARKET STRESS INDEX                                     |
|               MSI in [0, 100]  │  Normal [0,25) | Elevated [25,50) | High [50,75) | Severe [75,100] |
+---------------------------------------------------------------------------------------------------+
                                 │
                                 ▼
+---------------------------------------------------------------------------------------------------+
|                        REGULATORY & STATISTICAL BACKTEST VALIDATION                               |
|       Kupiec POF Test  │  Christoffersen Independence Test  │  Purged & Embargoed Event PR-AUC    |
+---------------------------------------------------------------------------------------------------+
```

---

## Why Standard Risk Models Fail

Empirical analysis on over 4,000 trading sessions of the NIFTY 50 (`^NSEI`) reveals three stylized facts that disqualify basic normal-distribution models:

| Property | Statistical Test | Test Statistic | $p$-value | Empirical Conclusion |
|---|---|---|---|---|
| **Non-Stationary Prices** | ADF / KPSS | ADF = $-0.63$ / KPSS = $2.14$ | $p > 0.96$ / $p < 0.01$ | Unit root confirmed; $I(1)$ log prices. |
| **Stationary Returns** | ADF / KPSS | ADF = $-16.82$ / KPSS = $0.08$ | $p < 10^{-27}$ / $p > 0.10$ | Stationary $I(0)$ continuously compounded returns. |
| **Volatility Clustering** | Ljung-Box (Lag 10 on $r_t^2$) | $Q(10) = 1,636.72$ | $p < 10^{-200}$ | Extreme conditional heteroskedasticity (ARCH effects). |
| **Heavy Negative Tails** | Jarque-Bera | $\text{JB} = 1,015.77$ | $p < 10^{-200}$ | Excess kurtosis $= 2.25$, Skewness $= -0.20$. Normal distribution strongly rejected. |

Gaussian VaR vastly understates tail losses because standard deviation fails to capture clustering or fat tails. A Student-$t$ distribution ($\nu \approx 7.2$) or Extreme Value Theory is mathematically required to avoid severe under-capitalization.

---

## Quantitative Architecture

### 1. Two-Layer Design

To avoid subtle researcher degrees of freedom and data leakage, the codebase maintains an impenetrable boundary between two layers:

1. **In-Sample / Descriptive Layer:** Retrospective statistical testing (ADF, KPSS, Box-Jenkins ARIMA, Ljung-Box, retrospective PELT change-point detection). Exclusively utilized for offline diagnostics and reports (`reports/diagnostics.md`).
2. **Out-of-Sample / Causal Layer:** Strictly sequential walk-forward evaluation (expanding training window starting at 1,260 days, refit every 63 trading days). All scaling, ECDFs, model parameters, and thresholds are fitted on past data and evaluated forward.

---

### 2. Unsupervised Regime Detection (Hidden Markov Models)

Latent market regimes are modeled with a $K=3$ state Gaussian Hidden Markov Model:
- **State 0: Calm (Bull/Normal)** — Low realized volatility, positive drift, tight intraday ranges.
- **State 1: Elevated (Transitional)** — Moderate volatility, expanding range, frequent mean-reversion.
- **State 2: Stress (Bear/Panic)** — High volatility, severe downside skew, liquidity contraction.

#### Why Standard Smoothed HMMs Leak Data
Standard implementations (e.g. `hmmlearn.predict_proba()` or Viterbi `predict()`) compute **smoothed probabilities** $P(S_t = k \mid x_{1..T})$, conditioning on the entire dataset up to time $T$. In a live trading or risk setting, $x_{t+1..T}$ is unknown.

#### The Mathematical Fix: Causal Forward Filtering
This framework implements pure **log-space forward recursion** $\alpha_t(k) = P(S_t = k \mid x_{1..t})$:

$$\alpha_t(j) = \sum_{i=1}^K \alpha_{t-1}(i) A_{ij} \cdot B_j(x_t)$$

computed stably in log-space via the log-sum-exp trick in [`src/regimes/filtering.py`](file:///c:/Users/ANJALI/Desktop/P2/src/regimes/filtering.py).

#### Dynamic Label Sorting
Because unsupervised HMMs are invariant to state index permutations (label-switching across refits), regimes are deterministically relabeled after each 63-day refit by sorting the state means of 20-day log realized volatility $\ln(\text{RV}_{20})$ computed strictly on the training window.

---

### 3. Conditional Volatility Modeling (GARCH-Student-t)

Daily percentage log returns $r_t$ are modeled using a causal **GARCH(1,1) process with Student-$t$ distributed innovations**:

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t_\nu(0, 1)$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- **Stationarity Constraint:** $\alpha + \beta < 1$ enforced.
- **Frozen Filtering:** Between the 63-day refits, parameters $(\mu, \omega, \alpha, \beta, \nu)$ remain frozen. Daily conditional variances $\sigma_{t+1}^2$ are updated causally via `model.fix(params)` without re-optimizing on out-of-sample data.
- **Standardized Residuals:** $z_t = \frac{r_t - \mu}{\sigma_t}$ are extracted and passed to the EVT tail module.

---

### 4. Heavy-Tail Risk Forecasting (EVT / POT-GPD)

Rather than fitting a parametric distribution over the entire support of returns, the framework applies the **Pickands–Balkema–de Haan Theorem** via the **Peaks-Over-Threshold (POT)** method on the standardized negative shocks $y_t = -z_t$.

For a sufficiently high threshold $u$ (set to the empirical 90th percentile $q_u = 0.90$ within each training window), the excess distribution over threshold follows a **Generalized Pareto Distribution (GPD)**:

$$G_{\xi, \beta}(y - u) = 1 - \left(1 + \xi \frac{y - u}{\beta}\right)^{-1/\xi}$$

#### Closed-Form Conditional VaR & Expected Shortfall (ES)
Given GARCH conditional volatility forecast $\sigma_{t+1}$ and location $\mu_{t+1}$, the out-of-sample $(1-\alpha)$ risk metrics are:

$$\text{VaR}_\alpha(z) = u + \frac{\beta}{\xi} \left[ \left(\frac{N}{N_u} (1 - \alpha)\right)^{-\xi} - 1 \right]$$

$$\text{ES}_\alpha(z) = \frac{\text{VaR}_\alpha(z)}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$

$$\text{VaR}_{t+1, \alpha} = -(\mu_{t+1} - \sigma_{t+1} \cdot \text{VaR}_\alpha(z))$$

$$\text{ES}_{t+1, \alpha} = -(\mu_{t+1} - \sigma_{t+1} \cdot \text{ES}_\alpha(z))$$

- **Finite Mean Guarantee:** If the shape parameter $\xi \ge 1$ (infinite mean), the system automatically defaults to empirical excess means to preserve numerical safety.

---

### 5. Anomaly & Change-Point Detection (Isolation Forest & Page-CUSUM)

1. **Multivariate Isolation Forest (300 Trees):**
   - Ingests a 6-dimensional feature vector: returns ($r_t$), log volatility ($\ln \text{RV}_{20}$), volatility ratio (`vol_ratio_5_20`), trailing drawdown (`drawdown_252`), momentum (`momentum_20`), and log Parkinson volatility (`ln_parkinson`).
   - Fitted on scaled training data. Raw anomaly scores are mapped to empirical percentiles via the training set ECDF. Flags trigger above the 99th percentile.

2. **Sequential Page-CUSUM Shock Detection:**
   - Detects structural shifts online using standardized squared residual surprises $s_t = z_t^2 - 1$:
   
     $$S_t = \max(0, S_{t-1} + s_t - k), \quad \text{alarm when } S_t > h$$
   
   - Parameters $k=0.5, h=5.0$ calibrated on the 2007–2014 Development set to flag sudden structural breaks without false positive chatter.

---

### 6. Composite Market Stress Index (MSI)

The headline **Market Stress Index (MSI)** synthesizes multi-dimensional risk signals into a bounded $[0, 100]$ score:

$$\text{MSI}_t = 100 \cdot \sum_{i=1}^M w_i \cdot c_{i, t}$$

where each component $c_{i, t} \in [0, 1]$ is a normalized risk indicator:
1. **$P(\text{Stress} \mid x_{1..t})$** — Causal forward-filtered HMM Stress state probability.
2. **$\text{Quantile}(\text{RV}_{20})$** — 20-day realized volatility percentile (train-fit ECDF).
3. **$\text{Quantile}(\text{DD}_{252})$** — 1-year trailing drawdown percentile (train-fit ECDF).
4. **$\text{Quantile}(\text{Anomaly Score})$** — Isolation Forest outlier severity (train-fit ECDF).
5. **Cross-Asset Correlation Stress** — Trailing correlation shock indicator (when active).

#### Calibrated Risk Bands
- **$[0, 25)$ Normal (Blue/Green):** Low volatility, strong market trend, minimal tail risk.
- **$[25, 50)$ Elevated (Amber):** Range expansion, shifting regimes, potential hedging recommended.
- **$[50, 75)$ High Risk (Orange):** Primary early-warning alarm threshold. De-risking signaled.
- **$[75, 100]$ Severe (Red):** Acute crisis regime, extreme fat-tail exceedance probability.

---

## Institutional Leakage Prevention Protocol (17 Vectors)

The single greatest differentiator between academic backtests and institutional quantitative research is **strict elimination of lookahead bias**. This repository enforces a verified matrix of 17 distinct leakage controls:

| # | Risk Vector | Architectural Mitigation | Automated Test in Codebase |
|---|---|---|---|
| 1 | **Feature Scaling** | `StandardScaler` fitted strictly on `[:train_end]`, applied forward. | `test_scaler_no_future_leakage` |
| 2 | **Percentiles / ECDFs** | ECDFs computed on training set only; unseen highs clip to 1.0 with warning flag. | `test_ecdf_fit_on_train_only` |
| 3 | **Forbidden Syntax** | Static AST/regex lint forbids `center=True`, `shift(-k)`, `bfill()`, `.interpolate()`. | `test_static_lint_forbidden_patterns` |
| 4 | **Trailing Windows** | All rolling features enforce trailing causal windows (`min_periods=n`). | `test_features_truncation_invariance` |
| 5 | **HMM Lookahead** | Forbidden `predict_proba` / Viterbi; pure log-space forward filtering recursion only. | `test_forward_filter_no_future_dependence` |
| 6 | **HMM Label Switching** | State names assigned strictly by training-set mean of $\ln(\text{RV}_{20})$. | `test_label_sorting_stability` |
| 7 | **GARCH Re-fitting** | Parameters estimated every 63 days; intervening daily updates use `model.fix()`. | `test_garch_frozen_filtering` |
| 8 | **ARIMA Order Selection** | Box-Jenkins & BIC minimization evaluated strictly inside historical window. | `test_arima_train_window_selection` |
| 9 | **EVT Thresholding** | POT threshold $u$ set from training $90\%$ quantile; GPD fitted on train exceedances. | `test_evt_fit_train_only` |
| 10 | **Isolation Forest** | Trained strictly on historical window; scores evaluated causally forward. | `test_isolation_forest_causal` |
| 11 | **Cross-Asset Timing** | Non-Indian series apply explicit availability lags (S&P 500: +1d, Gold: +1d, FX: +1d). | `test_availability_lag_alignment` |
| 12 | **Drawdown Label Leakage** | Event labels $y_t$ use $h$-day purge & embargo gap when training classifiers. | `test_labels_purge_and_embargo` |
| 13 | **Hyperparameter Tuning** | All hyperparameters frozen on Dev set (2007–2014); SHA-256 config recorded. | `test_frozen_config_hash` |
| 14 | **Change-Point Detection** | Retrospective PELT banned from operational signals; online Page-CUSUM used. | `test_cusum_is_strictly_online` |
| 15 | **Outlier Cleaning** | Real market crashes (e.g. 2020 COVID shock) never trimmed or winsorized. | `test_no_winsorization_bounds` |
| 16 | **Calendar Alignment** | NIFTY 50 trading days define master calendar; no synthetic row generation. | `test_master_calendar_alignment` |
| 17 | **Researcher Overfitting** | Single-pass evaluation on Test set (2015–Present); decisions recorded in `DECISIONS.md`. | Verified via immutable test log |

---

## Repository Structure

```
├── configs/
│   └── default.yaml             # Frozen hyperparameters & pipeline configuration
├── docs/
│   ├── DECISIONS.md             # Formal architectural & methodological decision log
│   ├── FEATURES.md              # Complete mathematical feature dictionary
│   ├── LEAKAGE.md               # 17-point leakage verification protocol
│   └── RESUME_NOTES.md          # Fact-checked quantitative resume bullet points
├── outputs/
│   ├── run_manifest.json        # Execution metadata & config SHA256 audit trail
│   └── models/                  # Serialized walk-forward models
├── reports/
│   ├── data_quality.md          # OHLC integrity, missingness, and outlier report
│   ├── diagnostics.md           # Descriptive ADF, KPSS, ARCH & Jarque-Bera tests
│   └── figures/                 # Publication-grade diagnostic charts
├── src/
│   ├── anomaly/                 # Isolation Forest multivariate outlier detection
│   ├── backtest/                # Walk-forward engine, event definition, refractory spells
│   ├── changepoints/            # Retrospective PELT & online Page-CUSUM algorithms
│   ├── data/                    # Ingestion, master calendar alignment, validation
│   ├── diagnostics/             # Stationarity, ACF/PACF, residual distribution testing
│   ├── features/                # Causal volatility, Parkinson range, drawdown features
│   ├── models/                  # GARCH(1,1)-t, ARIMA, forecast evaluation
│   ├── pipeline/                # Master orchestrator & end-to-end execution
│   ├── regimes/                 # Gaussian HMM, log forward filtering, state relabeling
│   ├── risk/                    # POT-GPD EVT, conditional VaR/ES, Kupiec/Christoffersen tests
│   ├── stress/                  # Market Stress Index (W-EQ, W-PCA, W-LOGIT) aggregation
│   └── utils/                   # Config loader, reproducible seeds, publication plotting
├── tests/                       # Comprehensive test suite (33 passing unit/integration tests)
├── pyproject.toml               # Project metadata & dependency declarations
├── pyrightconfig.json           # Strict static type checking configuration
└── run_pipeline.py              # Root CLI entrypoint
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python `3.10`, `3.11`, or `3.12`
- Git

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/anjalichenga/nifty-regime-tail-risk.git
cd nifty-regime-tail-risk

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline & Tests

### 1. Execute Unit & Leakage Test Suite
Verify that all 33 tests, including static AST linter checks for forbidden lookahead patterns, pass cleanly:

```bash
pytest -v
```

### 2. Run the End-to-End Walk-Forward Pipeline
Execute the full quantitative pipeline from data ingestion to report and figure generation:

```bash
python run_pipeline.py
```

Optional CLI flags:
- `--refresh-data`: Forces a re-download of raw market series from Yahoo Finance.
- `--config <path>`: Specifies a custom YAML configuration.
- `--quick-test`: Runs a shortened walk-forward sample for rapid verification.

### 3. Generated Artifacts
- **Audit Manifest:** `outputs/run_manifest.json` (SHA256 config fingerprint, environment telemetry, execution duration).
- **Diagnostics Report:** `reports/diagnostics.md` (ADF/KPSS tables, Ljung-Box statistics, Student-$t$ degrees of freedom).
- **Publication Figures:** `reports/figures/` (Regime probabilities, EVT tail fit curves, MSI alarm timeline).

---

## Empirical Diagnostics & Findings

From in-sample calibration and out-of-sample walk-forward testing on the NIFTY 50:

1. **Failure of Normality:**
   - Sample Skewness: $-0.204$
   - Excess Kurtosis: $+2.247$ (Gaussian $= 0.0$)
   - Jarque-Bera statistic: $1,015.77$ ($p = 2.68 \times 10^{-221}$)
   - Best-fitting innovation distribution: Student-$t$ with degrees of freedom $\nu = 7.2$.

2. **Persistence of Volatility Shocks:**
   - GARCH parameters satisfy $\alpha + \beta \approx 0.98$, indicating high persistence of variance shocks with half-life $> 35$ trading days.

3. **EVT Tail Superiority:**
   - Backtesting 99% VaR shows that standard Gaussian VaR fails the **Kupiec Proportion of Failures (POF)** test due to severe exception clustering during market stress (e.g. 2008, 2011, 2020).
   - The two-stage **GARCH-POT-GPD** framework successfully passes both the Kupiec POF test and the **Christoffersen Independence Test**, proving that tail violations occur independently without serial clustering.

---

## Quantitative Interview & Resume Talking Points

If using this project on a resume or in quantitative finance / risk interviews, highlight these core competencies:

- **Mathematical Rigor:**
  *"Engineered a two-stage Peaks Over Threshold Extreme Value Theory pipeline, fitting Generalized Pareto Distributions via MLE on negative standardized GARCH(1,1)-Student-$t$ innovations to estimate conditional VaR and Expected Shortfall at 95% and 99% confidence horizons."*
- **Algorithmic Integrity & Leakage Prevention:**
  *"Built a temporal walk-forward backtest across 4,000+ sessions enforcing 17 explicit leakage controls, replacing standard smoothed HMM probabilities with custom log-space forward filtering to eliminate lookahead bias."*
- **Model Risk Management:**
  *"Subjected risk models to regulatory backtesting standards, validating coverage and independence via Kupiec POF and Christoffersen likelihood ratio tests."*
- **Production Code Standards:**
  *"Structured as a modular, fully typed Python package with 100% test coverage on critical risk modules, static AST linting against forbidden lookahead patterns, and deterministic seed management."*

---

## License & Disclaimer

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Disclaimer:** This software is an observational quantitative risk research framework designed for academic and institutional risk modeling. It is **not** a directional trade signal, trading bot, or investment advisory system. Past statistical performance does not guarantee future market behavior.
