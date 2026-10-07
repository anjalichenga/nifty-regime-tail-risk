# Quantitative Resume Notes & Technical Highlights (RESUME_NOTES.md)

This document provides concise, fact-checked resume bullets detailing the exact quantitative methods implemented and validated in this project. All figures will be populated directly from verified pipeline outputs.

### Headline Accomplishments & Architecture
- **Market Regime Detection:** Engineered an unsupervised regime-classification framework for NIFTY 50 based on Gaussian Hidden Markov Models ($K=3$: Calm, Elevated, Stress), implementing custom numerical forward-filtering $\alpha_t(k) = P(S_t=k \mid x_{1..t})$ in log space to eliminate lookahead bias inherent in standard smoothed probabilities.
- **Tail-Risk Forecasting (EVT & VaR/ES):** Implemented a two-stage Peaks Over Threshold (POT) Extreme Value Theory pipeline, fitting Generalized Pareto Distributions (GPD) via MLE on negative standardized GARCH(1,1)-Student-$t$ innovations to generate conditional Value-at-Risk (VaR) and Expected Shortfall (ES) at 95% and 99% confidence horizons.
- **Walk-Forward Validation & Leakage Prevention:** Built a temporal walk-forward backtesting engine with expanding training windows (1,260-day warm-up, 63-day refits) enforcing 17 explicit leakage controls, including frozen-parameter GARCH filtering, train-fit ECDFs, availability lags for cross-asset series, and purged/embargoed event labeling.
- **Market Stress Index (MSI):** Designed and calibrated a composite 0–100 Market Stress Index fusing HMM regime probabilities, conditional volatility, drawdown severity, and Isolation Forest anomaly scores, achieving disciplined early-warning detection for market drawdown episodes.
- **Interactive Risk Dashboard:** Developed a multi-page Streamlit & Plotly risk dashboard displaying real-time filtered regime probabilities, volatility term structures, EVT tail fits, backtest metrics, and alarm lead-time distributions across historical market episodes.

*(Note: Exact quantitative metrics such as Kupiec $p$-values, Christoffersen LR test statistics, PR-AUC, and lead-time distributions are generated dynamically by `src/utils/render_results.py` and reflected in `reports/RESULTS.md`.)*
