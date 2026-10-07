# Feature Dictionary & Definitions (FEATURES.md)

All features implemented in this framework are strictly causal, computed over trailing rolling windows (`center=False`) with zero lookahead bias. Features are computed without backward shifts, backward fills, or interpolation.

| Feature Identifier | Mathematical Definition | Economic / Financial Interpretation | Window Length | Leakage Mitigation & Bounds | Model Usage (HMM / IF / MSI) |
|---|---|---|---|---|---|
| `return` ($r_t$) | $\ln(P_t / P_{t-1})$ | Daily continuously compounded price change. Weakly predictable mean shock. | 1 day | Strictly uses $P_t$ and $P_{t-1}$. First observation is NaN. | **HMM**, **IF** |
| `return_pct` | $100 \cdot r_t$ | Percentage log return for numerical stability in GARCH estimation. | 1 day | Causal 1-day difference. | GARCH target |
| `rv_5` | $\sqrt{252} \cdot \text{std}(r_{t-4..t}, \text{ddof}=1)$ | Short-term 1-week realized volatility proxy. Rapid reaction to shocks. | 5 days | Trailing window, `min_periods=5`. | **IF** (via ratio) |
| `rv_20` | $\sqrt{252} \cdot \text{std}(r_{t-19..t}, \text{ddof}=1)$ | 1-month realized annualized volatility benchmark. Standard risk measure. | 20 days | Trailing window, `min_periods=20`. | Baseline |
| `rv_60` | $\sqrt{252} \cdot \text{std}(r_{t-59..t}, \text{ddof}=1)$ | 1-quarter realized annualized volatility. Baseline persistence. | 60 days | Trailing window, `min_periods=60`. | Vol Panel |
| `ln_rv20` | $\ln(\text{rv\_20})$ | Log annualized volatility. Approximately Gaussian-distributed moment. | 20 days | Non-negative clipping before log transform. | **HMM**, **IF**, **MSI** |
| `vol_ratio_5_20` | $\ln(\text{rv\_5} / \text{rv\_20})$ | Volatility term-structure slope / shock acceleration indicator. | 5d / 20d | Trailing ratio of causal standard deviations. | **IF** |
| `parkinson_daily` | $\frac{(\ln(H_t / L_t))^2}{4 \ln 2}$ | Intraday high-low variance proxy (more efficient than close-to-close). | 1 day | Intraday OHLC strictly from session $t$. Requires $H_t \ge L_t > 0$. | Vol Panel |
| `parkinson_20` | $\sqrt{252} \cdot \sqrt{\frac{1}{20}\sum_{i=0}^{19} \text{parkinson\_daily}_{t-i}}$ | 20-day smoothed Parkinson volatility proxy. | 20 days | Trailing rolling mean of daily Parkinson variance. | Vol Panel |
| `ln_parkinson` | $\ln(\text{parkinson\_20})$ | Log 20-day Parkinson volatility for anomalous range expansion. | 20 days | Causal logarithmic transform. | **IF** |
| `downside_semi_dev` | $\sqrt{252} \cdot \sqrt{\frac{1}{20}\sum_{i=0}^{19} (\min(r_{t-i}, 0))^2}$ | Downside semi-deviation (20d). Captures asymmetric downside volatility. | 20 days | Trailing negative-shock squared dispersion. | Vol Panel |
| `drawdown` | $\frac{P_t}{\max_{s \le t} P_s} - 1$ | Path-dependent cumulative loss from all-time historical peak. | Full history to $t$ | Running maximum strictly from index start to $t$. | **HMM** |
| `drawdown_252` | $\frac{P_t}{\max_{s \in [t-251, t]} P_s} - 1$ | 1-year trailing drawdown. Independent of arbitrary origin date. | 252 days | Rolling maximum over trailing 252 trading days. | **IF**, **MSI** |
| `rolling_ret_5` | $\sum_{i=0}^{4} r_{t-i}$ | 1-week cumulative log return. Short-term trend/reversion. | 5 days | Trailing rolling sum. | Feature Set |
| `momentum_20` | $\sum_{i=0}^{19} r_{t-i}$ | 1-month momentum / trend signal. Trend exhaustion / crash indicator. | 20 days | Trailing rolling sum. | **IF** |
| `momentum_60` | $\sum_{i=0}^{59} r_{t-i}$ | 1-quarter medium-term momentum. | 60 days | Trailing rolling sum. | Feature Set |
| `ln_p_sma50` | $\ln(P_t / \text{SMA}_{50, t})$ | Medium-term trend divergence from 50-day moving average. | 50 days | Trailing rolling mean of close price. | Technical |
| `ln_sma50_sma200` | $\ln(\text{SMA}_{50, t} / \text{SMA}_{200, t})$ | Long-term macro regime divergence (Golden / Death Cross proxy). | 200 days | Trailing ratio of 50-day and 200-day rolling means. | Technical |
| `vol_zscore_252` | $\frac{\ln(\text{rv\_20}_t) - \mu_{252}(\ln \text{rv\_20})}{\sigma_{252}(\ln \text{rv\_20})}$ | Standardized volatility relative to trailing 1-year distribution. | 252 days | Trailing rolling mean & std over 252 days. | Technical |
| `volume_z_60` | $\frac{\ln(V_t) - \mu_{60}(\ln V)}{\sigma_{60}(\ln V)}$ | Normalized trading volume surge. Active only if valid coverage $\ge 90\%$. | 60 days | Trailing 60-day stats. Disabled gracefully if coverage $<90\%$. | **IF** (optional) |
| `corr_banknifty_60` | $\text{corr}(r_{\text{NIFTY}}, r_{\text{BANKNIFTY}})$ | 60-day rolling correlation with Bank NIFTY (financial sector stress). | 60 days | Synchronous close (lag 0). Trailing covariance / std. | Cross-Asset |
| `corr_sp500_60` | $\text{corr}(r_{\text{NIFTY}, t}, r_{\text{SP500}, t-1})$ | Global equity co-movement. Lags S&P by 1 trading day. | 60 days | Availability lag 1 applied strictly before alignment. | Cross-Asset |
| `corr_usdinr_60` | $\text{corr}(r_{\text{NIFTY}, t}, r_{\text{USDINR}, t-1})$ | Currency risk correlation. Lags USD/INR fix by 1 trading day. | 60 days | Availability lag 1 applied strictly before alignment. | Cross-Asset |
| `vix_level` | $\text{INDIAVIX}_t$ | Baseline implied volatility index. Synchronous closing value. | 1 day | Baseline comparator and optional MSI variant. | Baseline |
| `vix_change_20` | $\ln(\text{VIX}_t / \text{VIX}_{t-20})$ | 20-day change in implied volatility fear gauge. | 20 days | Trailing difference in log VIX. | Cross-Asset |
| `skew_60` & `kurt_60` | 60-day rolling sample skewness & kurtosis | Third and fourth moment shifts. Highly noisy; **descriptive only**. | 60 days | Excluded from models due to estimation variance. | Descriptive only |
