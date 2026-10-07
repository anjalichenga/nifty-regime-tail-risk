# Statistical Diagnostics Report: NIFTY 50 Dynamics

**Status:** IN-SAMPLE / DESCRIPTIVE ANALYSIS

## 1. Stationarity & Integration Order (ADF & KPSS)

| Series | ADF (Constant) p-val | KPSS (Constant) p-val | Decision | Conclusion |
|---|---|---|---|---|
| Log Prices ($\ln P$) | 0.9663 | 0.0100 | Unit Root | Non-stationary I(1) |
| Log Returns ($r_t$) | 7.9332e-28 | 0.1000 | Stationary | Stationary I(0) |

- **Log Price Interpretation:** Both tests agree on non-stationarity / unit root (ADF fails to reject; KPSS rejects stationarity).
- **Log Return Interpretation:** Both tests agree on stationarity (ADF rejects unit root; KPSS fails to reject stationarity).

## 2. Autocorrelation & Volatility Clustering

Ljung-Box p-values at lag 10: returns p=0.0000, squared returns p=0.0000e+00. Returns exhibit weak linear autocorrelation, but squared/absolute returns demonstrate highly significant, persistent autocorrelation. This confirms the presence of volatility clustering (ARCH effects), justifying explicit conditional heteroskedasticity modeling (GARCH).

| Metric | Lag 10 Stat | Lag 10 p-value | Lag 20 Stat | Lag 20 p-value |
|---|---|---|---|---|
| Returns ($r_t$) | 39.62 | 0.0000 | 72.18 | 0.0000 |
| Squared Returns ($r_t^2$) | 1636.72 | 0.0000e+00 | 2138.33 | 0.0000e+00 |

## 3. ARCH Effects & Heteroskedasticity

ARCH effects confirmed via Ljung-Box test on squared returns.

## 4. Return Distribution & Heavy Tails

- **Sample Skewness:** -0.204
- **Excess Kurtosis:** 2.247 (Gaussian = 0.0)
- **Fitted Student-t Degrees of Freedom:** $\nu = 7.2$
- **Normality Test:** Jarque-Bera test (stat=1015.77, p=2.6831e-221): Strongly rejects normality. Residuals exhibit negative skewness (-0.20) and substantial excess kurtosis (2.25), fitting a Student-t with df=7.2. This heavy-tailed distribution confirms that Gaussian tail assumptions underestimate downside risk.
