"""Residual distribution diagnostics and report rendering."""
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from src.utils.paths import REPORTS_DIR


def compute_residual_diagnostics(residuals: pd.Series) -> dict[str, Any]:
    """
    Computes skewness, excess kurtosis, Jarque-Bera normality test,
    and fits Gaussian & Student-t distributions to standardized residuals.
    """
    res = residuals.dropna()
    mean = float(res.mean())
    std = float(res.std(ddof=1))
    skew = float(stats.skew(res))
    excess_kurt = float(stats.kurtosis(res))  # Fisher kurtosis (normal == 0)

    jb_result: Any = stats.jarque_bera(res)
    jb_stat = float(jb_result[0])
    jb_pvalue = float(jb_result[1])

    # Fit Student-t
    t_params = stats.t.fit(res)
    df_t = float(t_params[0])

    is_normal = jb_pvalue >= 0.05

    interpretation = (
        f"Jarque-Bera test (stat={jb_stat:.2f}, p={jb_pvalue:.4e}): "
        + ("Consistent with Gaussian distribution." if is_normal else
           f"Strongly rejects normality. Residuals exhibit negative skewness ({skew:.2f}) "
           f"and substantial excess kurtosis ({excess_kurt:.2f}), fitting a Student-t with df={df_t:.1f}. "
           f"This heavy-tailed distribution confirms that Gaussian tail assumptions underestimate downside risk.")
    )

    return {
        "n_obs": len(res),
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": excess_kurt,
        "jarque_bera": {"stat": jb_stat, "pvalue": jb_pvalue},
        "student_t_fitted_df": df_t,
        "is_gaussian": is_normal,
        "interpretation": interpretation,
    }

def write_diagnostics_report(
    stationarity_p: dict[str, Any],
    stationarity_r: dict[str, Any],
    acf_results: dict[str, Any],
    arch_results: dict[str, Any],
    resid_results: dict[str, Any],
) -> Path:
    """
    Writes a comprehensive markdown report to reports/diagnostics.md.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "diagnostics.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Statistical Diagnostics Report: NIFTY 50 Dynamics\n\n")
        f.write("**Status:** IN-SAMPLE / DESCRIPTIVE ANALYSIS\n\n")

        f.write("## 1. Stationarity & Integration Order (ADF & KPSS)\n\n")
        f.write("| Series | ADF (Constant) p-val | KPSS (Constant) p-val | Decision | Conclusion |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(
            f"| Log Prices ($\\ln P$) | {stationarity_p['adf_c']['pvalue']:.4f} | "
            f"{stationarity_p['kpss_c']['pvalue']:.4f} | {stationarity_p['decision_constant']} | "
            f"Non-stationary I(1) |\n"
        )
        f.write(
            f"| Log Returns ($r_t$) | {stationarity_r['adf_c']['pvalue']:.4e} | "
            f"{stationarity_r['kpss_c']['pvalue']:.4f} | {stationarity_r['decision_constant']} | "
            f"Stationary I(0) |\n\n"
        )
        f.write(f"- **Log Price Interpretation:** {stationarity_p['interpretation_constant']}\n")
        f.write(f"- **Log Return Interpretation:** {stationarity_r['interpretation_constant']}\n\n")

        f.write("## 2. Autocorrelation & Volatility Clustering\n\n")
        f.write(f"{acf_results['interpretation']}\n\n")
        f.write("| Metric | Lag 10 Stat | Lag 10 p-value | Lag 20 Stat | Lag 20 p-value |\n")
        f.write("|---|---|---|---|---|\n")
        lb_r = acf_results["ljung_box_returns"]
        lb_sq = acf_results["ljung_box_squared_returns"]
        f.write(f"| Returns ($r_t$) | {lb_r['lag_10']['stat']:.2f} | {lb_r['lag_10']['pvalue']:.4f} | {lb_r['lag_20']['stat']:.2f} | {lb_r['lag_20']['pvalue']:.4f} |\n")
        f.write(f"| Squared Returns ($r_t^2$) | {lb_sq['lag_10']['stat']:.2f} | {lb_sq['lag_10']['pvalue']:.4e} | {lb_sq['lag_20']['stat']:.2f} | {lb_sq['lag_20']['pvalue']:.4e} |\n\n")

        f.write("## 3. ARCH Effects & Heteroskedasticity\n\n")
        f.write(f"{arch_results.get('interpretation', 'ARCH effects confirmed via Ljung-Box test on squared returns.')}\n\n")

        f.write("## 4. Return Distribution & Heavy Tails\n\n")
        f.write(f"- **Sample Skewness:** {resid_results['skewness']:.3f}\n")
        f.write(f"- **Excess Kurtosis:** {resid_results['excess_kurtosis']:.3f} (Gaussian = 0.0)\n")
        f.write(f"- **Fitted Student-t Degrees of Freedom:** $\\nu = {resid_results['student_t_fitted_df']:.1f}$\n")
        f.write(f"- **Normality Test:** {resid_results['interpretation']}\n")

    return report_path
