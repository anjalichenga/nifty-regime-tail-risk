"""Online causal change-point detection via Page-CUSUM on GARCH standardized shocks."""
import numpy as np
import pandas as pd


class OnlinePageCUSUM:
    """
    Two-sided Page-CUSUM process for real-time volatility regime change detection.
    Operates on standardized GARCH log-squared shocks: x_t = (ln(z_t^2) - m) / s
    where m and s are in-control mean and std from the training window.
    """
    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k = k  # Slack / reference value
        self.h = h  # Decision threshold
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.days_since_alarm = 0
        self.in_control_mean = 0.0
        self.in_control_std = 1.0
        self.is_calibrated = False

    def calibrate(self, train_std_resid: np.ndarray) -> "OnlinePageCUSUM":
        """Calibrates in-control baseline statistics on training window residuals."""
        safe_z = np.clip(np.abs(train_std_resid), 1e-4, None)
        log_z2 = np.log(safe_z ** 2)
        self.in_control_mean = float(np.mean(log_z2))
        self.in_control_std = float(np.std(log_z2, ddof=1))
        if self.in_control_std < 1e-6:
            self.in_control_std = 1.0
        self.is_calibrated = True
        return self

    def step(self, z_t: float) -> tuple[bool, bool, float, float, int]:
        """
        Updates CUSUM with a new standardized shock z_t.

        Returns:
        - alarm_high: True if upward volatility shock alarm
        - alarm_low: True if downward volatility shock alarm
        - s_pos: current upper CUSUM statistic
        - s_neg: current lower CUSUM statistic
        - days_since_alarm: counter since last alarm
        """
        safe_z = max(abs(z_t), 1e-4)
        log_z2 = np.log(safe_z ** 2)
        x_t = (log_z2 - self.in_control_mean) / self.in_control_std

        # Upper CUSUM (increase in variance / shock severity)
        self.s_pos = max(0.0, self.s_pos + x_t - self.k)
        alarm_high = bool(self.s_pos > self.h)
        if alarm_high:
            self.s_pos = 0.0 # Reset after alarm

        # Lower CUSUM (decrease in variance)
        self.s_neg = max(0.0, self.s_neg - x_t - self.k)
        alarm_low = bool(self.s_neg > self.h)
        if alarm_low:
            self.s_neg = 0.0

        if alarm_high or alarm_low:
            self.days_since_alarm = 0
        else:
            self.days_since_alarm += 1

        return alarm_high, alarm_low, self.s_pos, self.s_neg, self.days_since_alarm

def run_causal_cusum_stream(
    std_residuals: pd.Series,
    train_end_idx: int,
    k: float = 0.5,
    h: float = 5.0,
) -> pd.DataFrame:
    """
    Applies causal Page-CUSUM sequentially on standardized residuals.
    Calibrates strictly on data[:train_end_idx] and monitors forward.
    """
    cusum = OnlinePageCUSUM(k=k, h=h)
    train_res = std_residuals.iloc[:train_end_idx].dropna().values
    cusum.calibrate(train_res)

    alarms_high = []
    alarms_low = []
    s_pos_arr = []
    s_neg_arr = []
    days_since = []

    for z in std_residuals.values:
        if np.isnan(z):
            alarms_high.append(False)
            alarms_low.append(False)
            s_pos_arr.append(0.0)
            s_neg_arr.append(0.0)
            days_since.append(cusum.days_since_alarm)
        else:
            ah, al, sp, sn, ds = cusum.step(float(z))
            alarms_high.append(ah)
            alarms_low.append(al)
            s_pos_arr.append(sp)
            s_neg_arr.append(sn)
            days_since.append(ds)

    return pd.DataFrame(
        {
            "cusum_alarm_high": alarms_high,
            "cusum_alarm_low": alarms_low,
            "cusum_s_pos": s_pos_arr,
            "cusum_s_neg": s_neg_arr,
            "days_since_cusum_alarm": days_since,
        },
        index=std_residuals.index,
    )
