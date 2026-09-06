"""Diebold-Mariano test: is a forecast's edge over another actually
statistically significant, or could it be noise?

Every "beats naive" claim in this project reports a point estimate
(MAPE/RMSE), but some of those margins are tiny (e.g. the univariate
SARIMAX's 3.54% vs. naive's 3.59% MAPE) - this tests whether the
difference in squared forecast errors is distinguishable from zero,
using a Newey-West HAC variance estimate and the Harvey-Leybourne-Newbold
(1997) small-sample correction (compared against a Student's t
distribution rather than the standard normal, which is more accurate at
finite sample sizes).

Reference: Diebold, F.X. and Mariano, R.S. (1995), "Comparing Predictive
Accuracy", Journal of Business & Economic Statistics 13(3).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test comparing forecast errors from model A
    against model B, under `power`-loss (2 = squared error, matching RMSE;
    1 = absolute error, matching MAE).

    `h` is the forecast horizon (1 for the one-step-ahead forecasts used
    throughout this project) - it sets the number of autocorrelation lags
    included in the long-run variance estimate (h-1), per the original
    Diebold-Mariano specification.

    A negative statistic means model A's average loss is lower (better)
    than model B's. Returns `(dm_statistic, p_value)`.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)

    loss_a = np.abs(errors_a) ** power
    loss_b = np.abs(errors_b) ** power
    d = loss_a - loss_b
    n = len(d)
    d_mean = d.mean()

    long_run_variance = np.mean((d - d_mean) ** 2)  # gamma_0, divisor n
    for lag in range(1, h):
        gamma_lag = np.sum((d[lag:] - d_mean) * (d[:-lag] - d_mean)) / n
        long_run_variance += 2 * gamma_lag

    # Identical error series -> zero variance -> genuinely undefined (0/0),
    # not a bug; let it produce nan quietly rather than warn.
    with np.errstate(invalid="ignore"):
        dm_stat = d_mean / np.sqrt(long_run_variance / n)

    # Harvey, Leybourne, Newbold (1997) small-sample correction.
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction

    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_value)
