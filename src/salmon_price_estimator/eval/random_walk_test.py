"""Formal tests of the "close to a random walk" claim this project's
README leans on to explain why naive is such a tough benchmark.

Two complementary tests, both already available via statsmodels (no new
dependency):

- **ADF (Augmented Dickey-Fuller) on log(price)**: tests for a unit
  root. Failing to reject the null (unit root present) is consistent
  with - necessary but not sufficient for - non-stationary, random-walk-
  like behavior in the price level.
- **Ljung-Box on weekly log-returns**: tests whether the *return* series
  shows significant autocorrelation. A random walk's increments should
  be unpredictable - no significant autocorrelation at any lag - which is
  the more direct evidence for "hard to forecast," since it's actually
  about predictability of the thing being forecast (the change), not
  just the level's stochastic trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def adf_test(price: pd.Series) -> dict:
    """Augmented Dickey-Fuller test on log(price). Lag order chosen by AIC."""
    log_price = np.log(price)
    statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(
        log_price, autolag="AIC", result_object=False
    )
    return {
        "test": "ADF (log price level)",
        "statistic": statistic,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_value_5pct": critical_values["5%"],
        "unit_root_rejected_at_5pct": bool(p_value < 0.05),
    }


def ljung_box_test(price: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation in weekly log-returns, at each
    lag in `lags`. Returns one row per lag: `lag`, `lb_stat`, `lb_pvalue`,
    `autocorrelation_rejected_at_5pct`."""
    log_return = np.log(price / price.shift(1)).dropna()
    result = acorr_ljungbox(log_return, lags=lags, return_df=True)
    result = result.reset_index(names="lag")
    result["autocorrelation_rejected_at_5pct"] = result["lb_pvalue"] < 0.05
    return result
