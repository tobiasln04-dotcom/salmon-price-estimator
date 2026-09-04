"""Autoregressive + seasonality features for the XGBoost weekly model.

All lag/rolling features use only information already known at forecast
time (t-1 and earlier) - same leakage discipline as `weekly_panel.py`'s
1-week exog lag. The target is the 1-week-ahead log-return, not the raw
price level (see config/model.yaml for why) - reconstruct the price-level
prediction as `price_{t-1} * exp(predicted_return)` before scoring.

Works on any frame with `week_start_date` and the price column - the
plain SSB frame for the autoregressive variant, or `weekly_panel.py`'s
output for the exogenous variant, whose already-lagged exog columns just
pass through untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "log_return_target"


def feature_columns(lags: list[int], rolling_windows: list[int]) -> list[str]:
    """Deterministic names of the autoregressive/seasonality feature columns
    `build_features` produces, in a fixed order."""
    names = [f"price_lag_{lag}" for lag in lags]
    names += [f"rolling_mean_{w}" for w in rolling_windows]
    names += [f"rolling_std_{w}" for w in rolling_windows]
    names += ["log_return_lag_1", "week_of_year_sin", "week_of_year_cos"]
    return names


def build_features(
    df: pd.DataFrame,
    price_col: str,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Add autoregressive/seasonality feature columns and the log-return
    target to `df`. `df` must be sorted by `week_start_date` ascending with
    no gaps (lags are positional, not date-based).

    Rows without enough history for the largest lag/rolling window contain
    NaNs and should be dropped by the caller before training.
    """
    out = df.copy()
    price = out[price_col]

    for lag in lags:
        out[f"price_lag_{lag}"] = price.shift(lag)

    for window in rolling_windows:
        # Exclude the current week from its own rolling stat.
        shifted = price.shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()

    out["log_return_lag_1"] = np.log(price.shift(1) / price.shift(2))

    week_of_year = out["week_start_date"].dt.isocalendar().week.astype(float)
    out["week_of_year_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    out["week_of_year_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    out[TARGET_COL] = np.log(price / price.shift(1))

    return out
