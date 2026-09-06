"""Prediction intervals for models that don't provide one natively
(XGBoost), plus a calibration check usable for *any* interval - SARIMAX's
native model-based `conf_int` (see `eval/backtest.py`'s `interval_alpha`)
or XGBoost's empirical residual-quantile interval below.

XGBoost's point-prediction API has no native interval, so this uses a
rolling window of *already-realized* out-of-sample residuals (no
look-ahead: the interval for week i only ever uses residuals from weeks
strictly before it) to set the interval width empirically. A rolling
(not expanding) window is used deliberately: this series' volatility has
grown substantially as the price level roughly tripled since 2003, so an
expanding window would dilute recent, more-volatile residuals with a
long tail of calmer historical ones and understate current uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_interval_backtest(
    df: pd.DataFrame,
    pred_col: str,
    alpha: float = 0.05,
    window: int = 104,
    min_history: int = 52,
) -> pd.DataFrame:
    """Add `lower`/`upper` columns to an existing backtest results frame,
    using the trailing `window` (capped, not expanding past that) realized
    residuals as of each row - never including that row's own residual.
    Rows before `min_history` residuals have accumulated get NaN bounds.
    """
    out = df.reset_index(drop=True).copy()
    residuals = (out["actual"] - out[pred_col]).to_numpy()
    pred = out[pred_col].to_numpy()

    lower = np.full(len(out), np.nan)
    upper = np.full(len(out), np.nan)
    for i in range(len(out)):
        if i < min_history:
            continue
        past_residuals = residuals[max(0, i - window) : i]
        lower[i] = pred[i] + np.quantile(past_residuals, alpha / 2)
        upper[i] = pred[i] + np.quantile(past_residuals, 1 - alpha / 2)

    out["lower"] = lower
    out["upper"] = upper
    return out


def compute_coverage(df: pd.DataFrame, lower_col: str = "lower", upper_col: str = "upper") -> dict:
    """Empirical coverage (fraction of `actual` within `[lower, upper]`)
    and interval width (mean *and* median - reported separately since a
    handful of numerically unstable intervals can drag the mean far from
    what a typical interval actually looks like), over rows where both
    bounds are present. Compares directly against the interval's nominal
    level - e.g. a 95% interval should show ~0.95 coverage if well-
    calibrated."""
    valid = df.dropna(subset=[lower_col, upper_col])
    within = (valid["actual"] >= valid[lower_col]) & (valid["actual"] <= valid[upper_col])
    width = valid[upper_col] - valid[lower_col]
    return {
        "n": len(valid),
        "coverage": float(within.mean()),
        "mean_width": float(width.mean()),
        "median_width": float(width.median()),
    }
