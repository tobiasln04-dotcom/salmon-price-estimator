"""SARIMAX weekly baseline: model-construction helpers only.

The walk-forward backtest loop (expanding window, periodic refit) lives in
`eval.backtest` - this module only knows how to fit a SARIMAX model and read
a one-step-ahead forecast off it.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


def fit_sarimax(
    y: np.ndarray,
    exog: np.ndarray | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> SARIMAXResultsWrapper:
    """Fit a SARIMAX model on `y` (optionally with exogenous regressors `exog`).

    `enforce_stationarity`/`enforce_invertibility` are relaxed: real-world
    weekly price data doesn't always satisfy them at the MLE optimum, and
    this is a baseline model, not one relying on those constraints for
    interpretation.
    """
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_one_step(results: SARIMAXResultsWrapper, exog: np.ndarray | None = None) -> float:
    """Read the one-step-ahead point forecast off a fitted/updated SARIMAX result."""
    forecast = results.get_forecast(steps=1, exog=exog)
    return float(forecast.predicted_mean[0])


def forecast_one_step_with_interval(
    results: SARIMAXResultsWrapper,
    exog: np.ndarray | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point forecast plus the model's native `(1-alpha)` prediction interval
    (e.g. alpha=0.05 -> 95%). Separate from `forecast_one_step` rather than
    an added parameter there, so existing callers/tests are untouched."""
    forecast = results.get_forecast(steps=1, exog=exog)
    point = float(forecast.predicted_mean[0])
    lower, upper = forecast.conf_int(alpha=alpha)[0]
    return point, float(lower), float(upper)
