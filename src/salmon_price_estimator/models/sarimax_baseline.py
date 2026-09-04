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
