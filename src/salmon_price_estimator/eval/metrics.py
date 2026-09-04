"""Forecast evaluation metrics: MAPE, RMSE, directional accuracy, naive benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_last_value_forecast(actual: pd.Series) -> pd.Series:
    """The naive benchmark: predict next week's price as this week's price."""
    return actual.shift(1)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def directional_accuracy(
    actual: np.ndarray, predicted: np.ndarray, previous_actual: np.ndarray
) -> float:
    """Fraction of weeks where the forecast got the direction of change right."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    previous_actual = np.asarray(previous_actual, dtype=float)
    actual_direction = np.sign(actual - previous_actual)
    predicted_direction = np.sign(predicted - previous_actual)
    return float(np.mean(actual_direction == predicted_direction))
