import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.eval.metrics import (
    directional_accuracy,
    mape,
    naive_last_value_forecast,
    rmse,
)


def test_naive_last_value_forecast_shifts_by_one():
    actual = pd.Series([10.0, 11.0, 12.0])

    naive = naive_last_value_forecast(actual)

    assert naive.tolist()[1:] == [10.0, 11.0]
    assert np.isnan(naive.iloc[0])


def test_mape_known_value():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])

    # |10/100| + |20/200| = 0.1 + 0.1 -> mean 0.1 -> 10%
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_rmse_known_value():
    actual = np.array([0.0, 0.0])
    predicted = np.array([3.0, 4.0])

    # sqrt(mean(9, 16)) = sqrt(12.5)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))


def test_directional_accuracy_all_correct():
    previous = np.array([10.0, 20.0, 30.0])
    actual = np.array([12.0, 18.0, 30.0])  # up, down, flat
    predicted = np.array([11.0, 19.0, 30.0])  # up, down, flat -> matches all

    assert directional_accuracy(actual, predicted, previous) == pytest.approx(1.0)


def test_directional_accuracy_all_wrong():
    previous = np.array([10.0, 20.0])
    actual = np.array([12.0, 18.0])  # up, down
    predicted = np.array([8.0, 25.0])  # down, up -> matches none

    assert directional_accuracy(actual, predicted, previous) == pytest.approx(0.0)


def test_naive_directional_accuracy_is_zero_by_construction():
    """The naive forecast repeats the previous value, so it has no directional
    information - `directional_accuracy(actual, naive_pred, naive_pred)`
    should score ~0 whenever the actual series actually moves."""
    previous = np.array([10.0, 20.0, 30.0])
    actual = np.array([12.0, 18.0, 33.0])

    assert directional_accuracy(actual, previous, previous) == pytest.approx(0.0)
