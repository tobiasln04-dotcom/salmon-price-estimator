import numpy as np

from salmon_price_estimator.models.sarimax_baseline import (
    fit_sarimax,
    forecast_one_step,
    forecast_one_step_with_interval,
)


def _synthetic_series(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=n))


def test_fit_sarimax_univariate_returns_fitted_results():
    y = _synthetic_series()

    results = fit_sarimax(y, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

    assert results.params is not None
    assert np.isfinite(results.llf)


def test_fit_sarimax_exogenous_returns_fitted_results():
    y = _synthetic_series()
    rng = np.random.default_rng(1)
    exog = rng.normal(size=(len(y), 2))

    results = fit_sarimax(y, exog=exog, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))

    assert results.params is not None
    assert np.isfinite(results.llf)


def test_forecast_one_step_returns_finite_float():
    y = _synthetic_series()
    results = fit_sarimax(y, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

    prediction = forecast_one_step(results)

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


def test_forecast_one_step_with_interval_brackets_the_point_forecast():
    y = _synthetic_series()
    results = fit_sarimax(y, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

    point, lower, upper = forecast_one_step_with_interval(results, alpha=0.05)

    assert np.isfinite([point, lower, upper]).all()
    assert lower < point < upper


def test_forecast_one_step_with_interval_widens_as_alpha_shrinks():
    """A 99% interval (alpha=0.01) must be wider than a 80% interval
    (alpha=0.2) around the same point forecast."""
    y = _synthetic_series()
    results = fit_sarimax(y, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

    _, narrow_lower, narrow_upper = forecast_one_step_with_interval(results, alpha=0.2)
    _, wide_lower, wide_upper = forecast_one_step_with_interval(results, alpha=0.01)

    assert (wide_upper - wide_lower) > (narrow_upper - narrow_lower)
