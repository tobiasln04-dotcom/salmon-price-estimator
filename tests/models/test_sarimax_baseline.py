import numpy as np

from salmon_price_estimator.models.sarimax_baseline import fit_sarimax, forecast_one_step


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
