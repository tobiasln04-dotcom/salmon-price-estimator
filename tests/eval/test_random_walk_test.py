import numpy as np
import pandas as pd

from salmon_price_estimator.eval.random_walk_test import adf_test, ljung_box_test


def _random_walk_price(n: int = 400, seed: int = 0) -> pd.Series:
    """Price whose log-returns are exactly i.i.d. noise - a textbook random walk."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(scale=0.02, size=n)
    log_price = np.cumsum(log_returns) + np.log(100.0)
    return pd.Series(np.exp(log_price))


def _mean_reverting_price(n: int = 400, seed: int = 0) -> pd.Series:
    """AR(1) with a small coefficient around a fixed level - clearly stationary."""
    rng = np.random.default_rng(seed)
    log_price = np.zeros(n)
    log_price[0] = np.log(100.0)
    for t in range(1, n):
        log_price[t] = 0.3 * log_price[t - 1] + 0.7 * np.log(100.0) + rng.normal(scale=0.02)
    return pd.Series(np.exp(log_price))


def _autocorrelated_return_price(n: int = 400, seed: int = 0) -> pd.Series:
    """Log-returns that are themselves strongly autocorrelated (momentum)."""
    rng = np.random.default_rng(seed)
    log_returns = np.zeros(n)
    for t in range(1, n):
        log_returns[t] = 0.7 * log_returns[t - 1] + rng.normal(scale=0.01)
    log_price = np.cumsum(log_returns) + np.log(100.0)
    return pd.Series(np.exp(log_price))


def test_adf_fails_to_reject_unit_root_for_a_true_random_walk():
    price = _random_walk_price()

    result = adf_test(price)

    assert result["p_value"] > 0.05
    assert result["unit_root_rejected_at_5pct"] is False


def test_adf_rejects_unit_root_for_a_mean_reverting_series():
    price = _mean_reverting_price()

    result = adf_test(price)

    assert result["p_value"] < 0.05
    assert result["unit_root_rejected_at_5pct"] is True


def test_ljung_box_fails_to_reject_for_iid_returns():
    price = _random_walk_price()

    result = ljung_box_test(price, lags=[1, 4, 12])

    assert (result["lb_pvalue"] > 0.05).all()
    assert not result["autocorrelation_rejected_at_5pct"].any()


def test_ljung_box_rejects_for_autocorrelated_returns():
    price = _autocorrelated_return_price()

    result = ljung_box_test(price, lags=[1, 4, 12])

    assert result["lb_pvalue"].iloc[0] < 0.05
    assert result["autocorrelation_rejected_at_5pct"].iloc[0]


def test_ljung_box_returns_expected_columns_and_lags():
    price = _random_walk_price()

    result = ljung_box_test(price, lags=[1, 4])

    assert result["lag"].tolist() == [1, 4]
    assert list(result.columns) == [
        "lag",
        "lb_stat",
        "lb_pvalue",
        "autocorrelation_rejected_at_5pct",
    ]
