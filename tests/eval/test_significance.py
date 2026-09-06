import numpy as np
import pytest

from salmon_price_estimator.eval.significance import diebold_mariano_test


def test_diebold_mariano_favors_the_clearly_better_model():
    rng = np.random.default_rng(0)
    errors_a = rng.normal(scale=0.1, size=200)  # small errors: much better model
    errors_b = rng.normal(scale=5.0, size=200)  # large errors: much worse model

    dm_stat, p_value = diebold_mariano_test(errors_a, errors_b)

    assert dm_stat < 0  # negative: model A's average loss is lower
    assert p_value < 0.05  # difference is statistically significant


def test_diebold_mariano_identical_errors_is_undefined_not_zero():
    """Identical error series -> zero variance in the loss differential ->
    the statistic is genuinely undefined (0/0), not a defined zero."""
    errors = np.array([1.0, -2.0, 3.0, -1.5, 2.5])

    dm_stat, p_value = diebold_mariano_test(errors, errors)

    assert np.isnan(dm_stat)
    assert np.isnan(p_value)


def test_diebold_mariano_zero_mean_difference_by_construction():
    # Squared losses of a: [1,4,9,4,1,4,9,4]; of b: [4,1,4,9,4,1,4,9].
    # d = a - b = [-3,3,5,-5,-3,3,5,-5] -> mean exactly 0, but not identical series.
    errors_a = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0])
    errors_b = np.array([2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])

    dm_stat, p_value = diebold_mariano_test(errors_a, errors_b)

    assert dm_stat == pytest.approx(0.0, abs=1e-10)
    assert p_value == pytest.approx(1.0)


def test_diebold_mariano_absolute_loss_power_1():
    rng = np.random.default_rng(1)
    errors_a = rng.normal(scale=0.1, size=200)
    errors_b = rng.normal(scale=5.0, size=200)

    dm_stat, p_value = diebold_mariano_test(errors_a, errors_b, power=1)

    assert dm_stat < 0
    assert p_value < 0.05


def test_diebold_mariano_multi_step_horizon_does_not_crash():
    rng = np.random.default_rng(2)
    errors_a = rng.normal(scale=1.0, size=100)
    errors_b = rng.normal(scale=1.0, size=100)

    dm_stat, p_value = diebold_mariano_test(errors_a, errors_b, h=4)

    assert np.isfinite(dm_stat)
    assert 0.0 <= p_value <= 1.0
