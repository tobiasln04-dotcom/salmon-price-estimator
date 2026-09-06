import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.eval.trading_strategy import (
    compute_strategy_metrics,
    newey_west_mean_test,
    simple_directional_strategy,
)


def test_simple_directional_strategy_correct_up_call_is_profitable():
    df = pd.DataFrame({"actual": [110.0], "pred": [105.0], "naive_pred": [100.0]})

    result = simple_directional_strategy(df, "pred")

    assert result.loc[0, "position"] == 1.0  # forecast (105) > naive (100): predicted a rise
    assert result.loc[0, "realized_return"] == pytest.approx(np.log(110.0 / 100.0))
    assert result.loc[0, "strategy_return"] > 0  # predicted up, actually went up


def test_simple_directional_strategy_wrong_up_call_loses():
    df = pd.DataFrame({"actual": [90.0], "pred": [105.0], "naive_pred": [100.0]})

    result = simple_directional_strategy(df, "pred")

    assert result.loc[0, "position"] == 1.0
    assert result.loc[0, "strategy_return"] < 0  # predicted up, actually went down


def test_simple_directional_strategy_correct_down_call_is_profitable():
    df = pd.DataFrame({"actual": [90.0], "pred": [95.0], "naive_pred": [100.0]})

    result = simple_directional_strategy(df, "pred")

    assert result.loc[0, "position"] == -1.0  # forecast (95) < naive (100): predicted a fall
    assert result.loc[0, "strategy_return"] > 0  # predicted down, actually went down


def test_newey_west_mean_test_detects_clearly_positive_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.01, scale=0.02, size=300)

    t_stat, p_value = newey_west_mean_test(x)

    assert t_stat > 0
    assert p_value < 0.05


def test_newey_west_mean_test_zero_mean_by_construction():
    x = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])

    t_stat, p_value = newey_west_mean_test(x)

    assert t_stat == pytest.approx(0.0, abs=1e-10)
    assert p_value == pytest.approx(1.0)


def test_newey_west_mean_test_widens_standard_error_for_autocorrelated_series():
    """Positively autocorrelated data should give a smaller |t-stat| than
    treating the same mean/variance as if it were i.i.d. - the HAC
    adjustment should make the test more conservative, not less."""
    rng = np.random.default_rng(1)
    n = 300
    noise = rng.normal(scale=1.0, size=n)
    autocorrelated = np.zeros(n)
    for t in range(1, n):
        autocorrelated[t] = 0.02 + 0.8 * autocorrelated[t - 1] + noise[t]

    t_stat_hac, _ = newey_west_mean_test(autocorrelated, lags=8)

    naive_se = autocorrelated.std(ddof=1) / np.sqrt(n)
    t_stat_naive = autocorrelated.mean() / naive_se

    assert abs(t_stat_hac) < abs(t_stat_naive)


def test_compute_strategy_metrics_hit_rate_and_buy_hold_return():
    df = pd.DataFrame(
        {
            "actual": [110.0, 121.0, 108.9],
            "naive_pred": [100.0, 110.0, 121.0],
            "pred": [105.0, 115.0, 115.0],  # predicts up, up, down
        }
    )
    df = simple_directional_strategy(df, "pred")

    result = compute_strategy_metrics(df)

    # Week 1: predicted up (105>100), actual up (110>100) -> hit.
    # Week 2: predicted up (115>110), actual up (121>110) -> hit.
    # Week 3: predicted down (115<121), actual down (108.9<121) -> hit.
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["n_weeks"] == 3
    assert "strategy_cumulative_return" not in result  # see docstring: not reported, misleading
    # Buy-and-hold telescopes correctly: 100 -> 108.9 over the period.
    assert result["buy_hold_cumulative_return"] == pytest.approx(108.9 / 100.0 - 1.0)
