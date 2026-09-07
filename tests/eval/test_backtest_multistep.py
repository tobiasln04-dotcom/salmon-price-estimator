import numpy as np
import pytest

from salmon_price_estimator.eval.backtest import walk_forward_backtest
from salmon_price_estimator.eval.backtest_multistep import walk_forward_multistep_backtest


def _synthetic_series(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=n))


def test_walk_forward_multistep_backtest_drops_out_of_bounds_targets():
    y = _synthetic_series(n=40)
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20
    horizons = [1, 2, 4]

    result = walk_forward_multistep_backtest(
        week_ids,
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
        horizons=horizons,
    )

    n = len(y)
    for h in horizons:
        expected_n_rows = n - h - min_train_weeks + 1
        assert (result["horizon"] == h).sum() == expected_n_rows


def test_walk_forward_multistep_backtest_naive_pred_constant_across_horizons():
    y = _synthetic_series(n=40)
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_multistep_backtest(
        week_ids,
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
        horizons=[1, 2, 4],
    )

    for _, group in result.groupby("as_of_week_id"):
        assert group["naive_pred"].nunique() == 1


def test_walk_forward_multistep_backtest_target_week_alignment():
    y = _synthetic_series(n=30)
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_multistep_backtest(
        week_ids,
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
        horizons=[1, 4],
    )

    # First origin: i = min_train_weeks = 20 (0-indexed), as_of = week_ids[19].
    first_origin = result[result["as_of_week_id"] == week_ids[min_train_weeks - 1]]
    h1_row = first_origin[first_origin["horizon"] == 1].iloc[0]
    h4_row = first_origin[first_origin["horizon"] == 4].iloc[0]
    assert h1_row["target_week_id"] == week_ids[min_train_weeks]
    assert h4_row["target_week_id"] == week_ids[min_train_weeks + 3]
    assert h1_row["actual"] == pytest.approx(y[min_train_weeks])
    assert h4_row["actual"] == pytest.approx(y[min_train_weeks + 3])


def test_walk_forward_multistep_backtest_horizon_1_matches_single_step_backtest():
    """Same underlying fit/refit/extend mechanics - horizon=1 output should
    match the existing single-step walk_forward_backtest exactly."""
    y = _synthetic_series(n=40, seed=3)
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    kwargs = dict(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), min_train_weeks=20, refit_every_n_weeks=5
    )

    single_step = walk_forward_backtest(week_ids=week_ids, y=y, exog=None, **kwargs)
    multistep = walk_forward_multistep_backtest(week_ids, y, horizons=[1], **kwargs)

    assert multistep["target_week_id"].tolist() == single_step["week_id"].tolist()
    assert np.allclose(multistep["sarimax_pred"], single_step["sarimax_pred"])
    assert np.allclose(multistep["naive_pred"], single_step["naive_pred"])
    assert np.allclose(multistep["actual"], single_step["actual"])
