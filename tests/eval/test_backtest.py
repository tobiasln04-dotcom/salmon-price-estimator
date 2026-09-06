import numpy as np

from salmon_price_estimator.eval.backtest import walk_forward_backtest


def _synthetic_series(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(size=n))


def test_walk_forward_backtest_univariate_shape_and_columns():
    y = _synthetic_series()
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_backtest(
        week_ids,
        y,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
    )

    assert list(result.columns) == ["week_id", "actual", "sarimax_pred", "naive_pred"]
    assert len(result) == len(y) - min_train_weeks
    assert result["week_id"].tolist() == week_ids[min_train_weeks:]


def test_walk_forward_backtest_naive_pred_is_previous_actual():
    y = _synthetic_series()
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_backtest(
        week_ids,
        y,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
    )

    expected_naive = y[min_train_weeks - 1 : -1]
    assert np.allclose(result["naive_pred"].to_numpy(), expected_naive)


def test_walk_forward_backtest_exogenous_shape():
    y = _synthetic_series(seed=2)
    rng = np.random.default_rng(3)
    exog = rng.normal(size=(len(y), 2))
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_backtest(
        week_ids,
        y,
        exog=exog,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
    )

    assert len(result) == len(y) - min_train_weeks
    assert result["sarimax_pred"].notna().all()


def test_walk_forward_backtest_interval_alpha_adds_lower_upper_columns():
    y = _synthetic_series()
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20

    result = walk_forward_backtest(
        week_ids,
        y,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
        interval_alpha=0.05,
    )

    assert list(result.columns) == [
        "week_id",
        "actual",
        "sarimax_pred",
        "naive_pred",
        "lower",
        "upper",
    ]
    assert (result["lower"] < result["sarimax_pred"]).all()
    assert (result["sarimax_pred"] < result["upper"]).all()


def test_walk_forward_backtest_default_interval_alpha_is_unchanged():
    """interval_alpha=None (the default) must produce the exact same
    output as before this parameter existed - no schema change."""
    y = _synthetic_series()
    week_ids = [f"2020U{i:02d}" for i in range(1, len(y) + 1)]
    min_train_weeks = 20
    kwargs = dict(
        week_ids=week_ids,
        y=y,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        min_train_weeks=min_train_weeks,
        refit_every_n_weeks=5,
    )

    explicit_none = walk_forward_backtest(**kwargs, interval_alpha=None)
    default = walk_forward_backtest(**kwargs)

    assert list(explicit_none.columns) == ["week_id", "actual", "sarimax_pred", "naive_pred"]
    assert list(default.columns) == list(explicit_none.columns)
