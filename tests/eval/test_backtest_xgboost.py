import numpy as np

from salmon_price_estimator.eval.backtest_xgboost import rolling_window_backtest

PARAMS = {
    "max_depth": 2,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
}


def _synthetic_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(size=n))
    features = rng.normal(size=(n, 3))
    target = np.zeros(n)
    target[1:] = np.log(price[1:] / price[:-1])
    week_ids = [f"W{i:02d}" for i in range(n)]
    return week_ids, price, features, target


def test_rolling_window_backtest_shape_and_columns():
    week_ids, price, features, target = _synthetic_data()
    train_window = 20

    result = rolling_window_backtest(
        week_ids, price, features, target, train_window, PARAMS, num_boost_round=20
    )

    assert list(result.columns) == ["week_id", "actual", "xgboost_pred", "naive_pred"]
    assert len(result) == len(price) - train_window
    assert result["week_id"].tolist() == week_ids[train_window:]


def test_rolling_window_backtest_naive_and_actual_match_price_series():
    week_ids, price, features, target = _synthetic_data()
    train_window = 20

    result = rolling_window_backtest(
        week_ids, price, features, target, train_window, PARAMS, num_boost_round=20
    )

    assert np.allclose(result["actual"].to_numpy(), price[train_window:])
    assert np.allclose(result["naive_pred"].to_numpy(), price[train_window - 1 : -1])


def test_rolling_window_backtest_predictions_are_finite_and_positive():
    week_ids, price, features, target = _synthetic_data()
    train_window = 20

    result = rolling_window_backtest(
        week_ids, price, features, target, train_window, PARAMS, num_boost_round=20
    )

    assert np.isfinite(result["xgboost_pred"]).all()
    assert (result["xgboost_pred"] > 0).all()
