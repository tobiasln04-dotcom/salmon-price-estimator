import numpy as np
import pandas as pd

from salmon_price_estimator.eval.backtest_nowcast import walk_forward_nowcast_backtest
from salmon_price_estimator.features.daily_nowcast_features import FEATURE_COLUMNS, TARGET_COL

PARAMS = {
    "max_depth": 2,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
}


def _synthetic_panel(n_weeks: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_weeks):
        baseline_pred = 100 + w * 0.1
        actual = baseline_pred * np.exp(rng.normal(scale=0.02))
        for day in range(1, 5):
            rows.append(
                {
                    "week_id": f"W{w:03d}",
                    "days_elapsed_in_week": day,
                    "baseline_pred": baseline_pred,
                    "actual": actual,
                    "fx_usd_5d_return": rng.normal(scale=0.01),
                    "fx_eur_5d_return": rng.normal(scale=0.01),
                    "stock_5d_return_mean": rng.normal(scale=0.02),
                    "stock_5d_return_dispersion": rng.uniform(0, 0.02),
                    TARGET_COL: np.log(actual / baseline_pred),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_nowcast_backtest_shape_and_columns():
    panel = _synthetic_panel(n_weeks=30)
    train_window = 10

    result = walk_forward_nowcast_backtest(panel, train_window, PARAMS, num_boost_round=20)

    assert list(result.columns) == [
        "week_id",
        "days_elapsed_in_week",
        "actual",
        "nowcast_pred",
        "static_baseline_pred",
    ]
    n_test_weeks = 30 - train_window
    assert len(result) == n_test_weeks * 4


def test_walk_forward_nowcast_backtest_skips_only_full_training_window():
    panel = _synthetic_panel(n_weeks=15)
    train_window = 12

    result = walk_forward_nowcast_backtest(panel, train_window, PARAMS, num_boost_round=20)

    assert sorted(result["week_id"].unique()) == [f"W{w:03d}" for w in range(12, 15)]


def test_walk_forward_nowcast_backtest_static_baseline_matches_input():
    panel = _synthetic_panel(n_weeks=20)
    train_window = 10

    result = walk_forward_nowcast_backtest(panel, train_window, PARAMS, num_boost_round=20)

    merged = result.merge(
        panel[["week_id", "days_elapsed_in_week", "baseline_pred"]],
        on=["week_id", "days_elapsed_in_week"],
    )
    assert np.allclose(merged["static_baseline_pred"], merged["baseline_pred"])


def test_walk_forward_nowcast_backtest_predictions_finite_and_positive():
    panel = _synthetic_panel(n_weeks=20)
    train_window = 10

    result = walk_forward_nowcast_backtest(panel, train_window, PARAMS, num_boost_round=20)

    assert np.isfinite(result["nowcast_pred"]).all()
    assert (result["nowcast_pred"] > 0).all()


def test_feature_columns_present_in_synthetic_panel():
    panel = _synthetic_panel(n_weeks=5)
    assert set(FEATURE_COLUMNS).issubset(panel.columns)
