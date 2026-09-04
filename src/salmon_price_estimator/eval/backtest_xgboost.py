"""Rolling-window (not expanding) backtest for the XGBoost weekly model.

Unlike SARIMAX's Kalman-filter fitting, training XGBoost on a rolling
window of a few hundred rows is cheap (well under a second), so this
retrains from scratch every single week - no periodic-refit trick needed
(contrast with `eval/backtest.py`'s `.extend()`-based approach, which
exists specifically to work around SARIMAX's much higher per-refit cost).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from salmon_price_estimator.models.xgboost_baseline import fit_xgboost, predict_one_step


def rolling_window_backtest(
    week_ids: pd.Series,
    price: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    train_window_weeks: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> pd.DataFrame:
    """One-step-ahead backtest over a fixed-size rolling training window.

    `features`/`target` must already be free of NaN (caller drops the
    warm-up rows lacking full lag history). `price` is the actual price
    level (same length/index as `features`/`target`), used to reconstruct
    a price-level prediction from the predicted log-return and to build
    the naive benchmark.

    Returns one row per forecasted week: `week_id`, `actual`,
    `xgboost_pred`, `naive_pred` (previous week's actual price) - same
    schema as `eval/backtest.py`'s SARIMAX backtest.
    """
    week_ids = np.asarray(week_ids)
    price = np.asarray(price, dtype=float)
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)

    records = []
    for i in range(train_window_weeks, len(target)):
        train_slice = slice(i - train_window_weeks, i)
        booster = fit_xgboost(features[train_slice], target[train_slice], params, num_boost_round)

        predicted_return = predict_one_step(booster, features[i])
        predicted_price = price[i - 1] * np.exp(predicted_return)

        records.append(
            {
                "week_id": week_ids[i],
                "actual": price[i],
                "xgboost_pred": predicted_price,
                "naive_pred": price[i - 1],
            }
        )

    return pd.DataFrame.from_records(records)
