"""XGBoost weekly model: model-construction helpers only.

Uses the low-level Booster/DMatrix API rather than the sklearn wrapper,
to avoid adding scikit-learn as a dependency just for base classes the
low-level API doesn't need. The rolling-window backtest loop lives in
`eval.backtest_xgboost`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    num_boost_round: int,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=num_boost_round)


def predict_one_step(booster: xgb.Booster, x_row: np.ndarray) -> float:
    """`x_row`: 1D array of feature values for a single row."""
    dmatrix = xgb.DMatrix(x_row.reshape(1, -1))
    return float(booster.predict(dmatrix)[0])
