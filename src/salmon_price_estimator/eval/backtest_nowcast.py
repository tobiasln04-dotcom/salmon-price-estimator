"""Rolling-window walk-forward backtest for the daily nowcast layer.

Retrains once per week, not once per within-week day: a week's Mon-Thu
training set can only include *completed* weeks (target already known),
which doesn't change across that week's 4 predictions - refitting more
often than that would just re-fit an identical model on the same data.
Same efficiency reasoning as the `.extend()` vs `.append()` lesson from
the SARIMAX backtest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from salmon_price_estimator.features.daily_nowcast_features import FEATURE_COLUMNS, TARGET_COL
from salmon_price_estimator.models.xgboost_baseline import fit_xgboost, predict_one_step


def walk_forward_nowcast_backtest(
    panel: pd.DataFrame,
    train_window_weeks: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> pd.DataFrame:
    """`panel` must be `daily_nowcast_features.build_nowcast_panel`'s
    output with NaN feature rows already dropped.

    Returns one row per (week, weekday): `week_id`, `days_elapsed_in_week`,
    `actual`, `nowcast_pred`, `static_baseline_pred` (the weekly baseline
    held flat all week, i.e. no daily updating - what the nowcast needs to
    beat to justify updating at all).
    """
    week_ids_ordered = panel["week_id"].drop_duplicates().tolist()

    records = []
    for i, week_id in enumerate(week_ids_ordered):
        if i < train_window_weeks:
            continue

        train_week_ids = week_ids_ordered[i - train_window_weeks : i]
        train_rows = panel[panel["week_id"].isin(train_week_ids)]
        booster = fit_xgboost(
            train_rows[FEATURE_COLUMNS].to_numpy(),
            train_rows[TARGET_COL].to_numpy(),
            params,
            num_boost_round,
        )

        week_rows = panel[panel["week_id"] == week_id]
        for _, row in week_rows.iterrows():
            predicted_correction = predict_one_step(booster, row[FEATURE_COLUMNS].to_numpy())
            nowcast_pred = row["baseline_pred"] * np.exp(predicted_correction)
            records.append(
                {
                    "week_id": week_id,
                    "days_elapsed_in_week": row["days_elapsed_in_week"],
                    "actual": row["actual"],
                    "nowcast_pred": nowcast_pred,
                    "static_baseline_pred": row["baseline_pred"],
                }
            )

    return pd.DataFrame.from_records(records)
