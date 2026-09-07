"""Multi-step-ahead walk-forward backtest for the univariate SARIMAX -
the one model that already beat naive at 1-week-ahead. Does that edge
hold, grow, or shrink at longer horizons? Naive (repeat the last
observed value) ignores trend entirely, so it tends to get worse as a
benchmark the further out you forecast - this is the one analysis in
the project where the answer isn't a foregone conclusion.

Same walk-forward mechanics as `eval/backtest.py` (periodic refit,
`.extend()` between refits - see that module's docstring for why), but
one `get_forecast(steps=max(horizons))` call per origin returns *every*
horizon from a single already-fitted state at essentially the same cost
as a single-step forecast, so the whole multi-horizon backtest costs
about the same as the existing 1-step backtest, not a multiple of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from salmon_price_estimator.models.sarimax_baseline import fit_sarimax


def walk_forward_multistep_backtest(
    week_ids: pd.Series,
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    min_train_weeks: int,
    refit_every_n_weeks: int,
    horizons: list[int],
) -> pd.DataFrame:
    """Long format: one row per (forecast origin, horizon).

    At origin `i` (state fitted through index `i-1`), for each `h` in
    `horizons`: the target is index `i + h - 1`, the forecast is
    `predicted_mean[h-1]`, and naive-at-h is `y[i-1]` (repeat the last
    known value, regardless of horizon). Rows whose target index runs
    past the end of `y` are dropped (the last few origins can't produce
    a valid long-horizon target).

    Columns: `as_of_week_id`, `horizon`, `target_week_id`, `actual`,
    `sarimax_pred`, `naive_pred`.
    """
    week_ids = np.asarray(week_ids)
    y = np.asarray(y, dtype=float)
    max_horizon = max(horizons)

    results = fit_sarimax(y[:min_train_weeks], None, order, seasonal_order)

    records = []
    for i in range(min_train_weeks, len(y)):
        forecast = results.get_forecast(steps=max_horizon).predicted_mean
        for h in horizons:
            target_idx = i + h - 1
            if target_idx >= len(y):
                continue
            records.append(
                {
                    "as_of_week_id": week_ids[i - 1],
                    "horizon": h,
                    "target_week_id": week_ids[target_idx],
                    "actual": y[target_idx],
                    "sarimax_pred": float(forecast[h - 1]),
                    "naive_pred": y[i - 1],
                }
            )

        steps_done = i - min_train_weeks + 1
        if steps_done % refit_every_n_weeks == 0:
            results = fit_sarimax(y[: i + 1], None, order, seasonal_order)
        else:
            results = results.extend(y[i : i + 1])

    return pd.DataFrame.from_records(records)
