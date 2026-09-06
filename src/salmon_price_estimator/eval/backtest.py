"""Walk-forward (expanding window) backtest for the SARIMAX baseline.

Refits the full parameter set only every `refit_every_n_weeks`; in between,
`SARIMAXResultsWrapper.extend()` applies the existing (unchanged) parameters
to just the one new observation, using the already-filtered state as the
starting point - O(1) per step. `.append()` looks like the obvious choice
for this but was measured to recreate and re-filter the model over the
*entire* expanding dataset on every call (confirmed empirically: ~0.6s/call
at 160 obs growing to ~1.6s/call at 800 obs) - that made it O(n^2) overall
and impractical for a ~1300-week backtest. `.extend()` stayed flat at
~5-10ms/call regardless of how much history had accumulated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from salmon_price_estimator.models.sarimax_baseline import (
    fit_sarimax,
    forecast_one_step,
    forecast_one_step_with_interval,
)


def walk_forward_backtest(
    week_ids: pd.Series,
    y: np.ndarray,
    exog: np.ndarray | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    min_train_weeks: int,
    refit_every_n_weeks: int,
    interval_alpha: float | None = None,
) -> pd.DataFrame:
    """Expanding-window one-step-ahead backtest.

    Returns one row per forecasted week: `week_id`, `actual`, `sarimax_pred`,
    `naive_pred` (previous week's actual). If `interval_alpha` is given (e.g.
    0.05 for a 95% interval), also includes `lower`/`upper` columns from the
    model's native prediction interval at that step - purely additive, so
    the default (`None`) leaves the schema and behavior unchanged for
    existing callers.
    """
    week_ids = np.asarray(week_ids)
    y = np.asarray(y, dtype=float)
    exog = np.asarray(exog, dtype=float) if exog is not None else None

    results = fit_sarimax(
        y[:min_train_weeks],
        exog[:min_train_weeks] if exog is not None else None,
        order,
        seasonal_order,
    )

    records = []
    for i in range(min_train_weeks, len(y)):
        exog_forecast = exog[i : i + 1] if exog is not None else None
        if interval_alpha is not None:
            prediction, lower, upper = forecast_one_step_with_interval(
                results, exog=exog_forecast, alpha=interval_alpha
            )
        else:
            prediction = forecast_one_step(results, exog=exog_forecast)

        record = {
            "week_id": week_ids[i],
            "actual": y[i],
            "sarimax_pred": prediction,
            "naive_pred": y[i - 1],
        }
        if interval_alpha is not None:
            record["lower"] = lower
            record["upper"] = upper
        records.append(record)

        steps_done = i - min_train_weeks + 1
        exog_new = exog[i : i + 1] if exog is not None else None
        if steps_done % refit_every_n_weeks == 0:
            results = fit_sarimax(
                y[: i + 1], exog[: i + 1] if exog is not None else None, order, seasonal_order
            )
        else:
            results = results.extend(y[i : i + 1], exog=exog_new)

    return pd.DataFrame.from_records(records)
