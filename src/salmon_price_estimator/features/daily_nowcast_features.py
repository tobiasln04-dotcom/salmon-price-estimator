"""Daily-nowcast features: 5-trading-day returns + cross-sectional
dispersion, as-of aligned onto each week's Mon/Tue/Wed/Thu.

Each series keeps its own native trading calendar (FX and Oslo Bors have
different holiday sets), so the 5-day return is computed on each column's
own non-null observations, then as-of aligned (backward, same idiom as
`weekly_panel.py`'s fishmeal join) onto the target dates - never using
data from after that day's close.

The target is a log-ratio *correction* on top of the weekly baseline's own
forecast (`log(actual / baseline_pred)`), not the raw price level -
reconstruct via `baseline_pred * exp(predicted_correction)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "log_correction_target"
WEEKDAY_OFFSETS = [0, 1, 2, 3]  # days after week_start_date: Mon, Tue, Wed, Thu

FEATURE_COLUMNS = [
    "fx_usd_5d_return",
    "fx_eur_5d_return",
    "stock_5d_return_mean",
    "stock_5d_return_dispersion",
    "days_elapsed_in_week",
]


def compute_5day_return(series: pd.Series) -> pd.Series:
    """Log return over the most recent 5 observations of `series`'s own
    (already-dropna'd) native index - trading days, not calendar days."""
    clean = series.dropna()
    return np.log(clean / clean.shift(5))


def asof_lookup(series: pd.Series, target_dates: pd.Series) -> np.ndarray:
    """For each date in `target_dates`, the most recent non-null value of
    `series` at or before that date (backward as-of - never a future
    value). NaN if `series` has no observation on or before that date."""
    clean = series.dropna().sort_index()
    positions = clean.index.searchsorted(pd.DatetimeIndex(target_dates), side="right") - 1
    return np.where(positions >= 0, clean.to_numpy()[np.clip(positions, 0, None)], np.nan)


def aggregate_stock_returns(stock_returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cross-sectional mean and dispersion (std) of the tickers' 5-day
    returns, per date. `skipna=False`: a date only gets a valid value once
    *every* ticker has traded (e.g. a not-yet-listed stock correctly blocks
    the whole cross-section rather than silently averaging over the rest)."""
    return (
        stock_returns.mean(axis=1, skipna=False),
        stock_returns.std(axis=1, skipna=False),
    )


def build_nowcast_panel(
    daily_market: pd.DataFrame,
    weekly_baseline: pd.DataFrame,
    stock_columns: list[str],
) -> pd.DataFrame:
    """One row per (week, weekday in Mon-Thu) with nowcast features, the
    log-ratio correction target, and the actual price.

    `weekly_baseline` must have `week_id`, `week_start_date`, `actual`, and
    `baseline_pred` columns (the weekly model's own forecast for that
    week, made the preceding Sunday night). Rows before a ticker's listing
    date (or otherwise missing data) get NaN features - drop those before
    training.
    """
    daily = daily_market.set_index("date")

    usd_return = compute_5day_return(daily["usdnok"])
    eur_return = compute_5day_return(daily["eurnok"])
    stock_returns = pd.concat([compute_5day_return(daily[c]) for c in stock_columns], axis=1)
    stock_returns.columns = stock_columns
    stock_mean, stock_dispersion = aggregate_stock_returns(stock_returns)

    n_weeks = len(weekly_baseline)
    rows = weekly_baseline.loc[weekly_baseline.index.repeat(len(WEEKDAY_OFFSETS))].reset_index(
        drop=True
    )
    rows["days_elapsed_in_week"] = np.tile(WEEKDAY_OFFSETS, n_weeks) + 1
    rows["asof_date"] = rows["week_start_date"] + pd.to_timedelta(
        rows["days_elapsed_in_week"] - 1, unit="D"
    )

    rows["fx_usd_5d_return"] = asof_lookup(usd_return, rows["asof_date"])
    rows["fx_eur_5d_return"] = asof_lookup(eur_return, rows["asof_date"])
    rows["stock_5d_return_mean"] = asof_lookup(stock_mean, rows["asof_date"])
    rows["stock_5d_return_dispersion"] = asof_lookup(stock_dispersion, rows["asof_date"])
    rows[TARGET_COL] = np.log(rows["actual"] / rows["baseline_pred"])

    return rows[
        [
            "week_id",
            "week_start_date",
            "days_elapsed_in_week",
            "asof_date",
            "baseline_pred",
            "actual",
            *FEATURE_COLUMNS[:-1],
            TARGET_COL,
        ]
    ]
