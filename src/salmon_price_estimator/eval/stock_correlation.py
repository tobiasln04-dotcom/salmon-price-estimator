"""Cross-sectional side analysis: how much does each salmon-farming
stock's weekly return co-move with the SSB salmon export price's weekly
return? Not part of the core forecasting pipeline (see
`scripts/run_backtest.py`) - a standalone descriptive analysis reusing
already-fetched data.

Uses the same backward as-of alignment idiom as
`features/daily_nowcast_features.py` to resample each stock's daily
close to a weekly (end-of-week) value, matched to the SSB week grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from salmon_price_estimator.features.daily_nowcast_features import asof_lookup


def compute_week_end_return(daily_close: pd.Series, week_end_dates: pd.Series) -> pd.Series:
    """Week-over-week log return of a daily series' value as of each week's
    last day (backward as-of - never a future value)."""
    week_end_values = pd.Series(asof_lookup(daily_close, week_end_dates))
    return np.log(week_end_values / week_end_values.shift(1))


def compute_stock_salmon_correlations(
    daily_market: pd.DataFrame,
    ssb: pd.DataFrame,
    stock_columns: list[str],
    common_start: str | None = None,
) -> pd.DataFrame:
    """Correlation and regression beta of each stock's weekly return against
    the salmon export price's weekly return, over a common window so every
    stock is compared on the same period regardless of its listing date.

    Returns one row per stock, sorted by correlation descending: `stock`,
    `n_weeks`, `correlation`, `beta`, `r_squared`.
    """
    daily = daily_market.set_index("date")
    week_end_dates = ssb["week_start_date"] + pd.Timedelta(days=6)

    salmon_price = ssb["price_nok_per_kg"]
    salmon_return = np.log(salmon_price / salmon_price.shift(1))

    if common_start is not None:
        in_window = (ssb["week_start_date"] >= pd.Timestamp(common_start)).to_numpy()
    else:
        in_window = np.ones(len(ssb), dtype=bool)

    rows = []
    for col in stock_columns:
        stock_return = compute_week_end_return(daily[col], week_end_dates)
        df = pd.DataFrame(
            {
                "salmon_return": salmon_return.to_numpy(),
                "stock_return": stock_return.to_numpy(),
                "in_window": in_window,
            }
        )
        df = df[df["in_window"]].dropna(subset=["salmon_return", "stock_return"])

        corr = df["salmon_return"].corr(df["stock_return"])
        beta, _intercept = np.polyfit(df["salmon_return"], df["stock_return"], 1)

        rows.append(
            {
                "stock": col.upper(),
                "n_weeks": len(df),
                "correlation": corr,
                "beta": beta,
                "r_squared": corr**2,
            }
        )

    return pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
