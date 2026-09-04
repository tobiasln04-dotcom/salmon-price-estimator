import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.features.daily_nowcast_features import (
    TARGET_COL,
    aggregate_stock_returns,
    asof_lookup,
    build_nowcast_panel,
    compute_5day_return,
)


def test_compute_5day_return_uses_own_native_index():
    series = pd.Series(
        [10.0, 10.05, 10.10, 10.15, 10.20, 10.25],
        index=pd.date_range("2020-12-28", periods=6, freq="D"),
    )

    returns = compute_5day_return(series)

    assert returns.iloc[-1] == pytest.approx(np.log(10.25 / 10.0))


def test_asof_lookup_never_uses_future_values():
    """The core leakage guarantee: a gap on the target date must fall back
    to the *prior* observation, never a later one."""
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-07"]),  # gap on 01-06
    )

    result = asof_lookup(series, pd.to_datetime(["2021-01-06"]))

    assert result[0] == pytest.approx(2.0)  # 01-05's value, not 01-07's


def test_asof_lookup_exact_match():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-01-04", "2021-01-05"]))

    result = asof_lookup(series, pd.to_datetime(["2021-01-05"]))

    assert result[0] == pytest.approx(2.0)


def test_asof_lookup_returns_nan_before_series_starts():
    series = pd.Series([1.0], index=pd.to_datetime(["2021-01-05"]))

    result = asof_lookup(series, pd.to_datetime(["2021-01-01"]))

    assert np.isnan(result[0])


@pytest.fixture
def daily_market() -> pd.DataFrame:
    dates = pd.date_range("2020-12-28", "2021-01-07", freq="D")
    n = len(dates)
    # Simple increasing series so 5-day returns are all well-defined and distinct.
    base = np.linspace(10.0, 10.5, n)
    return pd.DataFrame(
        {
            "date": dates,
            "usdnok": base,
            "eurnok": base + 1,
            "mowi": base * 10,
            "salm": base * 11,
            "lsg": base * 12,
            "gsf": base * 13,
            "bakka": base * 14,
        }
    )


@pytest.fixture
def weekly_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": ["2021U01"],
            "week_start_date": pd.to_datetime(["2021-01-04"]),  # Monday
            "baseline_pred": [95.0],
            "actual": [100.0],
        }
    )


def test_build_nowcast_panel_shape_and_target(daily_market, weekly_baseline):
    panel = build_nowcast_panel(
        daily_market, weekly_baseline, stock_columns=["mowi", "salm", "lsg", "gsf", "bakka"]
    )

    assert len(panel) == 4  # Mon-Thu for the one week
    assert panel["days_elapsed_in_week"].tolist() == [1, 2, 3, 4]
    assert panel["asof_date"].tolist() == list(
        pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"])
    )
    expected_target = np.log(100.0 / 95.0)
    assert panel[TARGET_COL].tolist() == pytest.approx([expected_target] * 4)


def test_build_nowcast_panel_features_evolve_across_the_week(daily_market, weekly_baseline):
    panel = build_nowcast_panel(
        daily_market, weekly_baseline, stock_columns=["mowi", "salm", "lsg", "gsf", "bakka"]
    )

    # Each successive weekday's 5-day return should use one more day of
    # market data than the previous, so they shouldn't all be identical.
    assert panel["fx_usd_5d_return"].nunique() == 4


def test_aggregate_stock_returns_gates_on_every_ticker():
    """A date only gets a valid mean/dispersion once *every* ticker has a
    value - one not-yet-listed stock (NaN) must block the whole
    cross-section rather than silently averaging over the rest."""
    stock_returns = pd.DataFrame(
        {
            "mowi": [0.01, 0.02],
            "salm": [0.02, 0.01],
            "bakka": [np.nan, 0.03],  # not listed yet on the first date
        }
    )

    mean, dispersion = aggregate_stock_returns(stock_returns)

    assert np.isnan(mean.iloc[0])
    assert np.isnan(dispersion.iloc[0])
    assert mean.iloc[1] == pytest.approx((0.02 + 0.01 + 0.03) / 3)
