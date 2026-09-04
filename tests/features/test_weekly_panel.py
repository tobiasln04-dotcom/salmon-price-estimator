import pandas as pd
import pytest

from salmon_price_estimator.features.weekly_panel import (
    build_weekly_panel,
    resample_fishmeal_to_weekly,
)


@pytest.fixture
def ssb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": ["2021U01", "2021U02", "2021U03", "2021U04", "2021U05"],
            "week_start_date": pd.to_datetime(
                ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01"]
            ),
            "price_nok_per_kg": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


@pytest.fixture
def sst() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_start_date": pd.to_datetime(
                ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01"]
            ),
            "sst_anomaly_c": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )


@pytest.fixture
def fishmeal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_date": pd.to_datetime(["2020-12-01", "2021-01-01"]),
            "fishmeal_price_usd_per_tonne": [100.0, 110.0],
        }
    )


def test_resample_fishmeal_to_weekly_uses_backward_asof():
    fishmeal = pd.DataFrame(
        {
            "month_date": pd.to_datetime(["2020-12-01", "2021-01-01"]),
            "fishmeal_price_usd_per_tonne": [100.0, 110.0],
        }
    )
    weeks = pd.Series(pd.to_datetime(["2020-12-15", "2021-01-05"]))

    weekly = resample_fishmeal_to_weekly(fishmeal, weeks)

    assert weekly["fishmeal_price_usd_per_tonne"].tolist() == [100.0, 110.0]


def test_build_weekly_panel_lags_exog_by_one_week(ssb, sst, fishmeal):
    panel = build_weekly_panel(ssb, sst, fishmeal, exog_lag_weeks=1)

    # Inner join drops the first ssb week (no lagged exog observed yet) and
    # the exog weeks that would fall past the last ssb week.
    assert panel["week_start_date"].tolist() == list(
        pd.to_datetime(["2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01"])
    )

    # Row for 2021-01-11 should carry sea-temp from 2021-01-04 (the previous
    # week), not 2021-01-11's own (not-yet-known) value.
    row = panel[panel["week_start_date"] == "2021-01-11"].iloc[0]
    assert row["sst_anomaly_c"] == pytest.approx(1.0)
    assert row["price_nok_per_kg"] == pytest.approx(11.0)


def test_build_weekly_panel_no_nulls_or_duplicates(ssb, sst, fishmeal):
    panel = build_weekly_panel(ssb, sst, fishmeal, exog_lag_weeks=1)

    assert not panel.isnull().values.any()
    assert not panel["week_start_date"].duplicated().any()
