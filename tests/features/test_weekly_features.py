import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.features.weekly_features import (
    TARGET_COL,
    build_features,
    feature_columns,
)


@pytest.fixture
def df() -> pd.DataFrame:
    n = 15
    return pd.DataFrame(
        {
            "week_id": [f"2021U{i:02d}" for i in range(1, n + 1)],
            "week_start_date": pd.date_range("2021-01-04", periods=n, freq="7D"),
            "price_nok_per_kg": [10.0 + i for i in range(n)],  # 10, 11, 12, ...
        }
    )


def test_feature_columns_names(df):
    cols = feature_columns(lags=[1, 2], rolling_windows=[3])
    assert cols == [
        "price_lag_1",
        "price_lag_2",
        "rolling_mean_3",
        "rolling_std_3",
        "log_return_lag_1",
        "week_of_year_sin",
        "week_of_year_cos",
    ]


def test_build_features_lags_use_only_prior_weeks(df):
    feats = build_features(df, price_col="price_nok_per_kg", lags=[1, 2], rolling_windows=[3])

    # Row 5 (price=15) should see lag_1=14 (row 4) and lag_2=13 (row 3) -
    # never its own value.
    row = feats.iloc[5]
    assert row["price_nok_per_kg"] == pytest.approx(15.0)
    assert row["price_lag_1"] == pytest.approx(14.0)
    assert row["price_lag_2"] == pytest.approx(13.0)


def test_build_features_rolling_stats_exclude_current_week(df):
    feats = build_features(df, price_col="price_nok_per_kg", lags=[1], rolling_windows=[3])

    # Row 5 (price=15): rolling window over the 3 weeks *before* it (14,13,12).
    row = feats.iloc[5]
    assert row["rolling_mean_3"] == pytest.approx(np.mean([12.0, 13.0, 14.0]))


def test_build_features_target_is_log_return(df):
    feats = build_features(df, price_col="price_nok_per_kg", lags=[1], rolling_windows=[3])

    row = feats.iloc[5]
    expected = np.log(15.0 / 14.0)
    assert row[TARGET_COL] == pytest.approx(expected)


def test_build_features_seasonality_is_bounded(df):
    feats = build_features(df, price_col="price_nok_per_kg", lags=[1], rolling_windows=[3])

    assert feats["week_of_year_sin"].between(-1.0, 1.0).all()
    assert feats["week_of_year_cos"].between(-1.0, 1.0).all()


def test_build_features_warmup_rows_are_nan(df):
    lags = [1, 2, 4]
    rolling_windows = [3]
    feats = build_features(
        df, price_col="price_nok_per_kg", lags=lags, rolling_windows=rolling_windows
    )
    cols = feature_columns(lags, rolling_windows) + [TARGET_COL]

    # The largest lag (4) plus the rolling/log-return shifts mean the first
    # few rows can't have complete features.
    assert feats.iloc[4:][cols].isnull().any(axis=1).sum() == 0
    assert feats.iloc[0][cols].isnull().any()
