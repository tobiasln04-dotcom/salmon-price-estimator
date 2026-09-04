import pandas as pd

from salmon_price_estimator.data.daily_market_data import combine_series, normalize_close_index


def test_normalize_close_index_strips_tz_and_time_of_day():
    closes = pd.Series(
        [100.0, 101.0],
        index=pd.to_datetime(["2026-07-01 00:00:00+02:00", "2026-07-02 00:00:00+02:00"]),
    )

    normalized = normalize_close_index(closes)

    assert normalized.index.tz is None
    assert normalized.index.tolist() == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")]


def test_combine_series_outer_joins_on_date_keeping_gaps():
    a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-07-01", "2026-07-02"]))
    b = pd.Series([10.0], index=pd.to_datetime(["2026-07-02"]))  # missing 07-01

    combined = combine_series({"a": a, "b": b})

    assert list(combined.columns) == ["date", "a", "b"]
    assert combined["a"].tolist() == [1.0, 2.0]
    row1 = combined[combined["date"] == "2026-07-01"].iloc[0]
    assert pd.isna(row1["b"])


def test_combine_series_sorts_by_date():
    a = pd.Series([2.0, 1.0], index=pd.to_datetime(["2026-07-02", "2026-07-01"]))

    combined = combine_series({"a": a})

    assert combined["date"].tolist() == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")]
    assert combined["a"].tolist() == [1.0, 2.0]
