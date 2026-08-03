import pandas as pd
import pytest

from salmon_price_estimator.data.sea_surface_temperature import (
    average_across_points,
    build_point_query_url,
    build_year_chunks,
    parse_erddap_csv,
    resample_to_weekly,
)

ERDDAP_CSV_FIXTURE = """time,depth,latitude,longitude,anom
UTC,m,degrees_north,degrees_east,Celsius
2026-07-01T12:00:00Z,0.0,62.625,6.125,1.79
2026-07-02T12:00:00Z,0.0,62.625,6.125,1.86
2026-07-03T12:00:00Z,0.0,62.625,6.125,1.67
"""


def test_build_point_query_url():
    url = build_point_query_url(
        csv_url="https://example.org/erddap/griddap/x.csv",
        variable="anom",
        start_date="2020-02-28",
        end_date="2026-08-03",
        lat=62.625,
        lon=6.125,
    )
    assert url == (
        "https://example.org/erddap/griddap/x.csv?anom"
        "[(2020-02-28):1:(2026-08-03)][(0.0)][(62.625):1:(62.625)][(6.125):1:(6.125)]"
    )


def test_parse_erddap_csv_skips_units_row_and_normalizes_dates():
    df = parse_erddap_csv(ERDDAP_CSV_FIXTURE, "anom")

    assert list(df.columns) == ["date", "anom"]
    assert len(df) == 3
    assert df["anom"].tolist() == [1.79, 1.86, 1.67]
    assert df.loc[0, "date"] == pd.Timestamp("2026-07-01")


def test_build_year_chunks_splits_by_calendar_year_and_clamps_endpoints():
    chunks = build_year_chunks("2020-02-28", "2022-06-15")

    assert chunks == [
        ("2020-02-28", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-06-15"),
    ]


def test_build_year_chunks_single_year():
    chunks = build_year_chunks("2026-01-05", "2026-03-01")

    assert chunks == [("2026-01-05", "2026-03-01")]


def test_average_across_points_takes_mean():
    df_a = pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-07-02"]), "anom": [1.0, 2.0]})
    df_b = pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-07-02"]), "anom": [3.0, 4.0]})

    avg = average_across_points({"a": df_a, "b": df_b}, "anom")

    assert avg["sst_anomaly_c"].tolist() == [2.0, 3.0]


def test_resample_to_weekly_aggregates_iso_week_mean():
    # 2026-07-01 is a Wednesday, ISO week 27 of 2026. Give it two weeks of data.
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-06", "2026-07-07"]
            ),
            "sst_anomaly_c": [1.0, 2.0, 3.0, 10.0, 20.0],
        }
    )

    weekly = resample_to_weekly(daily)

    assert weekly["week_id"].tolist() == ["2026U27", "2026U28"]
    assert weekly.loc[0, "sst_anomaly_c"] == pytest.approx(2.0)
    assert weekly.loc[1, "sst_anomaly_c"] == pytest.approx(15.0)
    assert weekly.loc[0, "week_start_date"] == pd.Timestamp("2026-06-29")
    assert weekly.loc[1, "week_start_date"] == pd.Timestamp("2026-07-06")
