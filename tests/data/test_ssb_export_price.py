import pandas as pd
import pytest

from salmon_price_estimator.data.ssb_export_price import parse_json_stat2, trim_to_range


@pytest.fixture
def raw_json_stat2() -> dict:
    """Minimal json-stat2 fixture shaped like the real SSB table 03024 response."""
    return {
        "dimension": {
            "VareGrupper2": {"category": {"index": {"01": 0}}},
            "ContentsCode": {"category": {"index": {"Kilopris": 0}}},
            "Tid": {
                "category": {
                    "index": {
                        "2026U01": 0,
                        "2026U02": 1,
                        "2026U03": 2,
                    }
                }
            },
        },
        "value": [80.12, 79.5, 81.0],
    }


def test_parse_json_stat2_shape_and_types(raw_json_stat2):
    df = parse_json_stat2(raw_json_stat2)

    assert list(df.columns) == [
        "week_id",
        "year",
        "week",
        "week_start_date",
        "price_nok_per_kg",
    ]
    assert len(df) == 3
    assert df["week_id"].tolist() == ["2026U01", "2026U02", "2026U03"]
    assert df["year"].tolist() == [2026, 2026, 2026]
    assert df["week"].tolist() == [1, 2, 3]
    assert df["price_nok_per_kg"].tolist() == [80.12, 79.5, 81.0]


def test_parse_json_stat2_week_start_is_monday(raw_json_stat2):
    df = parse_json_stat2(raw_json_stat2)

    # ISO week 1 of 2026 starts on Monday 2025-12-29.
    assert df.loc[0, "week_start_date"] == pd.Timestamp(2025, 12, 29)
    for d in df["week_start_date"]:
        assert d.weekday() == 0


def test_parse_json_stat2_value_count_mismatch_raises():
    bad = {
        "dimension": {"Tid": {"category": {"index": {"2026U01": 0, "2026U02": 1}}}},
        "value": [1.0],
    }
    with pytest.raises(ValueError, match="Expected 2 values, got 1"):
        parse_json_stat2(bad)


def test_trim_to_range_filters_inclusive():
    df = pd.DataFrame(
        {
            "week_id": ["2026U01", "2026U02", "2026U03", "2026U04"],
            "price_nok_per_kg": [1.0, 2.0, 3.0, 4.0],
        }
    )

    trimmed = trim_to_range(df, start_week="2026U02", end_week="2026U03")

    assert trimmed["week_id"].tolist() == ["2026U02", "2026U03"]


def test_trim_to_range_no_bounds_returns_all():
    df = pd.DataFrame({"week_id": ["2026U01", "2026U02"], "price_nok_per_kg": [1.0, 2.0]})

    trimmed = trim_to_range(df, start_week=None, end_week=None)

    assert len(trimmed) == 2
