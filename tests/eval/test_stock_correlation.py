import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.eval.stock_correlation import (
    compute_stock_salmon_correlations,
    compute_week_end_return,
)


@pytest.fixture
def ssb() -> pd.DataFrame:
    n = 10
    week_start = pd.date_range("2021-01-04", periods=n, freq="7D")  # Mondays
    price = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0, 15.0]
    return pd.DataFrame(
        {
            "week_id": [f"2021U{i:02d}" for i in range(1, n + 1)],
            "week_start_date": week_start,
            "price_nok_per_kg": price,
        }
    )


@pytest.fixture
def daily_market(ssb) -> pd.DataFrame:
    week_end_dates = ssb["week_start_date"] + pd.Timedelta(days=6)  # Sundays
    price = ssb["price_nok_per_kg"].to_numpy()

    # perfectly_tracks: same log-returns as salmon price (scaled) -> corr = 1.0
    perfectly_tracks = price * 2.0
    # unrelated: a pattern with no resemblance to the salmon price's shape
    unrelated = np.array([100.0, 105.0, 95.0, 110.0, 90.0, 120.0, 80.0, 130.0, 70.0, 140.0])

    return pd.DataFrame(
        {
            "date": week_end_dates,
            "perfectly_tracks": perfectly_tracks,
            "unrelated": unrelated,
        }
    )


def test_compute_week_end_return_matches_asof_values(ssb, daily_market):
    week_end_dates = ssb["week_start_date"] + pd.Timedelta(days=6)
    series = daily_market.set_index("date")["perfectly_tracks"]

    returns = compute_week_end_return(series, week_end_dates)

    expected = np.log(daily_market["perfectly_tracks"] / daily_market["perfectly_tracks"].shift(1))
    assert returns.to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


def test_compute_stock_salmon_correlations_ranks_the_tracking_stock_first(ssb, daily_market):
    results = compute_stock_salmon_correlations(
        daily_market, ssb, stock_columns=["perfectly_tracks", "unrelated"]
    )

    assert results.iloc[0]["stock"] == "PERFECTLY_TRACKS"
    assert results.iloc[0]["correlation"] == pytest.approx(1.0)
    assert results.iloc[0]["r_squared"] == pytest.approx(1.0)
    assert results.iloc[0]["correlation"] > results.iloc[1]["correlation"]


def test_compute_stock_salmon_correlations_sorted_descending(ssb, daily_market):
    results = compute_stock_salmon_correlations(
        daily_market, ssb, stock_columns=["unrelated", "perfectly_tracks"]
    )

    assert results["correlation"].is_monotonic_decreasing


def test_compute_stock_salmon_correlations_common_start_filters_weeks(ssb, daily_market):
    full = compute_stock_salmon_correlations(daily_market, ssb, stock_columns=["perfectly_tracks"])
    restricted = compute_stock_salmon_correlations(
        daily_market, ssb, stock_columns=["perfectly_tracks"], common_start="2021-02-15"
    )

    assert restricted.iloc[0]["n_weeks"] < full.iloc[0]["n_weeks"]
