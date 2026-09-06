import numpy as np
import pandas as pd
import pytest

from salmon_price_estimator.eval.prediction_intervals import (
    compute_coverage,
    empirical_interval_backtest,
)


def test_empirical_interval_backtest_nan_before_min_history():
    df = pd.DataFrame({"actual": np.arange(10.0), "pred": np.arange(10.0)})

    result = empirical_interval_backtest(df, "pred", min_history=5)

    assert result["lower"].iloc[:5].isna().all()
    assert result["lower"].iloc[5:].notna().all()
    assert result["upper"].iloc[5:].notna().all()


def test_empirical_interval_backtest_uses_only_past_residuals():
    # Constant residual of +2 for the first 6 rows, then a residual for row 6
    # itself that must NOT leak into its own interval.
    actual = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 100.0])
    pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    df = pd.DataFrame({"actual": actual, "pred": pred})

    result = empirical_interval_backtest(df, "pred", alpha=0.5, min_history=5)

    # Row 5: past residuals (rows 0-4) are all exactly +2 -> interval collapses to pred+2.
    assert result.loc[5, "lower"] == pytest.approx(2.0)
    assert result.loc[5, "upper"] == pytest.approx(2.0)
    # Row 6's own huge residual (+100) must not affect its own bounds -
    # its interval is still based on the prior (all +2) residuals.
    assert result.loc[6, "lower"] == pytest.approx(2.0)
    assert result.loc[6, "upper"] == pytest.approx(2.0)


def test_empirical_interval_backtest_window_caps_history():
    # One huge early residual (row 0), then many small ones. Once the huge
    # residual falls outside the rolling `window`, it should stop affecting
    # the interval.
    actual = np.concatenate([[1000.0], np.ones(10)])
    pred = np.zeros(11)
    df = pd.DataFrame({"actual": actual, "pred": pred})

    result = empirical_interval_backtest(df, "pred", alpha=0.5, window=3, min_history=3)

    # Row 3: window=3 looks back at rows [0,1,2] - still includes the huge
    # residual from row 0.
    assert result.loc[3, "upper"] > 100
    # Row 5: window=3 looks back at rows [2,3,4] - the huge residual has
    # rolled out of view.
    assert result.loc[5, "upper"] < 10


def test_compute_coverage_all_within():
    df = pd.DataFrame(
        {"actual": [1.0, 2.0, 3.0], "lower": [0.0, 1.0, 2.0], "upper": [2.0, 3.0, 4.0]}
    )

    result = compute_coverage(df)

    assert result["coverage"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["mean_width"] == pytest.approx(2.0)
    assert result["median_width"] == pytest.approx(2.0)


def test_compute_coverage_median_width_resists_outliers():
    # One huge-width outlier row shouldn't drag the median the way it
    # drags the mean.
    df = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 3.0, 4.0],
            "lower": [0.0, 1.0, 2.0, -1e9],
            "upper": [2.0, 3.0, 4.0, 1e9],
        }
    )

    result = compute_coverage(df)

    assert result["median_width"] == pytest.approx(2.0)
    assert result["mean_width"] > 1e8


def test_compute_coverage_none_within():
    df = pd.DataFrame({"actual": [10.0, 10.0], "lower": [0.0, 0.0], "upper": [1.0, 1.0]})

    result = compute_coverage(df)

    assert result["coverage"] == pytest.approx(0.0)


def test_compute_coverage_ignores_nan_rows():
    df = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 3.0],
            "lower": [np.nan, 1.0, 2.0],
            "upper": [np.nan, 3.0, 4.0],
        }
    )

    result = compute_coverage(df)

    assert result["n"] == 2
