import pandas as pd
import pytest

from salmon_price_estimator.eval.ensemble import build_ensemble_predictions


def test_build_ensemble_predictions_averages_aligned_weeks():
    df_a = pd.DataFrame(
        {
            "week_id": ["W1", "W2", "W3"],
            "actual": [100.0, 101.0, 102.0],
            "naive_pred": [99.0, 100.0, 101.0],
            "sarimax_pred": [100.0, 102.0, 104.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "week_id": ["W1", "W2", "W3"],
            "xgboost_pred": [102.0, 100.0, 106.0],
        }
    )

    merged = build_ensemble_predictions(df_a, "sarimax_pred", df_b, "xgboost_pred")

    assert merged["ensemble_pred"].tolist() == pytest.approx([101.0, 101.0, 105.0])


def test_build_ensemble_predictions_inner_joins_on_week_id():
    df_a = pd.DataFrame(
        {
            "week_id": ["W1", "W2", "W3"],
            "actual": [100.0, 101.0, 102.0],
            "naive_pred": [99.0, 100.0, 101.0],
            "sarimax_pred": [100.0, 102.0, 104.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "week_id": ["W2", "W3", "W4"],  # W1 missing, W4 extra
            "xgboost_pred": [100.0, 106.0, 108.0],
        }
    )

    merged = build_ensemble_predictions(df_a, "sarimax_pred", df_b, "xgboost_pred")

    assert merged["week_id"].tolist() == ["W2", "W3"]


def test_build_ensemble_predictions_keeps_expected_columns():
    df_a = pd.DataFrame(
        {
            "week_id": ["W1"],
            "actual": [100.0],
            "naive_pred": [99.0],
            "sarimax_pred": [100.0],
        }
    )
    df_b = pd.DataFrame({"week_id": ["W1"], "xgboost_pred": [102.0]})

    merged = build_ensemble_predictions(df_a, "sarimax_pred", df_b, "xgboost_pred")

    assert set(merged.columns) == {
        "week_id",
        "actual",
        "naive_pred",
        "sarimax_pred",
        "xgboost_pred",
        "ensemble_pred",
    }
