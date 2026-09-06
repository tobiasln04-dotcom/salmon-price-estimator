"""Simple forecast combination: does averaging the univariate SARIMAX's
and the autoregressive XGBoost's predictions do any better than either
alone? Cheap to test since both backtests already exist - classic
"forecast combination puzzle" territory: combining a strong model with a
weak one sometimes still reduces variance even when the weak model alone
never beats naive.
"""

from __future__ import annotations

import pandas as pd


def build_ensemble_predictions(
    df_a: pd.DataFrame,
    pred_col_a: str,
    df_b: pd.DataFrame,
    pred_col_b: str,
    ensemble_col: str = "ensemble_pred",
) -> pd.DataFrame:
    """Inner-join two backtests on `week_id` and average their predictions.

    Only weeks present in *both* backtests are kept (they cover different
    date ranges). Returns `week_id`, `actual`, `naive_pred`, both source
    prediction columns, and `ensemble_col`.
    """
    merged = df_a[["week_id", "actual", "naive_pred", pred_col_a]].merge(
        df_b[["week_id", pred_col_b]], on="week_id", how="inner"
    )
    merged[ensemble_col] = (merged[pred_col_a] + merged[pred_col_b]) / 2
    return merged
