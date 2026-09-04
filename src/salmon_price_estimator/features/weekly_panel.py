"""Join the weekly target series with exogenous features into one panel.

The exogenous columns (sea surface temperature anomaly, fishmeal price) are
lagged before joining so the panel never carries look-ahead: the week-ahead
forecast is made before that week's sea-temp/fishmeal data exists, so only
information already known as of `exog_lag_weeks` earlier is used to predict
a given week's target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_CONFIG_PATH = REPO_ROOT / "config" / "data.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "config" / "model.yaml"


def load_data_config(config_path: Path = DEFAULT_DATA_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_config(config_path: Path = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _lag_by_weeks(df: pd.DataFrame, date_col: str, weeks: int) -> pd.DataFrame:
    """Shift a date column forward by `weeks`, so a value observed at week `d`
    becomes available as a predictor for week `d + weeks`."""
    out = df.copy()
    out[date_col] = out[date_col] + pd.Timedelta(weeks=weeks)
    return out


def resample_fishmeal_to_weekly(
    fishmeal: pd.DataFrame, week_start_dates: pd.Series
) -> pd.DataFrame:
    """As-of map each week onto the most recently published monthly fishmeal price.

    Uses backward as-of matching (never a future month's price for a given week).
    """
    weeks = pd.to_datetime(sorted(week_start_dates.unique())).astype("datetime64[ns]")
    weeks = pd.DataFrame({"week_start_date": weeks})
    fishmeal = fishmeal.assign(month_date=fishmeal["month_date"].astype("datetime64[ns]"))
    fishmeal = fishmeal.sort_values("month_date")
    merged = pd.merge_asof(
        weeks,
        fishmeal,
        left_on="week_start_date",
        right_on="month_date",
        direction="backward",
    )
    return merged[["week_start_date", "fishmeal_price_usd_per_tonne"]]


def build_weekly_panel(
    ssb: pd.DataFrame,
    sst: pd.DataFrame,
    fishmeal: pd.DataFrame,
    exog_lag_weeks: int = 1,
) -> pd.DataFrame:
    """Inner-join target + lagged exogenous features onto the SSB weekly grid."""
    ssb = ssb.assign(week_start_date=ssb["week_start_date"].astype("datetime64[ns]"))
    sst = sst.assign(week_start_date=sst["week_start_date"].astype("datetime64[ns]"))

    fishmeal_weekly = resample_fishmeal_to_weekly(fishmeal, ssb["week_start_date"])

    sst_cols = sst[["week_start_date", "sst_anomaly_c"]]
    sst_lagged = _lag_by_weeks(sst_cols, "week_start_date", exog_lag_weeks)
    fishmeal_lagged = _lag_by_weeks(fishmeal_weekly, "week_start_date", exog_lag_weeks)

    panel = ssb[["week_id", "week_start_date", "price_nok_per_kg"]]
    panel = panel.merge(sst_lagged, on="week_start_date", how="inner")
    panel = panel.merge(fishmeal_lagged, on="week_start_date", how="inner")
    return panel.sort_values("week_start_date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(
    data_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load the three processed weekly-feature sources and build the joined panel."""
    data_config = data_config or load_data_config()
    model_config = model_config or load_model_config()

    ssb = pd.read_parquet(REPO_ROOT / data_config["ssb_export_price"]["processed_path"])
    sst = pd.read_parquet(REPO_ROOT / data_config["sea_surface_temp"]["processed_path"])
    fishmeal = pd.read_parquet(REPO_ROOT / data_config["feed_cost_fishmeal"]["processed_path"])

    cfg = model_config["weekly_panel"]
    panel = build_weekly_panel(ssb, sst, fishmeal, exog_lag_weeks=cfg["exog_lag_weeks"])
    save_processed(panel, REPO_ROOT / cfg["processed_path"])

    return panel


def main() -> None:
    df = run()
    print(f"Saved {len(df)} weekly rows to data/processed/weekly_panel.parquet")
    print(df.tail())


if __name__ == "__main__":
    main()
