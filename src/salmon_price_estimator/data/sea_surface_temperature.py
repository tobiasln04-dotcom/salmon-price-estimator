"""Fetch NOAA OISST v2.1 daily sea surface temperature anomaly for the
Norwegian salmon-farming belt (via NCEI's ERDDAP mirror) and land it as a
weekly, coast-averaged parquet series.

Note: this ERDDAP mirror only keeps a rolling recent window (verified
2026-08-03: ~2020-02-28 to present), not the full 1981-present OISST
archive - see CLAUDE.md "Deferred items". The resulting weekly series will
have no data before that window.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "data.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_time_bounds(das_url: str, timeout: int = 30) -> tuple[str, str]:
    """Read the dataset's actual available [start, end] dates from its .das metadata."""
    response = requests.get(das_url, timeout=timeout)
    response.raise_for_status()
    match = re.search(r"time \{.*?actual_range ([\d.eE+-]+), ([\d.eE+-]+);", response.text, re.S)
    if not match:
        raise ValueError(f"Could not find time actual_range in DAS response from {das_url}")
    start = datetime.fromtimestamp(float(match.group(1)), tz=UTC).date()
    end = datetime.fromtimestamp(float(match.group(2)), tz=UTC).date()
    return start.isoformat(), end.isoformat()


def build_point_query_url(
    csv_url: str, variable: str, start_date: str, end_date: str, lat: float, lon: float
) -> str:
    return (
        f"{csv_url}?{variable}"
        f"[({start_date}):1:({end_date})]"
        f"[(0.0)]"
        f"[({lat}):1:({lat})]"
        f"[({lon}):1:({lon})]"
    )


def build_year_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split [start_date, end_date] into calendar-year chunks.

    A single request spanning the full ~5.5 year window gets rejected by the
    ERDDAP server with a 408 after ~2 minutes (verified 2026-08-03); one year
    per request (~15s) stays comfortably under that limit.
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return [
        (max(f"{year}-01-01", start_date), min(f"{year}-12-31", end_date))
        for year in range(start_year, end_year + 1)
    ]


def fetch_point_csv(url: str, timeout: int = 90) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_erddap_csv(csv_text: str, variable: str) -> pd.DataFrame:
    """Parse an ERDDAP griddap CSV response (header row + units row) into a tidy frame."""
    df = pd.read_csv(io.StringIO(csv_text), skiprows=[1])
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["time"]).dt.normalize().dt.tz_localize(None),
            variable: df[variable],
        }
    )


def average_across_points(point_frames: dict[str, pd.DataFrame], variable: str) -> pd.DataFrame:
    """Average a daily anomaly series across coastal points into one daily series."""
    combined = pd.concat(
        [df.set_index("date")[variable].rename(name) for name, df in point_frames.items()],
        axis=1,
    )
    return combined.mean(axis=1).rename("sst_anomaly_c").reset_index()


def resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily [date, sst_anomaly_c] series to ISO-week means."""
    df = daily_df.copy()
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"]
    df["week"] = iso["week"]

    weekly = df.groupby(["year", "week"], as_index=False)["sst_anomaly_c"].mean()
    weekly["week_id"] = weekly["year"].astype(str) + "U" + weekly["week"].astype(str).str.zfill(2)
    weekly["week_start_date"] = pd.to_datetime(
        weekly.apply(lambda r: date.fromisocalendar(int(r["year"]), int(r["week"]), 1), axis=1)
    )
    return (
        weekly[["week_id", "year", "week", "week_start_date", "sst_anomaly_c"]]
        .sort_values("week_id")
        .reset_index(drop=True)
    )


def save_raw(csv_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text)


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch, land raw + processed, and return the weekly coast-averaged SST anomaly."""
    config = config or load_config()
    cfg = config["sea_surface_temp"]
    variable = cfg["variable"]

    start_date, end_date = fetch_time_bounds(cfg["das_url"])
    year_chunks = build_year_chunks(start_date, end_date)

    point_frames: dict[str, pd.DataFrame] = {}
    for point in cfg["points"]:
        print(f"Fetching {point['name']} ({point['lat']}, {point['lon']})...")
        chunk_frames = []
        for chunk_start, chunk_end in year_chunks:
            url = build_point_query_url(
                cfg["csv_url"], variable, chunk_start, chunk_end, point["lat"], point["lon"]
            )
            csv_text = fetch_point_csv(url)
            save_raw(
                csv_text, REPO_ROOT / cfg["raw_dir"] / point["name"] / f"{chunk_start[:4]}.csv"
            )
            chunk_frames.append(parse_erddap_csv(csv_text, variable))
        point_frames[point["name"]] = pd.concat(chunk_frames, ignore_index=True)

    daily_avg = average_across_points(point_frames, variable)
    weekly = resample_to_weekly(daily_avg)
    save_processed(weekly, REPO_ROOT / cfg["processed_path"])

    return weekly


def main() -> None:
    df = run()
    print(f"Saved {len(df)} weekly rows to data/processed/sea_surface_temp_anomaly_weekly.parquet")
    print(df.tail())


if __name__ == "__main__":
    main()
