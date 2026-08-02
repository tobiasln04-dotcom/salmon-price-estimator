"""Fetch the SSB weekly export price of fresh farmed salmon (StatBank table 03024).

This is the target variable: the weighted average export price per kg,
FOB Norway, for fish-farm bred salmon, fresh or chilled, published weekly.
"""

from __future__ import annotations

import json
from datetime import date
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


def fetch_raw(
    api_url: str, commodity_code: str, contents_code: str, timeout: int = 30
) -> dict[str, Any]:
    """Query the PxWebApi table and return the raw json-stat2 response."""
    query = {
        "query": [
            {"code": "VareGrupper2", "selection": {"filter": "item", "values": [commodity_code]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": [contents_code]}},
            {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    response = requests.post(api_url, json=query, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_json_stat2(raw: dict[str, Any]) -> pd.DataFrame:
    """Flatten a single-series json-stat2 response into a tidy weekly DataFrame."""
    tid_index = raw["dimension"]["Tid"]["category"]["index"]
    week_codes = sorted(tid_index, key=tid_index.get)
    values = raw["value"]
    if len(values) != len(week_codes):
        raise ValueError(f"Expected {len(week_codes)} values, got {len(values)}")

    years = [int(code[:4]) for code in week_codes]
    weeks = [int(code[5:]) for code in week_codes]
    week_start_dates = [date.fromisocalendar(y, w, 1) for y, w in zip(years, weeks, strict=True)]

    return pd.DataFrame(
        {
            "week_id": week_codes,
            "year": years,
            "week": weeks,
            "week_start_date": pd.to_datetime(week_start_dates),
            "price_nok_per_kg": values,
        }
    )


def trim_to_range(df: pd.DataFrame, start_week: str | None, end_week: str | None) -> pd.DataFrame:
    """Filter to [start_week, end_week] inclusive, comparing week_id lexicographically.

    week_id codes (YYYYUww) sort lexicographically the same as chronologically.
    """
    if start_week is not None:
        df = df[df["week_id"] >= start_week]
    if end_week is not None:
        df = df[df["week_id"] <= end_week]
    return df.reset_index(drop=True)


def save_raw(raw: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2))


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch, land raw + processed, and return the tidy weekly export price DataFrame."""
    config = config or load_config()
    cfg = config["ssb_export_price"]

    raw = fetch_raw(
        api_url=cfg["api_url"],
        commodity_code=cfg["commodity_code"],
        contents_code=cfg["contents_code"],
    )
    save_raw(raw, REPO_ROOT / cfg["raw_path"])

    df = parse_json_stat2(raw)
    df = trim_to_range(df, cfg.get("start_week"), cfg.get("end_week"))
    save_processed(df, REPO_ROOT / cfg["processed_path"])

    return df


def main() -> None:
    df = run()
    print(f"Saved {len(df)} weekly rows to data/processed/ssb_weekly_export_price.parquet")
    print(df.tail())


if __name__ == "__main__":
    main()
