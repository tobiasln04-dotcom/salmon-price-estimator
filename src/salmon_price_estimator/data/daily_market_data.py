"""Fetch daily FX pairs and Oslo Bors salmon-stock closes for the daily
nowcast layer (via yfinance).

Fish Pool/Euronext salmon futures data was investigated and deferred
2026-09-04 - no free historical source exists, see CLAUDE.md "Deferred
items" - so this covers only FX + stocks, not a futures curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "data.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_close_index(closes: pd.Series) -> pd.Series:
    """Strip tz-awareness and time-of-day from a yfinance close-price index
    so different tickers' calendars can be combined/compared by plain date."""
    out = closes.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def fetch_ticker_close(ticker: str, period: str) -> pd.Series:
    """Fetch a single ticker's daily close prices as a date-indexed Series."""
    history = yf.Ticker(ticker).history(period=period)
    return normalize_close_index(history["Close"])


def combine_series(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Outer-join named daily series into one wide, date-sorted frame.

    Each series keeps its own native trading calendar - non-trading days
    for a given series are NaN rather than forward-filled here.
    `features/daily_nowcast_features.py` handles per-series alignment.
    """
    named = {name: series.rename(name) for name, series in series_map.items()}
    combined = pd.concat(named.values(), axis=1, sort=True)
    combined.index.name = "date"
    return combined.reset_index()


def save_raw(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name="close").to_csv(path)


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch, land raw + processed, and return the wide daily market-data frame."""
    config = config or load_config()
    cfg = config["daily_market_data"]

    series_map: dict[str, pd.Series] = {}
    for entry in cfg["fx_pairs"] + cfg["stock_tickers"]:
        ticker, column = entry["ticker"], entry["column"]
        closes = fetch_ticker_close(ticker, cfg["period"])
        save_raw(closes, REPO_ROOT / cfg["raw_dir"] / f"{column}.csv")
        series_map[column] = closes

    combined = combine_series(series_map)
    save_processed(combined, REPO_ROOT / cfg["processed_path"])
    return combined


def main() -> None:
    df = run()
    print(f"Saved {len(df)} daily rows to data/processed/daily_market_data.parquet")
    print(df.tail())


if __name__ == "__main__":
    main()
