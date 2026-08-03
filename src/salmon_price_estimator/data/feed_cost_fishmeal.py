"""Fetch the monthly fishmeal price (feed cost proxy) from IndexMundi.

IndexMundi has no dedicated CSV/API export - the price history is embedded
as an HTML table on the commodity page, so we parse that directly. There is
no "fish-oil" commodity on IndexMundi at all; see CLAUDE.md "Deferred items".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "data.yaml"

USER_AGENT = "Mozilla/5.0 (compatible; salmon-price-estimator data pipeline)"
MONTH_PRICE_ROW = re.compile(r"<td>([A-Za-z]{3} \d{4})</td><td>([\d,]+\.\d+)</td>")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_page_url(page_url: str, commodity: str, months: int) -> str:
    return f"{page_url}?commodity={commodity}&months={months}"


def fetch_page_html(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text


def parse_price_table(html: str) -> pd.DataFrame:
    """Extract the (Month, Price) rows from the embedded gvPrices HTML table."""
    match = re.search(r'id="gvPrices".*?</table>', html, re.S)
    if not match:
        raise ValueError("Could not find the gvPrices data table in the page HTML")

    rows = MONTH_PRICE_ROW.findall(match.group(0))
    if not rows:
        raise ValueError("gvPrices table found but no (month, price) rows matched")

    months, prices = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "month_date": pd.to_datetime(months, format="%b %Y"),
            "fishmeal_price_usd_per_tonne": [float(p.replace(",", "")) for p in prices],
        }
    )


def save_raw(html: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch, land raw + processed, and return the monthly fishmeal price DataFrame."""
    config = config or load_config()
    cfg = config["feed_cost_fishmeal"]

    url = build_page_url(cfg["page_url"], cfg["commodity"], cfg["months"])
    html = fetch_page_html(url)
    save_raw(html, REPO_ROOT / cfg["raw_path"])

    df = parse_price_table(html)
    save_processed(df, REPO_ROOT / cfg["processed_path"])

    return df


def main() -> None:
    df = run()
    print(f"Saved {len(df)} monthly rows to data/processed/fishmeal_price_monthly.parquet")
    print(df.tail())


if __name__ == "__main__":
    main()
