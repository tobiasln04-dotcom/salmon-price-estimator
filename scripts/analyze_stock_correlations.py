"""Side analysis: which salmon-farming stock is most exposed to weekly
moves in the SSB salmon export price?

Not part of the core forecasting pipeline (see `scripts/run_backtest.py`)
- a standalone descriptive analysis reusing already-fetched data. Requires
`data/processed/daily_market_data.parquet` and
`data/processed/ssb_weekly_export_price.parquet` to already exist (run
`scripts/run_backtest.py` first, or the individual fetchers, if not).

Usage:
    uv run python scripts/analyze_stock_correlations.py

Chart colors follow the project's dataviz-skill palette: a single accent
hue for all bars, since the "job" here is comparing one metric
(correlation) across named categories (stock tickers), not distinguishing
multiple series - see the skill's color-formula.md on nominal categoricals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from salmon_price_estimator.data import ssb_export_price
from salmon_price_estimator.eval.stock_correlation import compute_stock_salmon_correlations
from salmon_price_estimator.features import weekly_panel

REPO_ROOT = Path(__file__).resolve().parents[1]

BAR_COLOR = "#2a78d6"  # dataviz skill: sequential/slot-1 blue
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_correlations(results: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values("correlation")  # ascending: highest ends up on top

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.barh(ordered["stock"], ordered["correlation"], color=BAR_COLOR, height=0.55, zorder=3)

    max_value = ordered["correlation"].max()
    for bar, value in zip(bars, ordered["correlation"], strict=True):
        ax.text(
            bar.get_width() + max_value * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            color=TEXT_PRIMARY,
            fontsize=9,
        )

    ax.set_xlabel(
        "Correlation with weekly salmon export price return", color=TEXT_MUTED, fontsize=9
    )
    ax.set_title(
        "Which salmon stock tracks the salmon price?",
        color=TEXT_PRIMARY,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.text(
        0.0,
        1.03,
        "Weekly return correlation, common window 2010–present - all close to zero",
        transform=ax.transAxes,
        color=TEXT_MUTED,
        fontsize=9,
    )

    ax.tick_params(colors=TEXT_PRIMARY, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_value * 1.35)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    data_config = ssb_export_price.load_config()
    model_config = weekly_panel.load_model_config()
    cfg = model_config["stock_correlation"]

    daily_market = pd.read_parquet(REPO_ROOT / data_config["daily_market_data"]["processed_path"])
    ssb = pd.read_parquet(REPO_ROOT / data_config["ssb_export_price"]["processed_path"])

    results = compute_stock_salmon_correlations(
        daily_market, ssb, cfg["stock_columns"], common_start=cfg["common_start"]
    )

    results_path = REPO_ROOT / cfg["results_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    pd.set_option("display.width", 120)
    print(results.to_string(index=False))

    plot_correlations(results, REPO_ROOT / cfg["chart_path"])


if __name__ == "__main__":
    main()
