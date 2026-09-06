"""Single entry point reproducing the weekly baseline and nowcast backtests.

Runs four weekly variants side by side (see CLAUDE.md session log):
  - SARIMAX univariate (2026-08-31): SSB export price only, full history.
  - SARIMAX exogenous (2026-08-31): + fishmeal price and sea surface
    temperature anomaly (1-week lagged, see features.weekly_panel), which
    naturally truncates the backtest window to ~2020-present.
  - XGBoost autoregressive (2026-09-03): lagged price/rolling stats/
    seasonality features only, full history, rolling training window.
  - XGBoost exogenous (2026-09-03): same features + the exogenous columns,
    truncated to ~2020-present like its SARIMAX counterpart - built to
    test whether XGBoost's more flexible functional form lets the
    exogenous features help where SARIMAX's linear/seasonal-ARMA
    specification couldn't fairly test them (see session log 2026-09-03).

Then the daily nowcast layer (2026-09-04): anchored on the univariate
SARIMAX forecast (the one weekly baseline that beats naive), corrected
daily using FX + Oslo Bors salmon-stock returns as the week progresses
Mon-Thu. Evaluated by RMSE-by-days-elapsed (should decline Mon->Thu) and
against the "static" comparison of just holding the baseline flat all
week (see eval/backtest_nowcast.py).

Usage:
    uv run python scripts/run_backtest.py

Config (model orders/hyperparameters, training windows, refit cadence)
lives in config/model.yaml. Raw data sources are fetched on first run if
the processed parquet files under data/processed/ don't already exist;
re-run the individual fetchers directly (e.g. `python -m
salmon_price_estimator.data.ssb_export_price`) to refresh them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from salmon_price_estimator.data import (
    daily_market_data,
    feed_cost_fishmeal,
    sea_surface_temperature,
    ssb_export_price,
)
from salmon_price_estimator.eval import metrics
from salmon_price_estimator.eval.backtest import walk_forward_backtest
from salmon_price_estimator.eval.backtest_nowcast import walk_forward_nowcast_backtest
from salmon_price_estimator.eval.backtest_xgboost import rolling_window_backtest
from salmon_price_estimator.eval.ensemble import build_ensemble_predictions
from salmon_price_estimator.eval.prediction_intervals import (
    compute_coverage,
    empirical_interval_backtest,
)
from salmon_price_estimator.eval.random_walk_test import adf_test, ljung_box_test
from salmon_price_estimator.eval.significance import diebold_mariano_test
from salmon_price_estimator.eval.trading_strategy import (
    compute_strategy_metrics,
    simple_directional_strategy,
)
from salmon_price_estimator.features import weekly_panel
from salmon_price_estimator.features.daily_nowcast_features import (
    FEATURE_COLUMNS as NOWCAST_FEATURE_COLUMNS,
)
from salmon_price_estimator.features.daily_nowcast_features import (
    TARGET_COL as NOWCAST_TARGET_COL,
)
from salmon_price_estimator.features.daily_nowcast_features import build_nowcast_panel
from salmon_price_estimator.features.weekly_features import (
    TARGET_COL,
    build_features,
    feature_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_processed(module, data_config: dict, key: str) -> None:
    processed_path = REPO_ROOT / data_config[key]["processed_path"]
    if not processed_path.exists():
        print(f"{processed_path} not found, fetching...")
        module.run(data_config)


def compute_or_load(results_path: Path, compute_fn) -> pd.DataFrame:
    """Skip an expensive backtest if its results are already on disk.

    SARIMAX backtests take tens of minutes (see config/model.yaml); this
    makes reruns after an interruption (e.g. the machine sleeping mid-run,
    see CLAUDE.md session log) resume instead of redoing finished work.
    Delete the results file to force a recompute.
    """
    if results_path.exists():
        print(f"{results_path} already exists, skipping recompute")
        return pd.read_parquet(results_path)

    df = compute_fn()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path, index=False)
    return df


def summarize(variant: str, df: pd.DataFrame, pred_col: str) -> dict:
    return {
        "variant": variant,
        "n_weeks": len(df),
        "start_week": df["week_id"].iloc[0],
        "end_week": df["week_id"].iloc[-1],
        "model_mape": metrics.mape(df["actual"], df[pred_col]),
        "model_rmse": metrics.rmse(df["actual"], df[pred_col]),
        "model_directional_accuracy": metrics.directional_accuracy(
            df["actual"], df[pred_col], df["naive_pred"]
        ),
        "naive_mape": metrics.mape(df["actual"], df["naive_pred"]),
        "naive_rmse": metrics.rmse(df["actual"], df["naive_pred"]),
        "naive_directional_accuracy": metrics.directional_accuracy(
            df["actual"], df["naive_pred"], df["naive_pred"]
        ),
    }


def significance_row(variant: str, df: pd.DataFrame, pred_col: str, naive_col: str) -> dict:
    """Is `pred_col`'s edge over `naive_col` statistically significant, or
    could the point-estimate improvement be noise? Diebold-Mariano test,
    squared-error loss (matches RMSE), h=1 (all forecasts here are one-step-
    ahead)."""
    errors_model = (df["actual"] - df[pred_col]).to_numpy()
    errors_naive = (df["actual"] - df[naive_col]).to_numpy()
    dm_stat, p_value = diebold_mariano_test(errors_model, errors_naive, h=1)
    return {
        "variant": variant,
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "significant_at_5pct": bool(p_value < 0.05) if not np.isnan(p_value) else False,
    }


def strategy_row(variant: str, df: pd.DataFrame, pred_col: str) -> dict:
    """Notional long/short directional-strategy P&L for one variant - see
    eval/trading_strategy.py."""
    strategy_df = simple_directional_strategy(df, pred_col)
    return {"variant": variant} | compute_strategy_metrics(strategy_df)


def random_walk_summary(price: pd.Series) -> pd.DataFrame:
    """Combine the ADF and Ljung-Box tests into one printable/saveable table."""
    adf = adf_test(price)
    rows = [
        {
            "test": adf["test"],
            "statistic": adf["statistic"],
            "p_value": adf["p_value"],
            "rejected_at_5pct": adf["unit_root_rejected_at_5pct"],
            "interpretation": (
                "unit root rejected (NOT consistent with a random walk)"
                if adf["unit_root_rejected_at_5pct"]
                else "unit root not rejected (consistent with a random walk)"
            ),
        }
    ]
    for _, lb_row in ljung_box_test(price, lags=[1, 4, 12, 52]).iterrows():
        rows.append(
            {
                "test": f"Ljung-Box (weekly returns, lag {int(lb_row['lag'])})",
                "statistic": lb_row["lb_stat"],
                "p_value": lb_row["lb_pvalue"],
                "rejected_at_5pct": bool(lb_row["autocorrelation_rejected_at_5pct"]),
                "interpretation": (
                    "significant autocorrelation (NOT consistent with a random walk)"
                    if lb_row["autocorrelation_rejected_at_5pct"]
                    else "no significant autocorrelation (consistent with a random walk)"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_xgboost_variant(df: pd.DataFrame, feat_cfg: dict, xgb_cfg: dict) -> pd.DataFrame:
    """Build features on `df`, drop warm-up rows, run the rolling backtest."""
    feats = build_features(
        df,
        price_col=feat_cfg["price_col"],
        lags=feat_cfg["lags"],
        rolling_windows=feat_cfg["rolling_windows"],
    )
    cols = feature_columns(feat_cfg["lags"], feat_cfg["rolling_windows"]) + xgb_cfg["exog_cols"]
    feats = feats.dropna(subset=[*cols, TARGET_COL]).reset_index(drop=True)

    return rolling_window_backtest(
        week_ids=feats["week_id"],
        price=feats[feat_cfg["price_col"]].to_numpy(),
        features=feats[cols].to_numpy(),
        target=feats[TARGET_COL].to_numpy(),
        train_window_weeks=xgb_cfg["train_window_weeks"],
        params=xgb_cfg["params"],
        num_boost_round=xgb_cfg["num_boost_round"],
    )


def plot_weekly_forecast_vs_actual(
    results: pd.DataFrame, week_start_dates: pd.Series, path: Path
) -> None:
    """Headline visual: actual vs. one-step-ahead SARIMAX forecast, full history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week_start_dates, results["actual"], label="Actual", linewidth=1)
    ax.plot(
        week_start_dates,
        results["sarimax_pred"],
        label="SARIMAX univariate forecast",
        linewidth=1,
        alpha=0.8,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("NOK/kg")
    ax.set_title("Weekly SSB export price: actual vs. one-step-ahead forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recent_forecast_with_interval(
    results: pd.DataFrame, week_start_dates: pd.Series, n_weeks: int, path: Path
) -> None:
    """Fan chart: actual vs. forecast with its prediction interval, last
    `n_weeks` only - the full multi-decade history makes a ~3%-wide band
    visually indistinguishable from the line itself."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recent = results.tail(n_weeks)
    dates = week_start_dates.tail(n_weeks)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(
        dates, recent["lower"], recent["upper"], color="#2a78d6", alpha=0.15, label="95% interval"
    )
    ax.plot(dates, recent["actual"], label="Actual", linewidth=1.5, color="#0b0b0b")
    ax.plot(
        dates,
        recent["sarimax_pred"],
        label="SARIMAX forecast",
        linewidth=1.5,
        color="#2a78d6",
        alpha=0.9,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("NOK/kg")
    ax.set_title(f"Weekly forecast with 95% prediction interval (last {n_weeks} weeks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def nowcast_rmse_by_day(nowcast_results: pd.DataFrame) -> pd.DataFrame:
    """RMSE by days-elapsed-in-week, for the nowcast and the static
    (baseline-held-flat-all-week) comparison - the headline nowcast table."""
    rows = []
    for day, group in nowcast_results.groupby("days_elapsed_in_week"):
        rows.append(
            {
                "days_elapsed_in_week": day,
                "n_weeks": len(group),
                "nowcast_rmse": metrics.rmse(group["actual"], group["nowcast_pred"]),
                "static_baseline_rmse": metrics.rmse(
                    group["actual"], group["static_baseline_pred"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("days_elapsed_in_week").reset_index(drop=True)


def plot_nowcast_rmse_by_day(rmse_by_day: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        rmse_by_day["days_elapsed_in_week"],
        rmse_by_day["nowcast_rmse"],
        marker="o",
        label="Nowcast",
    )
    ax.plot(
        rmse_by_day["days_elapsed_in_week"],
        rmse_by_day["static_baseline_rmse"],
        marker="o",
        linestyle="--",
        label="Static baseline (no daily update)",
    )
    ax.set_xticks(rmse_by_day["days_elapsed_in_week"])
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu"][: len(rmse_by_day)])
    ax.set_xlabel("Day of week")
    ax.set_ylabel("RMSE (NOK/kg)")
    ax.set_title("Nowcast RMSE by day of week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    data_config = ssb_export_price.load_config()
    model_config = weekly_panel.load_model_config()

    ensure_processed(ssb_export_price, data_config, "ssb_export_price")
    ensure_processed(sea_surface_temperature, data_config, "sea_surface_temp")
    ensure_processed(feed_cost_fishmeal, data_config, "feed_cost_fishmeal")
    ensure_processed(daily_market_data, data_config, "daily_market_data")

    ssb = pd.read_parquet(REPO_ROOT / data_config["ssb_export_price"]["processed_path"])
    panel = weekly_panel.run(data_config, model_config)

    rw_summary = random_walk_summary(ssb["price_nok_per_kg"])
    print(rw_summary.to_string(index=False))
    rw_path = REPO_ROOT / model_config["backtest"]["random_walk_test_path"]
    rw_path.parent.mkdir(parents=True, exist_ok=True)
    rw_summary.to_csv(rw_path, index=False)
    print()

    interval_cfg = model_config["prediction_intervals"]

    uni_cfg = model_config["sarimax_univariate"]
    univariate_results = compute_or_load(
        REPO_ROOT / uni_cfg["results_path"],
        lambda: walk_forward_backtest(
            week_ids=ssb["week_id"],
            y=ssb[uni_cfg["target_col"]].to_numpy(),
            exog=None,
            order=tuple(uni_cfg["order"]),
            seasonal_order=tuple(uni_cfg["seasonal_order"]),
            min_train_weeks=uni_cfg["min_train_weeks"],
            refit_every_n_weeks=uni_cfg["refit_every_n_weeks"],
            interval_alpha=interval_cfg["alpha"],
        ),
    )

    exo_cfg = model_config["sarimax_exogenous"]
    exogenous_results = compute_or_load(
        REPO_ROOT / exo_cfg["results_path"],
        lambda: walk_forward_backtest(
            week_ids=panel["week_id"],
            y=panel[exo_cfg["target_col"]].to_numpy(),
            exog=panel[exo_cfg["exog_cols"]].to_numpy(),
            order=tuple(exo_cfg["order"]),
            seasonal_order=tuple(exo_cfg["seasonal_order"]),
            min_train_weeks=exo_cfg["min_train_weeks"],
            refit_every_n_weeks=exo_cfg["refit_every_n_weeks"],
            interval_alpha=interval_cfg["alpha"],
        ),
    )

    feat_cfg = model_config["weekly_features"]

    xgb_auto_cfg = model_config["xgboost_autoregressive"]
    xgb_auto_results = compute_or_load(
        REPO_ROOT / xgb_auto_cfg["results_path"],
        lambda: run_xgboost_variant(ssb, feat_cfg, xgb_auto_cfg),
    )

    xgb_exo_cfg = model_config["xgboost_exogenous"]
    xgb_exo_results = compute_or_load(
        REPO_ROOT / xgb_exo_cfg["results_path"],
        lambda: run_xgboost_variant(panel, feat_cfg, xgb_exo_cfg),
    )

    # --- Ensemble: simple average of sarimax_univariate + xgboost_autoregressive ---
    ensemble_results = build_ensemble_predictions(
        univariate_results, "sarimax_pred", xgb_auto_results, "xgboost_pred"
    )
    ensemble_path = REPO_ROOT / model_config["backtest"]["ensemble_results_path"]
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_results.to_parquet(ensemble_path, index=False)

    summary = pd.DataFrame(
        [
            summarize("sarimax_univariate", univariate_results, "sarimax_pred"),
            summarize("sarimax_exogenous", exogenous_results, "sarimax_pred"),
            summarize("xgboost_autoregressive", xgb_auto_results, "xgboost_pred"),
            summarize("xgboost_exogenous", xgb_exo_results, "xgboost_pred"),
            summarize("ensemble_sarimax_xgboost", ensemble_results, "ensemble_pred"),
        ]
    )
    metrics_path = REPO_ROOT / model_config["backtest"]["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_path, index=False)

    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))

    significance = pd.DataFrame(
        [
            significance_row(
                "sarimax_univariate", univariate_results, "sarimax_pred", "naive_pred"
            ),
            significance_row("sarimax_exogenous", exogenous_results, "sarimax_pred", "naive_pred"),
            significance_row(
                "xgboost_autoregressive", xgb_auto_results, "xgboost_pred", "naive_pred"
            ),
            significance_row("xgboost_exogenous", xgb_exo_results, "xgboost_pred", "naive_pred"),
            significance_row(
                "ensemble_sarimax_xgboost_vs_naive", ensemble_results, "ensemble_pred", "naive_pred"
            ),
            significance_row(
                "ensemble_vs_sarimax_alone", ensemble_results, "ensemble_pred", "sarimax_pred"
            ),
        ]
    )
    print()
    print(significance.to_string(index=False))

    # --- Economic value: does directional accuracy translate into P&L? ---
    trading_strategy = pd.DataFrame(
        [
            strategy_row("sarimax_univariate", univariate_results, "sarimax_pred"),
            strategy_row("sarimax_exogenous", exogenous_results, "sarimax_pred"),
            strategy_row("xgboost_autoregressive", xgb_auto_results, "xgboost_pred"),
            strategy_row("xgboost_exogenous", xgb_exo_results, "xgboost_pred"),
            strategy_row("ensemble_sarimax_xgboost", ensemble_results, "ensemble_pred"),
        ]
    )
    print()
    print(trading_strategy.to_string(index=False))
    trading_strategy_path = REPO_ROOT / model_config["backtest"]["trading_strategy_path"]
    trading_strategy_path.parent.mkdir(parents=True, exist_ok=True)
    trading_strategy.to_csv(trading_strategy_path, index=False)

    univariate_dates = univariate_results[["week_id"]].merge(
        ssb[["week_id", "week_start_date"]], on="week_id", how="left"
    )["week_start_date"]
    weekly_chart_path = REPO_ROOT / model_config["backtest"]["weekly_chart_path"]
    plot_weekly_forecast_vs_actual(univariate_results, univariate_dates, weekly_chart_path)

    # --- Prediction intervals: SARIMAX native (already in univariate/exogenous
    # results above), XGBoost via empirical rolling-residual quantiles ---
    xgb_auto_with_interval = empirical_interval_backtest(
        xgb_auto_results,
        "xgboost_pred",
        alpha=interval_cfg["alpha"],
        window=interval_cfg["empirical_window"],
        min_history=interval_cfg["empirical_min_history"],
    )
    xgb_exo_with_interval = empirical_interval_backtest(
        xgb_exo_results,
        "xgboost_pred",
        alpha=interval_cfg["alpha"],
        window=interval_cfg["empirical_window"],
        min_history=interval_cfg["empirical_min_history"],
    )

    coverage = pd.DataFrame(
        [
            {"variant": "sarimax_univariate", "interval_type": "native (SARIMAX conf_int)"}
            | compute_coverage(univariate_results),
            {"variant": "sarimax_exogenous", "interval_type": "native (SARIMAX conf_int)"}
            | compute_coverage(exogenous_results),
            {
                "variant": "xgboost_autoregressive",
                "interval_type": "empirical (rolling residual quantiles)",
            }
            | compute_coverage(xgb_auto_with_interval),
            {
                "variant": "xgboost_exogenous",
                "interval_type": "empirical (rolling residual quantiles)",
            }
            | compute_coverage(xgb_exo_with_interval),
        ]
    )
    coverage["nominal_coverage"] = 1 - interval_cfg["alpha"]
    print()
    print(coverage.to_string(index=False))
    coverage_path = REPO_ROOT / interval_cfg["coverage_path"]
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)

    plot_recent_forecast_with_interval(
        univariate_results,
        univariate_dates,
        n_weeks=interval_cfg["fan_chart_weeks"],
        path=REPO_ROOT / interval_cfg["fan_chart_path"],
    )

    # --- Daily nowcast layer, anchored on the univariate SARIMAX forecast ---
    nowcast_cfg = model_config["nowcast"]
    daily_market = pd.read_parquet(REPO_ROOT / data_config["daily_market_data"]["processed_path"])
    weekly_baseline = univariate_results.rename(columns={"sarimax_pred": "baseline_pred"}).merge(
        ssb[["week_id", "week_start_date"]], on="week_id", how="left"
    )

    nowcast_panel = build_nowcast_panel(
        daily_market, weekly_baseline, stock_columns=nowcast_cfg["stock_columns"]
    )
    nowcast_panel = nowcast_panel.dropna(
        subset=[*NOWCAST_FEATURE_COLUMNS, NOWCAST_TARGET_COL]
    ).reset_index(drop=True)

    nowcast_results = compute_or_load(
        REPO_ROOT / nowcast_cfg["results_path"],
        lambda: walk_forward_nowcast_backtest(
            nowcast_panel,
            train_window_weeks=nowcast_cfg["train_window_weeks"],
            params=nowcast_cfg["params"],
            num_boost_round=nowcast_cfg["num_boost_round"],
        ),
    )

    rmse_by_day = nowcast_rmse_by_day(nowcast_results)
    print()
    print(rmse_by_day.to_string(index=False))
    plot_nowcast_rmse_by_day(rmse_by_day, REPO_ROOT / nowcast_cfg["chart_path"])

    nowcast_significance = pd.DataFrame(
        [
            significance_row(
                f"nowcast_day_{day}",
                nowcast_results[nowcast_results["days_elapsed_in_week"] == day],
                "nowcast_pred",
                "static_baseline_pred",
            )
            for day in sorted(nowcast_results["days_elapsed_in_week"].unique())
        ]
    )
    print()
    print(nowcast_significance.to_string(index=False))

    significance = pd.concat([significance, nowcast_significance], ignore_index=True)
    significance_path = REPO_ROOT / model_config["backtest"]["significance_path"]
    significance_path.parent.mkdir(parents=True, exist_ok=True)
    significance.to_csv(significance_path, index=False)


if __name__ == "__main__":
    main()
