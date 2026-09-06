"""Economic value, not just statistical accuracy: does a model's
directional edge translate into any actual P&L?

A simple notional long/short strategy - go long for a week if the model
predicted a rise, short if it predicted a fall - directly tests whether
directional skill (already reported as "directional accuracy" elsewhere
in this project) is worth anything economically. This is a paper/notional
exercise: the SSB export price is a statistical index, not a directly
tradable instrument, and this ignores transaction costs entirely - it
answers "is there monetizable skill here at all," not "here is an
implementable trading strategy."

Reuses the already-computed `naive_pred` column (= previous week's actual)
as "last known price," so the position and realized-return logic needs no
new backtest machinery - it's a pure post-hoc analysis on the existing
walk-forward results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEEKS_PER_YEAR = 52


def simple_directional_strategy(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Add `position` (+1/-1, sign of the forecast's implied direction),
    `realized_return` (log return actually realized that week - the
    buy-and-hold return), and `strategy_return` (`position *
    realized_return`) columns."""
    out = df.copy()
    out["position"] = np.sign(out[pred_col] - out["naive_pred"])
    out["realized_return"] = np.log(out["actual"] / out["naive_pred"])
    out["strategy_return"] = out["position"] * out["realized_return"]
    return out


def newey_west_mean_test(x: np.ndarray, lags: int = 8) -> tuple[float, float]:
    """One-sample t-test for whether the mean of `x` differs significantly
    from zero, using a Newey-West HAC standard error (Bartlett kernel) to
    account for autocorrelation in `x` - e.g. weekly strategy returns,
    which this project's own Ljung-Box test already showed are
    autocorrelated. Returns `(t_statistic, p_value)`."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    demeaned = x - mean

    long_run_variance = np.mean(demeaned**2)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett kernel
        gamma_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        long_run_variance += 2 * weight * gamma_lag

    with np.errstate(invalid="ignore"):
        t_stat = mean / np.sqrt(long_run_variance / n)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 1)
    return float(t_stat), float(p_value)


def compute_strategy_metrics(df: pd.DataFrame) -> dict:
    """Summarize a `simple_directional_strategy`-augmented backtest:
    hit rate (should match this variant's already-reported directional
    accuracy - a useful cross-check), annualized return/Sharpe for both
    the strategy and a buy-and-hold baseline, and a Newey-West test of
    whether the strategy's mean weekly return is significantly different
    from zero.

    Deliberately does NOT report a compounded multi-decade "cumulative
    return" for the strategy: exponentiating the sum of ~20+ years of
    independently-signed weekly log-returns produces an astronomical,
    non-credible number (it implies fully reinvesting the entire notional
    every week with no capital constraints, transaction costs, or risk
    limits for two decades) - a well-known artifact of naive backtest P&L
    math, not a real result. Buy-and-hold's cumulative return has no such
    problem (the position never flips, so the telescoping sum of weekly
    log-returns is just the real total price return over the period) and
    is reported as useful context."""
    hits = np.sign(df["realized_return"]) == df["position"]
    strategy = df["strategy_return"].to_numpy()
    buy_hold = df["realized_return"].to_numpy()
    t_stat, p_value = newey_west_mean_test(strategy)

    return {
        "n_weeks": len(df),
        "hit_rate": float(hits.mean()),
        "strategy_annualized_return": float(strategy.mean() * WEEKS_PER_YEAR),
        "strategy_annualized_sharpe": float(
            strategy.mean() / strategy.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        ),
        "strategy_mean_return_t_stat": t_stat,
        "strategy_mean_return_p_value": p_value,
        "buy_hold_annualized_return": float(buy_hold.mean() * WEEKS_PER_YEAR),
        "buy_hold_annualized_sharpe": float(
            buy_hold.mean() / buy_hold.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        ),
        "buy_hold_cumulative_return": float(np.exp(buy_hold.sum()) - 1),
    }
