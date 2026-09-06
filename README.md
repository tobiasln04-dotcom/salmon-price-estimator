# Salmon Price Estimator

Week-ahead forecast and daily nowcast of the SSB weekly export price of
fresh Norwegian farmed salmon (NOK/kg) — published Sunday night, ahead of
the following Thursday's official print.

**Status: weekly baselines (SARIMAX, XGBoost) and the daily nowcast layer
are all built and backtested.** See "Roadmap" below and `CLAUDE.md` for
the full build log.

## Data sources

| Source | Frequency | Coverage | Used for |
|---|---|---|---|
| [SSB StatBank table 03024](https://data.ssb.no/api/v0/en/table/03024) | Weekly | 2003-present | Target variable: export price, NOK/kg |
| [NOAA OISST v2.1](https://www.ncei.noaa.gov/erddap) sea surface temp anomaly (5-point average across the farming belt) | Weekly (resampled from daily) | 2020-02-present | Exogenous feature |
| [IndexMundi](https://www.indexmundi.com/commodities/) fishmeal price | Monthly | 1996-present | Exogenous feature (feed-cost proxy) |
| NOK/USD, NOK/EUR and 5 Oslo Børs salmon stocks (MOWI, SALM, LSG, GSF, BAKKA) via [yfinance](https://github.com/ranaroussi/yfinance) | Daily | 2000/2010-present (BAKKA is the shortest) | Daily nowcast features |

All sources are fetched fresh by their own pipeline in
`src/salmon_price_estimator/data/` and landed as parquet — nothing is
committed to the repo, so the data is always reproduced from source, not
checked in. Fish Pool/Euronext salmon futures data (the fourth planned
daily input) was investigated and deferred — no free historical source
exists, see `CLAUDE.md` "Deferred items".

## Methodology

**Baseline model: SARIMAX(1,1,1)×(0,1,1,52)**, backtested with an
expanding-window walk-forward loop (`src/salmon_price_estimator/eval/backtest.py`):
fit on an initial window, forecast one week ahead, then extend the fitted
model with the newly-observed actual before forecasting the next week.
Parameters are re-estimated periodically (not every week — see
`config/model.yaml` for the runtime rationale) rather than only once,
so the model keeps adapting as more history accumulates.

Two variants are backtested side by side, not just one:

- **Univariate** — the SSB price series alone, using its full available
  history (2003-present).
- **Exogenous** — adds fishmeal price and sea-surface-temperature
  anomaly as regressors, both lagged one week before use so the model
  never sees information it wouldn't actually have at publication time
  (the forecast is made before that week's sea-temp or fishmeal data
  exists). This variant's window is shorter (2022-present) because sea
  temperature data only goes back to 2020, and the model needs a few
  years of training data before its first forecast.

A second model family, **XGBoost**, is backtested the same way but with a
rolling (not expanding) training window, retrained from scratch every
week (`eval/backtest_xgboost.py`) — cheap enough with gradient-boosted
trees on a few hundred rows that no periodic-refit shortcut is needed,
unlike SARIMAX. Rather than the raw price level, it predicts the
1-week-ahead **log-return** and reconstructs the price level from that
(`price_{t-1} * exp(predicted_return)`) — a raw-level tree model can't
extrapolate beyond the price range it was trained on, which matters on a
series that's roughly tripled since 2003. Its features
(`features/weekly_features.py`) are lagged price levels (including a
year-ago lag for seasonality), lagged log-returns, rolling mean/std, and
week-of-year sin/cos — all built from information already known at
forecast time, same leakage discipline as the SARIMAX exogenous variant.

Both model families are backtested as **autoregressive/univariate** (full
available history) and **exogenous** (+ fishmeal and sea-temp, truncated
to ~2020-present) variants — four backtests in total, all compared
against the same **naive benchmark**: predict next week's price as this
week's price. Metrics: MAPE, RMSE, and directional accuracy (did the
forecast get the sign of the week-over-week move right?).

**Daily nowcast** (`eval/backtest_nowcast.py`): anchored on the
univariate SARIMAX forecast (the model that actually beats naive), and
corrected daily Monday-Thursday using FX and Oslo Børs salmon-stock
5-day returns as the week progresses toward Thursday's official release.
The model predicts a **log-ratio correction** on top of the baseline
(`log(actual / baseline_forecast)`, reconstructed as
`baseline_forecast * exp(predicted_correction)`) rather than the raw
price level — the same "predict a transform, reconstruct the level"
pattern as the weekly XGBoost model, just applied to a correction instead
of a return. It retrains once per completed week on a 104-week rolling
window (not 4x per week — see the module docstring for why that's
provably equivalent, not a shortcut) and is compared against **holding
the baseline flat all week** (no daily updating at all), to check whether
updating on daily data helps at all.

## Headline result

![Weekly SSB export price: actual vs. one-step-ahead forecast](assets/weekly_forecast_vs_actual.png)

| Variant | Weeks backtested | Date range | MAPE | RMSE | Directional accuracy | vs. naive (Diebold-Mariano) |
|---|---|---|---|---|---|---|
| **SARIMAX univariate** | 1,230 | 2003 – 2026 | **3.54%** (naive: 3.59%) | **2.79** (naive: 2.91) | **58.9%** (naive: 0.5%) | **significantly better** (p = 0.019) |
| SARIMAX exogenous | 230 | 2022 – 2026 | 4.96% (naive: 4.29%) | 5.84 (naive: 5.14) | 54.3% (naive: 0.0%) | significantly **worse** (p = 0.032) |
| XGBoost autoregressive | 1,126 | 2004 – 2026 | 3.92% (naive: 3.75%) | 3.11 (naive: 3.04) | 54.8% (naive: 0.4%) | no significant difference (p = 0.368) |
| XGBoost exogenous | 178 | 2023 – 2026 | 4.50% (naive: 4.31%) | 5.32 (naive: 5.26) | 55.1% (naive: 0.0%) | no significant difference (p = 0.840) |

**Only the univariate SARIMAX beats naive on all three criteria — and
that edge is statistically real, not a lucky point estimate.** A
[Diebold-Mariano test](https://en.wikipedia.org/wiki/Diebold%E2%80%93Mariano_test)
(`eval/significance.py`, squared-error loss, Harvey-Leybourne-Newbold
small-sample correction) on the two forecast-error series confirms the
3.54%-vs-3.59% MAPE gap is significant at the 5% level (p = 0.019), not
noise. The other three results get more nuanced with a significance test
applied, rather than less: the **SARIMAX exogenous variant is
significantly *worse* than naive** (p = 0.032 — this isn't just a worse
point estimate, it's a real effect), while **both XGBoost variants are
statistically indistinguishable from naive** (p = 0.37 and 0.84) — not
proven worse, just not proven better either. Every variant comfortably
beats naive on directional accuracy, though, since the naive forecast is
structurally incapable of ever predicting a price move (it just repeats
the last value).

This is a genuinely useful negative result, not a disappointing one: it
shows the naive "last observed value" benchmark is a legitimately hard
target on this series (see "Is this actually a random walk?" below for
what that claim precisely means), and that model sophistication doesn't
automatically buy accuracy here.

**One more check, since it's cheap now that both backtests exist: does
averaging SARIMAX's and XGBoost's predictions help?** Not clearly.

| Variant | Weeks | MAPE | RMSE | Directional accuracy | vs. naive (DM) | vs. SARIMAX alone (DM) |
|---|---|---|---|---|---|---|
| Ensemble (SARIMAX + XGBoost avg) | 1,126 | 3.66% (naive: 3.75%) | 2.92 (naive: 3.04) | 57.5% (naive: 0.4%) | not significant (p = 0.060) | not significant (p = 0.733) |

The ensemble's p-value against naive (0.060) is the closest of any
non-SARIMAX-univariate variant to conventional significance — closer
than either XGBoost variant alone — but it doesn't clear the 5%
threshold, and more importantly it's statistically indistinguishable
from just using SARIMAX by itself. No evidence here that blending in
XGBoost buys anything over the simpler choice; the classic "forecast
combination" bet (a weak model can still reduce a strong one's variance
when averaged in) doesn't pay off on this series.

## Why the exogenous features and XGBoost didn't help — and what that does and doesn't mean

The SARIMAX exogenous variant's shortfall was investigated in detail
(full write-up in `CLAUDE.md`'s session log, 2026-09-03):

- Restricting the **univariate** SARIMAX to the exact same 230 weeks the
  exogenous variant was evaluated on, it *still* beats naive (MAPE 4.17%
  vs. 4.29%) — so it isn't just "recent years are harder to forecast."
- A control backtest — same short 2022+ window, same SARIMAX order, but
  with the exogenous regressors removed — *also* underperforms naive
  (MAPE 4.54%). The short window, forced by sea-temperature data only
  existing from 2020 onward, isn't long enough to reliably fit this
  model's annual seasonal component (the seasonal MA coefficient sits at
  its numerical boundary with an enormous standard error).
- The two exogenous coefficients themselves are statistically
  indistinguishable from zero in a full-sample fit (p = 0.95 and p =
  0.53) — consistent with there being essentially no linear relationship
  between *changes* in price and the *level* of either regressor, even
  though the raw level-to-level correlations look deceptively strong (a
  classic artifact of comparing two trending series).

XGBoost was built specifically to give these features a fairer test,
since it doesn't need an explicit parametric seasonal term the way
SARIMAX does. The result: XGBoost's exogenous variant (4.50% MAPE) comes
much closer to naive than SARIMAX's did (4.96%) — the gap shrank by
roughly two-thirds — and a feature-importance check on the fitted model
confirms `fishmeal_price_usd_per_tonne` and `sst_anomaly_c` aren't dead
weight (they rank mid-pack among all 14 features, not last). So XGBoost
*does* extract some real signal from them. And per the Diebold-Mariano
test above, XGBoost's shortfall (both variants) **isn't even
statistically distinguishable from naive** — a materially different, and
more defensible, position than SARIMAX-X's, which is significantly worse.

**Conclusion**: this isn't "fishmeal and sea temperature are useless" or
"XGBoost failed" — it's "the exogenous features carry weak signal,
XGBoost extracts a bit more of it than a linear model can (enough to turn
a significant loss into statistical noise, even if not into a win), and
neither approach clears naive on this particular series." The honest
headline model remains the univariate SARIMAX.

## Is this actually a random walk?

"Behaves close to a random walk" has been stated qualitatively so far —
worth testing formally rather than leaving it as a hand-wave
(`eval/random_walk_test.py`), and the answer turned out more precise
than the informal version, not just confirmed:

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| ADF (log price level) | -1.58 | 0.49 | Unit root **not rejected** — consistent with a non-stationary, trending price level |
| Ljung-Box, weekly returns (lags 1/4/12/52) | 40–250 | all < 10⁻⁹ | **Rejects** "no autocorrelation" at every lag |

Put together, this is **not a pure random walk** — a true random walk's
week-to-week changes would be unpredictable, and the Ljung-Box test would
fail to reject at every lag. It doesn't. Weekly returns have real,
statistically detectable autocorrelation. The price level *is*
non-stationary (the ADF result, the necessary-but-not-sufficient half of
"random walk"), but its changes aren't pure noise.

That combination — non-stationary *and* weakly autocorrelated — is, not
coincidentally, exactly what SARIMAX(1,1,1)×(0,1,1,52) is built to
capture: the AR/MA terms exist specifically to model autocorrelation in
the differenced series. **That's why SARIMAX can extract a statistically
significant edge over naive at all** (the Diebold-Mariano result above).
But the edge is still small (3.54% vs. 3.59% MAPE) — telling you the
autocorrelation, while real, is thin. "Close to a random walk" was the
right intuition; "non-stationary with weak but real autocorrelation" is
the more precise description these two tests actually support, and it's
a better explanation for the whole project's pattern than "random walk"
alone would be: there's a real, exploitable signal (hence SARIMAX's
significant win), it's just too thin for more data or more model
flexibility to meaningfully improve on.

## Daily nowcast result

![Nowcast RMSE by day of week](assets/nowcast_rmse_by_day.png)

| Day | Weeks | Nowcast RMSE | Static baseline RMSE (no daily update) | vs. static (Diebold-Mariano) |
|---|---|---|---|---|
| Mon | 747 | 3.71 | 3.47 | significantly worse (p = 0.030) |
| Tue | 747 | 3.63 | 3.47 | significantly worse (p = 0.050) |
| Wed | 747 | 3.63 | 3.47 | significantly worse (p = 0.029) |
| Thu | 747 | 3.73 | 3.47 | significantly worse (p < 0.001) |

This is a third instance of the same pattern as above, and it's reported
just as plainly: **the nowcast doesn't beat the static baseline on any
day**, and RMSE doesn't decline monotonically Mon → Thu either (it dips
Tue/Wed, then rises again Thursday). The Diebold-Mariano test confirms
this isn't noise — the underperformance is statistically significant
every single day, and, notably, **gets more significant as the week
progresses** (Thursday's p-value is two orders of magnitude smaller than
Monday's) even though the RMSE gap itself doesn't move monotonically.
Daily FX moves and Oslo Børs salmon-stock returns turn out to be a fairly
indirect, noisy proxy for the actual weekly export-price surprise —
equity prices for these companies reflect a lot more than just the spot
salmon price (forward earnings expectations, general market moves,
company-specific news), and a feature-importance check on the fitted
model shows no single daily feature dominates (importances span a narrow
0.008–0.018 range), the same diffuse-signal pattern seen in the weekly
exogenous features.

**Taken together with the weekly results above, this project's honest
finding is a coherent one, not three unrelated disappointments**: the
naive last-observed-value benchmark is a genuinely tough target on this
series (non-stationary with only weak, thin autocorrelation — see "Is
this actually a random walk?" above), and every attempt to beat it with
more data or more model flexibility (exogenous features, XGBoost, daily
nowcasting, even a simple SARIMAX+XGBoost ensemble) came up short against
it. The univariate SARIMAX remains the one model in this project that
clears the bar.

## Side analysis: which salmon stock tracks the salmon price?

![Which salmon stock tracks the salmon price?](assets/stock_salmon_price_correlation.png)

A natural follow-up question, and one the daily-market data already
answers (`scripts/analyze_stock_correlations.py`, not part of the core
backtest pipeline): of the 5 Oslo Børs salmon-farming stocks, which one's
weekly return co-moves most with the salmon export price's own weekly
move? Same-week correlation, all 5 compared over an identical window
(2010–present, since BAKKA's 2010 listing bounds it):

| Stock | Correlation | R² |
|---|---|---|
| **BAKKA** (Bakkafrost) | 0.046 | 0.2% |
| **GSF** (Grieg Seafood) | 0.036 | 0.1% |
| SALM (SalMar) | 0.018 | 0.03% |
| LSG (Lerøy Seafood) | 0.015 | 0.02% |
| MOWI (Mowi) | 0.013 | 0.02% |

The ranking itself is a minor, plausible detail — Bakkafrost and Grieg
Seafood are more concentrated, vertically-integrated pure-plays; Mowi,
the largest and most diversified, dilutes pure spot-price sensitivity
across feed, VAP, and international operations. **The real finding is
that every correlation is close to zero** — even the top stock's weekly
return has salmon price explaining only 0.2% of its variance. Stock
prices are forward-looking (pricing in *expected future* margins, not
this week's realized export price) and these companies hedge via forward
contracts, so this isn't surprising in hindsight — but it does explain,
from the other direction, why the daily nowcast above (which leans on
these same 5 stocks' returns) struggled: if salmon-company equities
barely track the salmon spot price even in the *same* week, they were
never going to be a strong predictor of it.

## Reproducing these results

```bash
uv sync
uv run python scripts/run_backtest.py
```

This fetches all data sources if they're not already cached locally,
builds the joined weekly panel/features, runs all four weekly backtest
variants plus the ensemble and the daily nowcast backtest, writes
`data/processed/backtest_*.parquet`, `data/processed/backtest_metrics.csv`,
`data/processed/significance_tests.csv` (the Diebold-Mariano results
above), and `data/processed/random_walk_tests.csv` (the ADF/Ljung-Box
results above), and regenerates both chart images under `assets/`.
**A fresh run takes roughly 40 minutes** — SARIMAX refit cost scales superlinearly with
training window size on this ~1,300-week series (see `config/model.yaml`
and `eval/backtest.py` for the measured numbers and the runtime
tradeoffs that shaped the default config); XGBoost and the nowcast layer
add only a few more minutes on top since retraining them is cheap. Each
variant's results are cached under `data/processed/` and skipped on
subsequent runs unless deleted, so an interrupted run resumes rather than
starting over.

Unlike everything else in `data/processed/` (gitignored, regenerated on
demand), the three PNGs under `assets/` **are committed** — they're the
artifacts this README embeds directly, so they need to actually be in
the repo rather than regenerated-and-ignored. Re-run the relevant script
and commit the updated PNGs if the underlying results change.

The stock-correlation side analysis is a separate, optional script (it
isn't part of the core forecasting pipeline, so it isn't in
`run_backtest.py`):

```bash
uv run python scripts/analyze_stock_correlations.py
```

## Roadmap

- [x] SSB export price target pipeline
- [x] Sea surface temperature and fishmeal exogenous feature pipelines
- [x] SARIMAX weekly baseline (univariate + exogenous), walk-forward backtest
- [x] XGBoost weekly baseline (autoregressive + exogenous), rolling-window backtest
- [x] Daily nowcast layer (FX + Oslo Børs salmon stocks; futures deferred, see below)
- [ ] Fish Pool/Euronext salmon futures (deferred — no free historical
      source, see `CLAUDE.md` "Deferred items")
- [ ] Harvest volume / standing biomass features (deferred — needs
      BarentsWatch API registration, see `CLAUDE.md`)

## Development

```bash
uv sync
uv run python -m pytest
uv run ruff check .
```

A `Dockerfile` is also provided (`docker build -t salmon-price-estimator .`
then `docker run salmon-price-estimator`, which runs
`scripts/run_backtest.py`) — **not build-tested**, since Docker isn't
available in this project's development environment. It's driven by
`uv.lock` so it should pick up all dependencies automatically, and
includes `libgomp1` for xgboost's OpenMP dependency, but treat it as
best-effort until someone actually runs it.
