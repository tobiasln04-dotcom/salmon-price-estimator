# CLAUDE.md

Context for any Claude Code session resuming work on this repo.

## Goal

Portfolio project for Nordic investment banking internship applications.
Code quality, README clarity, and reproducibility matter as much as
accuracy — this is being evaluated as a work sample, not just a model.

Build an AI price estimator for the Norwegian salmon farming market:

- **Week-ahead forecast**: published Sunday night, forecasting the SSB
  weekly export price of fresh Norwegian farmed salmon (NOK/kg) that will
  be released the following Thursday.
- **Daily nowcast**: updated Monday-Thursday as higher-frequency inputs
  arrive (futures, FX, equities), converging toward the actual SSB print.
- **Success criteria**: beat a naive "last observed value" benchmark on
  MAPE and directional accuracy. Nowcast RMSE should decline monotonically
  as the week progresses (Mon -> Thu).

## Architecture

1. **`src/salmon_price_estimator/data/`** — fetch and land raw data as
   tidy parquet in `data/processed/`, one clear schema per source.
   - Target + weekly features: SSB export price (StatBank) — done. Sea
     surface temp anomaly (NOAA OISST via ERDDAP) — done, see session log
     for the ~6.5-year data window caveat. Feed cost proxy: fishmeal price
     (IndexMundi) — done; fish oil deferred, see "Deferred items" below.
     Harvest volumes and biomass (Fiskeridirektoratet) also deferred.
   - Daily nowcast inputs: NOK/USD & NOK/EUR and Oslo Bors daily returns
     for MOWI/SALM/LSG/GSF/BAKKA (all via yfinance) — done, see session
     log 2026-09-04 (`data/daily_market_data.py`). Fish Pool/Euronext
     salmon futures deferred, see "Deferred items" - no free historical
     source exists.
   - Secondary: NQSALMON weekly history (scraped, Jan 2013-Aug 2024) for
     a longer backtest window and cross-check against SSB.
2. **`src/salmon_price_estimator/features/`** — weekly features (lagged
   target, log-returns, rolling stats, seasonality, harvest deltas,
   biomass ratios) and daily nowcast features (FX 5-day return, stock
   5-day return, cross-sectional dispersion - futures curve slope dropped
   per the Fish Pool deferral). Both done - see session logs 2026-09-03
   (`features/weekly_features.py`) and 2026-09-04
   (`features/daily_nowcast_features.py`). Harvest deltas/biomass ratios
   still deferred (no source, see "Deferred items").
3. **`src/salmon_price_estimator/models/`** — weekly baseline (SARIMAX,
   then XGBoost with lags, rolling-window train, predict t+1) and a daily
   nowcast layer: a second XGBoost predicting the same weekly SSB target
   from (weekly baseline output + latest daily observations), retrained
   daily on a rolling window. MIDAS regression is a v2 comparison, not
   MVP. All three now done - see session logs 2026-08-31 (SARIMAX),
   2026-09-03 (XGBoost weekly), 2026-09-04 (daily nowcast). The two weekly
   baselines are each built as **two variants** (pure
   autoregressive/univariate on full history, plus an exogenous variant
   with fishmeal + sea-temp on 2020+). Headline finding, confirmed a third
   time by the nowcast: only the univariate SARIMAX beats naive on all
   three success criteria - neither the exogenous features, XGBoost's
   extra flexibility, nor the daily nowcast's daily updating close the
   gap. See 2026-09-04 log for the full comparison and why that's a real,
   coherent finding, not three separate failures.
4. **`src/salmon_price_estimator/eval/`** — walk-forward backtest with
   expanding window. Metrics: MAPE, RMSE, directional accuracy, vs. naive
   last-week benchmark. Nowcast eval: RMSE by days-elapsed-in-week
   (Mon-Thu) — this is the headline README chart. All backtests done:
   `eval/backtest.py` (SARIMAX, expanding window), `eval/backtest_xgboost.py`
   (weekly XGBoost, rolling window), `eval/backtest_nowcast.py` (daily
   nowcast, rolling window, retrained per-week not per-day - see its
   docstring). All three share `eval/metrics.py`. **The nowcast RMSE does
   NOT decline monotonically Mon->Thu** (see 2026-09-04 log) - that
   success criterion from the Goal section was not met; reported honestly
   in the README rather than adjusted until it looked right.
   `eval/significance.py` (added 2026-09-06) tests whether each variant's
   point-estimate edge over its benchmark is statistically real (Diebold-
   Mariano test) - see that session log entry, the answer turned out more
   nuanced than the point estimates alone suggested.
5. **`scripts/run_backtest.py`** — single entry point that reproduces the
   headline result end-to-end. Config lives in `config/*.yaml` so
   hyperparameters and date ranges are auditable, not hardcoded. Done for
   all four weekly-baseline variants (2 SARIMAX + 2 XGBoost) plus the
   daily nowcast; each variant's results are cached under
   `data/processed/` and skipped on rerun if already present (see
   `compute_or_load()`, added 2026-09-04) so an interrupted run resumes
   instead of redoing finished work. Also regenerates
   `assets/nowcast_rmse_by_day.png` - the one output in this project that
   gets committed rather than gitignored, since the README embeds it.
6. **`notebooks/`** — exploration only, never the source of truth.

## Build order (in progress)

Building bottom-up: scaffolding -> target variable alone -> features ->
models -> backtest -> README. Do not jump ahead to modeling before the
target data pipeline is solid, and do not add data sources beyond what
the current step calls for.

## Conventions

- Python 3.11, dependency management via `uv` (not pip/poetry/conda).
- `src/` layout, package name `salmon_price_estimator`.
- Data lands as parquet in `data/processed/`; `data/raw/` holds untouched
  source downloads. Neither is committed to git (see `.gitignore`) —
  pipelines must be able to regenerate both from scratch.
- Config (date ranges, hyperparameters, source URLs) belongs in
  `config/*.yaml`, not hardcoded in scripts — this is an auditability
  requirement, not a style preference.
- Lint/format: `ruff` (config in `pyproject.toml`). Tests: `pytest`,
  under `tests/`, mirroring the `src/` package structure.
- Commit `uv.lock` for reproducibility.

## Session log

- **2026-08-02**: Scaffolding + SSB weekly export price pipeline done and
  pushed (see "Reproducing the headline result" section for how to run
  it). Investigated Fiskeridirektoratet harvest/biomass data — deferred,
  see "Deferred items" below. Investigated sea surface temperature
  sources: decided on **NOAA OISST v2.1 via ERDDAP**, no auth required,
  over MET Norway's Frost/Havvarsel-Frost APIs (which need self-serve but
  still separate registration) — simpler and one less credential to
  manage. Verified a live query works:
  `https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.csv?anom[(2026-07-01):1:(2026-07-10)][(0.0)][(62.625):1:(62.625)][(6.125):1:(6.125)]`
  — dataset has an `anom` variable that is literally "Daily sea surface
  temperature anomalies" (Celsius), so no manual anomaly computation
  needed. Longitude is in 0-360 convention, but Norway's ~5-7°E maps
  directly (no conversion needed since it's already the eastern
  hemisphere). Tested point (62.625N, 6.125E, mid-Norway coast) returns
  real (non-NaN) values, so it's clear of the land mask. **Not yet
  decided**: single representative coastal point vs. a few points spanning
  the salmon-farming belt (Rogaland up to Nordland/Troms) averaged
  together — pick this up next session before writing the fetcher.
- **2026-08-03**: Sea surface temperature fetcher done and pushed
  (`src/salmon_price_estimator/data/sea_surface_temperature.py`). Decided
  to average 5 points spanning the farming belt (Rogaland, Vestland, Møre
  og Romsdal, Trøndelag, Nordland — all verified clear of the land mask).
  Two things discovered while building it, both documented inline in
  config/code comments:
  1. **NCEI's ERDDAP mirror only covers ~2020-02-28 to present** (~6.5
     years), not the full 1981-present OISST archive — that's a rolling
     window ERDDAP happens to keep, not the full record. The full record
     lives on NOAA PSL's THREDDS server as yearly netCDF files with no
     ready-made anomaly variable (would need xarray/netCDF4 + our own
     climatology baseline) — decided against that added complexity for one
     feature; SST anomaly is simply unavailable before 2020 in this
     pipeline. The backtest/model need to tolerate a feature with partial
     history.
  2. **A single request spanning the full ~5.5-year range gets rejected by
     the ERDDAP server itself with an HTTP 408 after ~2 minutes** (verified
     directly with curl) — this is a server-side processing cap, not a
     network/timeout-config issue. Fixed by chunking each point's query
     into one request per calendar year (~15s each) and concatenating.
  Verified end-to-end: 334 weekly rows (2020-02-24 to 2026-07-13), no
  nulls, no duplicate weeks.
  Also built the fishmeal feed-cost proxy
  (`src/salmon_price_estimator/data/feed_cost_fishmeal.py`). Discovered
  IndexMundi has no CSV/API export at all (every `type`/`format`/`export`
  query param just returns the same HTML page) — parsed the embedded
  `gvPrices` HTML table directly instead. Also discovered IndexMundi
  doesn't track "fish oil" as a commodity at all (checked the full ~75
  commodity list; only `fish-meal` and `fish` [salmon] exist) — deferred,
  see "Deferred items" below. Found the `months` query param has an
  undocumented server-side cap of 360 (~30yr): anything higher silently
  returns a broken 3-row response instead of an error. Verified end-to-end:
  357 monthly rows (1996-07 to 2026-03), no nulls, no duplicates.
  All three weekly-feature data sources are now done: SSB target, sea
  surface temp anomaly, fishmeal. Next up is the SARIMAX weekly baseline
  model (`src/salmon_price_estimator/models/`) plus the walk-forward
  backtest harness (`src/salmon_price_estimator/eval/`). **Decision
  pending, ask the user before starting**: should the SARIMAX baseline be
  pure univariate (just the SSB price series, full 2000-present backtest
  range) or SARIMAX-X with fishmeal/sea-temp as exogenous regressors (which
  would truncate the backtest to 2020+ since sea temp only goes back that
  far, or require a two-variant approach)? Leaning toward pure univariate
  per the "baseline" framing in CLAUDE.md's architecture section, with
  exogenous features saved for the XGBoost stage — but this wasn't decided
  yet when the session ended, so don't assume it.
- **2026-08-31**: Resolved the SARIMAX-scope question above — asked the
  user directly; decided to build **both** variants rather than pick one,
  so there's a direct comparison of whether the exogenous features help
  instead of only finding out at the XGBoost stage (see Architecture item
  3). Built:
  - `features/weekly_panel.py` — joins SSB target + sea-temp + fishmeal
    onto one weekly grid. The two exogenous columns are lagged 1 week
    before joining (config: `weekly_panel.exog_lag_weeks`), so the panel
    never carries information the week-ahead forecast wouldn't actually
    have yet at publication time — verified with a leakage spot-check
    (a given week's panel row carries the *previous* week's raw sea-temp
    value, not its own).
  - `models/sarimax_baseline.py`, `eval/metrics.py`, `eval/backtest.py`,
    `scripts/run_backtest.py` — fit/forecast helpers, MAPE/RMSE/
    directional-accuracy/naive-benchmark metrics, the walk-forward
    expanding-window loop, and the single entry point running both
    variants. Config in `config/model.yaml`: order/seasonal_order fixed
    at `(1,1,1)`/`(0,1,1,52)` as a documented baseline default (not
    auto-tuned — real tuning is the XGBoost stage's job).
  Two implementation findings worth knowing before touching
  `eval/backtest.py` again:
  1. **`SARIMAXResultsWrapper.append()` is O(n) per call, not O(1)** — it
     re-filters the entire expanding dataset every time rather than
     incrementally updating from the existing filtered state. Measured
     ~0.6s/call at 160 obs growing to ~1.6s/call at 800 obs, which would
     have made the full walk-forward loop O(n²) and pushed the
     univariate backtest past 2 hours. `.extend()` does what `.append()`
     looks like it should: it reuses the filtered state and only
     processes the new observation, measured flat at ~5-10ms/call
     regardless of accumulated history. `backtest.py` uses `.extend()`
     for non-refit steps.
  2. **A full refit's cost itself grows superlinearly with training
     window size** (measured on the real SSB series: ~12s at 156 obs,
     ~20s at 500, ~82s at 1000, ~144s at 1385 — the full univariate
     series is ~1300 weeks long). That's why `refit_every_n_weeks`
     differs between the two variants in `config/model.yaml`: 52
     (annual) for univariate's long window vs. 8 for the much shorter
     (~330-week) exogenous window — both choices are runtime-driven and
     documented inline in the config comments with the measured numbers.
  Also hit and resolved a new local-machine issue this session — see
  "Known environment quirks": Smart App Control started blocking pandas
  itself (not just the previously-documented `pytest.exe` case) after a
  venv rebuild triggered by adding the `statsmodels` dependency; the user
  disabled Smart App Control in Windows Security settings to fix it.
  Verification: full pytest suite (31 tests) passes, `ruff check .` clean.
  `uv run python scripts/run_backtest.py` ran end-to-end in ~32 min
  (univariate ~15 min, exogenous ~17 min — matches the runtime estimate
  from the `.append()`/`.extend()` and refit-cost findings above).
  **Headline result** (`data/processed/backtest_metrics.csv`):

  | variant | weeks | range | SARIMAX MAPE | naive MAPE | SARIMAX RMSE | naive RMSE | SARIMAX dir. acc. | naive dir. acc. |
  |---|---|---|---|---|---|---|---|---|
  | univariate | 1230 | 2003U01-2026U30 | 3.54% | 3.59% | 2.79 | 2.91 | 58.9% | 0.5% |
  | exogenous | 230 | 2022U09-2026U30 | 4.96% | 4.29% | 5.84 | 5.14 | 54.3% | 0.0% |

  **The univariate SARIMAX beats naive on all three success criteria**
  (MAPE, RMSE, directional accuracy — the last one dramatically, since the
  naive forecast structurally can't ever get direction right except on the
  rare week where price doesn't move at all). **The exogenous variant does
  NOT beat naive on MAPE/RMSE** (though it still clears naive easily on
  directional accuracy) - adding lagged fishmeal + sea-temp as simple
  linear exogenous regressors hurt point-forecast accuracy relative to the
  univariate baseline over its shorter 2022+ eval window. This is a real
  finding, not a bug (both variants pass their unit tests and the leakage
  spot-check) - worth stating honestly in the README rather than only
  reporting the flattering number. Possible reasons worth exploring before
  the XGBoost stage: the exogenous variant's window is much shorter (230
  vs. 1230 weeks) and starts later, so it's not a fully like-for-like
  comparison; a linear exogenous term in SARIMAX may just be the wrong
  functional form for these features, which is exactly the kind of
  nonlinear relationship XGBoost is suited to instead.
- **2026-09-03**: Diagnosed the exogenous underperformance above with three
  follow-up checks (all on `data/processed/weekly_panel.parquet` /
  `backtest_univariate.parquet`, no config changes):
  1. Restricting the **full-history univariate** backtest to the exact same
     230 weeks the exogenous variant was evaluated on: MAPE 4.17%, RMSE
     4.90 — still comfortably beats naive (4.29%/5.14) on this window.
     So the exogenous variant's underperformance is *not* just "2022+ is a
     harder period" - a model with long history still wins on it.
  2. A **full-sample fit's coefficients** on the two exogenous regressors
     are statistically indistinguishable from zero: `sst_anomaly_c`
     coef=0.062, std err=0.905, p=0.946; `fishmeal_price_usd_per_tonne`
     coef=0.011, std err=0.017, p=0.527. Consistent with a near-zero
     correlation between price *changes* (what the differenced model
     actually predicts) and the exogenous *levels* (-0.077 and -0.014
     respectively) - the strong-looking level correlations (0.58, -0.35)
     are the classic spurious-regression artifact of two trending series.
     The same fit also showed `ma.S.L52` pinned at the -1.0 invertibility
     boundary with std err=854 - a red flag that the annual seasonal
     component is poorly identified from only ~334 weekly observations
     (~6.4 annual cycles).
  3. A **control backtest**: same truncated 2020+ window, same order/
     seasonal_order/cadence as the exogenous config, but `exog=None`.
     Result: MAPE 4.54%, RMSE 5.30 - *still worse than naive*, though
     better than the exogenous variant (4.96%/5.84).
  **Conclusion**: two compounding, separable problems, not one. (a) The
  short 2020+ window alone (forced by sea-temp's data start) is already
  insufficient to reliably fit a period-52 seasonal SARIMAX - this alone
  flips the model from beating naive to losing to naive, independent of
  any exogenous features. (b) The exogenous regressors add further harm
  on top of that, consistent with them carrying no real linear signal.
  This means the SARIMAX-X result **isn't a fair test of whether
  fishmeal/sea-temp actually help** - the comparison is confounded by
  window length. State this honestly in the README rather than concluding
  "exogenous features don't help": the real finding is "this window is
  too short to test that with a seasonal ARIMA-style model." XGBoost
  doesn't need to estimate an explicit seasonal component the same way,
  so it's a fairer place to actually test these features' value - not
  something to bother re-testing further at the SARIMAX stage.
  Two throwaway diagnostic scripts were run for this (not added to the
  codebase - ad hoc `python -c` one-offs) except for one output file kept
  for reference: `data/processed/backtest_control_short_window_no_exog.parquet`.
  Also note for future background runs on this machine: two attempts at
  the control backtest were killed mid-run by the machine going to sleep
  before the ~15-17 min job finished; had to ask the user to keep the
  machine awake for the retry. Not a code issue - just something to expect
  again for any future long-running background command here.
- **2026-09-04**: Logged the SARIMAX findings in the README, then built
  the XGBoost weekly model stage (Architecture item 3). Mirrored the
  SARIMAX split - `xgboost_autoregressive` (full history) and
  `xgboost_exogenous` (~2020+, reuses `weekly_panel.py`) - specifically to
  let XGBoost's more flexible functional form re-test whether the
  exogenous features help, since the SARIMAX-X result was confounded by
  window length (2026-09-03 finding). Built `features/weekly_features.py`
  (lags incl. a year-ago lag for seasonality, log-returns, rolling
  mean/std, week-of-year sin/cos - all leakage-safe by the same discipline
  as `weekly_panel.py`), `models/xgboost_baseline.py` (low-level
  `xgboost.train`/`Booster` API, not the sklearn wrapper - avoids an
  scikit-learn dependency for base classes not otherwise needed),
  `eval/backtest_xgboost.py` (rolling window, refit every week - unlike
  SARIMAX, retraining XGBoost on a few hundred rows is sub-second, so no
  periodic-refit trick was needed), and extended `scripts/run_backtest.py`
  to run all four variants. Config in `config/model.yaml`: XGBoost
  predicts the 1-week log-return (not raw price level) and reconstructs
  price via `price_{t-1} * exp(predicted_return)` - a raw-level tree model
  can't extrapolate past its training range, which matters on a series
  that's roughly tripled since 2003. `xgboost_exogenous` uses a shorter
  `train_window_weeks` (104 vs. 208) than `xgboost_autoregressive` since
  the exogenous panel only has ~330 rows total and a 208-week window would
  leave too few weeks for evaluation - both numbers documented in the
  config comments.
  Verification: full pytest suite (42 tests) passes, `ruff check .` clean.
  **Headline result** (all four variants, `data/processed/backtest_metrics.csv`):

  | variant | weeks | range | MAPE | naive MAPE | RMSE | naive RMSE | dir. acc. | naive dir. acc. |
  |---|---|---|---|---|---|---|---|---|
  | sarimax_univariate | 1230 | 2003-2026 | 3.54% | 3.59% | 2.79 | 2.91 | 58.9% | 0.5% |
  | sarimax_exogenous | 230 | 2022-2026 | 4.96% | 4.29% | 5.84 | 5.14 | 54.3% | 0.0% |
  | xgboost_autoregressive | 1126 | 2004-2026 | 3.92% | 3.75% | 3.11 | 3.04 | 54.8% | 0.4% |
  | xgboost_exogenous | 178 | 2023-2026 | 4.50% | 4.31% | 5.32 | 5.26 | 55.1% | 0.0% |

  **Only `sarimax_univariate` beats naive on all three success criteria.**
  Neither XGBoost variant beats naive on MAPE/RMSE either (both land
  within ~1-5% of naive, sometimes marginally worse) - so this isn't
  unique to SARIMAX-X. All four variants comfortably beat naive on
  directional accuracy. A feature-importance check on the fitted
  `xgboost_exogenous` model (gain-based, full-sample fit) shows
  `fishmeal_price_usd_per_tonne` and `sst_anomaly_c` ranked mid-pack among
  all 14 features (not last) - real but weak signal, consistent with (not
  contradicting) the SARIMAX coefficient-significance finding. XGBoost's
  exogenous MAPE (4.50%) is meaningfully closer to naive than SARIMAX-X's
  was (4.96%) - the flexible functional form did help close much of the
  gap, just not all the way. **Conclusion, and what to tell the README/any
  future session**: this is not "exogenous features are useless" or
  "XGBoost is broken" - it's "the exogenous features carry weak signal, a
  naive last-observed-value forecast is a genuinely tough benchmark to
  beat on this series (consistent with something close to a weekly random
  walk), and neither exogenous features nor extra model flexibility
  reliably clears it here." The univariate SARIMAX remains the
  best-performing model against the stated success criteria - don't let a
  future session assume XGBoost superseded it just because it's the more
  sophisticated model.
  Operational note: hit the same background-job-killed-by-machine-sleep
  issue as 2026-09-03, twice in a row, mid-run on the *full* 4-variant
  script (which redoes the ~32 min SARIMAX portion every time). Fixed
  properly this time instead of just asking the user to stay at the
  keyboard again: added `compute_or_load()` to `scripts/run_backtest.py`,
  which skips recomputing a variant if its results parquet already exists
  on disk (same pattern as the existing `ensure_processed()` for raw data)
  - an interrupted run now resumes instead of redoing finished work. This
  is a permanent fix, not a one-off workaround - keep it when touching
  this script again.
- **2026-09-04 (continued)**: Built the daily nowcast layer (Architecture
  items 1-5), the last unbuilt piece per the build order. First
  investigated Fish Pool futures data (the fourth planned daily input) -
  deferred, see "Deferred items": Fish Pool itself is now just an
  informational front for Euronext (absorbed salmon futures trading in
  2025, contract `ESF`); Euronext's settlement-prices page is
  JS-rendered and only exposes current/future delivery months anyway, not
  a historical series; real historical data is paid-only (Barchart,
  Undercurrent News). Confirmed via `yfinance` that both FX pairs
  (`USDNOK=X`, `EURNOK=X`) and all 5 Oslo Bors tickers (`MOWI.OL`,
  `SALM.OL`, `LSG.OL`, `GSF.OL`, `BAKKA.OL`) work; `BAKKA.OL` has the
  shortest history (starts 2010-03-26), bounding the nowcast backtest
  window to ~2010-present (~16 years - still much longer than the
  SARIMAX-exogenous variant's 230 weeks). User decision: skip futures,
  build the nowcast with just FX + stocks.
  Built `data/daily_market_data.py` (wide daily parquet, one outer-joined
  column per series - each keeps its own native trading calendar rather
  than being forward-filled at fetch time), `features/daily_nowcast_features.py`
  (5-trading-day returns computed per-series on its own native index,
  then backward as-of aligned onto each week's Mon/Tue/Wed/Thu - same
  leakage discipline as `weekly_panel.py`; cross-sectional mean/dispersion
  use `skipna=False` so a date only gets a valid value once *all 5*
  tickers have traded), and `eval/backtest_nowcast.py` (rolling 104-week
  window, retrains once per completed week and reuses that fit for all 4
  within-week predictions - provably identical to retraining 4x/week
  since the training set of *completed* weeks can't change mid-week, so
  doing so would just be wasted compute; same reasoning as the
  `.extend()`-vs-`.append()` lesson from the SARIMAX stage). Reused
  `models/xgboost_baseline.py` and `eval/metrics.py` unchanged - both were
  already generic enough. The model predicts `log(actual / baseline_pred)`
  (a correction on top of the univariate SARIMAX's own forecast, not the
  raw price level) and reconstructs via `baseline_pred * exp(pred)`.
  Extended `scripts/run_backtest.py` to build the daily data, run the
  nowcast backtest, and generate `assets/nowcast_rmse_by_day.png` (new
  `matplotlib` dependency) - the one committed (non-gitignored) output in
  this project, since the README embeds it directly.
  Verification: full pytest suite (57 tests) passes, `ruff check .` clean.
  **Headline result** (`data/processed/backtest_nowcast.parquet`, 747
  weeks per day, 2010-present):

  | day | nowcast RMSE | static baseline RMSE (no daily update) |
  |---|---|---|
  | Mon | 3.71 | 3.47 |
  | Tue | 3.63 | 3.47 |
  | Wed | 3.63 | 3.47 |
  | Thu | 3.73 | 3.47 |

  **The nowcast does not beat the static baseline on any day, and RMSE
  does not decline monotonically Mon->Thu** (it dips Tue/Wed then rises
  again Thursday) - the Goal section's nowcast success criterion was not
  met. A feature-importance check (gain-based, full-sample fit) shows no
  daily feature dominates (0.008-0.018 range, same diffuse pattern as the
  weekly exogenous features) - daily FX/salmon-stock moves are a noisy,
  indirect proxy for the actual weekly export-price surprise. This is the
  **third** time in this project that added complexity/data failed to
  beat a simpler benchmark (after SARIMAX-exogenous and both XGBoost
  variants) - reported honestly in the README as one coherent finding
  (naive is a genuinely tough benchmark on this series, consistent with
  something close to a random walk) rather than three separate
  disappointments, and definitely not adjusted/re-tuned until it looked
  better. **Don't let a future session read "daily nowcast" in the
  architecture and assume it worked** - check this result first.
  This closes out the build order's core architecture (data -> features
  -> models -> backtest -> README) for all three model layers (SARIMAX,
  XGBoost, nowcast). Remaining open items are the deferred data sources
  (Fish Pool futures, harvest/biomass, fish oil) - see "Deferred items" -
  and whatever the user wants to prioritize next (e.g. MIDAS v2, revisiting
  a deferred source, or considering the project done as an honest,
  rigorously-tested negative result).
- **2026-09-04 (wrap-up)**: Added a weekly headline chart (actual vs.
  one-step-ahead SARIMAX forecast, full history) to the README and
  committed the entire build (SARIMAX, XGBoost, nowcast - commit
  `ee3b311`). Then worked through the remaining items this file flags as
  needing attention before considering the project done, per its own
  "ask before locking in" / "ask for a heads-up" instructions rather than
  assuming:
  - **Fish oil feed-cost proxy**: followed up on the "World Bank Pink
    Sheet used to include a Fish oil series" lead - it doesn't (checked
    both the monthly and annual historical files directly, free/no-auth
    at `thedocs.worldbank.org`), and neither does FAO GLOBEFISH (narrative
    only, no data). Three sources checked now (IndexMundi, World Bank,
    FAO), none have it - see updated "Deferred items" entry. Not worth a
    fourth attempt without a specific new lead.
  - **Harvest/biomass (BarentsWatch)**: asked the user per this file's own
    instruction (pipeline is now fully working end-to-end, the condition
    for asking). Decision: skip it, don't register for BarentsWatch -
    settled, not just still-open.
  - **Dockerfile**: noticed it was orphaned scaffolding from the very
    first commit (`59c1649`) - never referenced in the README, never
    updated since statsmodels/xgboost/yfinance/matplotlib were added.
    Added `libgomp1` (xgboost's OpenMP dependency, commonly missing on
    Debian-slim images) and a README mention. **Not build-tested** -
    Docker isn't installed on this machine - flagged as best-effort in
    the README rather than silently presented as verified.
  With those resolved, the project is now complete relative to
  everything CLAUDE.md specifies as MVP or as requiring a decision.
  Anything further (MIDAS v2, a config framework, revisiting a deferred
  source) needs the user to actively propose it, not a future session
  assuming there's more to do.
- **2026-09-06**: Pushed the full build to GitHub (`origin/main`, commit
  `7e77f75`) - confirmed both chart images actually render (checked the
  raw file URLs directly, HTTP 200 + correct content-type; a WebFetch
  render-check gave a false "images broken" reading, which turned out to
  be an artifact of its HTML-to-markdown conversion, not a real GitHub
  problem - don't trust that method for verifying rendered images again).
  Then added a side analysis the user asked for: which of the 5 salmon
  stocks' weekly returns co-moves most with the SSB salmon price's own
  weekly return. Built as a **separate** script
  (`scripts/analyze_stock_correlations.py`, not part of
  `run_backtest.py` - this is a descriptive analysis, not a forecast),
  with the testable logic in `eval/stock_correlation.py`
  (`compute_stock_salmon_correlations`, reuses `asof_lookup` from
  `features/daily_nowcast_features.py` for the daily-to-weekly
  alignment) and config in `config/model.yaml`'s `stock_correlation`
  block. Used the `dataviz` skill for the chart (single accent hue
  `#2a78d6` for all 5 bars, since the job is comparing one metric across
  named categories, not distinguishing series - see the skill's
  color-formula.md on nominal categoricals - plus its palette's
  text/gridline/surface tokens).
  **Result**: BAKKA (Bakkafrost) and GSF (Grieg Seafood) show the
  highest correlation to salmon price (0.046, 0.036) over the common
  2010-present window, MOWI the lowest (0.013) - but the real finding is
  that **all five are close to zero** (R² 0.02-0.2%). This is a fourth,
  independent confirmation of this project's central theme: salmon-price-
  adjacent signals carry very little exploitable weekly information, this
  time from the other direction (salmon-company stocks barely track the
  spot price even contemporaneously, which explains in hindsight why they
  were weak nowcast predictors too). Verification: full pytest suite (61
  tests) passes, `ruff check .` clean, chart visually confirmed.
  Added to the README as its own section, between the nowcast result and
  "Reproducing these results".
- **2026-09-06 (continued)**: Added statistical rigor the project was
  missing - every "beats/loses to naive" claim so far was a bare point
  estimate (e.g. univariate SARIMAX's 3.54% vs. 3.59% MAPE), with no test
  of whether that margin is distinguishable from noise. Built
  `eval/significance.py`: a Diebold-Mariano test (squared-error loss,
  Harvey-Leybourne-Newbold small-sample correction, compared against a
  Student's t rather than normal distribution) on each variant's forecast
  errors vs. its benchmark's. Applied to all 4 weekly variants (vs. naive)
  and all 4 nowcast day-buckets (vs. the static baseline), wired into
  `scripts/run_backtest.py`, results saved to
  `data/processed/significance_tests.csv`.
  **This made the story more nuanced, not just more confirmed**:
  - `sarimax_univariate` vs naive: p=0.019, **significantly better** -
    the headline claim is now confirmed statistically real, not a lucky
    point estimate.
  - `sarimax_exogenous` vs naive: p=0.032, **significantly worse** - this
    one previously read as "a worse point estimate"; the test shows it's
    a real effect, not noise either.
  - `xgboost_autoregressive` and `xgboost_exogenous` vs naive: p=0.368
    and p=0.840, **no significant difference either way**. This is a
    materially different conclusion than "XGBoost doesn't beat naive" -
    it's "we can't statistically distinguish XGBoost from naive," which
    is a more defensible, weaker claim than SARIMAX-X's proven-worse
    result. Don't conflate these two "doesn't beat naive" cases in the
    README or future write-ups - they're statistically different findings.
  - All 4 nowcast day-buckets vs. static baseline: significantly worse on
    every day (p from 0.030 down to 0.00004) - and notably, **significance
    strengthens Mon->Thu even though the RMSE gap itself isn't
    monotonic** - an interesting wrinkle worth keeping in mind if the
    nowcast is ever revisited.
  Verification: full pytest suite (66 tests, incl. deliberately-constructed
  known-answer cases for the DM test itself - a clearly-better-model case,
  an identical-errors case that must return NaN not 0 since it's genuinely
  a 0/0 degenerate input, and a zero-mean-but-not-identical case
  constructed to give an exact analytical answer) passes, `ruff check .`
  clean. README updated throughout (headline table, the exogenous/XGBoost
  discussion, and the nowcast section) to reflect the more precise
  significant/worse vs. not-significant distinction rather than treating
  every non-win as equivalent.
- **2026-09-06 (continued)**: Two more additions, both extending the
  significance-testing theme from the same session. Built
  `eval/random_walk_test.py` (ADF test on log price level + Ljung-Box on
  weekly log-returns at lags 1/4/12/52) to formally test the "close to a
  random walk" claim the README had been asserting qualitatively since
  the SARIMAX stage. **This didn't just confirm the claim - it corrected
  it**: ADF fails to reject a unit root (p=0.49, consistent with a
  non-stationary price level), but Ljung-Box **rejects "no
  autocorrelation" overwhelmingly at every lag** (p < 1e-9 throughout) -
  weekly returns are *not* white noise, so this is **not a pure random
  walk**. Re-read as "non-stationary with weak but real autocorrelation,"
  this actually explains the whole project better than "random walk"
  did: the real autocorrelation is exactly what SARIMAX(1,1,1)x(0,1,1,52)
  is built to extract (hence its significant win over naive), and the
  fact that the edge is still small tells you the autocorrelation, while
  real, is thin. **Don't let a future session revert the README's
  language back to a bare "random walk" claim** - the more precise
  framing ("non-stationary, weakly autocorrelated") is what the tests
  actually support and is a better explanation for the project's overall
  pattern.
  Also built `eval/ensemble.py` (simple average of `sarimax_univariate` +
  `xgboost_autoregressive`, inner-joined on `week_id` since they cover
  different windows) and tested it the same rigorous way. Result: MAPE
  3.66% (naive 3.75%), and DM tests show **no significant difference vs.
  naive** (p=0.060 - closest of any non-SARIMAX-univariate variant to
  conventional significance, but doesn't clear it) **and no significant
  difference vs. SARIMAX alone** (p=0.733). Conclusion: blending in
  XGBoost neither helps nor hurts significantly - no evidence to prefer
  the ensemble over plain SARIMAX, so simplicity wins by default.
  Config additions in `config/model.yaml`'s `backtest:` block:
  `random_walk_test_path`, `ensemble_results_path`. Verification: full
  pytest suite (74 tests) passes, `ruff check .` clean. README updated:
  new "Is this actually a random walk?" section (between the exogenous/
  XGBoost discussion and the nowcast section), an ensemble row/paragraph
  added to "Headline result", and every remaining "random walk" mention
  elsewhere in the README repointed to the new section instead of
  restating the claim loosely.

## Deferred items

- **Fish oil feed-cost proxy** — deferred, not MVP, and now fully checked
  rather than just IndexMundi. IndexMundi (chosen source for fishmeal)
  does not track fish oil at all — confirmed 2026-08-03 against its full
  ~75-commodity list. Followed up 2026-09-04 on the "World Bank Pink
  Sheet used to include a Fish oil series" lead from that session: it
  doesn't - downloaded and inspected both `CMO-Historical-Data-Monthly.xlsx`
  and `CMO-Historical-Data-Annual.xlsx` directly from
  `thedocs.worldbank.org` (both free, no auth) and neither has a Fish oil
  column, only "Fish meal" (confirmed via the header row and the
  Description sheet's source notes). Also checked FAO GLOBEFISH's
  fishmeal/fish oil page - narrative market commentary only, no
  downloadable series or API. No free, structured fish oil price source
  was found across three attempts - if this is revisited, it likely needs
  a paid source (e.g. Tridge, aquafeed.com price reports) rather than
  more free-source searching alongside the BarentsWatch harvest/biomass
  item and the extended (pre-2020) sea surface temperature history.
- **Harvest volumes and standing biomass (Fiskeridirektoratet)** — deferred,
  not MVP. Investigated 2026-08-02: Fiskeridirektoratet's public, no-auth
  statbank (`statistikkbanken.fiskeridir.no`, PxWebApi at
  `https://statistikkbanken.fiskeridir.no/PxWeb/api/v1/no/Fiskeridirektoratet`)
  only publishes harvest/sales and biomass at **annual** granularity (`Tid`
  variable is `År`, e.g. table `A06002b.px` "Salg av laks... (Fylke)" and
  `A07002b.px` "Beholdning per 31.12..."). The actual weekly/monthly data
  lives behind BarentsWatch's Fishhealth/AquaInfo APIs
  (https://developer.barentswatch.no/docs/fishhealth), which require
  OAuth2 app registration (client id/secret) — not free/open access.
  Decision: skip for now, revisit as a v1.1 item once the rest of the
  pipeline (features, models, backtest) is working end-to-end. **Ask the
  user for a heads-up/go-ahead at that point** rather than silently adding
  it — it requires them to register a BarentsWatch account and app, and
  credentials would need to go in a local `.env` (gitignored) with setup
  steps documented in the README for reproducibility.
  **Asked 2026-09-04** (pipeline now fully working end-to-end, per the
  condition above): user chose to skip it and consider the project
  complete as-is, rather than register for BarentsWatch. Don't re-raise
  this without a reason to revisit - it's a settled decision, not just an
  still-open question.
- **Fish Pool / Euronext salmon futures (curve slope, daily nowcast
  input)** — deferred, not MVP. Investigated 2026-09-04: Fish Pool itself
  (fishpool.eu) is now just an informational front for Euronext, which
  absorbed salmon futures trading in 2025 (contract code `ESF`).
  Euronext's live settlement-prices page renders its table via
  client-side JavaScript (not fetchable as static content), and even a
  scraped snapshot would only cover *current/future* delivery months, not
  a historical time series usable for backtesting. Actual historical data
  lives behind paid providers (Barchart Premier, Undercurrent News API).
  Unlike IndexMundi (a public info page), Euronext is a regulated
  exchange - scraping its live trading platform isn't the same kind of
  thing as parsing a public commodity-price page, so this wasn't pursued
  further without asking. Decision (user, 2026-09-04): skip it and build
  the daily nowcast layer with just FX (NOK/USD, NOK/EUR) and Oslo Børs
  salmon-stock returns (both confirmed working via `yfinance` - see
  session log 2026-09-04), dropping the futures-curve-slope feature from
  CLAUDE.md architecture item 2. Revisit as a v1.1 item alongside the
  other deferred sources if a free/affordable historical source turns up.

## Architectural decisions still open

Ask before locking these in — don't assume:
- Final ML library choice for the weekly baseline beyond
  SARIMAX/XGBoost (e.g. whether to add LightGBM, a DL model, etc.)
- Whether to introduce a config framework (e.g. Hydra) vs. plain
  YAML + a small loader.
- MIDAS implementation approach (v2, not MVP — revisit later).

## Known environment quirks (this machine)

These are local-machine workarounds, not project design choices —
don't "fix" them by changing project code:

- **TLS interception**: this machine's network path breaks `uv`'s
  default TLS verification (`invalid peer certificate: UnknownIssuer`),
  likely corporate AV/proxy SSL inspection. Fixed via
  `UV_SYSTEM_CERTS=1`, set persistently as a user env var. If `uv`
  network commands fail with a cert error in a new shell, `setx
  UV_SYSTEM_CERTS 1` (or check it's still set) before assuming it's a
  real problem.
- **Same TLS interception breaks Python's `requests` library** (used by
  the data fetchers) with `SSL: CERTIFICATE_VERIFY_FAILED: unable to
  get local issuer certificate` — `requests`/`urllib3` use their own
  bundled `certifi` CA list rather than the Windows trust store, so
  `UV_SYSTEM_CERTS` doesn't cover it. `curl` was unaffected (it uses
  Windows' schannel, which already trusts the intercepting cert). Fixed
  by exporting the Windows Root+CA cert stores to
  `C:\Users\tobia\.certs\windows-root-ca-bundle.pem` and setting
  `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` to that path, both persisted
  as user env vars. If a data fetcher hits this error in a new shell,
  check those two env vars are still set before assuming it's a real
  problem.
- **venv location**: the project directory is inside OneDrive. OneDrive
  syncing `.venv` mid-write caused a transient file-lock error during
  setup. Fixed by pointing `.venv` outside the synced tree via
  `UV_PROJECT_ENVIRONMENT=C:\Users\tobia\.venvs\salmon-price-estimator`,
  set persistently as a user env var. The venv itself is disposable
  (`uv sync` recreates it); only the project directory needs to be in
  OneDrive.
- **Windows Smart App Control blocked `pytest.exe`, then blocked pandas
  itself**: originally seen (before 2026-08-31) as `uv run pytest` failing
  with "An Application Control policy has blocked this file" — worked
  around with `uv run python -m pytest` instead of the compiled
  console-script stub. On 2026-08-31, adding a new dependency
  (`statsmodels`, which pulled in `scipy`) triggered a venv rebuild, after
  which even `import pandas` failed with the *identical* message — this
  time on pandas' own compiled `.pyd` files, not a launcher stub, so
  there's no `python -m` style workaround. Root cause confirmed via
  registry: `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy
  VerifiedAndReputablePolicyState` = 1 (Enforced) — Windows 11's Smart App
  Control blocks binaries without established Microsoft reputation, and
  every compiled file in a freshly-rebuilt venv looks "new" to it.
  **Resolved 2026-08-31**: user disabled Smart App Control in Windows
  Security settings. Note this is one-way — Microsoft only supports
  re-enabling it via a full Windows reinstall — so don't suggest
  toggling it back on as a troubleshooting step. If a fresh clone/venv on
  a *different* machine hits the same "Application Control policy" error,
  that machine likely still has Smart App Control enabled; check
  `ruff`/`pre-commit` still work (they were unaffected here) before
  assuming it's this same issue.

## Reproducing the headline result

Once the backtest exists: `uv run python scripts/run_backtest.py`
(see script docstring for config overrides).
