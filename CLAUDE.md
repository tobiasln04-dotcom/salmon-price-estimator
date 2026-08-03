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
   - Daily nowcast inputs: Fish Pool futures, NOK/USD & NOK/EUR (yfinance),
     Oslo Bors daily returns for MOWI/SALM/LSG/GSF/BAKKA (yfinance).
   - Secondary: NQSALMON weekly history (scraped, Jan 2013-Aug 2024) for
     a longer backtest window and cross-check against SSB.
2. **`src/salmon_price_estimator/features/`** — weekly features (lagged
   target, log-returns, rolling stats, seasonality, harvest deltas,
   biomass ratios) and daily nowcast features (futures curve slope, FX
   5-day return, stock 5-day return, cross-sectional dispersion).
3. **`src/salmon_price_estimator/models/`** — weekly baseline (SARIMAX,
   then XGBoost with lags, rolling-window train, predict t+1) and a daily
   nowcast layer: a second XGBoost predicting the same weekly SSB target
   from (weekly baseline output + latest daily observations), retrained
   daily on a rolling window. MIDAS regression is a v2 comparison, not
   MVP.
4. **`src/salmon_price_estimator/eval/`** — walk-forward backtest with
   expanding window. Metrics: MAPE, RMSE, directional accuracy, vs. naive
   last-week benchmark. Nowcast eval: RMSE by days-elapsed-in-week
   (Mon-Thu) — this is the headline README chart.
5. **`scripts/run_backtest.py`** — single entry point that reproduces the
   headline result end-to-end. Config lives in `config/*.yaml` so
   hyperparameters and date ranges are auditable, not hardcoded.
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

## Deferred items

- **Fish oil feed-cost proxy** — deferred, not MVP. IndexMundi (chosen
  source for fishmeal) does not track fish oil as a commodity at all —
  confirmed 2026-08-03 against its full ~75-commodity list. Would need a
  different source (e.g. a World Bank Pink Sheet historical file, which
  used to include a Fish oil series) — revisit as a v1.1 item alongside
  the BarentsWatch harvest/biomass item and the extended (pre-2020) sea
  surface temperature history.
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
- **Application Control policy blocks `pytest.exe`**: running `uv run
  pytest` fails with "An Application Control policy has blocked this
  file" (likely WDAC/EDR blocking the compiled console-script launcher
  stub in `.venv/Scripts/`). Workaround: use `uv run python -m pytest`
  instead — this was verified to work. `ruff` and `pre-commit` were not
  affected. If other console-script tools get blocked the same way,
  try `uv run python -m <tool>` first before troubleshooting further.

## Reproducing the headline result

Once the backtest exists: `uv run python scripts/run_backtest.py`
(see script docstring for config overrides).
