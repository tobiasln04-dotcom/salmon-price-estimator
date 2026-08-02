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
   - Target + weekly features: SSB export price (StatBank), harvest
     volumes and biomass (Fiskeridirektoratet), sea surface temp anomaly
     (MET Norway/NOAA), feed cost proxies (IndexMundi fishmeal/fish oil).
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
