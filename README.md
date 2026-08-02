# Salmon Price Estimator

Week-ahead forecast and daily nowcast of the SSB weekly export price of
fresh Norwegian farmed salmon (NOK/kg).

**Status: scaffolding only.** Data pipeline, models, backtest, and the
headline results/chart are not yet built. This README will be filled in
(pitch, methodology, data sources, headline chart, limitations) once
there's a working backtest to report on — see `CLAUDE.md` for the full
build plan and current progress.

## Development

```bash
uv sync
uv run python -m pytest
uv run ruff check .
```
