FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

# libgomp1: xgboost's shared library dynamically links against OpenMP,
# which python:3.11-slim doesn't include by default - a common silent
# failure on Debian-slim images. Not build-tested on this machine (no
# Docker available); this is the standard fix for that known failure mode.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["uv", "run", "python", "scripts/run_backtest.py"]
