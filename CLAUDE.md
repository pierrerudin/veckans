# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Campaign sales forecasting system ("Veckans") that predicts 3-week campaign impact on item-level and cluster-level sales using LightGBM models. Reads historical data from Azure Data Lake (Delta Lake format), trains separate models per campaign week, and compares campaign vs. baseline scenarios to quantify uplift.

## Commands

```bash
# Install dependencies
poetry install

# Run forecast via main orchestrator
python src/main.py --item-ids 528208,414235 --forecast-date 2024-11-18 --campaign-start 2025-01-13

# Run forecast via simpler entry point
python src/forecast.py --items "123456,789012" --start "2025-11-18" --end "2025-12-08"

# Force data refresh (skip parquet cache)
python src/forecast.py --items "..." --start "..." --end "..." --refresh

# Validate historical data
python src/validate_historical.py --sample

# Docker
docker build -t campaign-forecast .
docker run --rm -v $PWD/data:/app/data --env-file .env campaign-forecast python src/forecast.py --items "123456" --start "2025-11-18" --end "2025-12-08"
```

No formal test framework (pytest) is configured. Validation is done via `validate_historical.py` and `data_test.py`.

## Architecture

**Pipeline flow:** Input (item IDs + dates) → `fetch_data.py` → `preprocess.py` → `forecast.py` → `main.py` (output)

Key source files in `src/`:

- **main.py** — Orchestration entry point. Parses args, validates inputs, coordinates the full pipeline, formats output tables.
- **forecast.py** — Model training (`train_models`), prediction (`predict`, `predict_baseline`), evaluation. Trains separate LightGBM models per campaign week (1, 2, 3). Also usable as a standalone CLI entry point.
- **preprocess.py** — Heavy lifting: weekly aggregation, baseItemId resolution, unit conversion (kg/L), lag/rolling feature engineering (horizon-aware), log transformation, campaign flag creation. ~1080 lines.
- **fetch_data.py** — Queries Azure Data Lake for `dim_item` and `fact_order` tables. Results cached as parquet files keyed by MD5 of query params in `data/cache/`.
- **azure_datalake_connector.py** — Azure SDK wrapper for Delta Lake & Parquet I/O via adlfs/deltalake.
- **config_forecast.py** — All model hyperparameters, feature lists, data split ratios, evaluation metric functions. Two param sets: `LGBM_PARAMS` (item-level) and `LGBM_PARAMS_CLUSTER` (heavier regularization).
- **config.py** — Reads Azure credentials from environment variables (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, etc.).

**Model design:**
- Separate models per campaign week (1, 2, 3) at both item and cluster level
- Quantile regression (10th, 50th, 90th percentiles) for uncertainty intervals
- Predictions in log-space (`log1p`), inverse-transformed for output
- Campaign effect = campaign forecast minus baseline forecast
- Sales rounded up to nearest integer

**Data:**
- `data/weekly_deals.xlsx` — Campaign calendar (master input)
- `data/cache/` — Parquet query cache (~770MB, auto-generated)
- `data/output/` — Forecast result exports
- `.env` — Azure credentials (not committed)

**Environment variables required** (in `.env`): `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `WORKSPACE_NAME`, `STORAGE_ACCOUNT_NAME`

## Key Conventions

- All source imports assume `src/` is on PYTHONPATH (Dockerfile sets this; locally run from `src/` or set PYTHONPATH)
- Target column is `salesQuantityKgL_log` (log-transformed); original scale is `salesQuantityKgL`
- Data split: 60% train / 20% val / 20% test (temporal, not random)
- Minimum ~10 weeks historical data per item required
- `src-old/` and `src-old2/` are previous iterations, not active code
