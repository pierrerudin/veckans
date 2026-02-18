# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Campaign sales forecasting system ("Veckans") that predicts 3-week campaign impact on item-level and cluster-level sales using LightGBM models. Reads historical data from Azure Data Lake (Delta Lake format), and compares model-based campaign vs. baseline counterfactual scenarios to quantify uplift. Supports both 3-model (one per campaign week) and single-model (unified with `campaign_week_number` feature) approaches.

## Commands

```bash
# Install dependencies
poetry install

# Load Azure credentials (required before running)
set -a && source .env && set +a

# Run forecast (forecast-date defaults to today)
poetry run python src/main.py --item-ids 528208 --campaign-start 2025-03-10

# Run forecast with explicit forecast date (backtesting)
poetry run python src/main.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13

# Use single unified model instead of 3 separate models
poetry run python src/main.py --item-ids 528208 --campaign-start 2025-03-10 --single-model

# Save results to CSV
poetry run python src/main.py --item-ids 528208,414235 --campaign-start 2025-03-10 --output-file data/output/results.csv

# Hyperparameter tuning with Optuna (walk-forward CV)
poetry run python src/tune.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13 --n-trials 50

# Docker
docker build -t campaign-forecast .
docker run --rm -v $PWD/data:/app/data --env-file .env campaign-forecast python src/main.py --item-ids 528208 --campaign-start 2025-03-10
```

No formal test framework (pytest) is configured. Validation is done via `validate_historical.py` and `data_test.py`.

**Note:** `deltalake` 1.1.4 has a macOS-specific bug in the `object_store` Rust crate. For local macOS development, use `deltalake==0.25.5` (`poetry run pip install 'deltalake==0.25.5'`). Docker (Linux) works fine with 1.1.4.

## Architecture

**Pipeline flow:** Input (item IDs + dates) → `fetch_data.py` → `preprocess.py` → `forecast.py` → `main.py` (output)

Key source files in `src/`:

- **main.py** — Orchestration entry point. Parses args, validates inputs, coordinates the full pipeline, formats output tables. Supports `--single-model` flag and optional `--forecast-date` (defaults to today).
- **forecast.py** — Model training and prediction. `train_models()` trains 3 separate LightGBM models (one per campaign week). `train_models_single()` trains one unified model with `campaign_week_number` as feature + interaction terms. `predict()` handles both modes and uses model-based counterfactual (campaign flags ON vs OFF) for uplift estimation.
- **preprocess.py** — Heavy lifting: weekly aggregation, baseItemId resolution, unit conversion (kg/L), lag/rolling feature engineering (horizon-aware), log transformation, campaign flag creation, category hierarchy lift features with shrinkage, post-campaign recovery week exclusion from normalizers.
- **fetch_data.py** — Queries Azure Data Lake for `dim_item` (including categoryLevel1-4) and `fact_order` tables. Results cached as parquet files keyed by MD5 of query params in `data/cache/`.
- **azure_datalake_connector.py** — Azure SDK wrapper for Delta Lake & Parquet I/O via adlfs/deltalake.
- **config_forecast.py** — All model hyperparameters, feature lists (`COMMON_FEATURES`, `CATEGORY_FEATURES`, `CLUSTER_SPECIFIC_FEATURES`, etc.), data split ratios, evaluation metric functions. Two param sets: `LGBM_PARAMS` (item-level) and `LGBM_PARAMS_CLUSTER` (heavier regularization).
- **config.py** — Reads Azure credentials from environment variables (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, etc.).
- **tune.py** — Optuna hyperparameter tuning with walk-forward cross-validation. Campaign-aware evaluation (SMAPE on campaign weeks).

**Model design:**
- Two modes: 3 separate models per campaign week (default) or single unified model (`--single-model`) with `campaign_week_number` feature + interaction terms (`cwn_x_lag`, `cwn_x_rolling`)
- Both item-level and cluster-level models; cluster campaign intensity uses fractional values (`1/num_items_in_cluster`) instead of binary flags
- Quantile regression (10th, 50th, 90th percentiles) for uncertainty intervals
- Predictions in log-space (`log1p`), inverse-transformed for output
- Model-based counterfactual: campaign effect = same model with campaign ON minus campaign OFF
- Category hierarchy features with hierarchical shrinkage (L4 → L3 → L2 → L1) for cold-start items
- Temporal train/val/test split (60/20/20), no random shuffling

**Data:**
- `data/weekly_deals.xlsx` — Campaign calendar (master input)
- `data/cache/` — Parquet query cache (~770MB, auto-generated)
- `data/output/` — Forecast result exports
- `.env` — Azure credentials (not committed)

**Environment variables required** (in `.env`): `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `WORKSPACE_NAME`, `STORAGE_ACCOUNT_NAME`

## Key Conventions

- Run from project root with `poetry run python src/main.py ...` (Dockerfile sets `WORKDIR /app` and adds `src/` to PYTHONPATH)
- Target column is `salesQuantityKgL_log` (log-transformed); original scale is `salesQuantityKgL`
- Per-item normalization: `log1p(sales / baseline_median)` for transferable campaign effects
- Data split: 60% train / 20% val / 20% test (temporal, not random)
- Minimum ~10 weeks historical data per item required
- `--forecast-date` defaults to today; no minimum forecast horizon — can forecast at any point before campaign start
- `src-old/` and `src-old2/` are previous iterations, not active code
