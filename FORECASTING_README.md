# Campaign Sales Forecasting System

Complete forecasting system for predicting campaign sales impact with uncertainty intervals.

## Features

- ✅ **3-week campaign forecasts** with week-by-week predictions
- ✅ **Flexible week alignment** based on campaign start day (Tuesday campaigns → Tuesday-Monday weeks)
- ✅ **Baseline comparison** (with vs without campaign)
- ✅ **Uncertainty intervals** (10th and 90th percentiles)
- ✅ **Log normalization** for handling wide sales ranges (basil to milk)
- ✅ **Horizon-aware lag features** (8-10 week lags matching forecast timing)
- ✅ **Multiple metrics** (SMAPE, MAE, MAPE, RMSE)
- ✅ **Single-item cluster handling** (train on them, don't forecast twice)

## Quick Start

### Basic Usage

```bash
# Forecast milk item for January 2025 campaign
python src/main.py \
  --item-ids 528208 \
  --forecast-date 2024-11-18 \
  --campaign-start 2025-01-13
```

### Multiple Items

```bash
# Forecast multiple items at once
python src/main.py \
  --item-ids 528208,414235,262168 \
  --forecast-date 2024-12-01 \
  --campaign-start 2025-02-10
```

### Save Results

```bash
# Export forecasts to CSV
python src/main.py \
  --item-ids 528208 \
  --forecast-date 2024-11-18 \
  --campaign-start 2025-01-13 \
  --output-file ../data/output/forecast_results.csv
```

## Output Format

For each cluster, you'll see:

```
--------------------------------------------------------------------------------
Item 528208 - STANDARDMJÖLK 3% KRT 6X1L ARA
--------------------------------------------------------------------------------

Type            Campaign   Week 1       Week 2       Week 3        Total
--------------------------------------------------------------------------------
Cluster         Yes        32,450 kg    35,670 kg    33,890 kg    101,010 kg
(in units)                  5,408        5,945        5,648        17,001

Cluster         No         30,700 kg    31,200 kg    31,500 kg     93,400 kg

Uncertainty     10th %ile                                         85,200 kg
                90th %ile                                        117,300 kg

Campaign Effects
----------------------------------------
Cluster sales change:  +7,610 kg (+8.1%)
```

## Configuration

All forecasting parameters are in `config_forecast.py`:
- Model hyperparameters (LGBM_PARAMS)
- Feature sets (week-specific features)
- Metrics (SMAPE, MAE, MAPE, RMSE)
- Splits (60/20/20 train/val/test)
- Quantiles for uncertainty intervals

For detailed information on how week alignment works (e.g., Tuesday campaigns with Tuesday-Monday weeks), see **WEEK_ALIGNMENT.md**.

### What you can customize

- Features per campaign week
- Lag feature alignment

### Model tuning

## Architecture

### Data Flow

```
fetch_data.py
    ↓
    Fetches historical orders, items, campaigns from Azure Data Lake
    ↓
preprocess.py
    ↓
    - Converts to baseItemId, kg/L units
    - Applies log1p transformation
    - Aggregates to weekly level
    - Creates horizon-aware lag features (week1/2/3)
    - Adds time features (cyclical encoding)
    - Adds cluster context
    ↓
forecast.py
    ↓
    - Trains 3 LightGBM models (one per campaign week)
    - Quantile regression for uncertainty (10th, 50th, 90th percentile)
    - Evaluates on test set (SMAPE, MAE, MAPE, RMSE)
    ↓
main.py
    ↓
    - Orchestrates workflow
    - Validates inputs
    - Generates predictions
    - Calculates campaign effects
    - Formats results
```

### Key Design Decisions

1. **Separate models per week**: Week 1 uses 8-week lags, week 2 uses 9-week, week 3 uses 10-week
2. **Log transformation**: Handles items selling tons (milk) vs grams (saffron)
3. **Cluster-level forecasting**: Focuses on cluster impact, not individual item cannibalization
4. **Random stratified split**: 60/20/20 split by cluster to ensure generalization across all time periods
5. **Quantile regression**: Provides uncertainty intervals for risk assessment

## Validation

The system reports test set performance for each campaign week:

- **SMAPE**: Symmetric Mean Absolute Percentage Error (lower is better)
- **MAE**: Mean Absolute Error in kg/L
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error in kg/L

Check these metrics to assess model reliability before using forecasts for purchasing decisions.

## Input Requirements

- **item-ids**: Comma-separated string (e.g., "528208,414235")
- **forecast-date**: Date when forecast is made (YYYY-MM-DD, must be ≤ today)
- **campaign-start**: Campaign start date (YYYY-MM-DD, must be after forecast-date)
  - **Week alignment**: The day-of-week of campaign-start determines week boundaries
  - Example: If campaign starts on Tuesday, all weeks are aligned Tuesday-Monday
- **campaign-duration**: Fixed at 3 weeks (21 days)
- **Forecast horizon**: Automatically calculated, must be 4-12 weeks

## Warnings

The system will warn you if:
- ⚠️ Very high campaign effect (>200%) - verify inputs
- ⚠️ Negative campaign effect - unusual, check data
- ⚠️ No historical data for item/cluster
- ⚠️ Insufficient lag data for prediction

## Example Workflow

```bash
# 1. Test with single item
python src/main.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13

# 2. Review results and model performance

# 3. Run for all planned campaign items
python src/main.py \
  --item-ids 528208,414235,262168,117903 \
  --forecast-date 2024-11-25 \
  --campaign-start 2025-01-20 \
  --output-file ../data/output/january_campaign_forecast.csv

# 4. Use forecasts for purchasing decisions
```

## Development

### Preprocessing test

```bash
# Run preprocessing simulation
cd src
docker exec -it veckans-forecast python preprocess.py
```

This shows:
- Feature engineering pipeline
- Training data preparation
- Forecast input simulation
- Lag feature alignment

### Model tuning

Edit `config_forecast.py` to adjust:
- `LGBM_PARAMS`: Model hyperparameters
- `COMMON_FEATURES`: Which features to include
- `QUANTILES`: Uncertainty interval percentiles

Retrain and compare test set metrics to find optimal configuration.

## Troubleshooting

**"Forecast horizon too short/long"**
→ Adjust forecast-date or campaign-start to be 4-12 weeks apart

**"No historical data for cluster"**
→ Item is too new or has no sales history, cannot forecast

**"Only X samples for week Y"**
→ Not enough historical campaigns, model skips this week

**Very high SMAPE (>50%)**
→ Model struggling, check feature engineering or increase training data

## Next Steps

- Integrate with purchasing system
- A/B test forecasts vs actual outcomes
- Tune model parameters based on business metrics
- Add price elasticity features (if price data becomes available)
- Expand to 4+ week campaigns

---

**Questions?** Check logs for detailed diagnostics or review model performance metrics in the output.
