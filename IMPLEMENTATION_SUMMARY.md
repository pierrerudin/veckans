# Campaign Forecast Implementation Summary

## What We Built

A complete forecasting pipeline that predicts 3-week campaign sales for both individual items and their clusters, with counterfactual (no-campaign) scenarios for effect analysis.

## Key Components

### 1. **forecast.py** - Main Forecasting Script
The core script that:
- Loads/preprocesses data automatically
- Trains separate models for item and cluster sales
- Generates forecasts with campaign ON and OFF
- Outputs formatted tables with campaign effect calculations

**Key Functions:**
- `load_or_preprocess_data()` - Smart caching of preprocessed data
- `train_simple_model()` - LightGBM with regularization for interpretability
- `forecast_item()` - End-to-end forecasting for one item
- `format_forecast_output()` - Pretty CLI tables

### 2. **validate.py** - Testing & Validation
Helper script to:
- Validate preprocessed data
- Find sample items with good campaign history
- Generate test commands

### 3. **config_forecast.py** - Configuration
Centralized config for easy tuning:
- Model hyperparameters
- Feature definitions
- Data filtering rules

## How It Works

### Data Flow
```
Item IDs + Campaign Dates
    ↓
fetch_data.py (Azure Data Lake)
    ↓
preprocess.py (Weekly aggregation, features)
    ↓
forecast.py (Model training & prediction)
    ↓
Formatted output tables
```

### Model Approach

**Simple & Interpretable:**
- LightGBM with regularization (transparent feature importance)
- Key features: campaign flag, cluster sales, seasonality, campaign history
- Separate models for item-level and cluster-level

**Counterfactual Estimation:**
- Same model used for both scenarios
- Toggle `is_campaign_week` flag between 0 and 1
- Direct comparison shows campaign effect

### Features Used

1. **Campaign Signal:**
   - `is_campaign_week` - Binary flag (THIS IS THE KEY!)

2. **Temporal Patterns:**
   - `week` - Week of year (seasonality)
   - `month` - Month (seasonality)
   - `year` - Year (trend)

3. **Cluster Context:**
   - `cluster_sales_lag_1` - Previous week cluster sales
   - `cluster_sales_rolling4w` - 4-week rolling average
   - (Captures substitution/cannibalization effects)

4. **Campaign History:**
   - `weeks_since_campaign` - Time since last campaign
   - `campaigns_ytd` - Cumulative campaigns this year
   - (Captures fatigue effects)

## Output Format

```
================================================================================
Campaign and effect forecast for ITEM 123456 in cluster ABC789
================================================================================
                         Week 1      Week 2      Week 3      Total       Campaign
Item sale                   150         200         180         530      True
Cluster sale               1200        1350        1280        3830      True
Item sale                   100         120         115         335      False
Cluster sale               1150        1300        1250        3700      False
================================================================================
Expected campaign effect
================================================================================
Item sales: +58.21%
Cluster sales: +3.51%
================================================================================
```

**Interpretation:**
- **Week 1** (campaign=True): Sales force + competition (typically highest)
- **Week 2-3** (campaign=True): Online + stores available
- **campaign=False**: Baseline forecast without campaign
- **Effect %**: Direct uplift from campaign

## Why This Approach?

✅ **Simple & Fast:** Can train and forecast in seconds
✅ **Interpretable:** Clear feature importance, easy to explain
✅ **Flexible:** Same model for campaign ON/OFF scenarios
✅ **Handles Key Effects:** 
   - Week 1 sales force boost
   - Cluster cannibalization
   - Campaign fatigue
✅ **Production Ready:** Caching, error handling, logging

## Usage Examples

### Basic Forecast
```bash
python src/forecast.py \
  --items "123456,789012,345678" \
  --start "2025-11-18" \
  --end "2025-12-08"
```

### Find Test Items
```bash
python src/validate.py --sample
```

### Force Data Refresh
```bash
python src/forecast.py --items "..." --start "..." --end "..." --refresh
```

## Files Delivered

```
src/
├── forecast.py              # Main forecasting script
├── fetch_data.py           # Data loading from Azure
├── preprocess.py           # Feature engineering
├── validate.py             # Testing utilities
└── config_forecast.py      # Configuration

README.md                    # Full documentation
QUICKSTART.md               # Quick start guide
```

## For Your Presentation (14:00)

### Key Talking Points:

1. **"We forecast at two levels: item and cluster"**
   - Item: Direct sales of the campaign item
   - Cluster: Total category sales (shows cannibalization)

2. **"Week 1 is special - sales force only with competition"**
   - Typically shows highest uplift
   - Weeks 2-3 include online + physical stores

3. **"The model learns from historical campaigns"**
   - Uses 2+ years of campaign history
   - Captures seasonality, trends, and campaign patterns
   - Feature importance shows what drives predictions

4. **"We estimate campaign effect via counterfactuals"**
   - Same model, toggle campaign flag on/off
   - Direct comparison shows incremental impact
   - Accounts for cannibalization at cluster level

5. **"Simple, interpretable, and fast"**
   - Can forecast 10-30 items in under a minute
   - Clear feature contributions
   - Easy to validate and trust

### What Users See:
- Clean formatted tables (ready to copy/paste)
- Sales in actual units (not kg/l)
- Clear campaign effect percentages
- Week-by-week breakdown

## Next Steps / Future Enhancements

- [ ] Add prediction intervals (confidence bands)
- [ ] Save models to disk for reuse
- [ ] Export forecasts to Excel/CSV
- [ ] Add more sophisticated time series features
- [ ] Ensemble multiple models
- [ ] Web interface / API
- [ ] Automatic model retraining schedule

## Technical Notes

**Data Requirements:**
- Minimum 10 weeks of historical sales per item
- Campaign data with start/end dates
- Cluster assignments (stable for last 2 years)

**Performance:**
- First run: 5-15 min (includes preprocessing)
- Cached runs: 30-60 seconds for multiple items
- Scales to 10-30 items easily

**Limitations:**
- New items (no history) can't be forecasted
- Cluster definition changes before 2023 may cause artifacts
- Assumes campaign structure remains similar to history

## Support

Questions? Check:
1. QUICKSTART.md - Getting started
2. README.md - Full documentation
3. validate.py --sample - Find test data

Good luck with your presentation! 🎉
