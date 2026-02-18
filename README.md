# Campaign Sales Forecasting

This tool generates forecasts for 3-week campaigns, predicting both item-level and cluster-level sales.

## Features

- **Item & Cluster Forecasts**: Predicts sales for specific items and their clusters
- **Campaign Effect Analysis**: Compares campaign vs. no-campaign scenarios
- **Weekly Breakdown**: Shows week 1 (sales force), week 2, and week 3 (online + stores)
- **Simple Interpretable Models**: Uses LightGBM with regularization

## Usage

### Basic Command

```bash
python src/forecast.py --items "item1,item2,item3" --start "2025-11-18" --end "2025-12-08"
```

### Docker Usage

```bash
docker build -t campaign-forecast .

docker run --rm \
  -v $PWD/data:/app/data \
  --env-file .env \
  campaign-forecast \
  python src/forecast.py --items "123456,789012" --start "2025-11-18" --end "2025-12-08"
```

### Arguments

- `--items`: Comma-separated list of item IDs to forecast
- `--start`: Campaign start date (YYYY-MM-DD format)
- `--end`: Campaign end date (YYYY-MM-DD format)  
- `--refresh`: (Optional) Force refresh of cached preprocessed data

## Output Format

The tool outputs a formatted table for each item:

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

## How It Works

1. **Data Loading**: Fetches historical order data and item information
2. **Preprocessing**: Aggregates to weekly level, adds campaign flags and features
3. **Model Training**: Trains separate models for item and cluster sales using:
   - Temporal features (week, month, year)
   - Campaign flag (binary)
   - Cluster lag features (previous week sales, rolling averages)
   - Campaign history (weeks since last campaign, campaigns YTD)
4. **Forecasting**: 
   - Generates forecasts with `is_campaign_week=1` (campaign scenario)
   - Generates forecasts with `is_campaign_week=0` (baseline scenario)
5. **Effect Calculation**: Compares campaign vs baseline to quantify uplift

## Model Interpretability

The models are primarily driven by:
- **Campaign flag**: Direct impact of being on campaign
- **Historical cluster sales**: Recent performance of similar items
- **Seasonality**: Week/month patterns
- **Campaign fatigue**: Impact of recent campaigns

Feature importance is logged during training for transparency.

## Data Requirements

- Historical order data (minimum 2 years recommended)
- Item master data with cluster assignments
- Campaign calendar (weekly_deals.xlsx)
- Stable cluster definitions (last 2 years)

## File Structure

```
.
├── data/
│   ├── weekly_deals.xlsx          # Campaign calendar
│   ├── processed/                 # Cached preprocessed data
│   ├── models/                    # (Future) Saved models
│   └── output/                    # (Future) Output files
├── src/
│   ├── fetch_data.py             # Data fetching logic
│   ├── preprocess.py             # Data preprocessing
│   └── forecast.py               # Main forecasting script
├── Dockerfile
└── pyproject.toml
```

## Notes

- **Week 1 Effect**: First week includes sales force competition, typically shows higher uplift
- **Cluster Effects**: May be smaller than item effects due to cannibalization
- **Rounding**: Sales units are rounded up to nearest integer
- **Ongoing Campaigns**: Only completed campaigns are used for training
