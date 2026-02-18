# Quick Start Guide

## Setup (First Time)

1. **Place the files in your project:**
   ```bash
   # Copy the src directory into your project
   # Make sure you have:
   # - src/forecast.py
   # - src/fetch_data.py  
   # - src/preprocess.py
   # - src/validate.py
   ```

2. **Ensure you have weekly_deals.xlsx:**
   ```bash
   # Should be at: data/weekly_deals.xlsx
   ```

3. **Your .env file should have Azure credentials:**
   ```
   AZURE_STORAGE_ACCOUNT_NAME=...
   AZURE_STORAGE_ACCOUNT_KEY=...
   # (or whatever credentials your AzureDataLakeConnector needs)
   ```

## Running Forecasts

### Step 1: Get Sample Item IDs (Optional)

If you want to test with items that have good campaign history:

```bash
python src/validate.py --sample
```

This will show you 5 items with campaign history and generate a test command.

### Step 2: Run Forecast

```bash
python src/forecast.py \
  --items "item1,item2,item3" \
  --start "2025-11-18" \
  --end "2025-12-08"
```

**Arguments:**
- `--items`: Comma-separated item IDs (no spaces!)
- `--start`: Campaign start date (YYYY-MM-DD)
- `--end`: Campaign end date (YYYY-MM-DD)
- `--refresh`: (Optional) Force data refresh

### Step 3: Read the Output

The forecast will print formatted tables to your console:

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

## Docker Usage

```bash
# Build
docker build -t campaign-forecast .

# Run
docker run --rm \
  -v $PWD/data:/app/data \
  --env-file .env \
  campaign-forecast \
  python src/forecast.py --items "123456,789012" --start "2025-11-18" --end "2025-12-08"
```

## Troubleshooting

### "No preprocessed data found"
Run the forecast once - it will automatically preprocess the data for you.

### "Insufficient training data"
The item doesn't have enough historical sales. Try items with more history.

### "Item not found"
Check that the item ID exists in your data. Use `validate.py --sample` to find valid items.

### Data seems stale
Use `--refresh` flag to force reprocessing:
```bash
python src/forecast.py --items "..." --start "..." --end "..." --refresh
```

## For Your Presentation

1. **Run forecasts for your campaign items** before the meeting
2. **Copy the output tables** - they're ready to paste into slides/docs
3. **Key talking points:**
   - Week 1 is sales force only (typically higher uplift)
   - Cluster effects show cannibalization impact
   - Models use historical campaigns + seasonality + cluster trends
   - Forecasts are in sales units (rounded up)

## Time Estimates

- First run (with preprocessing): ~5-15 minutes depending on data size
- Subsequent runs (cached data): ~30-60 seconds per item
- Each forecast: ~10-20 seconds

Good luck with your presentation! 🚀
