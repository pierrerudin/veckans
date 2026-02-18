# Week Alignment Logic

## Overview
The forecasting system supports flexible week alignment based on campaign start date. If a campaign starts on Tuesday, **all data** (historical and forecast) is aligned Tuesday-Monday. If it starts on Saturday, everything is aligned Saturday-Friday.

## Implementation Details

### Single Alignment Throughout
- **Decision point**: Campaign start date passed to preprocessing
- **Alignment**: Everything uses the **same day-of-week** as campaign start
- **Benefit**: No translation needed between training and prediction

### Preprocessing (aggregate_to_weekly)
- **File**: `preprocess.py` → `aggregate_to_weekly()`
- **Parameter**: `campaign_start_date` (optional, defaults to Monday if None)
- **Logic**:
  ```python
  if campaign_start_date is not None:
      campaign_weekday = campaign_start_date.weekday()  # 0=Monday, 1=Tuesday, etc.
      df_orders['week_start'] = df_orders['orderDate'] - pd.to_timedelta(
          (df_orders['orderDate'].dt.weekday - campaign_weekday) % 7, unit='d'
      )
  else:
      # Default to Monday alignment
      df_orders['week_start'] = df_orders['orderDate'] - pd.to_timedelta(
          df_orders['orderDate'].dt.weekday, unit='d'
      )
  ```

### Model Training
- Uses the campaign-aligned weekly data
- No special handling needed - features already aligned correctly

### Prediction (forecast.py)
- **Simplified**: No translation layer needed
- Campaign weeks and lag lookups directly use campaign-aligned dates:
  ```python
  campaign_week1_start = campaign_start_date  # e.g., Tuesday 2025-06-03
  lag_date = week_start - pd.Timedelta(weeks=lag_weeks)
  lag_data = cluster_data[cluster_data['week_start'] == lag_date]  # Direct lookup
  ```

## Example: Tuesday Campaign

**Campaign Details**:
- Start: Tuesday 2025-06-03
- End: Monday 2025-06-23 (21 days)
- Forecast date: Monday 2025-05-05
- Horizon: 4 weeks

**Week Alignment**:
- Campaign Week 1: Tuesday 2025-06-03 to Monday 2025-06-09
- Campaign Week 2: Tuesday 2025-06-10 to Monday 2025-06-16
- Campaign Week 3: Tuesday 2025-06-17 to Monday 2025-06-23

**Lag Calculation (Week 1)**:
- `week_start = 2025-06-03` (Tuesday)
- `lag_weeks = 4 + (1-1) = 4`
- `lag_date = 2025-06-03 - 4 weeks = 2025-05-06` (Tuesday)
- Historical data for week starting Tuesday 2025-05-06 is directly retrieved
- No translation needed!

**Cutoff Date**:
- `forecast_date = 2025-05-05` (Monday)
- `campaign_weekday = 1` (Tuesday)
- `forecast_weekday = 0` (Monday)
- `days_back = (0 - 1) % 7 = 6`
- `cutoff_date = 2025-05-05 - 6 days = 2025-04-29` (Tuesday)

This ensures we only use data available before the Tuesday cutoff, matching the campaign's weekly rhythm. Since all historical data is already Tuesday-aligned, this works seamlessly.

## Logging Output

When running main.py, you'll see:
```
Campaign: 2025-06-03 (Tuesday) to 2025-06-23 (Monday)
Week alignment: Tuesday to Monday
Forecast horizon: 4 weeks
```

This clearly shows:
1. Campaign start day-of-week
2. How weeks are aligned
3. The forecast horizon

## Why This Design?

**Benefits**:
1. **Simplicity**: Single alignment throughout the entire system
2. **Consistency**: Training and prediction use identical week boundaries
3. **No translation**: Direct lag feature lookups without day-of-week conversions
4. **Flexibility**: Each forecast run can use different campaign start days

**Trade-offs**:
- Preprocessing must be run for each forecast (can't preprocess once and reuse)
- However, this is already the design (fetch → preprocess → train → predict is one workflow)

**Why not preprocess once?**
- Each forecast specifies its own campaign start date
- Different campaigns may start on different days (Tuesday vs Saturday)
- The 4-12 week horizon requirement means preprocessing is fast enough to run each time

## Validation

To verify correct alignment:
1. Check preprocessing log: "Aligning weeks to Tuesday (matching campaign start)"
2. Check that `week_start` in forecast output matches campaign start day
3. Verify that lag features are retrieved from correct historical weeks (same day-of-week)
4. All week_start dates should share the same day-of-week throughout the dataset
