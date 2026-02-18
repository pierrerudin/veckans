# Pre-Presentation Checklist ✓

## Before You Run (First Time)

- [ ] Copy all files from the delivery to your project directory
- [ ] Ensure `data/weekly_deals.xlsx` exists
- [ ] Verify `.env` has Azure credentials configured
- [ ] Check your Docker setup (if using Docker)

## Testing the Pipeline (30 min before presentation)

### Step 1: Find Good Test Items
```bash
python src/validate.py --sample
```
✓ This shows 5 items with campaign history
✓ Copy the generated test command

### Step 2: Run First Forecast
```bash
python src/forecast.py --items "item1,item2,item3" --start "2025-11-18" --end "2025-12-08"
```
✓ First run will preprocess data (5-15 min)
✓ Subsequent runs use cache (much faster)

### Step 3: Validate Output
- [ ] Tables display correctly
- [ ] Numbers look reasonable (not 0 or negative)
- [ ] Campaign effect % makes sense
- [ ] Week 1 typically higher than weeks 2-3

## For Your Actual Campaign Items

### Get Your Item IDs
```bash
# From your campaign planning or database
ITEMS="item1,item2,item3,..."  # Your actual items for the campaign
```

### Run Production Forecast
```bash
python src/forecast.py \
  --items "$ITEMS" \
  --start "2025-11-18" \
  --end "2025-12-08"
```

### Copy Output for Presentation
- [ ] Copy the formatted tables
- [ ] Note the campaign effect percentages
- [ ] Prepare to explain week 1 vs weeks 2-3

## Presentation Talking Points

### Opening
"We've built a forecasting system that predicts sales for our 3-week campaigns at both the item and cluster level, with counterfactual analysis showing the expected campaign effect."

### Key Features to Highlight
- ✓ **Two-level forecasting**: Item (direct) + Cluster (cannibalization)
- ✓ **Week breakdown**: Week 1 (sales force) vs Weeks 2-3 (all channels)
- ✓ **Campaign effect**: Direct comparison of campaign vs no-campaign
- ✓ **Simple & fast**: Forecasts in seconds, easy to interpret

### Example Explanation
"For Item X in the milk cluster:
- With campaign: We forecast 530 units total (150 + 200 + 180)
- Without campaign: We'd expect 335 units (100 + 120 + 115)
- Campaign effect: +58% uplift on item sales
- Cluster shows +3.5%, indicating some cannibalization of other milk products"

### Technical Details (if asked)
- ✓ Uses LightGBM with 2+ years of historical campaign data
- ✓ Key features: campaign flag, seasonality, cluster trends, campaign history
- ✓ Counterfactual via toggling campaign flag in same model
- ✓ Interpretable: clear feature importance

### Limitations (be honest)
- Requires sufficient historical data (10+ weeks)
- New items can't be forecasted
- Assumes campaign structure similar to past
- Cluster definitions must be stable

## Backup Plans

### If Data Issues
"We're still refining the data pipeline, but the methodology is solid and we'll have full forecasts by [next week]"

### If Questioned on Accuracy
"These are initial forecasts. We'll validate against actual results and refine the models iteratively."

### If Asked About Automation
"Currently run on-demand. Future: automated weekly forecasts, saved models, web interface."

## Post-Presentation

- [ ] Note feedback and questions
- [ ] Identify model improvements needed
- [ ] Plan for validation (compare forecasts to actuals)
- [ ] Consider additional features requested

## Emergency Contacts

If something breaks:
1. Check logs (script prints detailed logging)
2. Try `--refresh` flag to reprocess data
3. Use `validate.py` to check data quality
4. Worst case: "Technical issue, will follow up with results"

## Time Management

- 13:30 - Final test run
- 13:40 - Copy outputs, prepare talking points
- 13:50 - Review presentation flow
- 14:00 - Showtime! 🎬

---

## Quick Reference Commands

```bash
# Find test items
python src/validate.py --sample

# Run forecast
python src/forecast.py --items "X,Y,Z" --start "2025-11-18" --end "2025-12-08"

# Force refresh
python src/forecast.py --items "X,Y,Z" --start "2025-11-18" --end "2025-12-08" --refresh

# Validate data
python src/validate.py
```

---

**You've got this! The hard work is done. Now just run it and present the results.** 💪🚀
