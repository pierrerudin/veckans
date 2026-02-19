"""
Main entry point for campaign sales forecasting.

Usage:
    python main.py --item-ids 528208,414235 --campaign-start 2025-01-13
    python main.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13

This script orchestrates the complete forecasting workflow:
1. Validate inputs
2. Fetch historical data
3. Preprocess data with correct forecast horizon
4. Train forecasting models
5. Generate predictions
6. Present results with campaign effects
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from fetch_data import fetch_data
from preprocess import preprocess_data
from forecast import train_models, train_models_single, predict
from config_forecast import (
    DEFAULT_FORECAST_HORIZON_WEEKS,
    MAX_FORECAST_HORIZON_WEEKS,
    MIN_FORECAST_HORIZON_WEEKS,
    EXCLUDE_SINGLE_ITEM_CLUSTERS_FROM_FORECAST
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Campaign Sales Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forecast using today as forecast date (default)
  python main.py --item-ids 528208 --campaign-start 2025-03-10

  # Forecast as if making decision on a past date (backtesting)
  python main.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13

  # Forecast multiple items
  python main.py --item-ids 528208,414235,262168 --campaign-start 2025-02-10
        """
    )
    
    parser.add_argument(
        '--item-ids',
        required=True,
        type=str,
        help='Comma-separated list of item IDs to forecast (e.g., "528208,414235")'
    )
    
    parser.add_argument(
        '--forecast-date',
        type=str,
        default=None,
        help='Date when forecast is made (YYYY-MM-DD). Defaults to today. Only data before this date is used.'
    )
    
    parser.add_argument(
        '--campaign-start',
        required=True,
        type=str,
        help='Campaign start date (YYYY-MM-DD, must be after forecast-date). Week alignment will use this day of week.'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Save results to CSV file (optional)'
    )

    parser.add_argument(
        '--single-model',
        action='store_true',
        default=False,
        help='Use a single model for all 3 campaign weeks (instead of 3 separate models)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """
    Validate all input parameters.
    
    Campaign is always 3 weeks (21 days).
    Week alignment uses the campaign start day-of-week.
    
    Returns:
        Tuple of (item_ids, forecast_date, campaign_start_date, campaign_end_date)
    
    Raises:
        ValueError if validation fails
    """
    logging.info("Validating inputs...")
    
    # Parse item IDs
    item_ids = [item_id.strip() for item_id in args.item_ids.split(',')]
    if not item_ids or any(not item_id for item_id in item_ids):
        raise ValueError("Invalid item IDs. Provide comma-separated list of non-empty IDs.")
    
    logging.info(f"  Item IDs: {item_ids}")
    
    # Parse dates
    today = pd.Timestamp.now().normalize()

    if args.forecast_date is not None:
        try:
            forecast_date = pd.Timestamp(args.forecast_date)
        except:
            raise ValueError(f"Invalid forecast-date format: {args.forecast_date}. Use YYYY-MM-DD.")
        if forecast_date > today:
            raise ValueError(f"Forecast date ({forecast_date.date()}) cannot be in the future. Today is {today.date()}.")
    else:
        forecast_date = today

    try:
        campaign_start = pd.Timestamp(args.campaign_start)
    except:
        raise ValueError(f"Invalid campaign-start format: {args.campaign_start}. Use YYYY-MM-DD.")

    # Validate campaign start is after forecast date
    if campaign_start <= forecast_date:
        raise ValueError(f"Campaign start ({campaign_start.date()}) must be after forecast date ({forecast_date.date()}).")

    # Calculate forecast horizon
    forecast_horizon_weeks = int((campaign_start - forecast_date).days / 7)

    if forecast_horizon_weeks > MAX_FORECAST_HORIZON_WEEKS:
        raise ValueError(f"Forecast horizon ({forecast_horizon_weeks} weeks) is too long. Maximum: {MAX_FORECAST_HORIZON_WEEKS} weeks.")
    
    # Campaign is always 3 weeks (21 days)
    campaign_end = campaign_start + timedelta(days=20)  # 21 days inclusive
    
    # Get campaign start day name for logging
    day_name = campaign_start.strftime('%A')
    
    logging.info(f"  Forecast date: {forecast_date.date()}")
    logging.info(f"  Campaign period: {campaign_start.date()} to {campaign_end.date()} (3 weeks)")
    logging.info(f"  Week alignment: {day_name} to {pd.Timestamp(campaign_start + timedelta(days=6)).strftime('%A')}")
    logging.info(f"  Forecast horizon: {forecast_horizon_weeks} weeks")
    logging.info("  ✅ All inputs valid")
    
    return item_ids, forecast_date, campaign_start, campaign_end, forecast_horizon_weeks


def format_results(df_item_predictions, df_item_baseline, df_cluster_predictions, df_cluster_baseline,
                  df_items, df_item_weekly, df_cluster_weekly, forecast_date,
                  item_normalizers, cluster_normalizers):
    """
    Format and display forecast results as a clean summary table.

    Returns:
        DataFrame with one row per item, suitable for CSV export.
    """
    if len(df_item_predictions) == 0 and len(df_cluster_predictions) == 0:
        logging.error("No forecasts generated.")
        return pd.DataFrame()

    if len(df_item_predictions) == 0 or 'baseItemId' not in df_item_predictions.columns:
        logging.warning("No item-level predictions available")
        return pd.DataFrame()

    rows = []

    for item_id in df_item_predictions['baseItemId'].unique():
        item_preds = df_item_predictions[df_item_predictions['baseItemId'] == item_id]
        item_bases = df_item_baseline[df_item_baseline['baseItemId'] == item_id]

        # Get item metadata
        item_info = df_items[df_items['baseItemId'] == item_id]
        item_desc = item_info.iloc[0]['itemDesc'] if len(item_info) > 0 else "Unknown"
        unit_conv = float(item_info.iloc[0]['itemUnitConversion']) if len(item_info) > 0 else 1.0
        cluster_id = item_info.iloc[0]['cluster1To1Id'] if len(item_info) > 0 else "Unknown"

        # Get cluster forecasts
        cluster_preds = df_cluster_predictions[df_cluster_predictions['cluster1To1Id'] == cluster_id] if len(df_cluster_predictions) > 0 else pd.DataFrame()
        cluster_bases = df_cluster_baseline[df_cluster_baseline['cluster1To1Id'] == cluster_id] if len(df_cluster_baseline) > 0 else pd.DataFrame()

        # Get cluster size
        cluster_rows = df_cluster_weekly[df_cluster_weekly['cluster1To1Id'] == cluster_id]
        num_items = int(cluster_rows['num_items_in_cluster'].iloc[0]) if len(cluster_rows) > 0 and 'num_items_in_cluster' in cluster_rows.columns else 1

        row = {'item_id': item_id, 'description': item_desc, 'cluster_id': cluster_id, 'cluster_size': num_items}

        # Per-week item forecasts (convert from kg/L back to sales units)
        for wk in [1, 2, 3]:
            wk_pred = item_preds[item_preds['campaign_week'] == wk]
            wk_base = item_bases[item_bases['campaign_week'] == wk]
            pred_kgl = wk_pred['prediction_kgl'].values[0] if len(wk_pred) > 0 else 0
            base_kgl = wk_base['prediction_kgl'].values[0] if len(wk_base) > 0 else 0
            row[f'item_w{wk}_campaign'] = pred_kgl / unit_conv if unit_conv > 0 else 0
            row[f'item_w{wk}_baseline'] = base_kgl / unit_conv if unit_conv > 0 else 0

        # Item totals (already in sales units)
        row['item_campaign_total'] = sum(row[f'item_w{wk}_campaign'] for wk in [1, 2, 3])
        row['item_baseline_total'] = sum(row[f'item_w{wk}_baseline'] for wk in [1, 2, 3])
        row['item_effect'] = row['item_campaign_total'] - row['item_baseline_total']
        row['item_effect_pct'] = (row['item_effect'] / row['item_baseline_total'] * 100) if row['item_baseline_total'] > 0 else 0

        # Cluster totals (kg/L — clusters mix different items so can't convert to one unit)
        row['cluster_campaign_total'] = cluster_preds['prediction_kgl'].sum() if len(cluster_preds) > 0 else 0
        row['cluster_baseline_total'] = cluster_bases['prediction_kgl'].sum() if len(cluster_bases) > 0 else 0
        row['cluster_effect_kg'] = row['cluster_campaign_total'] - row['cluster_baseline_total']
        row['cluster_effect_pct'] = (row['cluster_effect_kg'] / row['cluster_baseline_total'] * 100) if row['cluster_baseline_total'] > 0 else 0

        # Cannibalization verdict
        # For singleton clusters the item IS the cluster, so comparing item vs cluster
        # effect is meaningless (two different models predicting the same entity).
        item_eff = row['item_effect']
        cluster_eff = row['cluster_effect_kg']
        if num_items <= 1:
            if item_eff > 0:
                row['verdict'] = 'Positive (sole)'
            elif item_eff < 0:
                row['verdict'] = 'Negative (sole)'
            else:
                row['verdict'] = 'Neutral (sole)'
        elif item_eff > 0 and cluster_eff > 0:
            if item_eff > cluster_eff * 0.8:
                row['verdict'] = 'Halo'
            else:
                row['verdict'] = 'Positive'
        elif item_eff > 0 and cluster_eff < 0:
            row['verdict'] = 'Cannibalization'
        elif item_eff < 0:
            row['verdict'] = 'Negative'
        else:
            row['verdict'] = 'Neutral'

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # Print formatted table
    print("\n" + "=" * 168)
    print(f"CAMPAIGN FORECAST SUMMARY  (forecast date: {forecast_date.strftime('%Y-%m-%d')})")
    print("=" * 168)

    # Item detail table: per-week breakdown (in sales units)
    print(f"\n{'Item ID':<10} {'Description':<30} "
          f"{'W1 Camp':>9} {'W1 Base':>9} "
          f"{'W2 Camp':>9} {'W2 Base':>9} "
          f"{'W3 Camp':>9} {'W3 Base':>9} "
          f"{'Total C':>10} {'Total B':>10} {'Effect%':>8}")
    print("-" * 168)

    for _, r in df_summary.iterrows():
        print(f"{r['item_id']:<10} {str(r['description'])[:29]:<30} "
              f"{r['item_w1_campaign']:>9,.0f} {r['item_w1_baseline']:>9,.0f} "
              f"{r['item_w2_campaign']:>9,.0f} {r['item_w2_baseline']:>9,.0f} "
              f"{r['item_w3_campaign']:>9,.0f} {r['item_w3_baseline']:>9,.0f} "
              f"{r['item_campaign_total']:>10,.0f} {r['item_baseline_total']:>10,.0f} "
              f"{r['item_effect_pct']:>7.1f}%")

    # Summary table: item in sales units, cluster in kg/L (mixed items can't share a unit)
    print(f"\n{'Item ID':<10} {'Description':<30} "
          f"{'Camp(units)':>12} {'Base(units)':>12} {'Effect':>10} {'Item%':>7} "
          f"{'#Clust':>6} {'Camp(kgL)':>12} {'Base(kgL)':>12} {'Clust%':>8} "
          f"{'Verdict':<18}")
    print("-" * 168)

    for _, r in df_summary.iterrows():
        print(f"{r['item_id']:<10} {str(r['description'])[:29]:<30} "
              f"{r['item_campaign_total']:>12,.0f} {r['item_baseline_total']:>12,.0f} "
              f"{r['item_effect']:>10,.0f} {r['item_effect_pct']:>6.1f}% "
              f"{r['cluster_size']:>6} "
              f"{r['cluster_campaign_total']:>12,.0f} {r['cluster_baseline_total']:>12,.0f} "
              f"{r['cluster_effect_pct']:>7.1f}% "
              f"{r['verdict']:<18}")

    # Grand totals (item-level only — can't sum units across different items meaningfully)
    print("-" * 168)
    t_ic = df_summary['item_campaign_total'].sum()
    t_ib = df_summary['item_baseline_total'].sum()
    t_ie = t_ic - t_ib
    t_ip = (t_ie / t_ib * 100) if t_ib > 0 else 0
    print(f"{'TOTAL':<10} {'(mixed units — indicative only)':<30} "
          f"{t_ic:>12,.0f} {t_ib:>12,.0f} "
          f"{t_ie:>10,.0f} {t_ip:>6.1f}% "
          f"{'':>6}")
    print("=" * 168)

    return df_summary


def main():
    """Main execution function."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        item_ids, forecast_date, campaign_start, campaign_end, forecast_horizon_weeks = validate_inputs(args)

        # Step 1: Fetch data
        logging.info("\n" + "="*80)
        logging.info("STEP 1: Fetching historical data")
        logging.info("="*80)
        df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=item_ids)

        # Step 2: Preprocess
        logging.info("\n" + "="*80)
        logging.info("STEP 2: Preprocessing data")
        logging.info("="*80)
        df_item_weekly, df_cluster_weekly, df_items, item_normalizers, cluster_normalizers = preprocess_data(
            df_items, df_orders, df_campaigns,
            forecast_horizon_weeks=forecast_horizon_weeks,
            campaign_start_date=campaign_start,
            forecast_date=forecast_date,
            forecast_item_ids=item_ids  # Keep target items even if not in historical campaigns
        )
        # Get target IDs
        target_items = item_ids
        target_clusters = df_items[df_items['baseItemId'].isin(item_ids)]['cluster1To1Id'].unique().tolist()

        # Build mapping: how many of the forecasted items are in each cluster
        # This is needed for correct cluster campaign intensity and item-level competition features
        campaign_items_per_cluster = (
            df_items[df_items['baseItemId'].isin(item_ids)]
            .groupby('cluster1To1Id')['baseItemId']
            .nunique()
            .to_dict()
        )
        # Total cluster sizes (all items, not just those on campaign)
        cluster_sizes = (
            df_items
            .groupby('cluster1To1Id')['baseItemId']
            .nunique()
            .to_dict()
        )
        for cid, n in campaign_items_per_cluster.items():
            if n > 1:
                total = cluster_sizes.get(cid, n)
                logging.info(f"  Cluster {cid}: {n}/{total} items on campaign")
        
        # Step 3: Train item models
        logging.info("\n" + "="*80)
        train_fn = train_models_single if args.single_model else train_models
        logging.info(f"STEP 3: Training item models ({'single' if args.single_model else '3 separate'})")
        logging.info("="*80)
        item_models = train_fn(
            df_item_weekly, forecast_horizon_weeks, item_normalizers,
            level='item', target_ids=target_items
        )

        # Step 4: Train cluster models
        logging.info("\n" + "="*80)
        logging.info(f"STEP 4: Training cluster models ({'single' if args.single_model else '3 separate'})")
        logging.info("="*80)
        cluster_models = train_fn(
            df_cluster_weekly, forecast_horizon_weeks, cluster_normalizers,
            level='cluster'
        )
        
        # Step 5: Forecast items
        logging.info("\n" + "="*80)
        logging.info("STEP 5: Forecasting items")
        logging.info("="*80)
        df_item_predictions = predict(item_models, df_item_weekly, target_items, forecast_date, campaign_start, level='item', campaign_on=True, campaign_items_per_cluster=campaign_items_per_cluster, cluster_sizes=cluster_sizes)
        df_item_baseline = predict(item_models, df_item_weekly, target_items, forecast_date, campaign_start, level='item', campaign_on=False)

        # Step 6: Forecast clusters
        logging.info("\n" + "="*80)
        logging.info("STEP 6: Forecasting clusters")
        logging.info("="*80)
        df_cluster_predictions = predict(cluster_models, df_cluster_weekly, target_clusters, forecast_date, campaign_start, level='cluster', campaign_on=True, campaign_items_per_cluster=campaign_items_per_cluster)
        df_cluster_baseline = predict(cluster_models, df_cluster_weekly, target_clusters, forecast_date, campaign_start, level='cluster', campaign_on=False)
        
        # Step 7: Format and display results
        logging.info("\n" + "="*80)
        logging.info("STEP 7: Presenting results")
        logging.info("="*80)
        results = format_results(
            df_item_predictions, df_item_baseline,
            df_cluster_predictions, df_cluster_baseline,
            df_items, df_item_weekly, df_cluster_weekly, forecast_date,
            item_normalizers, cluster_normalizers
        )
        
        # Save to file if requested
        if args.output_file:
            results.to_csv(args.output_file, index=False)
            logging.info(f"Results saved to {args.output_file}")
        
        logging.info("✅ Forecasting complete!")
        
    except ValueError as e:
        logging.error(f"❌ Input validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
