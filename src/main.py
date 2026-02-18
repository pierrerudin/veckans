"""
Main entry point for campaign sales forecasting.

Usage:
    python main.py --item-ids 528208,414235 --forecast-date 2024-11-18 --campaign-start 2025-01-13

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
from forecast import train_models, predict, predict_baseline
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
  # Forecast single item for campaign in 8 weeks
  python main.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13
  
  # Forecast multiple items with custom horizon
  python main.py --item-ids 528208,414235,262168 --forecast-date 2024-12-01 --campaign-start 2025-02-10
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
        required=True,
        type=str,
        help='Date when forecast is made (YYYY-MM-DD, must be before today)'
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
    try:
        forecast_date = pd.Timestamp(args.forecast_date)
    except:
        raise ValueError(f"Invalid forecast-date format: {args.forecast_date}. Use YYYY-MM-DD.")
    
    try:
        campaign_start = pd.Timestamp(args.campaign_start)
    except:
        raise ValueError(f"Invalid campaign-start format: {args.campaign_start}. Use YYYY-MM-DD.")
    
    # Validate forecast date is not in the future
    today = pd.Timestamp.now().normalize()
    if forecast_date > today:
        raise ValueError(f"Forecast date ({forecast_date.date()}) cannot be in the future. Today is {today.date()}.")
    
    # Validate campaign start is after forecast date
    if campaign_start <= forecast_date:
        raise ValueError(f"Campaign start ({campaign_start.date()}) must be after forecast date ({forecast_date.date()}).")
    
    # Calculate forecast horizon
    forecast_horizon_weeks = int((campaign_start - forecast_date).days / 7)
    
    if forecast_horizon_weeks < MIN_FORECAST_HORIZON_WEEKS:
        raise ValueError(f"Forecast horizon ({forecast_horizon_weeks} weeks) is too short. Minimum: {MIN_FORECAST_HORIZON_WEEKS} weeks.")
    
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
    Format and display forecast results with campaign effects.
    
    Parameters:
        df_item_predictions: Item-level campaign forecasts from predict()
        df_item_baseline: Item-level baseline forecasts from predict_baseline()
        df_cluster_predictions: Cluster-level campaign forecasts from predict()
        df_cluster_baseline: Cluster-level baseline forecasts from predict_baseline()
        df_items: Item master data for descriptions and unit conversions
        df_item_weekly: Item-level weekly data for recent sales
        df_cluster_weekly: Cluster-level weekly data for recent sales
        forecast_date: Date when forecast was made
        item_normalizers: Dict mapping item ID to baseline kg/L for denormalization
        cluster_normalizers: Dict mapping cluster ID to baseline kg/L for denormalization
    
    Returns:
        DataFrame with formatted results
    """
    logging.info("\n" + "="*80)
    logging.info("FORECAST RESULTS")
    logging.info("="*80)
    
    # Check if we have any forecasts
    if len(df_item_predictions) == 0 and len(df_cluster_predictions) == 0:
        logging.error("❌ No forecasts generated! Check that:")
        logging.error("   - Items have sufficient historical data")
        logging.error("   - Lag features are available at forecast date")
        logging.error("   - Campaign is 4-12 weeks in the future")
        return pd.DataFrame()
    
    # Check if we have item predictions
    if len(df_item_predictions) == 0 or 'baseItemId' not in df_item_predictions.columns:
        logging.warning("⚠️  No item-level predictions available")
        logging.warning("   Using cluster-level forecasts only")
        # TODO: Handle cluster-only forecasts
        return pd.DataFrame()
    
    # Process each item
    for item_id in df_item_predictions['baseItemId'].unique():
        item_preds = df_item_predictions[df_item_predictions['baseItemId'] == item_id]
        item_bases = df_item_baseline[df_item_baseline['baseItemId'] == item_id]
        
        # Get item info
        item_info = df_items[df_items['baseItemId'] == item_id]
        if len(item_info) > 0:
            item_desc = item_info.iloc[0]['itemDesc']
            unit_conversion = float(item_info.iloc[0]['itemUnitConversion'])  # Convert Decimal to float
            cluster_id = item_info.iloc[0]['cluster1To1Id']
        else:
            item_desc = "Unknown"
            unit_conversion = 1.0
            cluster_id = "Unknown"
        
        # Get cluster forecasts if available
        cluster_preds = df_cluster_predictions[df_cluster_predictions['cluster1To1Id'] == cluster_id]
        cluster_bases = df_cluster_baseline[df_cluster_baseline['cluster1To1Id'] == cluster_id]
        
        # Print header
        print("\n" + "="*80)
        print(f"Item {item_id} - {item_desc}")
        print(f"Cluster: {cluster_id}")
        print("="*80)
        
        # Calculate recent historical sales (last 3 weeks before forecast date)
        # Denormalize from log space back to kg/L: expm1(log) * normalizer
        item_recent = df_item_weekly[
            (df_item_weekly['baseItemId'] == item_id) & 
            (df_item_weekly['week_start'] < forecast_date)
        ].sort_values('week_start', ascending=False).head(3)
        
        cluster_recent = df_cluster_weekly[
            (df_cluster_weekly['cluster1To1Id'] == cluster_id) & 
            (df_cluster_weekly['week_start'] < forecast_date)
        ].sort_values('week_start', ascending=False).head(3)
        
        if len(item_recent) > 0:
            # Denormalize: expm1(log) gives normalized value, multiply by normalizer to get kg/L
            item_normalizer = item_normalizers.get(item_id, 1000)
            recent_weeks = item_recent['week_start'].dt.strftime('%Y-%m-%d').tolist()
            recent_sales_normalized = np.expm1(item_recent['salesQuantityKgL_log'].values)
            recent_sales_kgl = recent_sales_normalized * item_normalizer
            
            print(f"\n📊 Recent Item Sales (last 3 weeks before {forecast_date.strftime('%Y-%m-%d')}):")
            for week, sales_kgl in zip(recent_weeks, recent_sales_kgl):
                print(f"   {week}: {sales_kgl:>10,.0f} kg")
            avg_recent = recent_sales_kgl.mean()
            print(f"   Average:  {avg_recent:>10,.0f} kg/week")
        
        if len(cluster_recent) > 0:
            # Denormalize cluster sales
            cluster_normalizer = cluster_normalizers.get(cluster_id, 1000)
            recent_weeks = cluster_recent['week_start'].dt.strftime('%Y-%m-%d').tolist()
            recent_sales_normalized = np.expm1(cluster_recent['salesQuantityKgL_log'].values)
            recent_sales_kgl = recent_sales_normalized * cluster_normalizer
            
            print(f"\n📊 Recent Cluster Sales (last 3 weeks before {forecast_date.strftime('%Y-%m-%d')}):")
            for week, sales_kgl in zip(recent_weeks, recent_sales_kgl):
                print(f"   {week}: {sales_kgl:>10,.0f} kg")
            avg_recent = recent_sales_kgl.mean()
            print(f"   Average:  {avg_recent:>10,.0f} kg/week")
        
        # Calculate totals for item
        item_pred_total = item_preds['prediction_kgl'].sum()
        item_base_total = item_bases['baseline_kgl'].sum()
        item_effect = item_pred_total - item_base_total
        item_effect_pct = (item_effect / item_base_total * 100) if item_base_total > 0 else 0
        
        # Calculate totals for cluster
        cluster_pred_total = cluster_preds['prediction_kgl'].sum() if len(cluster_preds) > 0 else 0
        cluster_base_total = cluster_bases['baseline_kgl'].sum() if len(cluster_bases) > 0 else 0
        cluster_effect = cluster_pred_total - cluster_base_total
        cluster_effect_pct = (cluster_effect / cluster_base_total * 100) if cluster_base_total > 0 else 0
        
        # Print summary table
        print(f"\n{'Level':<15} {'Campaign':<10} {'Total (kg)':>15} {'Effect (kg)':>15} {'Effect (%)':>12}")
        print("-"*80)
        print(f"{'Item':<15} {'Yes':<10} {item_pred_total:>15,.0f} {item_effect:>15,.0f} {item_effect_pct:>11.1f}%")
        print(f"{'Item':<15} {'No':<10} {item_base_total:>15,.0f} {'-':>15} {'-':>12}")
        
        if len(cluster_preds) > 0:
            print(f"{'Cluster':<15} {'Yes':<10} {cluster_pred_total:>15,.0f} {cluster_effect:>15,.0f} {cluster_effect_pct:>11.1f}%")
            print(f"{'Cluster':<15} {'No':<10} {cluster_base_total:>15,.0f} {'-':>15} {'-':>12}")
        
        # Convert to units
        if unit_conversion > 0:
            item_pred_units = item_pred_total / unit_conversion
            item_base_units = item_base_total / unit_conversion
            print(f"\nItem forecast: {item_pred_units:,.0f} units (with campaign) vs {item_base_units:,.0f} units (baseline)")
        
        # Cannibalization analysis
        if len(cluster_preds) > 0 and cluster_base_total > 0:
            print(f"\n{'Cannibalization Analysis':<30}")
            print("-"*40)
            
            item_share_with = (item_pred_total / cluster_pred_total * 100) if cluster_pred_total > 0 else 0
            item_share_without = (item_base_total / cluster_base_total * 100) if cluster_base_total > 0 else 0
            share_change = item_share_with - item_share_without
            
            print(f"Item share of cluster: {item_share_with:.1f}% (with campaign) vs {item_share_without:.1f}% (baseline)")
            print(f"Share change: {share_change:+.1f} percentage points")
            
            # Detect cannibalization or halo effect
            if item_effect > 0 and cluster_effect > 0:
                if item_effect > cluster_effect * 0.8:
                    print(f"✅ HALO EFFECT: Item drives most of cluster growth")
                else:
                    print(f"✅ POSITIVE: Both item and cluster grow")
            elif item_effect > 0 and cluster_effect < 0:
                print(f"⚠️  CANNIBALIZATION: Item gains at expense of cluster")
            elif item_effect < 0:
                print(f"❌ NEGATIVE: Item sales decline with campaign")
    
    logging.info("\n" + "="*80 + "\n")
    
    # Return combined results
    return df_item_predictions


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

        # list all itemids in item_ids, with counts of rows in df_orders along the earliest and last order date for each itemid
        # No need to make it fancy, just a quick check
        # Just so you know, df_orders has columns: orderDate, deliveryDate, itemId, itemSkey, salesQuantity and salesCampaignId 
        # So you need to do a row count for each itemId and add the min and max orderDate for each itemId

        # TESTING ONLY - COMMENT OUT LATER OR UNCOMMENT IF NEEDED

        print(df_items.loc[df_items['baseItemId'].isin(item_ids)][['itemId', 'baseItemId', 'itemDesc', 'itemSkey','weightKgPreparedItemComparisonUnit', 'netWeightKgComparisonUnit', 'grossWeightKgComparisonUnit']].sort_values('itemSkey', ascending=False).drop_duplicates(subset=['baseItemId']).to_string(index=False))
        

        items = df_items.loc[df_items['baseItemId'].isin(item_ids)][['itemId', 'baseItemId', 'itemDesc', 'itemSkey', 'cluster1To1Id', 'itemRegDate']]
        # For each item in items get the version with the highest itemSkey and print the most recent versions of each item
        items = items.sort_values('itemSkey', ascending=False).drop_duplicates(subset=['baseItemId'])
        logging.info(f"\nItem details for {len(item_ids)} items:\n{items.to_string(index=False)}")

        item_summary = df_orders[df_orders['itemId'].isin(item_ids)].groupby('itemId').agg(
            row_count=('itemId', 'size'),
            earliest_order=('orderDate', 'min'),
            last_order=('orderDate', 'max')
        ).reset_index()
        logging.info(f"\nItem summary for {len(item_ids)} items:\n{item_summary.to_string()}")

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
        
        # Step 3: Train item models
        logging.info("\n" + "="*80)
        logging.info("STEP 3: Training item models")
        logging.info("="*80)
        item_models = train_models(
            df_item_weekly, forecast_horizon_weeks, item_normalizers,
            level='item', target_ids=target_items
        )
        
        # Step 4: Train cluster models
        logging.info("\n" + "="*80)
        logging.info("STEP 4: Training cluster models")
        logging.info("="*80)
        cluster_models = train_models(
            df_cluster_weekly, forecast_horizon_weeks, cluster_normalizers,
            level='cluster'
        )
        
        # Step 5: Forecast items
        logging.info("\n" + "="*80)
        logging.info("STEP 5: Forecasting items")
        logging.info("="*80)
        df_item_predictions = predict(item_models, df_item_weekly, target_items, forecast_date, campaign_start, level='item')
        df_item_baseline = predict_baseline(df_item_weekly, target_items, forecast_date, campaign_start, item_normalizers, level='item')
        
        # Step 6: Forecast clusters
        logging.info("\n" + "="*80)
        logging.info("STEP 6: Forecasting clusters")
        logging.info("="*80)
        df_cluster_predictions = predict(cluster_models, df_cluster_weekly, target_clusters, forecast_date, campaign_start, level='cluster')
        df_cluster_baseline = predict_baseline(df_cluster_weekly, target_clusters, forecast_date, campaign_start, cluster_normalizers, level='cluster')
        
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
