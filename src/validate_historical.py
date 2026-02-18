"""
Validate forecasting model against historical campaigns.

For each completed campaign in the selected year:
1. Set forecast_date = campaign_start - 4 weeks
2. Replicate main.py workflow exactly
3. Compare predictions to actual outcomes
4. Calculate metrics
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

from fetch_data import fetch_data
from preprocess import preprocess_data
from forecast import train_models, predict, predict_baseline
from config_forecast import METRIC_FUNCTIONS
from main import format_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_future_campaigns(df_items, df_orders, df_campaigns, campaign_start, df_all_campaigns, df_all_orders):
    """
    Remove items and campaigns that start AFTER the current campaign.
    This simulates what the model would have known at that point in time.
    
    Logic:
    - Items on future campaigns only → exclude
    - Their cluster friends → exclude (unless they were on past campaigns)
    - Future campaigns → exclude from df_campaigns
    
    Args:
        df_items: Items fetched for current campaign
        df_orders: Orders fetched for current campaign
        df_campaigns: Campaigns fetched for current campaign
        campaign_start: Start date of campaign being validated
        df_all_campaigns: Full campaign list (for reference)
        df_all_orders: Full order list (for reference)
    
    Returns:
        Tuple of (df_items_filtered, df_orders_filtered, df_campaigns_filtered)
    """
    logging.info("\n" + "="*80)
    logging.info("STEP 1.5: Filtering out future campaign items and campaigns")
    logging.info("="*80)
    
    # Find all campaigns that start AFTER this campaign (these are "future")
    future_campaigns = df_all_campaigns[pd.to_datetime(df_all_campaigns['campaignStart']) > campaign_start]['campaignId'].tolist()
    
    # Find items on those future campaigns
    future_campaign_items = df_all_orders[df_all_orders['salesCampignId'].isin(future_campaigns)]['itemId'].unique()
    
    # Find items that were on campaigns BEFORE or AT this campaign
    past_campaigns = df_all_campaigns[pd.to_datetime(df_all_campaigns['campaignStart']) <= campaign_start]['campaignId'].tolist()
    past_campaign_items = df_all_orders[df_all_orders['salesCampignId'].isin(past_campaigns)]['itemId'].unique()
    
    # Items to exclude: in future campaigns AND NOT in past campaigns
    items_to_exclude = set(future_campaign_items) - set(past_campaign_items)
    
    logging.info(f"  Future campaigns: {len(future_campaigns)}")
    logging.info(f"  Future campaign items: {len(future_campaign_items)}")
    logging.info(f"  Past campaign items: {len(past_campaign_items)}")
    logging.info(f"  Items to exclude (future only): {len(items_to_exclude)}")
    
    # Get clusters of items to exclude
    clusters_to_exclude = df_items[df_items['itemId'].isin(items_to_exclude)]['cluster1To1Id'].dropna().unique()
    logging.info(f"  Clusters to exclude: {len(clusters_to_exclude)}")
    
    # Find all items in those clusters
    items_in_excluded_clusters = df_items[df_items['cluster1To1Id'].isin(clusters_to_exclude)]['itemId'].unique()
    
    # BUT keep items that were on past campaigns (even if in excluded cluster)
    final_items_to_exclude = set(items_in_excluded_clusters) - set(past_campaign_items)
    
    logging.info(f"  Total items to exclude (future items + cluster friends, excluding past campaign items): {len(final_items_to_exclude)}")
    
    # Filter out these items from df_items and df_orders
    df_items_filtered = df_items[~df_items['itemId'].isin(final_items_to_exclude)].copy()
    df_orders_filtered = df_orders[~df_orders['itemId'].isin(final_items_to_exclude)].copy()
    
    # Filter out future campaigns from df_campaigns
    df_campaigns_filtered = df_campaigns[pd.to_datetime(df_campaigns['campaignStart']) <= campaign_start].copy()
    
    logging.info(f"  Items: {len(df_items)} → {len(df_items_filtered)}")
    logging.info(f"  Orders: {len(df_orders)} → {len(df_orders_filtered)}")
    logging.info(f"  Campaigns: {len(df_campaigns)} → {len(df_campaigns_filtered)}")
    
    return df_items_filtered, df_orders_filtered, df_campaigns_filtered


def run_campaign_valitation(item_ids, forecast_date, forecast_horizon_weeks, campaign_start, df_all_campaigns, df_all_orders, output_file=None):
    logging.info(f"\n{'='*80}")
    logging.info(f"Running campaign validation:")
    logging.info(f"Items: {len(item_ids)}")
    logging.info(f"Forecast date: {forecast_date.date()}")
    logging.info(f"Forecast horizon (weeks): {forecast_horizon_weeks}")
    logging.info(f"{'='*80}")

    # Step 1: Fetch data
    logging.info("\n" + "="*80)
    logging.info("STEP 1: Fetching historical data")
    logging.info("="*80)
    df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=item_ids)

    # Step 1.5: Filter out "future" campaign items and their clusters
    df_items, df_orders, df_campaigns = filter_future_campaigns(
        df_items, df_orders, df_campaigns, campaign_start,
        df_all_campaigns, df_all_orders
    )

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
    
    logging.info("✅ Forecasting complete!")
    
    # Return predictions for comparison with actuals
    return {
        'df_item_predictions': df_item_predictions,
        'df_item_baseline': df_item_baseline,
        'df_cluster_predictions': df_cluster_predictions,
        'df_cluster_baseline': df_cluster_baseline,
        'df_items': df_items,
        'item_normalizers': item_normalizers,
        'cluster_normalizers': cluster_normalizers,
        'target_items': target_items,
        'target_clusters': target_clusters
    }


def compare_with_actuals(predictions_dict, campaign_id, campaign_start, campaign_end, df_all_items, df_all_orders, df_all_campaigns):
    """
    Compare predictions to actual sales during the campaign period.
    
    Args:
        predictions_dict: Dict returned from run_campaign_valitation
        campaign_id: Campaign ID
        campaign_start: Campaign start date
        campaign_end: Campaign end date
        df_all_items: Full items dataset (unfiltered)
        df_all_orders: Full orders dataset (unfiltered)
        df_all_campaigns: Full campaigns dataset (unfiltered)
    
    Returns:
        List of dicts with metrics for each item
    """
    logging.info("\n" + "="*80)
    logging.info("STEP 7: Comparing predictions to actual sales")
    logging.info("="*80)
    
    # Get actuals by preprocessing the FULL dataset (not time-filtered)
    df_item_weekly_full, df_cluster_weekly_full, _, _, _ = preprocess_data(
        df_all_items, df_all_orders, df_all_campaigns
    )
    
    results = []
    
    df_item_predictions = predictions_dict['df_item_predictions']
    df_item_baseline = predictions_dict['df_item_baseline']
    df_cluster_predictions = predictions_dict['df_cluster_predictions']
    df_cluster_baseline = predictions_dict['df_cluster_baseline']
    df_items = predictions_dict['df_items']
    item_normalizers = predictions_dict['item_normalizers']
    cluster_normalizers = predictions_dict['cluster_normalizers']
    
    for item_id in predictions_dict['target_items']:
        # Get cluster for this item
        item_cluster = df_items[df_items['baseItemId'] == item_id]['cluster1To1Id'].values
        if len(item_cluster) == 0:
            logging.warning(f"  Item {item_id} has no cluster - skipping")
            continue
        cluster_id = item_cluster[0]
        
        # Get actual sales during campaign period
        df_item_actual = df_item_weekly_full[
            (df_item_weekly_full['baseItemId'] == item_id) &
            (df_item_weekly_full['week_start'] >= campaign_start) &
            (df_item_weekly_full['week_start'] < campaign_end)
        ]
        
        df_cluster_actual = df_cluster_weekly_full[
            (df_cluster_weekly_full['cluster1To1Id'] == cluster_id) &
            (df_cluster_weekly_full['week_start'] >= campaign_start) &
            (df_cluster_weekly_full['week_start'] < campaign_end)
        ]
        
        if len(df_item_actual) == 0:
            logging.warning(f"  Item {item_id} has no actual sales during campaign - skipping")
            continue
        
        # Denormalize actuals (reverse log1p and multiply by normalizer)
        item_normalizer = item_normalizers.get(item_id, 1000)
        cluster_normalizer = cluster_normalizers.get(cluster_id, 1000)
        
        item_actual_total = (np.expm1(df_item_actual['salesQuantityKgL_log'].values) * item_normalizer).sum()
        cluster_actual_total = (np.expm1(df_cluster_actual['salesQuantityKgL_log'].values) * cluster_normalizer).sum() if len(df_cluster_actual) > 0 else 0
        
        # Get predictions (already denormalized)
        item_pred_total = df_item_predictions[df_item_predictions['baseItemId'] == item_id]['prediction_kgl'].sum()
        item_baseline_total = df_item_baseline[df_item_baseline['baseItemId'] == item_id]['baseline_kgl'].sum()
        
        cluster_pred_total = df_cluster_predictions[df_cluster_predictions['cluster1To1Id'] == cluster_id]['prediction_kgl'].sum()
        cluster_baseline_total = df_cluster_baseline[df_cluster_baseline['cluster1To1Id'] == cluster_id]['baseline_kgl'].sum()
        
        # Calculate metrics
        item_smape = METRIC_FUNCTIONS['SMAPE'](
            np.array([item_actual_total]),
            np.array([item_pred_total])
        )
        
        cluster_smape = METRIC_FUNCTIONS['SMAPE'](
            np.array([cluster_actual_total]),
            np.array([cluster_pred_total])
        ) if cluster_pred_total > 0 else np.nan
        
        item_lift_actual = (item_actual_total / item_baseline_total - 1) * 100 if item_baseline_total > 0 else 0
        item_lift_predicted = (item_pred_total / item_baseline_total - 1) * 100 if item_baseline_total > 0 else 0
        
        cluster_lift_actual = (cluster_actual_total / cluster_baseline_total - 1) * 100 if cluster_baseline_total > 0 else 0
        cluster_lift_predicted = (cluster_pred_total / cluster_baseline_total - 1) * 100 if cluster_baseline_total > 0 else 0
        
        # Log comparison
        logging.info(f"  Item {item_id}:")
        logging.info(f"    Item SMAPE: {item_smape:.1f}% | Lift: {item_lift_predicted:.1f}% pred vs {item_lift_actual:.1f}% actual")
        logging.info(f"    Cluster SMAPE: {cluster_smape:.1f}% | Lift: {cluster_lift_predicted:.1f}% pred vs {cluster_lift_actual:.1f}% actual")
        
        results.append({
            'campaign_id': campaign_id,
            'item_id': item_id,
            'cluster_id': cluster_id,
            'campaign_start': campaign_start,
            'campaign_end': campaign_end,
            'item_actual': item_actual_total,
            'item_predicted': item_pred_total,
            'item_baseline': item_baseline_total,
            'item_smape': item_smape,
            'item_lift_actual': item_lift_actual,
            'item_lift_predicted': item_lift_predicted,
            'cluster_actual': cluster_actual_total,
            'cluster_predicted': cluster_pred_total,
            'cluster_baseline': cluster_baseline_total,
            'cluster_smape': cluster_smape,
            'cluster_lift_actual': cluster_lift_actual,
            'cluster_lift_predicted': cluster_lift_predicted,
        })
    
    logging.info(f"  Validated {len(results)} items for campaign {campaign_id}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate forecasting model against historical campaigns')
    parser.add_argument('--year', type=int, default=datetime.now().year,
                        help='Year to validate (default: current year). Only finished campaigns will be included.')
    parser.add_argument('--max-campaigns', type=int, default=None,
                        help='Maximum number of campaigns to validate (default: all)')
    parser.add_argument('--output', type=str, default='../data/output/validation_results.csv',
                        help='Output CSV file for results')
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info(f"HISTORICAL VALIDATION - {args.year} CAMPAIGNS")
    logging.info("="*80)
    
    # Fetch all data (no date restrictions)
    logging.info("\nFetching all historical data...")
    df_items, df_orders, df_campaigns = fetch_data()
    logging.info(f"Fetched {len(df_items)} items, {len(df_orders)} orders, {len(df_campaigns)} campaigns")
    
    # Collect all validation results
    all_results = []
    successful_campaigns = 0
    failed_campaigns = 0
    
    # Loop through each campaign in the selected year
    campaigns_to_validate = []
    for campaignId in df_campaigns['campaignId'].to_list():
        campaign = df_campaigns[df_campaigns['campaignId'] == campaignId].iloc[0]
        campaign_start = pd.to_datetime(campaign['campaignStart'])
        campaign_end = pd.to_datetime(campaign['campaignEnd'])
        if campaign_start.year != args.year or campaign_end.year != args.year:
            continue
        campaigns_to_validate.append((campaignId, campaign_start, campaign_end))
    
    if args.max_campaigns:
        campaigns_to_validate = campaigns_to_validate[:args.max_campaigns]
    
    logging.info(f"\nValidating {len(campaigns_to_validate)} campaigns from {args.year}")
    
    for campaignId, campaign_start, campaign_end in campaigns_to_validate:
        try:
            forecast_horizon_weeks = 4
            item_ids = df_orders[df_orders['salesCampignId'] == campaignId]['itemId'].unique().tolist()
            forecast_date = campaign_start - timedelta(weeks=4)
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Campaign {campaignId}: {len(item_ids)} items, forecast_date={forecast_date.date()}")
            logging.info(f"{'='*80}")
            
            # Run forecasting
            predictions = run_campaign_valitation(
                item_ids, forecast_date, forecast_horizon_weeks, campaign_start,
                df_all_campaigns=df_campaigns,
                df_all_orders=df_orders,
                output_file=None
            )
            
            # Compare with actuals
            campaign_results = compare_with_actuals(
                predictions, campaignId, campaign_start, campaign_end,
                df_items, df_orders, df_campaigns
            )
            
            all_results.extend(campaign_results)
            successful_campaigns += 1
            
        except Exception as e:
            logging.error(f"Validation failed for campaign {campaignId}: {e}")
            import traceback
            traceback.print_exc()
            failed_campaigns += 1
            continue
    
    # Summary statistics
    if len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        
        logging.info("\n" + "="*80)
        logging.info("VALIDATION SUMMARY")
        logging.info("="*80)
        logging.info(f"\nCampaigns validated: {successful_campaigns}/{len(campaigns_to_validate)} ({failed_campaigns} failed)")
        logging.info(f"Total items validated: {len(all_results)}")
        
        logging.info(f"\nItem-level metrics:")
        logging.info(f"  Mean SMAPE: {df_results['item_smape'].mean():.1f}%")
        logging.info(f"  Median SMAPE: {df_results['item_smape'].median():.1f}%")
        logging.info(f"  Mean Lift Error: {(df_results['item_lift_predicted'] - df_results['item_lift_actual']).mean():.1f} pp")
        
        logging.info(f"\nCluster-level metrics:")
        logging.info(f"  Mean SMAPE: {df_results['cluster_smape'].mean():.1f}%")
        logging.info(f"  Median SMAPE: {df_results['cluster_smape'].median():.1f}%")
        logging.info(f"  Mean Lift Error: {(df_results['cluster_lift_predicted'] - df_results['cluster_lift_actual']).mean():.1f} pp")
        
        # Save results
        df_results.to_csv(args.output, index=False)
        logging.info(f"\n✅ Results saved to {args.output}")
        
        # Show best and worst predictions
        logging.info(f"\nBest predictions (lowest item SMAPE):")
        best = df_results.nsmallest(5, 'item_smape')
        for _, row in best.iterrows():
            logging.info(f"  Campaign {row['campaign_id']}, Item {row['item_id']}: SMAPE={row['item_smape']:.1f}%, "
                        f"Actual={row['item_actual']:.0f} kg, Predicted={row['item_predicted']:.0f} kg")
        
        logging.info(f"\nWorst predictions (highest item SMAPE):")
        worst = df_results.nlargest(5, 'item_smape')
        for _, row in worst.iterrows():
            logging.info(f"  Campaign {row['campaign_id']}, Item {row['item_id']}: SMAPE={row['item_smape']:.1f}%, "
                        f"Actual={row['item_actual']:.0f} kg, Predicted={row['item_predicted']:.0f} kg")
    else:
        logging.warning("No successful validations")
    
    logging.info("\n" + "="*80)
    logging.info("✅ Validation complete!")
    logging.info("="*80)
  

if __name__ == '__main__':
    main()
