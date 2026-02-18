"""
Campaign Forecast Script
Generates forecasts for item and cluster sales during campaigns,
including counterfactual (no-campaign) scenarios.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from preprocess import preprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_or_preprocess_data(item_ids: list[str], force_refresh: bool = False):
    """
    Load preprocessed data or run preprocessing if needed.
    
    Parameters:
        item_ids: List of item IDs to forecast
        force_refresh: If True, force reprocessing even if cached data exists
    
    Returns:
        df_orders, df_items
    """
    orders_path = Path("../data/processed/orders_processed.parquet")
    items_path = Path("../data/processed/items_processed.parquet")
    
    if not force_refresh and orders_path.exists() and items_path.exists():
        logging.info("Loading preprocessed data from cache...")
        df_orders = pd.read_parquet(orders_path)
        df_items = pd.read_parquet(items_path)
        
        # Check if requested items exist in cache
        existing_items = set(df_orders['itemId'].unique())
        requested_items = set(item_ids)
        missing_items = requested_items - existing_items
        
        if missing_items:
            logging.info(f"Found {len(missing_items)} new items not in cache. Reprocessing...")
            preprocess(item_ids=item_ids)
            df_orders = pd.read_parquet(orders_path)
            df_items = pd.read_parquet(items_path)
    else:
        logging.info("Preprocessed data not found. Running preprocessing pipeline...")
        preprocess(item_ids=item_ids)
        df_orders = pd.read_parquet(orders_path)
        df_items = pd.read_parquet(items_path)
    
    return df_orders, df_items


def filter_completed_campaigns(df_orders: pd.DataFrame, cutoff_date: pd.Timestamp = None):
    """
    Filter out ongoing campaigns - only keep campaigns that have ended.
    
    Parameters:
        df_orders: DataFrame with order data
        cutoff_date: Date before which campaigns must have ended (default: today)
    
    Returns:
        Filtered DataFrame
    """
    if cutoff_date is None:
        cutoff_date = pd.Timestamp.now()
    
    logging.info(f"Filtering to campaigns completed before {cutoff_date.date()}")
    
    # For training, we only want data up to the cutoff
    df_filtered = df_orders[df_orders['orderDate'] < cutoff_date].copy()
    
    logging.info(f"Filtered from {len(df_orders)} to {len(df_filtered)} rows")
    return df_filtered


def prepare_features(df: pd.DataFrame, is_training: bool = True):
    """
    Prepare features for modeling.
    
    Parameters:
        df: DataFrame with order data
        is_training: If True, this is training data. If False, this is prediction data.
    
    Returns:
        DataFrame with features ready for modeling
    """
    df = df.copy()
    
    # Add cyclical time features (week of year)
    if 'week' in df.columns:
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    # Add cyclical month features
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Ensure we have all necessary features
    required_features = [
        'week', 'month', 'year',
        'week_sin', 'week_cos',
        'month_sin', 'month_cos',
        'is_campaign_week',
        'cluster_sales_lag_1',
        'cluster_sales_rolling4w',
        'weeks_since_campaign',
        'campaigns_ytd'
    ]
    
    for feat in required_features:
        if feat not in df.columns:
            logging.warning(f"Missing feature: {feat}. Setting to 0.")
            df[feat] = 0
    
    # Fill any NaN values
    df = df.fillna(0)
    
    return df


def build_model_data(df_orders: pd.DataFrame, item_id: str, cluster_id: str):
    """
    Build training dataset for a specific item and its cluster.
    
    Parameters:
        df_orders: Full order DataFrame
        item_id: Item to forecast
        cluster_id: Cluster the item belongs to
    
    Returns:
        df_item_train: Training data for item-level model
        df_cluster_train: Training data for cluster-level model
        df_global_train: Training data for global model (all items)
    """
    # Item-level training data
    df_item_train = df_orders[df_orders['itemId'] == item_id].copy()
    df_item_train = df_item_train.sort_values('orderDate')
    
    # Log recent item sales to understand the scale
    recent_sales_summary = df_item_train.tail(12)['salesQuantityKgL'].describe()
    logging.info(f"Item recent sales (last 12 weeks): mean={recent_sales_summary['mean']:.1f} kg/L, "
                 f"median={recent_sales_summary['50%']:.1f}, max={recent_sales_summary['max']:.1f}")
    logging.info(f"Item training data: {len(df_item_train)} weeks of history")
    
    # Check if this looks like aggregated data (suspiciously high for a single SKU)
    if recent_sales_summary['mean'] > 10000:
        logging.warning(f"⚠️  Item sales seem very high ({recent_sales_summary['mean']:.0f} kg/week).")
        logging.warning(f"   This might be aggregated data across multiple stores/SKUs!")
        logging.warning(f"   Checking data structure...")
        
        # Check if there are multiple rows per week
        weeks_with_multiple_rows = df_item_train.groupby('orderDate').size()
        if (weeks_with_multiple_rows > 1).any():
            logging.warning(f"   Found multiple rows per week! Data might not be aggregated properly.")
            logging.warning(f"   Example: {weeks_with_multiple_rows[weeks_with_multiple_rows > 1].head()}")
    
    # Add item-specific lag features
    df_item_train['item_sales_lag_1'] = df_item_train['salesQuantityKgL'].shift(1)
    df_item_train['item_sales_lag_2'] = df_item_train['salesQuantityKgL'].shift(2)
    df_item_train['item_sales_rolling_4w'] = df_item_train['salesQuantityKgL'].shift(1).rolling(window=4, min_periods=1).mean()
    
    # Cluster-level training data (aggregate all items in cluster by week)
    df_cluster = df_orders[df_orders['cluster1To1Id'] == cluster_id].copy()
    
    # Log cluster composition
    items_in_cluster = df_cluster['itemId'].nunique()
    cluster_recent_avg = df_cluster[df_cluster['orderDate'] >= df_cluster['orderDate'].max() - pd.Timedelta(weeks=12)].groupby('orderDate')['salesQuantityKgL'].sum().mean()
    logging.info(f"Cluster composition: {items_in_cluster} unique items")
    logging.info(f"Cluster recent weekly avg: {cluster_recent_avg:.1f} kg/L")
    
    # First aggregate to get total cluster sales per week
    df_cluster_train = (
        df_cluster
        .groupby(['orderDate', 'week', 'month', 'year'])
        .agg({
            'salesQuantityKgL': 'sum',
            'is_campaign_week': 'max',  # If any item in cluster is on campaign
            'weeks_since_campaign': 'min',
            'campaigns_ytd': 'max'
        })
        .reset_index()
    )
    
    # Sort by date and calculate proper cluster-level lag features
    df_cluster_train = df_cluster_train.sort_values('orderDate')
    df_cluster_train['cluster_sales_lag_1'] = df_cluster_train['salesQuantityKgL'].shift(1).fillna(0)
    df_cluster_train['cluster_sales_rolling4w'] = (
        df_cluster_train['salesQuantityKgL']
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
        .fillna(0)
    )
    
    # GLOBAL training data (aggregate ALL items by week)
    df_global_train = (
        df_orders
        .groupby(['orderDate', 'week', 'month', 'year'])
        .agg({
            'salesQuantityKgL': 'sum',
            'is_campaign_week': 'max',  # If ANY item is on campaign
            'weeks_since_campaign': 'min',
            'campaigns_ytd': 'max'
        })
        .reset_index()
    )
    
    # Sort by date and calculate proper global-level lag features
    df_global_train = df_global_train.sort_values('orderDate')
    df_global_train['cluster_sales_lag_1'] = df_global_train['salesQuantityKgL'].shift(1).fillna(0)
    df_global_train['cluster_sales_rolling4w'] = (
        df_global_train['salesQuantityKgL']
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
        .fillna(0)
    )
    
    return df_item_train, df_cluster_train, df_global_train


def train_simple_model(df_train: pd.DataFrame, target_col: str = 'salesQuantityKgL'):
    """
    Train a simple LightGBM model for forecasting.
    
    Parameters:
        df_train: Training data
        target_col: Name of target column
    
    Returns:
        Trained model and feature names
    """
    from lightgbm import LGBMRegressor
    
    feature_cols = [
        # REMOVED: 'week', 'month', 'year' - redundant with cyclical encoding
        'week_sin', 'week_cos',         # Cyclical week
        'month_sin', 'month_cos',       # Cyclical month
        'is_campaign_week',             # KEY FEATURE!
        'item_sales_lag_1',             # Item's own recent sales
        'item_sales_lag_2',             # Item's sales 2 weeks ago
        'item_sales_rolling_4w',        # Item's 4-week trend
        'cluster_sales_lag_1',          # Cluster lag
        'cluster_sales_rolling4w',      # Cluster trend
        'weeks_since_campaign',         # Campaign recency
        'campaigns_ytd'                 # Campaign frequency
    ]
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df_train.columns:
            df_train[col] = 0
    
    X = df_train[feature_cols].fillna(0)
    y = df_train[target_col].fillna(0)
    
    # Model with stronger regularization to prevent overfitting to seasonality
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=4,                # Reduced from 5
        learning_rate=0.05,         # Reduced from 0.1 for better generalization
        min_child_samples=10,       # Increased from 5
        reg_alpha=0.5,              # Increased from 0.1
        reg_lambda=0.5,             # Increased from 0.1
        feature_fraction=0.8,       # Add feature sampling
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    # Log feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info(f"Top 5 feature importance:\n{importance.head().to_string(index=False)}")
    
    return model, feature_cols


def create_prediction_weeks(campaign_start: str, campaign_end: str):
    """
    Create a DataFrame with the three campaign weeks.
    
    Parameters:
        campaign_start: Start date (YYYY-MM-DD)
        campaign_end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with weekly periods
    """
    start = pd.to_datetime(campaign_start)
    end = pd.to_datetime(campaign_end)
    
    # Align to Monday
    start = start - pd.Timedelta(days=start.weekday())
    
    weeks = []
    current = start
    week_num = 1
    
    while current <= end and week_num <= 3:
        week_of_year = current.isocalendar().week
        month = current.month
        
        weeks.append({
            'campaign_week': week_num,
            'orderDate': current,
            'week': week_of_year,
            'month': month,
            'year': current.year,
            'week_sin': np.sin(2 * np.pi * week_of_year / 52),
            'week_cos': np.cos(2 * np.pi * week_of_year / 52),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        })
        current += pd.Timedelta(days=7)
        week_num += 1
    
    return pd.DataFrame(weeks)


def get_last_year_sales(
    df_orders: pd.DataFrame,
    item_id: str,
    cluster_id: str,
    campaign_start: str,
    campaign_end: str
):
    """
    Get actual sales from the same period last year for comparison.
    
    Parameters:
        df_orders: Historical order data
        item_id: Item ID
        cluster_id: Cluster ID
        campaign_start: Campaign start date
        campaign_end: Campaign end date
    
    Returns:
        Dictionary with last year's sales data
    """
    start = pd.to_datetime(campaign_start)
    end = pd.to_datetime(campaign_end)
    
    # Calculate same period last year
    ly_start = start - pd.DateOffset(years=1)
    ly_end = end - pd.DateOffset(years=1)
    
    # Align to Monday
    ly_start = ly_start - pd.Timedelta(days=ly_start.weekday())
    ly_end = ly_end - pd.Timedelta(days=ly_end.weekday())
    
    # Get item sales last year
    df_item_ly = df_orders[
        (df_orders['itemId'] == item_id) &
        (df_orders['orderDate'] >= ly_start) &
        (df_orders['orderDate'] <= ly_end)
    ].copy()
    
    # Get cluster sales last year
    df_cluster_ly = df_orders[
        (df_orders['cluster1To1Id'] == cluster_id) &
        (df_orders['orderDate'] >= ly_start) &
        (df_orders['orderDate'] <= ly_end)
    ].copy()
    
    if len(df_item_ly) == 0:
        logging.warning(f"No data found for item {item_id} in same period last year")
        return None
    
    # Aggregate by week
    df_item_ly['week_num'] = ((df_item_ly['orderDate'] - ly_start).dt.days // 7) + 1
    item_weekly = df_item_ly.groupby('week_num')['salesQuantityKgL'].sum()
    
    df_cluster_ly = df_cluster_ly.groupby('orderDate')['salesQuantityKgL'].sum().reset_index()
    df_cluster_ly['week_num'] = ((df_cluster_ly['orderDate'] - ly_start).dt.days // 7) + 1
    cluster_weekly = df_cluster_ly.groupby('week_num')['salesQuantityKgL'].sum()
    
    results = {
        'week_1_item': item_weekly.get(1, 0),
        'week_2_item': item_weekly.get(2, 0),
        'week_3_item': item_weekly.get(3, 0),
        'week_1_cluster': cluster_weekly.get(1, 0),
        'week_2_cluster': cluster_weekly.get(2, 0),
        'week_3_cluster': cluster_weekly.get(3, 0),
    }
    
    return results


def forecast_item(
    df_orders: pd.DataFrame,
    item_id: str,
    cluster_id: str,
    campaign_start: str,
    campaign_end: str
):
    """
    Generate forecasts for an item during a campaign period.
    
    Parameters:
        df_orders: Historical order data
        item_id: Item to forecast
        cluster_id: Cluster the item belongs to
        campaign_start: Campaign start date
        campaign_end: Campaign end date
    
    Returns:
        Dictionary with forecast results
    """
    logging.info(f"Forecasting item {item_id} in cluster {cluster_id}")
    
    # Build training data
    df_item_train, df_cluster_train, df_global_train = build_model_data(df_orders, item_id, cluster_id)
    
    if len(df_item_train) < 10:
        logging.warning(f"Insufficient training data for item {item_id} ({len(df_item_train)} rows)")
        return None
    
    # Check campaign history at all levels
    item_campaign_weeks = df_item_train[df_item_train['is_campaign_week'] == 1]
    cluster_campaign_weeks = df_cluster_train[df_cluster_train['is_campaign_week'] == 1]
    global_campaign_weeks = df_global_train[df_global_train['is_campaign_week'] == 1]
    
    logging.info(f"Campaign data available:")
    logging.info(f"  Item level: {len(item_campaign_weeks)} weeks")
    logging.info(f"  Cluster level: {len(cluster_campaign_weeks)} weeks")
    logging.info(f"  Global level: {len(global_campaign_weeks)} weeks (ALL items)")
    
    # Decide which level to use for campaign effect
    use_fallback = False
    fallback_level = None
    min_campaign_weeks_required = 3
    
    if len(item_campaign_weeks) < min_campaign_weeks_required:
        use_fallback = True
        if len(cluster_campaign_weeks) >= 10:
            fallback_level = 'cluster'
            logging.warning(f"⚠️  Item has insufficient campaign history. Using CLUSTER patterns.")
        else:
            fallback_level = 'global'
            logging.warning(f"⚠️  Item AND cluster have insufficient campaign history. Using GLOBAL patterns (all items).")
            logging.warning(f"   Rhode Island Dressing learns from milk, pork chops, and everything else! 🎯")
    
    # Train models
    logging.info("Training item-level model...")
    item_model, item_features = train_simple_model(df_item_train)
    
    logging.info("Training cluster-level model...")
    cluster_model, cluster_features = train_simple_model(df_cluster_train)
    
    logging.info("Training global-level model (all items)...")
    global_model, global_features = train_simple_model(df_global_train)
    
    # Create prediction weeks
    df_pred = create_prediction_weeks(campaign_start, campaign_end)
    
    # Get initial context for lag features (from training data)
    recent_item = df_item_train.tail(4)
    recent_cluster = df_cluster_train.tail(4)
    
    # Initialize with historical values
    last_item_sales = list(recent_item['salesQuantityKgL'].values) if len(recent_item) > 0 else [0, 0, 0, 0]
    last_cluster_sales = list(recent_cluster['salesQuantityKgL'].values) if len(recent_cluster) > 0 else [0, 0, 0, 0]
    
    # Get baseline campaigns_ytd
    base_campaigns_ytd = df_item_train['campaigns_ytd'].max() if 'campaigns_ytd' in df_item_train.columns else 0
    
    # Predict week-by-week with updating lag features
    item_sales_campaign = []
    item_sales_no_campaign = []
    cluster_sales_campaign = []
    cluster_sales_no_campaign = []
    
    for week_idx in range(len(df_pred)):
        # Prepare features for this week
        week_features = df_pred.iloc[[week_idx]].copy()
        
        # Item-level lags (use last predicted values after week 1)
        week_features['item_sales_lag_1'] = last_item_sales[-1]
        week_features['item_sales_lag_2'] = last_item_sales[-2] if len(last_item_sales) > 1 else 0
        week_features['item_sales_rolling_4w'] = np.mean(last_item_sales[-4:])
        
        # Cluster-level lags
        week_features['cluster_sales_lag_1'] = last_cluster_sales[-1]
        week_features['cluster_sales_rolling4w'] = np.mean(last_cluster_sales[-4:])
        
        # Campaign features - FOR CAMPAIGN SCENARIO
        week_features['weeks_since_campaign'] = 0  # During campaign, reset to 0
        week_features['campaigns_ytd'] = base_campaigns_ytd + week_idx + 1
        
        # Predict WITH campaign
        week_features['is_campaign_week'] = 1
        X_campaign = week_features[item_features].fillna(0)
        item_pred_campaign = item_model.predict(X_campaign)[0]
        
        X_cluster_campaign = week_features[cluster_features].fillna(0)
        cluster_pred_campaign = cluster_model.predict(X_cluster_campaign)[0]
        
        # Predict WITHOUT campaign - as if no campaign happened
        # Get the last weeks_since_campaign from training data and increment
        last_weeks_since = df_item_train['weeks_since_campaign'].iloc[-1] if len(df_item_train) > 0 else 0
        week_features['weeks_since_campaign'] = last_weeks_since + week_idx + 1
        week_features['is_campaign_week'] = 0
        week_features['campaigns_ytd'] = base_campaigns_ytd  # No new campaign in baseline
        
        X_no_campaign = week_features[item_features].fillna(0)
        item_pred_no_campaign = item_model.predict(X_no_campaign)[0]
        
        X_cluster_no_campaign = week_features[cluster_features].fillna(0)
        cluster_pred_no_campaign = cluster_model.predict(X_cluster_no_campaign)[0]
        
        # Store predictions
        item_sales_campaign.append(max(0, item_pred_campaign))
        item_sales_no_campaign.append(max(0, item_pred_no_campaign))
        cluster_sales_campaign.append(max(0, cluster_pred_campaign))
        cluster_sales_no_campaign.append(max(0, cluster_pred_no_campaign))
        
        # Update lag history with campaign prediction for next week
        last_item_sales.append(item_pred_campaign)
        last_cluster_sales.append(cluster_pred_campaign)
        
    # Convert to arrays
    item_sales_campaign = np.array(item_sales_campaign)
    item_sales_no_campaign = np.array(item_sales_no_campaign)
    cluster_sales_campaign = np.array(cluster_sales_campaign)
    cluster_sales_no_campaign = np.array(cluster_sales_no_campaign)
    
    # For global predictions (used in fallback), predict all weeks at once
    # Add item-level features (will be 0 for global aggregation)
    df_pred['item_sales_lag_1'] = last_item_sales[-1] if len(last_item_sales) > 0 else 0
    df_pred['item_sales_lag_2'] = last_item_sales[-2] if len(last_item_sales) > 1 else 0
    df_pred['item_sales_rolling_4w'] = np.mean(last_item_sales[-4:]) if len(last_item_sales) > 0 else 0
    df_pred['cluster_sales_lag_1'] = last_cluster_sales[-1] if len(last_cluster_sales) > 0 else 0
    df_pred['cluster_sales_rolling4w'] = np.mean(last_cluster_sales[-4:]) if len(last_cluster_sales) > 0 else 0
    df_pred['weeks_since_campaign'] = 0
    df_pred['campaigns_ytd'] = base_campaigns_ytd + 1
    df_pred['is_campaign_week'] = 1
    
    X_global_campaign = df_pred[global_features].fillna(0)
    global_sales_campaign = global_model.predict(X_global_campaign)
    
    df_pred['is_campaign_week'] = 0
    df_pred['weeks_since_campaign'] = 1
    X_global_no_campaign = df_pred[global_features].fillna(0)
    global_sales_no_campaign = global_model.predict(X_global_no_campaign)
    
    # Apply fallback if needed
    if use_fallback:
        logging.info(f"Applying {fallback_level}-level campaign effect...")
        
        # Strategy: Keep item model's baseline, but apply campaign uplift from cluster/global
        # This preserves the item's normal sales patterns while learning campaign effects from similar items
        
        if fallback_level == 'cluster':
            level_campaign = cluster_sales_campaign
            level_no_campaign = cluster_sales_no_campaign
            
            cluster_items_on_campaign = df_orders[
                (df_orders['cluster1To1Id'] == cluster_id) & 
                (df_orders['is_campaign_week'] == 1)
            ]['itemId'].nunique()
            logging.info(f"Learning from {cluster_items_on_campaign} items in cluster with campaign history")
            
        else:  # global
            level_campaign = cluster_sales_campaign  # Use cluster predictions (more stable than global)
            level_no_campaign = cluster_sales_no_campaign
            
            global_items_on_campaign = df_orders[df_orders['is_campaign_week'] == 1]['itemId'].nunique()
            logging.info(f"Learning from {global_items_on_campaign} items across ALL categories with campaign history")
        
        # Calculate campaign uplift ratio from the reference level
        # Apply this uplift to the item's own baseline predictions
        for week_idx in range(len(item_sales_campaign)):
            if level_no_campaign[week_idx] > 0:
                uplift_ratio = level_campaign[week_idx] / level_no_campaign[week_idx]
            else:
                uplift_ratio = 1.0
            
            # Apply uplift to item's baseline
            item_sales_campaign[week_idx] = item_sales_no_campaign[week_idx] * uplift_ratio
        
        avg_uplift = ((level_campaign.mean() / level_no_campaign.mean() - 1) * 100) if level_no_campaign.mean() > 0 else 0
        logging.info(f"Campaign uplift from {fallback_level} model: {avg_uplift:.1f}%")    # Compile results
    results = {
        'item_id': item_id,
        'cluster_id': cluster_id,
        'campaign_start': campaign_start,
        'campaign_end': campaign_end,
        'week_1_item_campaign': max(0, item_sales_campaign[0]),
        'week_2_item_campaign': max(0, item_sales_campaign[1]) if len(item_sales_campaign) > 1 else 0,
        'week_3_item_campaign': max(0, item_sales_campaign[2]) if len(item_sales_campaign) > 2 else 0,
        'week_1_cluster_campaign': max(0, cluster_sales_campaign[0]),
        'week_2_cluster_campaign': max(0, cluster_sales_campaign[1]) if len(cluster_sales_campaign) > 1 else 0,
        'week_3_cluster_campaign': max(0, cluster_sales_campaign[2]) if len(cluster_sales_campaign) > 2 else 0,
        'week_1_item_no_campaign': max(0, item_sales_no_campaign[0]),
        'week_2_item_no_campaign': max(0, item_sales_no_campaign[1]) if len(item_sales_no_campaign) > 1 else 0,
        'week_3_item_no_campaign': max(0, item_sales_no_campaign[2]) if len(item_sales_no_campaign) > 2 else 0,
        'week_1_cluster_no_campaign': max(0, cluster_sales_no_campaign[0]),
        'week_2_cluster_no_campaign': max(0, cluster_sales_no_campaign[1]) if len(cluster_sales_no_campaign) > 1 else 0,
        'week_3_cluster_no_campaign': max(0, cluster_sales_no_campaign[2]) if len(cluster_sales_no_campaign) > 2 else 0,
    }
    
    # Get last year sales for comparison
    last_year_sales = get_last_year_sales(df_orders, item_id, cluster_id, campaign_start, campaign_end)
    if last_year_sales:
        results.update({
            'week_1_item_last_year': last_year_sales['week_1_item'],
            'week_2_item_last_year': last_year_sales['week_2_item'],
            'week_3_item_last_year': last_year_sales['week_3_item'],
            'week_1_cluster_last_year': last_year_sales['week_1_cluster'],
            'week_2_cluster_last_year': last_year_sales['week_2_cluster'],
            'week_3_cluster_last_year': last_year_sales['week_3_cluster'],
        })
    
    return results


def format_forecast_output(results: dict, df_items: pd.DataFrame):
    """
    Format forecast results into a nice CLI table.
    
    Parameters:
        results: Forecast results dictionary
        df_items: Items DataFrame for unit conversion
    
    Returns:
        Formatted string for display
    """
    item_id = results['item_id']
    cluster_id = results['cluster_id']
    
    # Get item weight for conversion back to sales units
    item_info = df_items[df_items['itemId'] == item_id].iloc[0]
    
    # Calculate weight (same logic as in preprocess)
    if pd.notna(item_info.get('weightKgPreparedItemComparisonUnit')) and item_info.get('weightKgPreparedItemComparisonUnit', 0) != 0:
        weight = float(item_info['weightKgPreparedItemComparisonUnit'])
    elif pd.notna(item_info.get('netWeightKgComparisonUnit')) and item_info.get('netWeightKgComparisonUnit', 0) != 0:
        weight = float(item_info['netWeightKgComparisonUnit'])
    else:
        weight = float(item_info.get('grossWeightKgComparisonUnit', 1))
    
    # Determine unit type (kg or L)
    unit_type = 'kg' if weight > 0 else 'units'
    
    # Convert kg/l back to sales units and round up
    def to_units(kg_l):
        return int(np.ceil(kg_l / weight)) if weight > 0 else 0
    
    # Calculate totals in kg/L
    item_campaign_total_kgl = sum([
        results['week_1_item_campaign'],
        results['week_2_item_campaign'],
        results['week_3_item_campaign']
    ])
    
    cluster_campaign_total_kgl = sum([
        results['week_1_cluster_campaign'],
        results['week_2_cluster_campaign'],
        results['week_3_cluster_campaign']
    ])
    
    item_no_campaign_total_kgl = sum([
        results['week_1_item_no_campaign'],
        results['week_2_item_no_campaign'],
        results['week_3_item_no_campaign']
    ])
    
    cluster_no_campaign_total_kgl = sum([
        results['week_1_cluster_no_campaign'],
        results['week_2_cluster_no_campaign'],
        results['week_3_cluster_no_campaign']
    ])
    
    # Last year totals (if available)
    has_last_year = 'week_1_item_last_year' in results
    if has_last_year:
        item_last_year_total_kgl = sum([
            results['week_1_item_last_year'],
            results['week_2_item_last_year'],
            results['week_3_item_last_year']
        ])
        
        cluster_last_year_total_kgl = sum([
            results['week_1_cluster_last_year'],
            results['week_2_cluster_last_year'],
            results['week_3_cluster_last_year']
        ])
    
    # Calculate campaign effects
    if item_no_campaign_total_kgl > 0:
        item_effect_pct = ((item_campaign_total_kgl - item_no_campaign_total_kgl) / item_no_campaign_total_kgl) * 100
    else:
        item_effect_pct = 0
    
    if cluster_no_campaign_total_kgl > 0:
        cluster_effect_pct = ((cluster_campaign_total_kgl - cluster_no_campaign_total_kgl) / cluster_no_campaign_total_kgl) * 100
    else:
        cluster_effect_pct = 0
    
    # Calculate YoY comparison
    if has_last_year:
        if item_last_year_total_kgl > 0:
            item_yoy_pct = ((item_campaign_total_kgl - item_last_year_total_kgl) / item_last_year_total_kgl) * 100
        else:
            item_yoy_pct = 0
            
        if cluster_last_year_total_kgl > 0:
            cluster_yoy_pct = ((cluster_campaign_total_kgl - cluster_last_year_total_kgl) / cluster_last_year_total_kgl) * 100
        else:
            cluster_yoy_pct = 0
    
    # Format output - Main forecast in kg/L
    output = f"""
{'='*95}
Campaign Forecast for ITEM {item_id} in cluster {cluster_id}
{'='*95}
FORECAST (in {unit_type.upper()})           Week 1      Week 2      Week 3      Total       vs Last Year
{'='*95}
Item sale (w/ campaign)      {results['week_1_item_campaign']:8.1f}    {results['week_2_item_campaign']:8.1f}    {results['week_3_item_campaign']:8.1f}    {item_campaign_total_kgl:8.1f}    {item_last_year_total_kgl if has_last_year else 0:8.1f}
Cluster sale (w/ campaign)   {results['week_1_cluster_campaign']:8.1f}    {results['week_2_cluster_campaign']:8.1f}    {results['week_3_cluster_campaign']:8.1f}    {cluster_campaign_total_kgl:8.1f}    {cluster_last_year_total_kgl if has_last_year else 0:8.1f}

Item sale (baseline)         {results['week_1_item_no_campaign']:8.1f}    {results['week_2_item_no_campaign']:8.1f}    {results['week_3_item_no_campaign']:8.1f}    {item_no_campaign_total_kgl:8.1f}
Cluster sale (baseline)      {results['week_1_cluster_no_campaign']:8.1f}    {results['week_2_cluster_no_campaign']:8.1f}    {results['week_3_cluster_no_campaign']:8.1f}    {cluster_no_campaign_total_kgl:8.1f}
{'='*95}
Expected campaign effect
{'='*95}
Item sales: {item_effect_pct:+.2f}%
Cluster sales: {cluster_effect_pct:+.2f}%"""
    
    if has_last_year:
        output += f"""
{'='*95}
Year-over-Year comparison (vs same period last year)
{'='*95}
Item sales: {item_yoy_pct:+.2f}%
Cluster sales: {cluster_yoy_pct:+.2f}%"""
    
    # Add ordering recommendation in sales units
    output += f"""
{'='*95}
ORDERING RECOMMENDATION (in sales units/packs)
{'='*95}
Week 1: {to_units(results['week_1_item_campaign']):>6} units
Week 2: {to_units(results['week_2_item_campaign']):>6} units  
Week 3: {to_units(results['week_3_item_campaign']):>6} units
------
Total:  {to_units(item_campaign_total_kgl):>6} units

(Based on item weight: {weight:.3f} {unit_type} per unit)
{'='*95}
"""
    return output


def main():
    parser = argparse.ArgumentParser(description='Forecast campaign sales for items')
    parser.add_argument('--items', type=str, required=True, help='Comma-separated list of item IDs')
    parser.add_argument('--start', type=str, required=True, help='Campaign start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='Campaign end date (YYYY-MM-DD)')
    parser.add_argument('--forecast-date', type=str, default=None, help='Date when forecast is made (YYYY-MM-DD). Only data before this date will be used. Default: today')
    parser.add_argument('--refresh', action='store_true', help='Force refresh of preprocessed data')
    
    args = parser.parse_args()
    
    # Parse item IDs
    item_ids = [item.strip() for item in args.items.split(',')]
    
    # Parse forecast date
    if args.forecast_date:
        forecast_date = pd.to_datetime(args.forecast_date)
    else:
        forecast_date = pd.Timestamp.now()
    
    campaign_start = pd.to_datetime(args.start)
    campaign_end = pd.to_datetime(args.end)
    
    # Validate dates
    if forecast_date >= campaign_start:
        logging.warning(f"Forecast date ({forecast_date.date()}) is after or same as campaign start ({campaign_start.date()}). This simulates a very late forecast!")
    
    logging.info(f"Starting forecast for {len(item_ids)} items")
    logging.info(f"Forecast date (as-of): {forecast_date.date()}")
    logging.info(f"Campaign period: {args.start} to {args.end}")
    logging.info(f"Weeks before campaign: {(campaign_start - forecast_date).days / 7:.1f}")
    
    # Load data
    df_orders, df_items = load_or_preprocess_data(item_ids, force_refresh=args.refresh)
    
    # Filter to data available at forecast date
    df_orders = filter_completed_campaigns(df_orders, cutoff_date=forecast_date)
    
    # Prepare features
    df_orders = prepare_features(df_orders)
    
    # Forecast each item
    for item_id in item_ids:
        try:
            # Get cluster for this item
            item_info = df_items[df_items['itemId'] == item_id]
            
            if len(item_info) == 0:
                logging.error(f"Item {item_id} not found in item master data")
                continue
            
            cluster_id = item_info.iloc[0]['cluster1To1Id']
            
            # Generate forecast
            results = forecast_item(
                df_orders,
                item_id,
                cluster_id,
                args.start,
                args.end
            )
            
            if results is None:
                logging.error(f"Could not generate forecast for item {item_id}")
                continue
            
            # Display formatted output
            output = format_forecast_output(results, df_items)
            print(output)
            
        except Exception as e:
            logging.error(f"Error forecasting item {item_id}: {e}", exc_info=True)
            continue
    
    logging.info("Forecast complete!")


if __name__ == "__main__":
    main()
