import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
NORMALIZATION WORKFLOW:

Training Mode (calculate normalizers):
    df_item_weekly, df_cluster_weekly, df_items, item_normalizers, cluster_normalizers = preprocess_data(
        df_items, df_orders, df_campaigns,
        forecast_horizon_weeks=8,
        forecast_date=pd.Timestamp('2024-11-18')
    )
    # Save normalizers to disk for later use
    import joblib
    joblib.dump(item_normalizers, 'data/models/item_normalizers.joblib')
    joblib.dump(cluster_normalizers, 'data/models/cluster_normalizers.joblib')

Forecasting Mode (use saved normalizers):
    import joblib
    item_normalizers = joblib.load('data/models/item_normalizers.joblib')
    cluster_normalizers = joblib.load('data/models/cluster_normalizers.joblib')
    
    df_item_weekly, df_cluster_weekly, df_items, _, _ = preprocess_data(
        df_items, df_orders, df_campaigns,
        forecast_horizon_weeks=8,
        campaign_start_date=pd.Timestamp('2025-01-13'),
        forecast_date=pd.Timestamp.now(),
        item_normalizers=item_normalizers,      # Use training normalizers!
        cluster_normalizers=cluster_normalizers  # Use training normalizers!
    )
    
    # Make predictions (already normalized)
    predictions_log = model.predict(X_forecast)
    
    # Denormalize predictions
    predictions_normalized = np.expm1(predictions_log)  # Inverse of log1p
    predictions_kgL = predictions_normalized * item_normalizers[item_id]
"""

def select_item_version(df_items: pd.DataFrame) -> pd.DataFrame:
    # Select the item version of each item with the highest itemSkey
    logging.info("Selecting latest item versions based on itemSkey...")
    df_selected = df_items.loc[df_items.groupby('baseItemId')['itemSkey'].idxmax()].reset_index(drop=True)
    logging.info(f"Selected {len(df_selected)} latest item versions from {len(df_items)} total items.")
    return df_selected

def select_item_unit_conversion(df_items: pd.DataFrame) -> pd.DataFrame:
    # Select the item unit conversion in the following order:
    # 1. weightKgPreparedItemComparisonUnit (prepared weight) if not 0 or null
    # 2. netWeightKgComparisonUnit (net weight) if not 0 or null
    # 3. grossWeightKgComparisonUnit (gross weight) if not 0 or null
    # 4. 1 as fallback, but log a warning if none of the above are available
    logging.info("Selecting item unit conversions...")
    def determine_conversion(row):
        if pd.notnull(row['weightKgPreparedItemComparisonUnit']) and row['weightKgPreparedItemComparisonUnit'] > 0:
            return row['weightKgPreparedItemComparisonUnit']
        elif pd.notnull(row['netWeightKgComparisonUnit']) and row['netWeightKgComparisonUnit'] > 0:
            return row['netWeightKgComparisonUnit']
        elif pd.notnull(row['grossWeightKgComparisonUnit']) and row['grossWeightKgComparisonUnit'] > 0:
            return row['grossWeightKgComparisonUnit']
        else:
            logging.warning(f"Item {row['itemId']} has no valid weight; defaulting conversion to 1.")
            return 1.0
        
    df_items['itemUnitConversion'] = df_items.apply(determine_conversion, axis=1)
    logging.info("Item unit conversions selected.")

    return df_items[['itemId', 'itemDesc', 'itemSkey', 'baseItemId', 'cluster1To1Id', 'itemUnitConversion']]

def convert_order_itemIds_to_baseItemIds(df_orders: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Map order itemIds to baseItemIds using the item data.
    This handles variant items (with leading M/S/K prefixes) by mapping them to their base item.
    
    Parameters:
        df_orders: Order data with itemId column
        df_items: Item master data with itemId and baseItemId columns
    
    Returns:
        Orders DataFrame with baseItemId column added
    """
    logging.info("Converting order itemIds to baseItemIds...")
    
    # Create mapping from itemId to baseItemId
    item_mapping = df_items[['itemId', 'baseItemId']].drop_duplicates(subset='itemId').set_index('itemId')['baseItemId'].to_dict()
    
    # Log mapping stats
    total_items_in_mapping = len(item_mapping)
    variant_items = sum(1 for item_id, base_id in item_mapping.items() if item_id != base_id)
    logging.info(f"Item mapping: {total_items_in_mapping} total items, {variant_items} are variants (with M/S/K prefix)")
    
    # Check for unmapped items (should never happen)
    unmapped_items = set(df_orders['itemId'].unique()) - set(item_mapping.keys())
    if unmapped_items:
        logging.error(f"Found {len(unmapped_items)} items in orders that don't exist in item master data!")
        logging.error(f"Unmapped items: {list(unmapped_items)[:10]}")
        raise ValueError(f"Order data contains {len(unmapped_items)} items not found in item master. "
                        "This indicates a fundamental data fetch problem. Check fetch_data logic.")
    
    # Map itemIds to baseItemIds
    df_orders['baseItemId'] = df_orders['itemId'].map(item_mapping)
    
    # Count conversions
    total_rows = len(df_orders)
    rows_converted = (df_orders['itemId'] != df_orders['baseItemId']).sum()
    unique_items_converted = df_orders[df_orders['itemId'] != df_orders['baseItemId']]['itemId'].nunique()
    
    logging.info(f"Mapped {total_rows:,} order rows: {rows_converted:,} converted from variants, {total_rows - rows_converted:,} unchanged")
    logging.info(f"  {unique_items_converted} unique variant items → base items")
    logging.info(f"  Conversion rate: {rows_converted/total_rows*100:.2f}% of orders affected by variants")
    
    return df_orders

def convert_order_units(df_orders: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sales quantities in kg/L using item unit conversions.
    Merges item conversion factors and applies them to order quantities.
    
    Parameters:
        df_orders: Order data with baseItemId and salesQuantity columns
        df_items: Item master data with baseItemId and itemUnitConversion columns
    
    Returns:
        Orders DataFrame with salesQuantityKgL column added
    """
    logging.info("Converting order sales quantities to standardized units (kg/L)...")
    
    # Create mapping from baseItemId to unit conversion
    conversion_mapping = df_items[['baseItemId', 'itemUnitConversion']].drop_duplicates(subset='baseItemId').set_index('baseItemId')['itemUnitConversion'].to_dict()
    
    # Check for unmapped items
    unmapped_items = set(df_orders['baseItemId'].unique()) - set(conversion_mapping.keys())
    if unmapped_items:
        logging.error(f"Found {len(unmapped_items)} baseItemIds in orders without unit conversion!")
        logging.error(f"Unmapped baseItemIds: {list(unmapped_items)[:10]}")
        raise ValueError(f"Order data contains {len(unmapped_items)} baseItemIds without unit conversion. "
                        "This indicates item deduplication removed needed items.")
    
    # Map unit conversions to orders
    df_orders['itemUnitConversion'] = df_orders['baseItemId'].map(conversion_mapping)
    
    # Calculate sales in kg/L and ensure float type
    df_orders['salesQuantityKgL'] = (df_orders['salesQuantity'] * df_orders['itemUnitConversion']).astype(float)
    
    # Check for negative or invalid values before log transformation
    negative_sales = df_orders['salesQuantityKgL'] < 0
    if negative_sales.any():
        n_negative = negative_sales.sum()
        logging.warning(f"Found {n_negative:,} orders with negative sales quantities - setting to 0")
        df_orders.loc[negative_sales, 'salesQuantityKgL'] = 0
    
    # Apply log transformation for normalization (handles wide range of sales volumes)
    # log1p = log(1 + x) to handle zeros gracefully
    # df_orders['salesQuantityKgL_log'] = np.log1p(df_orders['salesQuantityKgL'])
    
    # Log statistics (convert to float to avoid Decimal issues)
    total_sales_units = float(df_orders['salesQuantity'].sum())
    total_sales_kgl = float(df_orders['salesQuantityKgL'].sum())
    avg_conversion = total_sales_kgl / total_sales_units if total_sales_units > 0 else 0
    
    logging.info(f"Converted {len(df_orders):,} order rows:")
    logging.info(f"  Total sales: {total_sales_units:,.0f} units → {total_sales_kgl:,.0f} kg/L")
    logging.info(f"  Average conversion factor: {avg_conversion:.3f} kg/L per unit")
    logging.info(f"  Applied log1p transformation for normalization")
    
    return df_orders


def add_campaign_features(df_orders: pd.DataFrame, df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """
    Add campaign-related features to order data.
    
    Features added:
    - is_campaign_week: 1 if order is during a campaign, 0 otherwise
    - campaign_week_number: 1, 2, or 3 for which week of the campaign, NULL if not on campaign
    - campaignId: Campaign identifier for joining
    
    Parameters:
        df_orders: Order data with salesCampignId
        df_campaigns: Campaign calendar with campaignId, campaignStart, campaignEnd
    
    Returns:
        Orders with campaign features added
    """
    logging.info("Adding campaign features to orders...")
    
    # Merge campaign dates into orders (dates already converted in main preprocess_data)
    df = df_orders.merge(
        df_campaigns[['campaignId', 'campaignStart', 'campaignEnd']],
        left_on='salesCampignId',
        right_on='campaignId',
        how='left'
    )
    
    # Handle edge case: orders placed during campaign but delivered after
    # Use orderDate instead of deliveryDate for these to keep campaign weeks "pure"
    edge_case = (df['orderDate'] >= df['campaignStart']) & (df['orderDate'] <= df['campaignEnd']) & (df['deliveryDate'] > df['campaignEnd'])
    if edge_case.any():
        n_adjusted = edge_case.sum()
        logging.info(f"  Found {n_adjusted:,} orders placed during campaign but delivered after - using orderDate for aggregation")
        df.loc[edge_case, 'deliveryDate'] = df.loc[edge_case, 'orderDate']
    
    # Calculate campaign week number based on (potentially adjusted) deliveryDate
    df['days_into_campaign'] = (df['deliveryDate'] - df['campaignStart']).dt.days
    in_campaign = (df['deliveryDate'] >= df['campaignStart']) & (df['deliveryDate'] <= df['campaignEnd'])
    df['campaign_week_number'] = ((df['days_into_campaign'] // 7) + 1).where(in_campaign, 0)

    # One-hot encode campaign weeks (0=no campaign, 1/2/3=campaign weeks)
    df['campaign_week_0'] = (df['campaign_week_number'] == 0).astype(int)
    df['campaign_week_1'] = (df['campaign_week_number'] == 1).astype(int)
    df['campaign_week_2'] = (df['campaign_week_number'] == 2).astype(int)
    df['campaign_week_3'] = (df['campaign_week_number'] == 3).astype(int)
    
    # Clean up temp columns
    df = df.drop(columns=['campaignStart', 'campaignEnd', 'days_into_campaign', 'campaign_week_number'])
    
    campaign_orders = (df[['campaign_week_1', 'campaign_week_2', 'campaign_week_3']].sum(axis=1) > 0).sum()
    logging.info(f"  {campaign_orders:,} orders ({campaign_orders/len(df)*100:.1f}%) during campaigns")
    
    return df


def aggregate_to_weekly(df_orders: pd.DataFrame, df_items: pd.DataFrame, 
                       campaign_start_date: pd.Timestamp = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate daily orders to weekly level.
    Creates both item-level and cluster-level aggregations.
    
    Parameters:
        df_orders: Order data with baseItemId, deliveryDate, salesQuantityKgL
        df_items: Item master for cluster mapping
        campaign_start_date: Start date of campaign for week alignment. 
                           If None, defaults to Monday alignment.
    
    Returns:
        df_item_weekly: Item-level weekly aggregation
        df_cluster_weekly: Cluster-level weekly aggregation
    """
    logging.info("Aggregating orders to weekly level...")
    
    # Add cluster info to orders first
    cluster_mapping = df_items[['baseItemId', 'cluster1To1Id']].drop_duplicates(subset='baseItemId')
    df_orders = df_orders.merge(cluster_mapping, on='baseItemId', how='left')
    
    # Determine week start alignment
    if campaign_start_date is not None:
        campaign_weekday = campaign_start_date.weekday()  # 0=Monday, 1=Tuesday, etc.
        logging.info(f"  Aligning weeks to {campaign_start_date.strftime('%A')} (matching campaign start)")
        # Calculate offset from each order date to the campaign's day-of-week
        df_orders['week_start'] = df_orders['deliveryDate'] - pd.to_timedelta(
            (df_orders['deliveryDate'].dt.weekday - campaign_weekday) % 7, unit='d'
        )
    else:
        # Default to Monday alignment
        logging.info("  Aligning weeks to Monday (default)")
        df_orders['week_start'] = df_orders['deliveryDate'] - pd.to_timedelta(
            df_orders['deliveryDate'].dt.weekday, unit='d'
        )
    
    # Item-level aggregation
    # Use max() for campaign flags since we want binary indicators for training
    # The effective_delivery_date adjustment above ensures weeks are "pure"
    df_item_weekly = df_orders.groupby(['baseItemId', 'week_start']).agg({
        'salesQuantityKgL': 'sum',
        'campaign_week_0': 'max',
        'campaign_week_1': 'max',
        'campaign_week_2': 'max',
        'campaign_week_3': 'max',
        'campaignId': 'first',
        'cluster1To1Id': 'first'
    }).reset_index()
    
    # Apply log transformation to weekly aggregated values (for modeling)
    # df_item_weekly['salesQuantityKgL_log'] = np.log1p(df_item_weekly['salesQuantityKgL'])
    
    # Cluster-level aggregation
    # First, calculate campaign intensity (what fraction of items in cluster are on campaign)
    # This is done per cluster-week by checking unique items
    cluster_agg = []
    for (cluster_id, week), group in df_orders.groupby(['cluster1To1Id', 'week_start']):
        # Get unique items in this cluster-week
        unique_items = group['baseItemId'].unique()
        n_items = len(unique_items)
        
        # For each campaign flag, count how many unique items have it set
        # Then divide by total unique items to get intensity (0.0 to 1.0)
        row = {
            'cluster1To1Id': cluster_id,
            'week_start': week,
            'salesQuantityKgL': group['salesQuantityKgL'].sum(),
            'baseItemId': group['baseItemId'].mode()[0] if len(group['baseItemId'].mode()) > 0 else None,
            'campaignId': group['campaignId'].iloc[0]
        }
        
        # Calculate campaign intensity for each week
        for campaign_week in [0, 1, 2, 3]:
            col = f'campaign_week_{campaign_week}'
            # Get max value per item (in case item appears multiple times in week)
            item_campaign_status = group.groupby('baseItemId')[col].max()
            # Intensity = fraction of items that have this flag set
            row[col] = item_campaign_status.sum() / n_items if n_items > 0 else 0.0
        
        cluster_agg.append(row)
    
    df_cluster_weekly = pd.DataFrame(cluster_agg)
    
    # Apply log transformation to weekly aggregated values
    
    logging.info(f"  Item weekly: {len(df_item_weekly):,} rows ({df_item_weekly['baseItemId'].nunique()} items)")
    logging.info(f"  Cluster weekly: {len(df_cluster_weekly):,} rows ({df_cluster_weekly['cluster1To1Id'].nunique()} clusters)")
    
    return df_item_weekly, df_cluster_weekly


def fill_missing_weeks(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Fill missing weeks with zero sales for each item/cluster.
    This ensures lag features and rolling averages work correctly (count calendar weeks, not rows).
    
    Without this, shift(5) might shift by 5 data points but not 5 calendar weeks if some weeks are missing.
    
    Parameters:
        df: DataFrame with weekly data (must have 'week_start' column)
        id_col: 'baseItemId' or 'cluster1To1Id'
    
    Returns:
        DataFrame with complete week coverage (missing weeks filled with zeros)
    """
    logging.info(f"Filling missing weeks for {id_col}...")
    
    # Get the full date range (min to max week across all items)
    min_week = df['week_start'].min()
    max_week = df['week_start'].max()
    
    # Create complete week range (7 days apart)
    all_weeks = pd.date_range(start=min_week, end=max_week, freq='7D')
    
    # Get all unique IDs
    all_ids = df[id_col].unique()
    
    # Create complete skeleton (all IDs × all weeks)
    skeleton = pd.DataFrame({
        id_col: np.repeat(all_ids, len(all_weeks)),
        'week_start': np.tile(all_weeks, len(all_ids))
    })
    
    # Merge with actual data (left join to keep all skeleton rows)
    df_complete = skeleton.merge(df, on=[id_col, 'week_start'], how='left')
    
    # Fill missing values
    # Sales-related columns: fill with 0
    sales_cols = ['salesQuantityKgL']
    for col in sales_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(0)
    
    # Campaign flags: fill with 0 (no campaign)
    campaign_cols = ['campaign_week_0', 'campaign_week_1', 'campaign_week_2', 'campaign_week_3']
    for col in campaign_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(0).astype(int)
    
    # Other columns: forward fill (use most recent known value)
    other_cols = ['cluster1To1Id', 'campaignId']
    for col in other_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete.groupby(id_col)[col].fillna(method='ffill')
    
    original_rows = len(df)
    filled_rows = len(df_complete)
    added_rows = filled_rows - original_rows
    
    logging.info(f"  Added {added_rows:,} missing week rows ({added_rows/original_rows*100:.1f}% increase)")
    logging.info(f"  Complete data: {filled_rows:,} rows = {len(all_ids)} {id_col}s × {len(all_weeks)} weeks")
    
    return df_complete


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time encoding for seasonality.
    
    Features added:
    - week_of_year: 1-52
    - week_of_year_sin, week_of_year_cos: Cyclical encoding
    - year: Calendar year
    
    Parameters:
        df: DataFrame with week_start column
    
    Returns:
        DataFrame with time features added
    """
    logging.info("Adding time features...")
    
    df['week_of_year'] = df['week_start'].dt.isocalendar().week
    df['year'] = df['week_start'].dt.year
    
    # Cyclical encoding (52 weeks in a year)
    df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # df.drop(columns=['week_of_year'], inplace=True)
    return df


def add_lag_features(df: pd.DataFrame, id_col: str, forecast_horizon_weeks: int = 4, target_col: str = 'salesQuantityKgL_log') -> pd.DataFrame:
    """
    Add lag and rolling average features for campaign forecasting.
    
    Since we train 3 separate models (one per campaign week), we need 3 sets of lag features:
    - Week 1 model: predicts week 1 of campaign (horizon + 1 weeks ahead)
    - Week 2 model: predicts week 2 of campaign (horizon + 2 weeks ahead)
    - Week 3 model: predicts week 3 of campaign (horizon + 3 weeks ahead)
    
    Example: If forecast_horizon = 4 weeks:
    - Week 1 lags: 5 weeks back (4 + 1)
    - Week 2 lags: 6 weeks back (4 + 2)
    - Week 3 lags: 7 weeks back (4 + 3)
    
    Features added per campaign week:
    - sales_lag_weekN: Recent sales at safe distance
    - sales_rolling_4w_weekN: 4-week rolling average ending at safe lag
    
    Also adds universal features:
    - sales_lag_12w: Medium-term lag (roughly 3 months back)
    - sales_lag_52w: Same week last year (seasonality)
    
    Parameters:
        df: DataFrame with weekly data
        id_col: Column to group by ('baseItemId' or 'cluster1To1Id')
        forecast_horizon_weeks: How many weeks before campaign starts (default 4)
        target_col: Sales column name
    
    Returns:
        DataFrame with lag features added
    """
    logging.info(f"Adding lag features for {id_col} (base horizon: {forecast_horizon_weeks} weeks)...")
    
    # Sort by ID and date
    df = df.sort_values([id_col, 'week_start'])
    
    # Create lag features for each campaign week (3 separate models)
    # Each model predicts a different week ahead, so needs different lag distance
    for week_num in [1, 2, 3]:
        lag_distance = forecast_horizon_weeks + week_num
        
        # Recent lag (specific to this campaign week)
        df[f'sales_lag_week{week_num}'] = df.groupby(id_col)[target_col].shift(lag_distance)
        
        # Rolling 4-week average ending at this lag distance
        df[f'sales_rolling_4w_week{week_num}'] = (
            df.groupby(id_col)[target_col]
            .shift(lag_distance)
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # Fill NaN with 0 for missing lags (new items, early weeks)
        df[f'sales_lag_week{week_num}'] = df[f'sales_lag_week{week_num}'].fillna(0)
        df[f'sales_rolling_4w_week{week_num}'] = df[f'sales_rolling_4w_week{week_num}'].fillna(0)
    
    # Universal features (same for all models)
    # Medium-term lag (roughly 3 months)
    df['sales_lag_12w'] = df.groupby(id_col)[target_col].shift(12)
    
    # Seasonal lag (same week last year)
    df['sales_lag_52w'] = df.groupby(id_col)[target_col].shift(52)
    
    # Fill NaN with 0
    df['sales_lag_12w'] = df['sales_lag_12w'].fillna(0)
    df['sales_lag_52w'] = df['sales_lag_52w'].fillna(0)
    
    return df

def add_cluster_context_to_items(df_item_weekly: pd.DataFrame, df_cluster_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster-level lag features to item-level data for context.
    
    Since we have 3 different lag offsets, we add all 3 cluster lags.
    The model will use the appropriate one based on which campaign week it's trained for.
    
    Features added:
    - cluster_sales_lag_week1/2/3: Cluster's lagged sales (matching forecast horizon)
    
    Parameters:
        df_item_weekly: Item-level weekly data
        df_cluster_weekly: Cluster-level weekly data with lag features
    
    Returns:
        Item data with cluster context added
    """
    logging.info("Adding cluster context to item-level data...")
    
    # Merge cluster lag features and rolling averages for all 3 campaign weeks
    cluster_lags = df_cluster_weekly[['cluster1To1Id', 'week_start', 
                                       'sales_lag_week1', 'sales_lag_week2', 'sales_lag_week3',
                                       'sales_rolling_4w_week1', 'sales_rolling_4w_week2', 'sales_rolling_4w_week3']].copy()
    cluster_lags = cluster_lags.rename(columns={
        'sales_lag_week1': 'cluster_sales_lag_week1',
        'sales_lag_week2': 'cluster_sales_lag_week2',
        'sales_lag_week3': 'cluster_sales_lag_week3',
        'sales_rolling_4w_week1': 'cluster_sales_rolling_4w_week1',
        'sales_rolling_4w_week2': 'cluster_sales_rolling_4w_week2',
        'sales_rolling_4w_week3': 'cluster_sales_rolling_4w_week3'
    })
    
    df = df_item_weekly.merge(
        cluster_lags,
        on=['cluster1To1Id', 'week_start'],
        how='left'
    )
    
    # Fill NaN with 0
    df['cluster_sales_lag_week1'] = df['cluster_sales_lag_week1'].fillna(0)
    df['cluster_sales_lag_week2'] = df['cluster_sales_lag_week2'].fillna(0)
    df['cluster_sales_lag_week3'] = df['cluster_sales_lag_week3'].fillna(0)
    df['cluster_sales_rolling_4w_week1'] = df['cluster_sales_rolling_4w_week1'].fillna(0)
    df['cluster_sales_rolling_4w_week2'] = df['cluster_sales_rolling_4w_week2'].fillna(0)
    df['cluster_sales_rolling_4w_week3'] = df['cluster_sales_rolling_4w_week3'].fillna(0)
    
    return df


def add_campaign_metadata_to_clusters(df_cluster_weekly: pd.DataFrame, df_item_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add campaign item metadata to cluster-level data.
    
    Features added:
    - campaign_item_share: Campaign item's recent % of cluster sales (rolling 8-12 week window)
    - num_items_in_cluster: Number of distinct items in each cluster (helps differentiate cluster sizes)
    
    This captures recent market dynamics (e.g., new private label stealing share from Arla)
    rather than historical dominance that may no longer reflect reality.
    
    Parameters:
        df_cluster_weekly: Cluster-level weekly data
        df_item_weekly: Item-level weekly data
    
    Returns:
        Cluster data with campaign metadata added
    """
    logging.info("Adding campaign metadata to cluster-level data...")
    
    # Add cluster size feature (number of items per cluster)
    # This helps the model differentiate between single-item clusters and large multi-item clusters
    cluster_sizes = df_item_weekly.groupby('cluster1To1Id')['baseItemId'].nunique().reset_index()
    cluster_sizes.columns = ['cluster1To1Id', 'num_items_in_cluster']
    df_cluster_weekly = df_cluster_weekly.merge(cluster_sizes, on='cluster1To1Id', how='left')
    df_cluster_weekly['num_items_in_cluster'] = df_cluster_weekly['num_items_in_cluster'].fillna(1).astype(int)
    
    logging.info(f"  Cluster sizes: min={df_cluster_weekly['num_items_in_cluster'].min()}, median={df_cluster_weekly['num_items_in_cluster'].median():.0f}, max={df_cluster_weekly['num_items_in_cluster'].max()}")
    
    # For each week, calculate campaign_item_share based on the PREVIOUS 8-12 weeks of non-campaign data
    # This ensures we capture recent market dynamics (new items, shifting shares)
    lookback_weeks = 12
    
    df_baseline = df_item_weekly[df_item_weekly['campaign_week_0'] == 1].copy()
    df_baseline = df_baseline.sort_values('week_start')
    
    # For cluster weeks with campaigns, identify which item was on campaign
    in_campaign = (df_item_weekly['campaign_week_0'] == 0)
    campaign_items = df_item_weekly[in_campaign][['cluster1To1Id', 'week_start', 'baseItemId']].drop_duplicates()
    
    df = df_cluster_weekly.merge(
        campaign_items,
        on=['cluster1To1Id', 'week_start'],
        how='left',
        suffixes=('', '_campaign')
    )
    
    # Calculate rolling share for each campaign week
    campaign_shares = []
    for idx, row in df[df['baseItemId_campaign'].notna()].iterrows():
        campaign_item = row['baseItemId_campaign']
        cluster_id = row['cluster1To1Id']
        week = row['week_start']
        
        # Get last 12 weeks of baseline data BEFORE this week
        recent_baseline = df_baseline[
            (df_baseline['week_start'] < week) &
            (df_baseline['week_start'] >= week - pd.Timedelta(weeks=lookback_weeks))
        ]
        
        # Item sales in recent baseline
        item_recent = recent_baseline[recent_baseline['baseItemId'] == campaign_item]['salesQuantityKgL_log'].sum()
        
        # Cluster sales in recent baseline  
        cluster_recent = recent_baseline[recent_baseline['cluster1To1Id'] == cluster_id]['salesQuantityKgL_log'].sum()
        
        share = item_recent / cluster_recent if cluster_recent > 0 else 0
        campaign_shares.append({'week_start': week, 'cluster1To1Id': cluster_id, 'baseItemId_campaign': campaign_item, 'campaign_item_share': share})
    
    # Merge the calculated shares back
    if campaign_shares:
        df_shares = pd.DataFrame(campaign_shares)
        df = df.drop(columns=['campaign_item_share'], errors='ignore')
        df = df.merge(df_shares[['week_start', 'cluster1To1Id', 'baseItemId_campaign', 'campaign_item_share']], 
                      on=['week_start', 'cluster1To1Id', 'baseItemId_campaign'], how='left')
    else:
        df['campaign_item_share'] = np.nan
    
    # Leave NaN for non-campaign weeks (not 0) - NaN means "feature doesn't apply"
    # 0 would be ambiguous: no campaign OR campaign item with no recent sales
    
    # Drop the temporary lookup column
    df = df.drop(columns=['baseItemId_campaign'])
    
    return df

def filter_to_campaign_items_only(df_item_weekly: pd.DataFrame, keep_items: list = None) -> pd.DataFrame:
    """
    Filter item weekly data to only items that were ever on campaign.
    Also removes data after the last campaign week for each item.
    
    Parameters:
        df_item_weekly: Item-level weekly data with campaign flags
        keep_items: Optional list of item IDs to keep even if they weren't on campaign
                   (useful for forecasting new campaign items)
    
    Returns:
        Filtered item-level weekly data containing only items that were ever on campaign,
        with data after the last campaign week for each item removed.
    """
    items_on_campaign = df_item_weekly[
        (df_item_weekly['campaign_week_1'] == 1) |
        (df_item_weekly['campaign_week_2'] == 1) |
        (df_item_weekly['campaign_week_3'] == 1)
    ]['baseItemId'].unique()
    
    # Add keep_items to the set if provided
    if keep_items:
        items_to_keep = set(items_on_campaign) | set(keep_items)
        items_to_keep = list(items_to_keep)
        logging.info(f"  Keeping {len(keep_items)} additional items requested for forecasting")
    else:
        items_to_keep = items_on_campaign

    df_item_weekly = df_item_weekly[df_item_weekly['baseItemId'].isin(items_to_keep)].copy()
    logging.info(f"Filtered item weekly data to {len(df_item_weekly):,} rows for {len(items_to_keep):,} items (ever on campaign + forecast targets).")
    
    # Remove data after last campaign week for each item
    # BUT: Keep ALL data for items in keep_items (forecast targets) - they need full history!
    last_campaign_weeks = df_item_weekly[
        (df_item_weekly['campaign_week_1'] == 1) |
        (df_item_weekly['campaign_week_2'] == 1) |
        (df_item_weekly['campaign_week_3'] == 1)
    ].groupby('baseItemId')['week_start'].max().reset_index().rename(columns={'week_start': 'last_campaign_week'})
    
    df_item_weekly = df_item_weekly.merge(
        last_campaign_weeks,
        on='baseItemId',
        how='left'
    )

    # Only filter out post-campaign data for items that:
    # 1. Have been on campaign (last_campaign_week is not NaN)
    # 2. Are NOT in the keep_items list (forecast targets need full history)
    keep_items_set = set(keep_items) if keep_items else set()
    is_forecast_target = df_item_weekly['baseItemId'].isin(keep_items_set)
    has_campaign = df_item_weekly['last_campaign_week'].notna()
    
    # Keep row if:
    # - It's a forecast target (keep ALL data), OR
    # - It has no campaign history (keep ALL data), OR  
    # - It's before the last campaign week
    df_item_weekly = df_item_weekly[
        is_forecast_target |  # Keep all data for forecast targets
        (~has_campaign) |  # Keep all data for items without campaigns
        (df_item_weekly['week_start'] <= df_item_weekly['last_campaign_week'])  # Filter post-campaign for others
    ].copy()
    df_item_weekly = df_item_weekly.drop(columns=['last_campaign_week'])
    logging.info(f"After removing data after last campaign week, item weekly data has {len(df_item_weekly):,} rows.")
    return df_item_weekly

def normalize_aggregated_sales(df: pd.DataFrame, group_by_column: str, normalizers: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Normalize aggregated sales by dividing by per-item/cluster baseline (median of non-campaign weeks).
    This centers the distribution around 1.0 for each item/cluster, making campaign effects
    transferable across different items/clusters (e.g., "1.5 = 50% increase").
    
    Parameters:
        df: DataFrame with salesQuantityKgL and campaign_week_0 columns
        group_by_column: 'baseItemId' or 'cluster1To1Id'
        normalizers: Optional dict mapping ID -> baseline value. If provided, these will be used
                    instead of calculating new ones. Use this for forecasting to ensure consistency
                    with training data.
    
    Returns:
        df: DataFrame with salesQuantityKgL_log column (normalized and log-transformed)
        normalizers: Dict mapping ID -> baseline median (for denormalization at prediction time)
    """
    logging.info(f"Normalizing aggregated sales by {group_by_column} baseline...")
    
    # If normalizers provided (forecasting mode), use them
    if normalizers is not None:
        logging.info(f"  Using provided normalizers for {len(normalizers)} {group_by_column}s (forecasting mode)")
    else:
        # Training mode: Calculate median baseline from NON-CAMPAIGN weeks only
        logging.info(f"  Calculating normalizers from data (training mode)")
        df_baseline = df[df['campaign_week_0'] == 1].copy()
        normalizers = df_baseline.groupby(group_by_column)['salesQuantityKgL'].median().to_dict()
        
        # Fallback: use mean for items with no baseline data (shouldn't happen after filtering)
        mean_normalizers = df_baseline.groupby(group_by_column)['salesQuantityKgL'].mean().to_dict()
        for id_val, norm_val in mean_normalizers.items():
            if id_val not in normalizers or normalizers[id_val] <= 0:
                normalizers[id_val] = norm_val
        
        logging.info(f"  Calculated normalizers for {len(normalizers)} {group_by_column}s")
    
    # Map normalizers to all rows
    df['normalizer'] = df[group_by_column].map(normalizers)
    
    # Handle missing normalizers (shouldn't happen, but be safe)
    missing_mask = df['normalizer'].isna() | (df['normalizer'] <= 0)
    if missing_mask.any():
        n_missing = missing_mask.sum()
        fallback = df['salesQuantityKgL'].median()
        logging.warning(f"  {n_missing} rows missing normalizer, using global median: {fallback:.2f}")
        df.loc[missing_mask, 'normalizer'] = fallback
    
    # Normalize: divide by baseline (result: baseline ≈ 1.0, campaign ≈ 1.2-1.8)
    df['salesQuantityKgL_normalized'] = df['salesQuantityKgL'] / df['normalizer']
    
    # Apply log transformation (helps with outliers, makes distribution more Gaussian)
    df['salesQuantityKgL_log'] = np.log1p(df['salesQuantityKgL_normalized'])
    
    # Clean up temporary columns
    df = df.drop(columns=['normalizer', 'salesQuantityKgL_normalized'])
    
    # Log statistics
    baseline_values = list(normalizers.values())
    logging.info(f"  Baseline median range: {min(baseline_values):.2f} - {max(baseline_values):.2f} kg/L")
    logging.info(f"  Mean baseline: {np.mean(baseline_values):.2f} kg/L")
    
    return df, normalizers


def preprocess_data(df_items: pd.DataFrame, df_orders: pd.DataFrame, df_campaigns: pd.DataFrame, 
                   forecast_horizon_weeks: int = 8, campaign_start_date: pd.Timestamp = None,
                   forecast_date: pd.Timestamp = None, item_normalizers: dict = None, 
                   cluster_normalizers: dict = None, forecast_item_ids: list = None):
    """
    Preprocess data for training or forecasting.
    
    This creates separate lag features for each campaign week to enable accurate modeling:
    - Train 3 separate models (week1, week2, week3)
    - Each model uses its corresponding lag features (sales_lag_week1, sales_lag_week2, etc.)
    - Ensures training conditions match prediction reality at different horizons
    
    Parameters:
        df_items: Item master data
        df_orders: Order history
        df_campaigns: Campaign definitions
        forecast_horizon_weeks: How many weeks before campaign week 1 we make the forecast (default 8)
                               Week 2 will be at horizon+1, week 3 at horizon+2
        campaign_start_date: Start date of the campaign being forecast. Used to align week boundaries.
                            If None, defaults to Monday alignment.
        forecast_date: Date when forecast is being made. Only data up to this date will be used.
                      If None, all data is used.
        item_normalizers: Optional dict of item normalization factors from training. If provided,
                         these will be used instead of calculating new ones (forecasting mode).
        cluster_normalizers: Optional dict of cluster normalization factors from training.
        forecast_item_ids: Optional list of item IDs to forecast. These will be kept even if
                          they weren't on campaign (useful for forecasting new items).
    
    Returns:
        df_item_weekly: Item-level weekly features
        df_cluster_weekly: Cluster-level weekly features
        df_items: Deduplicated item master
        item_normalizers: Item normalization factors (calculated or passed through)
        cluster_normalizers: Cluster normalization factors (calculated or passed through)
    """
    logging.info(f"Starting data preprocessing (forecast horizon: {forecast_horizon_weeks} weeks)...")
    # STEP 0: Ensure correct data types from raw data
    df_campaigns['campaignStart'] = pd.to_datetime(df_campaigns['campaignStart'])
    df_campaigns['campaignEnd'] = pd.to_datetime(df_campaigns['campaignEnd'])
    df_orders['orderDate'] = pd.to_datetime(df_orders['orderDate'])
    df_orders['deliveryDate'] = pd.to_datetime(df_orders['deliveryDate'])
    
    # Filter data to simulate forecast date as "today"
    if forecast_date is not None:
        original_rows = len(df_orders)
        df_orders = df_orders[df_orders['orderDate'] < forecast_date].copy()
        filtered_rows = original_rows - len(df_orders)
        if filtered_rows > 0:
            logging.info(f"  Filtered {filtered_rows:,} orders after forecast date {forecast_date.date()}")
            logging.info(f"  Simulating forecast as of {forecast_date.date()} (using {len(df_orders):,} historical orders)")
    

    # STEP 1: Convert orders to baseItemId BEFORE deduplicating items
    df_orders = convert_order_itemIds_to_baseItemIds(df_orders, df_items)
    
    # STEP 2: Deduplicate items (keep latest version per baseItemId)
    df_items = select_item_version(df_items)
    df_items = select_item_unit_conversion(df_items)
    
    # STEP 3: Convert order units using the deduplicated item list
    df_orders = convert_order_units(df_orders, df_items)

    # STEP 4: Add campaign features
    df_orders = add_campaign_features(df_orders, df_campaigns)

    # STEP 5: Aggregate to weekly level
    df_item_weekly, df_cluster_weekly = aggregate_to_weekly(df_orders, df_items, campaign_start_date=campaign_start_date)
    
    # STEP 5.5: Fill missing weeks with zeros (ensures lag features work correctly)
    df_item_weekly = fill_missing_weeks(df_item_weekly, id_col='baseItemId')
    df_cluster_weekly = fill_missing_weeks(df_cluster_weekly, id_col='cluster1To1Id')

    # STEP 6: Normalize sales by per-item/cluster baseline (BEFORE filtering!)
    # Must calculate normalizers from full historical data, not filtered subset
    # Pass through normalizers if provided (forecasting mode), otherwise calculate them (training mode)
    df_item_weekly, item_normalizers = normalize_aggregated_sales(
        df_item_weekly, group_by_column='baseItemId', normalizers=item_normalizers
    )
    df_cluster_weekly, cluster_normalizers = normalize_aggregated_sales(
        df_cluster_weekly, group_by_column='cluster1To1Id', normalizers=cluster_normalizers
    )
                
    # STEP 7: Filter to campaign items only and remove post-campaign data
    # Keep forecast_item_ids even if they weren't on campaign (for forecasting new items)
    # This must happen AFTER normalization to ensure proper baseline calculation
    df_item_weekly = filter_to_campaign_items_only(df_item_weekly, keep_items=forecast_item_ids)
    
    # Drop raw sales column (we only need normalized log values)
    df_item_weekly = df_item_weekly.drop(columns=['salesQuantityKgL'])
    df_cluster_weekly = df_cluster_weekly.drop(columns=['salesQuantityKgL'])

    # STEP 8: Add time features
    df_item_weekly = add_time_features(df_item_weekly)
    df_cluster_weekly = add_time_features(df_cluster_weekly)
    
    # STEP 9: Add lag features (matching forecast horizon)
    df_item_weekly = add_lag_features(df_item_weekly, id_col='baseItemId', forecast_horizon_weeks=forecast_horizon_weeks)
    df_cluster_weekly = add_lag_features(df_cluster_weekly, id_col='cluster1To1Id', forecast_horizon_weeks=forecast_horizon_weeks)
    
    # Add cluster_sales_* aliases for cluster-level data (model expects these names)
    for week_num in [1, 2, 3]:
        df_cluster_weekly[f'cluster_sales_lag_week{week_num}'] = df_cluster_weekly[f'sales_lag_week{week_num}']
        df_cluster_weekly[f'cluster_sales_rolling_4w_week{week_num}'] = df_cluster_weekly[f'sales_rolling_4w_week{week_num}']

    # STEP 10: Add cluster context to items
    df_item_weekly = add_cluster_context_to_items(df_item_weekly, df_cluster_weekly)    
    
    # STEP 11: Add campaign metadata to clusters
    df_cluster_weekly = add_campaign_metadata_to_clusters(df_cluster_weekly, df_item_weekly)
    
    logging.info("Preprocessing completed!")
    logging.info(f"  Item weekly data: {len(df_item_weekly):,} rows, {df_item_weekly['baseItemId'].nunique()} items")
    logging.info(f"  Cluster weekly data: {len(df_cluster_weekly):,} rows, {df_cluster_weekly['cluster1To1Id'].nunique()} clusters")
    
    # DEBUG: Analyze data quality for model training
    logging.info("\n" + "="*80)
    logging.info("🔍 DATA QUALITY ANALYSIS FOR MODEL TRAINING")
    logging.info("="*80)
    
    # Item-level analysis
    logging.info("\n📊 ITEM-LEVEL DATA:")
    campaign_items = df_item_weekly[
        (df_item_weekly['campaign_week_1'] == 1) | 
        (df_item_weekly['campaign_week_2'] == 1) | 
        (df_item_weekly['campaign_week_3'] == 1)
    ]
    baseline_items = df_item_weekly[df_item_weekly['campaign_week_0'] == 1]
    
    logging.info(f"  Campaign weeks: {len(campaign_items):,} rows from {campaign_items['baseItemId'].nunique()} items")
    logging.info(f"  Baseline weeks: {len(baseline_items):,} rows from {baseline_items['baseItemId'].nunique()} items")
    logging.info(f"  Campaign sales (normalized log): min={campaign_items['salesQuantityKgL_log'].min():.3f}, median={campaign_items['salesQuantityKgL_log'].median():.3f}, max={campaign_items['salesQuantityKgL_log'].max():.3f}")
    logging.info(f"  Baseline sales (normalized log): min={baseline_items['salesQuantityKgL_log'].min():.3f}, median={baseline_items['salesQuantityKgL_log'].median():.3f}, max={baseline_items['salesQuantityKgL_log'].max():.3f}")
    
    # Check for zeros/NaNs in features
    zero_sales = (df_item_weekly['salesQuantityKgL_log'] == 0).sum()
    logging.info(f"  Zero sales weeks: {zero_sales:,} ({zero_sales/len(df_item_weekly)*100:.1f}%)")
    
    # Cluster-level analysis
    logging.info("\n📊 CLUSTER-LEVEL DATA:")
    campaign_clusters = df_cluster_weekly[
        (df_cluster_weekly['campaign_week_1'] == 1) | 
        (df_cluster_weekly['campaign_week_2'] == 1) | 
        (df_cluster_weekly['campaign_week_3'] == 1)
    ]
    baseline_clusters = df_cluster_weekly[df_cluster_weekly['campaign_week_0'] == 1]
    
    logging.info(f"  Campaign weeks: {len(campaign_clusters):,} rows from {campaign_clusters['cluster1To1Id'].nunique()} clusters")
    logging.info(f"  Baseline weeks: {len(baseline_clusters):,} rows from {baseline_clusters['cluster1To1Id'].nunique()} clusters")
    logging.info(f"  Campaign sales (normalized log): min={campaign_clusters['salesQuantityKgL_log'].min():.3f}, median={campaign_clusters['salesQuantityKgL_log'].median():.3f}, max={campaign_clusters['salesQuantityKgL_log'].max():.3f}")
    logging.info(f"  Baseline sales (normalized log): min={baseline_clusters['salesQuantityKgL_log'].min():.3f}, median={baseline_clusters['salesQuantityKgL_log'].median():.3f}, max={baseline_clusters['salesQuantityKgL_log'].max():.3f}")
    
    zero_sales_clusters = (df_cluster_weekly['salesQuantityKgL_log'] == 0).sum()
    logging.info(f"  Zero sales weeks: {zero_sales_clusters:,} ({zero_sales_clusters/len(df_cluster_weekly)*100:.1f}%)")
    
    # Check specific forecast items if provided
    if forecast_item_ids:
        for item_id in forecast_item_ids:
            item_data = df_item_weekly[df_item_weekly['baseItemId'] == item_id]
            if len(item_data) > 0:
                logging.info(f"\n🎯 FORECAST TARGET ITEM {item_id}:")
                logging.info(f"  Total weeks: {len(item_data)}")
                logging.info(f"  Campaign weeks: {item_data[['campaign_week_1', 'campaign_week_2', 'campaign_week_3']].sum().sum()}")
                logging.info(f"  Sales (normalized log): min={item_data['salesQuantityKgL_log'].min():.3f}, median={item_data['salesQuantityKgL_log'].median():.3f}, max={item_data['salesQuantityKgL_log'].max():.3f}")
                logging.info(f"  Lag features (week1): min={item_data['sales_lag_week1'].min():.3f}, median={item_data['sales_lag_week1'].median():.3f}, max={item_data['sales_lag_week1'].max():.3f}")
                logging.info(f"  Cluster lag (week1): min={item_data['cluster_sales_lag_week1'].min():.3f}, median={item_data['cluster_sales_lag_week1'].median():.3f}, max={item_data['cluster_sales_lag_week1'].max():.3f}")
                
                # Check denormalization
                if item_id in item_normalizers:
                    normalizer = item_normalizers[item_id]
                    logging.info(f"  Normalizer: {normalizer:.2f} kg/L")
                    # Show what the predictions would look like
                    median_log = item_data['salesQuantityKgL_log'].median()
                    median_normalized = np.expm1(median_log)
                    median_kgl = median_normalized * normalizer
                    logging.info(f"  Median prediction path: log={median_log:.3f} -> normalized={median_normalized:.3f} -> kgL={median_kgl:.2f}")
    
    logging.info("="*80 + "\n")
    
    return df_item_weekly, df_cluster_weekly, df_items, item_normalizers, cluster_normalizers


if __name__ == "__main__":
    from fetch_data import fetch_data
    from datetime import datetime, timedelta
    
    print("="*80)
    print("SIMULATION: Campaign Forecasting Workflow")
    print("="*80)
    
    # ============================================================================
    # SCENARIO SETUP
    # ============================================================================
    forecast_date = pd.Timestamp('2024-11-18')  # Today: when we make the forecast
    campaign_start_date = pd.Timestamp('2025-01-13')  # Campaign starts in 8 weeks
    campaign_end_date = campaign_start_date + pd.Timedelta(days=20)  # 3-week campaign
    item_ids = ['528208']  # Milk item we want to forecast
    
    # Calculate forecast horizon
    forecast_horizon_weeks = int((campaign_start_date - forecast_date).days / 7)
    print(f"\n📅 Forecast Date: {forecast_date.strftime('%Y-%m-%d')}")
    print(f"🎯 Campaign Start: {campaign_start_date.strftime('%Y-%m-%d')}")
    print(f"⏰ Forecast Horizon: {forecast_horizon_weeks} weeks ahead")
    
    # ============================================================================
    # STEP 1: FETCH HISTORICAL DATA (for training)
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Fetching historical data for model training...")
    print(f"{'='*80}")
    
    df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=item_ids)
    
    # ============================================================================
    # STEP 2: PREPROCESS FOR TRAINING (with correct forecast horizon)
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: Preprocessing data (horizon: {forecast_horizon_weeks} weeks)...")
    print(f"{'='*80}")
    
    df_item_weekly, df_cluster_weekly, df_items, item_normalizers, cluster_normalizers = preprocess_data(
        df_items, df_orders, df_campaigns, 
        forecast_horizon_weeks=forecast_horizon_weeks
    )
    
    print(f"\n📊 Normalization factors calculated:")
    print(f"   Items: {len(item_normalizers)} normalizers")
    print(f"   Clusters: {len(cluster_normalizers)} normalizers")
    
    # ============================================================================
    # STEP 3: PREPARE TRAINING DATASETS (3 separate models)
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Preparing training data for 3 campaign week models...")
    print(f"{'='*80}")
    
    # Filter training data by campaign week=    
    train_week1 = df_item_weekly[df_item_weekly['campaign_week_1'] == 1].copy()
    train_week2 = df_item_weekly[df_item_weekly['campaign_week_2'] == 1].copy()
    train_week3 = df_item_weekly[df_item_weekly['campaign_week_3'] == 1].copy()
    
    print(f"\n📊 Training Data Summary:")
    print(f"   Week 1 model: {len(train_week1):,} training samples")
    print(f"   Week 2 model: {len(train_week2):,} training samples")
    print(f"   Week 3 model: {len(train_week3):,} training samples")
    
    # Define features for each model (use corresponding lag features)
    features_week1 = ['sales_lag_week1', 'sales_rolling_4w_week1', 'sales_lag_52w', 
                      'week_of_year_sin', 'week_of_year_cos', 'cluster_sales_lag_week1']
    features_week2 = ['sales_lag_week2', 'sales_rolling_4w_week2', 'sales_lag_52w',
                      'week_of_year_sin', 'week_of_year_cos', 'cluster_sales_lag_week2']
    features_week3 = ['sales_lag_week3', 'sales_rolling_4w_week3', 'sales_lag_52w',
                      'week_of_year_sin', 'week_of_year_cos', 'cluster_sales_lag_week3']
    
    print(f"\n🎯 Week 1 features: {features_week1[:3]} + time/cluster features")
    print(f"🎯 Week 2 features: {features_week2[:3]} + time/cluster features")
    print(f"🎯 Week 3 features: {features_week3[:3]} + time/cluster features")
    
    # Show sample training data
    if len(train_week1) > 0:
        print(f"\n📋 Week 1 Training Sample:")
        # Note: salesQuantityKgL column was dropped after normalization, only salesQuantityKgL_log exists
        sample_cols = ['baseItemId', 'week_start', 'salesQuantityKgL_log', 'sales_lag_week1', 'campaign_week_1']
        available_cols = [c for c in sample_cols if c in train_week1.columns]
        print(train_week1[available_cols].tail(3))
        
        # Show normalization effect
        if '528208' in item_normalizers:
            print(f"\n🔢 Normalization example for item 528208:")
            print(f"   Baseline (median of non-campaign weeks): {item_normalizers['528208']:.2f} kg/L")
            print(f"   Normalized values centered around log(1) = 0")
            print(f"   salesQuantityKgL_log ≈ 0 means typical baseline week")
            print(f"   salesQuantityKgL_log ≈ 0.4 means 1.5x baseline (50% increase)")
    
    # ============================================================================
    # STEP 4: SIMULATE FORECAST DATA PREPARATION
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 4: Preparing forecast input (future 3 weeks)...")
    print(f"{'='*80}")
    
    # Create forecast skeleton (3 future weeks)
    forecast_weeks = pd.date_range(
        start=campaign_start_date,
        periods=3,
        freq='W-MON'
    )
    
    print(f"\n🔮 Forecasting for weeks:")
    for i, week in enumerate(forecast_weeks, 1):
        print(f"   Week {i}: {week.strftime('%Y-%m-%d')}")
    
    # Get the most recent data point (as of forecast_date)
    cutoff_date = forecast_date - pd.Timedelta(days=forecast_date.weekday())  # Align to Monday
    recent_data = df_item_weekly[df_item_weekly['week_start'] <= cutoff_date].copy()
    
    print(f"\n📊 Most recent data available (as of {forecast_date.strftime('%Y-%m-%d')}):")
    print(f"   Latest week in data: {recent_data['week_start'].max().strftime('%Y-%m-%d')}")
    print(f"   Weeks until campaign: {forecast_horizon_weeks}")
    
    # For each forecast week, we'd extract the lag features from recent_data
    # Example for our milk item:
    milk_recent = recent_data[recent_data['baseItemId'] == '528208'].tail(60)  # Last 60 weeks
    
    if len(milk_recent) > 0:
        # Week 1 forecast: use sales from 8 weeks before campaign start
        week1_lag_date = campaign_start_date - pd.Timedelta(weeks=forecast_horizon_weeks)
        week1_lag_data = milk_recent[milk_recent['week_start'] == week1_lag_date]
        
        print(f"\n🔍 Forecast Week 1 Input Features:")
        print(f"   Lag source date: {week1_lag_date.strftime('%Y-%m-%d')} ({forecast_horizon_weeks}w before campaign)")
        if len(week1_lag_data) > 0:
            print(f"   sales_lag_week1: {week1_lag_data.iloc[0]['salesQuantityKgL']:.2f} kg/L")
            print(f"   sales_lag_52w: {week1_lag_data.iloc[0]['sales_lag_52w']:.2f} kg/L")
        
        # Week 2 forecast: use sales from 9 weeks before week 2
        week2_lag_date = (campaign_start_date + pd.Timedelta(weeks=1)) - pd.Timedelta(weeks=forecast_horizon_weeks+1)
        week2_lag_data = milk_recent[milk_recent['week_start'] == week2_lag_date]
        
        print(f"\n🔍 Forecast Week 2 Input Features:")
        print(f"   Lag source date: {week2_lag_date.strftime('%Y-%m-%d')} ({forecast_horizon_weeks+1}w before week 2)")
        if len(week2_lag_data) > 0:
            print(f"   sales_lag_week2: {week2_lag_data.iloc[0]['salesQuantityKgL']:.2f} kg/L")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("✅ SIMULATION COMPLETE")
    print(f"{'='*80}")
    print("\n📝 Next Steps in Production:")
    print("   1. Train 3 LightGBM models (one per campaign week)")
    print("   2. For each model, use corresponding lag features")
    print("   3. At forecast time, extract lag features from cutoff date")
    print("   4. Predict all 3 weeks using the 3 trained models")
    print("   5. Sum predictions for total campaign impact")
    print(f"\n⚠️  Key Insight: Week 1 uses {forecast_horizon_weeks}w old data, Week 3 uses {forecast_horizon_weeks+2}w old data")
    print("   Each model is trained to handle its specific lag offset!")
    print(f"{'='*80}\n")