from azure_datalake_connector import AzureDataLakeConnector
from config import LAKE_CONFIG
import logging
import pyarrow.dataset as ds
import pandas as pd
from datetime import date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure empty values are properly detected (NaN or empty string)
def _is_empty(val):
    return pd.isna(val) or val == ''

def _fetch_orders_by_campaigns(datalake_connector: AzureDataLakeConnector, campaign_ids: list):
    """
    Fetch orders for specific campaigns (to learn which items were on campaign).
    
    Parameters:
        datalake_connector: The connector to interact with the Data Lake
        campaign_ids: List of campaign IDs to fetch orders for
    
    Returns:
        DataFrame with order history for those campaigns
    """
    logging.info(f"Fetching campaign order data for {len(campaign_ids)} unique campaigns from Azure Data Lake...")
    
    filter_expression = (
        (ds.field("salesCampignId").isin(campaign_ids)) &
        (ds.field("orderCategoryId") == 1) &
        ds.field("orderCancelledDatetime").is_null() &
        (ds.field("cancelFlag") == "N") &
        (ds.field("deliveryDate") >= date(2021, 1, 1))
    )
    
    query = {
        "container_name": "datalake",
        "path_in_container": "dm/out/fact_order",
        "batch_size": 5_000_000,
        "columns": ['orderDate', 'itemId', 'salesCampignId', 'salesQuantity'],
        "filter_expression": filter_expression,
        "use_cache": True,
        "storage_format": "delta"
    }
    df_orders = datalake_connector.read_data(**query)
    logging.info(f"Successfully fetched {df_orders.shape[0]} rows of campaign order data.")
    return df_orders


def _fetch_orders_by_items(datalake_connector: AzureDataLakeConnector, item_ids: list):
    """
    Fetch complete order history for specific items (for training models).
    
    Parameters:
        datalake_connector: The connector to interact with the Data Lake
        item_ids: List of item IDs to fetch order history for
    
    Returns:
        DataFrame with complete order history for those items
    """
    logging.info(f"Fetching historical order data for {len(item_ids)} unique items from Azure Data Lake...")
    
    filter_expression = (
        (ds.field("itemId").isin(item_ids)) &
        (ds.field("orderCategoryId") == 1) &
        ds.field("orderCancelledDatetime").is_null() &
        (ds.field("cancelFlag") == "N") &
        (ds.field("deliveryDate") >= date(2021, 1, 1))
    )
    
    query = {
        "container_name": "datalake",
        "path_in_container": "dm/out/fact_order",
        "batch_size": 5_000_000,
        "columns": ['orderDate', 'itemId', 'salesCampignId', 'salesQuantity'],
        "filter_expression": filter_expression,
        "use_cache": True,
        "storage_format": "delta"
    }
    df_orders = datalake_connector.read_data(**query)
    logging.info(f"Successfully fetched {df_orders.shape[0]} rows of order data.")
    return df_orders


def _fetch_items_by_ids(datalake_connector, item_ids: list, columns: list = None):
    """
    Fetch specific items by their item IDs.
    
    Parameters:
        datalake_connector: The connector to interact with the Data Lake
        item_ids: List of item IDs to fetch
        columns: List of columns to return (None = all columns)
    
    Returns:
        DataFrame with requested items
    """
    if columns is None:
        columns = ['itemId', 'itemDesc', 'cluster1To1Id', 'weightKgPreparedItemComparisonUnit', 
                   'netWeightKgComparisonUnit', 'grossWeightKgComparisonUnit']
    
    logging.info(f"Fetching {len(item_ids)} specific items from Azure Data Lake...")
    
    query = {
        "container_name": "datalake",
        "path_in_container": "dm/out/dim_item",
        "batch_size": 5_000_000,
        "columns": columns,
        "filter_expression": ds.field("itemId").isin(item_ids),
        "use_cache": True,
        "storage_format": "delta"
    }
    df_items = datalake_connector.read_data(**query)
    
    logging.info(f"Fetched {len(df_items)} items")
    return df_items


def _fetch_items_by_clusters(datalake_connector, cluster_ids: list, columns: list = None):
    """
    Fetch all items belonging to specific clusters.
    
    Parameters:
        datalake_connector: The connector to interact with the Data Lake
        cluster_ids: List of cluster IDs to fetch items from
        columns: List of columns to return (None = all columns)
    
    Returns:
        DataFrame with all items in the specified clusters
    """
    if columns is None:
        columns = ['itemId', 'itemDesc', 'cluster1To1Id', 'weightKgPreparedItemComparisonUnit', 
                   'netWeightKgComparisonUnit', 'grossWeightKgComparisonUnit']
    
    logging.info(f"Fetching all items from {len(cluster_ids)} clusters from Azure Data Lake...")
    
    query = {
        "container_name": "datalake",
        "path_in_container": "dm/out/dim_item",
        "batch_size": 5_000_000,
        "columns": columns,
        "filter_expression": ds.field("cluster1To1Id").isin(cluster_ids),
        "use_cache": True,
        "storage_format": "delta"
    }
    df_items = datalake_connector.read_data(**query)
    
    logging.info(f"Fetched {len(df_items)} items from {len(cluster_ids)} clusters")
    
    # Log cluster composition for debugging
    for cluster_id in cluster_ids[:5]:  # Show first 5 clusters
        items_in_cluster = df_items[df_items['cluster1To1Id'] == cluster_id]['itemId'].nunique()
        logging.info(f"  Cluster {cluster_id}: {items_in_cluster} items")
    
    return df_items

def _read_campaign_data():
    """
    Reads campaign data from an Excel file.
    
    Returns:
        pandas DataFrame: DataFrame containing campaign data.
    """
    try:
        years = list(range(2022, date.today().year + 1))
        logging.info("Reading campaign data from Excel file...")
        campaign_data = {'campaignId': [], 'campaignStart': [], 'campaignEnd': []}
        for year in years:
            df_weekly_deals = pd.read_excel("../data/weekly_deals.xlsx", sheet_name=f"{year}")
            campaign_id, campaign_start, campaign_end = df_weekly_deals.iloc[[1, 4, 5], 1:].values
            campaign_data["campaignId"].extend(campaign_id)
            campaign_data["campaignStart"].extend(campaign_start)
            campaign_data["campaignEnd"].extend(campaign_end)
        campaign_df = pd.DataFrame(campaign_data)
        logging.info("Campaign data read successfully.")
        return campaign_df
    except Exception as e:
        logging.error(f"Error reading campaign data: {e}")
        raise

def fetch_data(simulate_item_ids: list[str] = None):
    """
    Main data fetching function with clear logic:
    1. Fetch campaign orders to know which items were on campaign
    2. Combine campaign items with simulated items (items we want to forecast)
    3. Fetch those focus items to get their cluster IDs
    4. Fetch ALL items in those clusters (to learn substitution patterns)
    5. Fetch all order history for all items
    
    Parameters:
        simulate_item_ids: List of item IDs to forecast (in addition to campaign items)
    
    Returns:
        df_items: All items (focus items + their cluster peers)
        df_orders: Order history for all items
        df_campaigns: Campaign calendar
    """
    logging.info("Starting data fetch process...")
    adl_connector = AzureDataLakeConnector(**LAKE_CONFIG)
    df_campaigns = _read_campaign_data()

    # Step 1: Fetch orders for actual campaigns to know which items were on campaign
    campaign_ids = df_campaigns['campaignId'].unique().tolist()
    df_campaign_orders = _fetch_orders_by_campaigns(adl_connector, campaign_ids)
    campaign_item_ids = df_campaign_orders['itemId'].unique().tolist()
    logging.info(f"Found {len(campaign_item_ids)} items from campaign history")

    # Step 2: Combine campaign items with simulated items
    simulate_item_ids = simulate_item_ids or []
    all_focus_item_ids = list(set(campaign_item_ids) | set(simulate_item_ids))
    logging.info(f"Total focus items (campaign + simulated): {len(all_focus_item_ids)}")

    # Step 3: Fetch those focus items to get their cluster IDs (minimal columns for speed)
    df_focus_items = _fetch_items_by_ids(
        adl_connector, 
        all_focus_item_ids,
        columns=['itemId', 'cluster1To1Id']
    )

    # Step 4: Extract cluster IDs and fetch ALL items in those clusters
    cluster_ids = df_focus_items['cluster1To1Id'].dropna().unique().tolist()
    logging.info(f"Found {len(cluster_ids)} unique clusters from focus items")
    
    if cluster_ids:
        # Fetch all items in these clusters (full details)
        df_items = _fetch_items_by_clusters(adl_connector, cluster_ids)
        
        # Log cluster composition for debugging
        for cluster_id in cluster_ids[:5]:
            items_in_cluster = df_items[df_items['cluster1To1Id'] == cluster_id]['itemId'].nunique()
            logging.info(f"  Cluster {cluster_id}: {items_in_cluster} items")
        
        # Handle focus items that don't have a cluster assigned
        items_fetched = set(df_items['itemId'].unique())
        missing_focus_items = set(all_focus_item_ids) - items_fetched
        
        if missing_focus_items:
            logging.warning(f"Fetching {len(missing_focus_items)} focus items without cluster assignments...")
            df_missing = _fetch_items_by_ids(adl_connector, list(missing_focus_items))
            df_items = pd.concat([df_items, df_missing], ignore_index=True)
    else:
        # No clusters found, just use focus items
        logging.warning("No cluster IDs found for focus items. Fetching items individually...")
        df_items = _fetch_items_by_ids(adl_connector, all_focus_item_ids)
    
    # Remove duplicates
    df_items = df_items.drop_duplicates(subset='itemId', keep='first')
    logging.info(f"Total unique items after fetching clusters: {len(df_items)}")

    # Step 5: Fetch orders for all items
    all_item_ids = df_items['itemId'].unique().tolist()
    df_orders = _fetch_orders_by_items(adl_connector, all_item_ids)

    logging.info("Data fetch process completed successfully.")
    return df_items, df_orders, df_campaigns

if __name__ == "__main__":
    df_items, df_orders, df_weekly_deals = fetch_data()