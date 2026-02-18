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

def _fetch_order_data(datalake_connector: AzureDataLakeConnector, df_campaigns: pd.DataFrame = None, df_items: pd.DataFrame = None):
    """
    Fetches raw data from the Azure Data Lake.
    
    Parameters:
        datalake_connector (AzureDataLakeConnector): The connector to interact with the Data Lake.
    """
    
    filter_expression = (
        (ds.field("orderCategoryId") == 1) &
        ds.field("orderCancelledDatetime").is_null() &
        (ds.field("cancelFlag") == "N") &
        (ds.field("deliveryDate") >= date(2021, 1, 1))
    )

    columns = ['orderDate', 'itemId', 'salesCampignId', 'salesQuantity']

    if df_campaigns is not None:
        campaign_ids = df_campaigns.campaignId.unique()
        logging.info(f"Fetching campaign order data for {len(campaign_ids)} unique campaigns, from Azure Data Lake...")
        filter_expression = (
            (ds.field("salesCampignId").isin(campaign_ids)) &
            (ds.field("orderCategoryId") == 1) &
            ds.field("orderCancelledDatetime").is_null() &
            (ds.field("cancelFlag") == "N") &
            (ds.field("deliveryDate") >= date(2021, 1, 1))
        )
    elif df_items is not None:
        item_ids = df_items.itemId.unique()
        logging.info(f"Fetching historical order data for {len(item_ids)} unique items, from Azure Data Lake...")
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
        "columns": columns,
        "filter_expression": filter_expression,
        "use_cache": True,
        "storage_format": "delta"
    }
    df_orders = datalake_connector.read_data(**query)
    logging.info(f"Successfully fetched {df_orders.shape[0]} rows of order data.")
    return df_orders


def _fetch_item_data(datalake_connector: AzureDataLakeConnector, df_campaign_items: pd.DataFrame = None, df_item_clusters: pd.DataFrame = None):
    """
    Fetches item data from the Azure Data Lake.
    
    Parameters:
        datalake_connector (AzureDataLakeConnector): The connector to interact with the Data Lake.
    """
    logging.info("Fetching item data from Azure Data Lake...")
    columns = ['itemId', 'itemDesc', 'cluster1To1Id', 'weightKgPreparedItemComparisonUnit', 'netWeightKgComparisonUnit', 'grossWeightKgComparisonUnit']
    filter_expression = None

    if df_campaign_items is not None:
        campaign_item_ids = df_campaign_items.itemId.unique()
        columns = ['itemId', 'cluster1To1Id']
        filter_expression = (ds.field("itemId").isin(campaign_item_ids))
    elif df_item_clusters is not None:
        cluster_ids = df_item_clusters.cluster1To1Id.unique()
        filter_expression = (ds.field("cluster1To1Id").isin(cluster_ids))

    query = {
        "container_name": "datalake",
        "path_in_container": "dm/out/dim_item",
        "batch_size": 5_000_000,
        "columns": columns,
        "filter_expression": filter_expression,
        "use_cache": True,
        "storage_format": "delta"
    }
    df_items = datalake_connector.read_data(**query)

    if df_item_clusters is not None and 'itemId' in df_item_clusters.columns:
        valid_item_ids = df_item_clusters['itemId'].unique()
        df_items = df_items[~(
            df_items['cluster1To1Id'].apply(_is_empty) &
            ~df_items['itemId'].isin(valid_item_ids)
        )]

    df_items = df_items.drop_duplicates(subset='itemId', keep='first')
    print(df_items.head())
    logging.info("Item data fetched successfully.")
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
            df_weekly_deals = pd.read_excel("data/weekly_deals.xlsx", sheet_name=f"{year}")
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
    Fetch item and order data from Azure Data Lake Storage.
    Read weekly deal data from excel file
    Returns:
    --------
    df_items: pandas DataFrame
    df_orders: pandas DataFrame
    df_weekly_deals: pandas DataFrame
    """
    logging.info("Starting data fetch process...")
    adl_connector = AzureDataLakeConnector(**LAKE_CONFIG)
    df_campaigns = _read_campaign_data()

    # 1. Fetch orders for actual campaigns to know which items were on campaign
    df_campaign_orders = _fetch_order_data(adl_connector, df_campaigns=df_campaigns)
    campaign_item_ids = df_campaign_orders['itemId'].unique().tolist()

    # 2. Combine campaign items with simulated items
    simulate_item_ids = simulate_item_ids or []
    all_focus_item_ids = list({*campaign_item_ids, *simulate_item_ids})

    # 3. Fetch those focus items to get their clusters
    df_focus_items = _fetch_item_data(
        adl_connector,
        df_campaign_items=pd.DataFrame({"itemId": all_focus_item_ids}),
        df_item_clusters=None
    )

    # 4. Extract cluster IDs from focus items
    cluster_ids = df_focus_items['cluster1To1Id'].dropna().unique().tolist()

    # 5. Fetch all items in those clusters (peers)
    if cluster_ids:
        df_cluster_items = pd.DataFrame({"cluster1To1Id": cluster_ids})
        df_items = _fetch_item_data(adl_connector, df_campaign_items=None, df_item_clusters=df_cluster_items)
    else:
        df_items = df_focus_items.copy()

    # 6. Fetch orders for the expanded item universe (campaign + simulated + their cluster peers)
    df_orders = _fetch_order_data(adl_connector, df_campaigns=None, df_items=df_items)

    # Remove unnecessary dataframes from memory
    # del df_campaign_items
    # del df_item_clusters

    logging.info("Data fetch process completed successfully.")
    return df_items, df_orders, df_campaigns

if __name__ == "__main__":
    df_items, df_orders, df_weekly_deals = fetch_data()