from azure_datalake_connector import AzureDataLakeConnector
from config import LAKE_CONFIG
import pandas as pd
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data():
    data_lake_connector = AzureDataLakeConnector(**LAKE_CONFIG)
    # Fetch item data
    logging.info("Fetching item data from Azure Data Lake...")

    columns = ['itemId', 'itemDesc', 'cluster1To1Id', 'itemSkey']
    filter_expression = None

    query = {
            "container_name": "datalake",
            "path_in_container": "dm/out/dim_item",
            "batch_size": 5_000_000,
            "columns": columns,
            "filter_expression": filter_expression,
            "use_cache": False,  # DISABLE CACHE to see real data
            "storage_format": "delta"
        }
    df_items = data_lake_connector.read_data(**query)

    logging.info(f"Successfully fetched {df_items.shape[0]} rows of item data.")
    return df_items

def main():
    df_items = fetch_data()
    cluster_id = df_items.loc[df_items['itemId'] == '528208', 'cluster1To1Id'].values[0]
    print(f"Cluster ID for item 528208: {cluster_id}")
    print(df_items[df_items['cluster1To1Id'] == cluster_id])
    print(f"Found {len(df_items[df_items['cluster1To1Id'] == cluster_id])} items in cluster {cluster_id}")


if __name__ == "__main__":
    main()