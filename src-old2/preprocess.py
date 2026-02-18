from fetch_data import fetch_data
import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from datetime import date, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _add_campaign_flag(df_orders: pd.DataFrame, campaign_calendar: pd.DataFrame):
    """
    Adds a binary flag to indicate if an item was on campaign during a given order date.
    """
    logging.info("Flagging campaign weeks in order data...")

    df = df_orders.merge(
        campaign_calendar,
        left_on=['itemId', 'salesCampignId'],
        right_on=['itemId', 'campaignId'],
        how='left'
    )

    df['is_campaign_week'] = (
        (df['orderDate'] >= df['campaignStartDate']) &
        (df['orderDate'] <= df['campaignEndDate'])
    ).astype(int)

    df = df.drop(columns=['campaignStartDate', 'campaignEndDate', 'campaignId'])

    logging.info("Campaign flag added successfully.")
    return df

def _add_cluster_temporal_features(df: pd.DataFrame, shift_weeks: int = 1):
    """Adds temporal features to the cluster sales data.
    This includes rolling averages, weeks since last campaign,
    and cumulative campaigns YTD.
    Parameters:
        df (pd.DataFrame): DataFrame containing cluster sales data with 'orderDate' column.
    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    logging.info("Adding temporal features to cluster sales data...")
    df = df.sort_values(["cluster1To1Id", "orderDate"])

    # Rolling mean (4 weeks)
    df["cluster_sales_rolling4w"] = (
        df.groupby("cluster1To1Id")["salesQuantityKgL"]
          .transform(lambda x: x.shift(shift_weeks).rolling(window=4, min_periods=1).mean())
    )

    # Weeks since last campaign (simplified)
    df["weeks_since_campaign"] = (
        df.groupby("cluster1To1Id")["is_campaign_week"]
        .transform(lambda x: x.ne(1).cumsum())
    )

    # Campaigns YTD (cumulative)
    df["campaigns_ytd"] = (
        df.sort_values("orderDate")
          .groupby("cluster1To1Id")["is_campaign_week"]
          .cumsum()
    )
    logging.info("Temporal features added successfully.")
    return df


def _add_cluster_features(df_orders: pd.DataFrame, df_items: pd.DataFrame, shift_weeks: int = 1):
    """
    Adds historical cluster sales (lag) as a feature to capture substitution effects.
    Ensures cluster1To1Id is filled without creating duplicate columns.
    """
    logging.info("Adding cluster-level context features...")

    df = df_orders.copy()

    if 'cluster1To1Id' not in df.columns:
        # merge in cluster1To1Id if it's absent
        df = df.merge(df_items[['itemId', 'cluster1To1Id']], on='itemId', how='left')
    else:
        # fill missing or empty cluster1To1Id from item master
        missing_mask = df['cluster1To1Id'].isna() | (df['cluster1To1Id'] == '')
        if missing_mask.any():
            helper = df_items[['itemId', 'cluster1To1Id']].rename(columns={'cluster1To1Id': 'cluster_from_items'})
            df = df.merge(helper, on='itemId', how='left')
            df.loc[missing_mask, 'cluster1To1Id'] = df.loc[missing_mask, 'cluster_from_items']
            df = df.drop(columns=['cluster_from_items'])

    # Sort by cluster and date
    df = df.sort_values(['cluster1To1Id', 'orderDate'])

    # Calculate previous week's cluster sales
    df['cluster_sales_lag_1'] = (
        df.groupby('cluster1To1Id')['salesQuantityKgL']
        .shift(shift_weeks)
        .fillna(0)
    )

    logging.info("Cluster-level features added successfully.")
    return df



def _add_time_features(df: pd.DataFrame):
    df['week'] = df['orderDate'].dt.isocalendar().week
    df['month'] = df['orderDate'].dt.month
    df['year'] = df['orderDate'].dt.year
    return df


def _add_missing_clusters(df_items):
    """
    Adds missing clusters to the item data by making items with missing `cluster1To1Id` take their own `itemId` as the cluster ID.
    This function ensures that all items have a valid cluster ID, which is essential for further processing in the data pipeline.
    
    Parameters:
        df_items (pd.DataFrame): DataFrame containing item data.
        
    Returns:
        pd.DataFrame: Updated DataFrame with missing clusters added.
    """
    logging.info("Adding missing clusters to item data...")    

    # Fill missing cluster1To1Id where cluster1To1Id is NaN or an empty string, using itemId as the cluster ID
    df_items['cluster1To1Id'] = df_items['cluster1To1Id'].replace('', pd.NA)
    df_items['cluster1To1Id'] = df_items['cluster1To1Id'].fillna(df_items['itemId'])

    logging.info("Missing clusters added successfully.")
    return df_items

def _create_campaign_calendar(df_campaigns, df_orders):
    """
    Creates a campaign calendar DataFrame that maps each item in an campaign 
    to the start and end dates of the campaign. df_orders is used to determine
    which items are in the campaign and df_campaigns is used to get the 
    campaign start and end dates.
    
    Parameters:
        df_campaigns (pd.DataFrame): DataFrame containing campaign data.
        df_orders (pd.DataFrame): DataFrame containing order data.
        
    Returns:
        pd.DataFrame: DataFrame with campaign IDs, start dates, and end dates.
    """
    logging.info("Creating campaign calendar...")

    # Cast salesCampignId and campaignId the same integer type
    df_orders['salesCampignId'] = df_orders['salesCampignId'].astype('Int64')
    df_campaigns['campaignId'] = df_campaigns['campaignId'].astype('Int64')
    print(df_orders.columns)
    # Get all unique compinations of itemId and salesCampignId from df_orders
    df_campaign_calendar = df_orders[['itemId', 'salesCampignId']].drop_duplicates()
    
    # Filter out rows where salesCampignId is 0, NaN or empty
    df_campaign_calendar = df_campaign_calendar[
        (df_campaign_calendar['salesCampignId'] != 0) &
        (df_campaign_calendar['salesCampignId'].notna()) &
        (df_campaign_calendar['salesCampignId'] != '')
    ]

    # Merge with df_campaigns to get start and end dates
    df_campaign_calendar = df_campaign_calendar.merge(
        df_campaigns,
        left_on='salesCampignId',
        right_on='campaignId',
        how='left'
    )

    # Remove rows where campaignId is not in df_campaigns
    df_campaign_calendar = df_campaign_calendar[
        df_campaign_calendar['campaignId'].notna()
    ]

    # Rename columns for clarity
    # Replace campaignId from df_campaigns with renamed salesCampignId
    df_campaign_calendar = df_campaign_calendar.drop(columns=['campaignId']).rename(
        columns={
            'salesCampignId': 'campaignId',
            'campaignStart': 'campaignStartDate',
            'campaignEnd': 'campaignEndDate'
        }
    )


    # Select relevant columns and drop duplicates
    df_campaign_calendar = df_campaign_calendar[['itemId', 'campaignId', 'campaignStartDate', 'campaignEndDate']]
    df_campaign_calendar = df_campaign_calendar.drop_duplicates()

    logging.info("Campaign calendar created successfully.")
    return df_campaign_calendar

def _aggregate_orders(df_orders: pd.DataFrame):
    """
    Aggregates order data by week and itemId, summing the salesQuantity.
    Adds the most common salesCampaignId during the week for each item.
    """
    logging.info("Aggregating order data...")

    df_orders['orderDate'] = pd.to_datetime(df_orders['orderDate'], errors='coerce')
    df_orders['orderDate'] = df_orders['orderDate'] - pd.to_timedelta(df_orders['orderDate'].dt.weekday, unit='d')

    # Sum salesQuantity per item and week
    df_weekly = (
        df_orders
        .groupby(['orderDate', 'itemId'])
        .agg({'salesQuantity': 'sum'})
        .reset_index()
    )

    # Förbered tom lista att fylla upp med varje artikels veckor
    rows = []
    for item_id, group in df_weekly.groupby('itemId'):
        start = group['orderDate'].min()
        end = group['orderDate'].max()
        weeks = pd.date_range(start=start, end=end, freq='W-MON')

        existing = group.set_index('orderDate')
        row = pd.DataFrame(index=weeks)
        row['itemId'] = item_id
        row['salesQuantity'] = existing['salesQuantity'] if not existing.empty else 0
        row.reset_index(names='orderDate', inplace=True)
        rows.append(row)

    df_filled = pd.concat(rows, ignore_index=True)
    df_filled['salesQuantity'] = df_filled['salesQuantity'].fillna(0)

    # Lägg till kampanj-ID från ursprungliga data, välj den vanligaste per vecka
    df_orders['orderDate'] = df_orders['orderDate'] - pd.to_timedelta(df_orders['orderDate'].dt.weekday, unit='d')
    df_campaign_mode = (
        df_orders[df_orders['salesCampignId'].notna() & (df_orders['salesCampignId'] != 0)]
        .groupby(['orderDate', 'itemId'])['salesCampignId']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
        .reset_index()
    )

    df_filled = df_filled.merge(df_campaign_mode, on=['orderDate', 'itemId'], how='left')
    df_filled['salesCampignId'] = df_filled['salesCampignId'].fillna(0).astype('Int64')

    logging.info("Order data aggregated successfully.")
    return df_filled



def _calc_weight(df_item):
    if 'weightKgPreparedItemComparisonUnit' in df_item and df_item['weightKgPreparedItemComparisonUnit'] != 0:
        return Decimal(df_item['weightKgPreparedItemComparisonUnit'])

    if 'netWeightKgComparisonUnit' in df_item and df_item['netWeightKgComparisonUnit'] != 0:
        return Decimal(df_item['netWeightKgComparisonUnit'])

    return Decimal(df_item.get('grossWeightKgComparisonUnit', 1))


def _convert_order_quantities(df_orders: pd.DataFrame, df_items: pd.DataFrame):
    """
    Converts order quantities to kg/l based on the _calc_weight function from df_items.
    This function ensures that the order quantities are in the correct units for further processing.
    Parameters:
        df_orders (pd.DataFrame): DataFrame containing order data.
        df_items (pd.DataFrame): DataFrame containing item data.
    Returns:
        pd.DataFrame: Updated DataFrame with order quantities converted to kg/l.
    """
    logging.info("Precomputing weights per item...")

    # Calculate weight once per item
    df_items = df_items.copy()
    df_items['weight'] = df_items.apply(_calc_weight, axis=1)

    logging.info("Merging item weights into order data...")
    df = df_orders.merge(df_items[['itemId', 'weight']], how='left', on='itemId')

    logging.info("Computing salesQuantityKgL...")
    df['salesQuantityKgL'] = (df['salesQuantity'] * df['weight']).astype(float)

    df = df.drop(columns=['salesQuantity', 'weight'])
    logging.info("Order quantities converted to kg/l successfully.")
    return df


def preprocess(item_ids=None):
    
    reference_date = date.today() - timedelta(days=date.today().weekday())


    if item_ids is None:
        raise ValueError("item_ids must be provided")
    logging.info("Starting preprocessing pipeline...")
    df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=item_ids)

    df_orders = _aggregate_orders(df_orders)
    df_items = _add_missing_clusters(df_items)
    campaign_calendar = _create_campaign_calendar(df_campaigns, df_orders)

    df_orders = _convert_order_quantities(df_orders, df_items)
    
    df_orders = _add_campaign_flag(df_orders, campaign_calendar)
    df_orders = _add_cluster_features(df_orders, df_items)
    df_orders = _add_time_features(df_orders)
    df_orders = _add_cluster_temporal_features(df_orders)

    df_orders.to_parquet("../data/processed/orders_processed.parquet", index=False)
    df_items.to_parquet("../data/processed/items_processed.parquet", index=False)

    dupes = df_orders[df_orders.duplicated(subset=["orderDate", "itemId", "cluster1To1Id"], keep=False)]
    print(f"{len(dupes)} duplicates found")
    print(dupes.head(10))
    print(df_items.head())
    print(df_orders.head())
    logging.info("Preprocessing completed successfully. Data saved to 'data/processed/' directory.")


if __name__ == "__main__":
    preprocess()