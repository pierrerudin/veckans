import pandas as pd
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _calc_weight(df_item):
    if 'weightKgPreparedItemComparisonUnit' in df_item and df_item['weightKgPreparedItemComparisonUnit'] != 0:
        return float(df_item['weightKgPreparedItemComparisonUnit'])

    if 'netWeightKgComparisonUnit' in df_item and df_item['netWeightKgComparisonUnit'] != 0:
        return float(df_item['netWeightKgComparisonUnit'])

    return float(df_item.get('grossWeightKgComparisonUnit', 1))


def load_predictions(item_forecast_path="data/output/item_sales_forecast.parquet", cluster_forecast_path="data/output/cluster_sales_forecast.parquet"):
    """Load cluster and item-level forecasts."""
    item_forecast = pd.read_parquet(item_forecast_path)
    cluster_forecast = pd.read_parquet(cluster_forecast_path)
    return item_forecast, cluster_forecast

def merge_forecasts(item_forecast, cluster_forecast):
    """
    Merge item and cluster forecasts on orderDate and cluster1To1Id.
    This calculates the predicted quantity based on item share and total cluster sales.
    Parameters:
        item_forecast: DataFrame containing item-level forecasts
        cluster_forecast: DataFrame containing cluster-level forecasts
    Returns:
        Merged DataFrame with predicted quantities.
    """
    logging.info("Merging item and cluster forecasts...")
    df = pd.merge(
        item_forecast,
        cluster_forecast,
        on=["orderDate", "cluster1To1Id"],
        suffixes=("_item", "_cluster")
    )

    df["predicted_quantity_kg"] = df["predicted_item_share"] * df["predicted_cluster_sales_kg"]
    logging.info("Merged forecast data successfully.")
    return df

def filter_campaign_items(df, orders_path="data/processed/orders_processed.parquet"):
    """Keep only rows where item is on campaign that week."""
    df_orders = pd.read_parquet(orders_path)
    df = df.merge(
        df_orders[["orderDate", "itemId", "is_campaign_week"]],
        on=["orderDate", "itemId"],
        how="left"
    )
    return df[df["is_campaign_week"] == 1].copy()

def _process_output(df: pd.DataFrame, df_items: pd.DataFrame):
    """Process the final output DataFrame."""
    logging.info("Processing final output DataFrame...")
    df_items = df_items.copy()
    df_items['weight'] = df_items.apply(_calc_weight, axis=1)

    logging.info("Merging item weights into order data...")
    df = df.merge(df_items[['itemId', 'weight']], how='left', on='itemId')

    logging.info("Computing predicted_quantity...")
    df['weight'] = df['weight'].replace(0, np.nan).fillna(1.0).astype(float)
    df['predicted_quantity'] = (df['predicted_quantity_kg'] / df['weight']).round().astype(float)
    logging.info("Final output DataFrame processed successfully.")
    return df

def save_output(df, output_path="data/output/campaign_item_forecast.parquet"):
    """Save final filtered forecast output."""
    logging.info(f"Saving filtered campaign item forecast to {output_path}...")
    output_cols = [
        "orderDate", "itemId", "cluster1To1Id", "predicted_item_share",
        "predicted_cluster_sales_kg", "predicted_quantity_kg", "predicted_quantity"
    ]
    df[output_cols].to_parquet(output_path, index=False)
    logging.info(f"Filtered campaign item forecast saved to {output_path}.")

def main(
    item_forecast_path="data/output/item_sales_forecast.parquet",
    cluster_forecast_path="data/output/cluster_sales_forecast.parquet",
    orders_path="data/processed/orders_processed.parquet",
    items_path="data/processed/items_processed.parquet",
    output_path="data/output/campaign_item_forecast.parquet"
):
    item_forecast, cluster_forecast = load_predictions(
        item_forecast_path=item_forecast_path,
        cluster_forecast_path=cluster_forecast_path
    )
    df = merge_forecasts(item_forecast, cluster_forecast)
    df = filter_campaign_items(df, orders_path=orders_path)
    df_items = pd.read_parquet(items_path)
    df = _process_output(df, df_items)
    save_output(df, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine item & cluster forecasts and filter campaign items.")
    parser.add_argument("--item_forecast_path", default="data/output/item_sales_forecast.parquet")
    parser.add_argument("--cluster_forecast_path", default="data/output/cluster_sales_forecast.parquet")
    parser.add_argument("--orders_path", default="data/processed/orders_processed.parquet")
    parser.add_argument("--items_path", default="data/processed/items_processed.parquet")
    parser.add_argument("--output_path", default="data/output/campaign_item_forecast.parquet")
    args = parser.parse_args()
    main(
        item_forecast_path=args.item_forecast_path,
        cluster_forecast_path=args.cluster_forecast_path,
        orders_path=args.orders_path,
        items_path=args.items_path,
        output_path=args.output_path
    )