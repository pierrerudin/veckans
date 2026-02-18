import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import argparse
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def _train_model(X_train: pd.DataFrame, y_train: pd.Series):
#     logging.info("Training LightGBM model...")
#     model = lgb.LGBMRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     # Save the model using pickle
#     with open("data/models/cluster_sales_model.pkl", "wb") as f:
#         pickle.dump(model, f)
#     logging.info("Model training completed successfully.")
#     return model

def _evaluate_model(X_test: pd.DataFrame, y_test: pd.Series, model: lgb.LGBMRegressor):
    """
    Evaluates the model and saves the predictions.
    Parameters:
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): Target variable for testing.
        model (lgb.LGBMRegressor): Trained LightGBM model.
    Returns:
        y_pred (np.ndarray): Predictions on the test set.
    """
    logging.info("Evaluating model and saving predictions...")

    # Predict
    log_preds = model.predict(X_test)
    log_preds = np.clip(log_preds, a_min=-10, a_max=20)  # prevent overflow
    y_pred = np.expm1(log_preds)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    smape = 100 * (2 * abs(y_test - y_pred) / (abs(y_test) + abs(y_pred))).mean()
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"SMAPE: {smape:.4f}")
    logging.info("Model evaluation completed successfully.")

def _prepare_horizon_datasets(df_cluster: pd.DataFrame, horizon: int, cutoff_date: str):
    """
    For a given horizon h, create X/y shifted so that features at t predict cluster_sales_kg at t+h.
    """
    df_cluster = df_cluster.sort_values(["cluster1To1Id", "orderDate"])
    target_col = f"target_h{horizon}"
    df_cluster[target_col] = df_cluster.groupby("cluster1To1Id")["cluster_sales_kg"].shift(-horizon)

    # Drop rows without a target (because of shift)
    df = df_cluster[df_cluster[target_col].notna()].copy()

    # Train/test split based on original orderDate (features are at time t)
    df['orderDate'] = pd.to_datetime(df['orderDate'])
    train = df[df["orderDate"] < cutoff_date]
    test = df[df["orderDate"] >= cutoff_date]

    X_train = train.drop(columns=["cluster_sales_kg", "orderDate"] + [c for c in df_cluster.columns if c.startswith("target_h")])
    y_train = np.log1p(train[target_col].clip(lower=1e-3))  # log1p transform
    X_test = test.drop(columns=["cluster_sales_kg", "orderDate"] + [c for c in df_cluster.columns if c.startswith("target_h")])
    y_test = test[target_col]  # keep in natural scale for evaluation

    # Categorical
    X_train['cluster1To1Id'] = X_train['cluster1To1Id'].astype('category')
    X_test['cluster1To1Id'] = X_test['cluster1To1Id'].astype('category')

    return X_train, y_train, X_test, y_test

def _split_data(df_cluster: pd.DataFrame, cutoff_date: str = "2025-06-01"):
    """
    Splits the cluster sales data into training and testing sets based on a cutoff date.
    
    Parameters:
        df_cluster (pd.DataFrame): DataFrame containing cluster sales data with 'orderDate' column.
    
    Returns:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): Target variable for testing.
    """
    logging.info("Splitting data into training and testing sets...")
    if 'orderDate' not in df_cluster.columns:
        raise ValueError("DataFrame must contain 'orderDate' column for splitting.")

    df_cluster['orderDate'] = pd.to_datetime(df_cluster['orderDate'])
    df_cluster = df_cluster.sort_values("orderDate")

    train = df_cluster[df_cluster["orderDate"] < cutoff_date]
    test = df_cluster[df_cluster["orderDate"] >= cutoff_date]

    X_train = train.drop(columns=["cluster_sales_kg", "orderDate"])
    train_target = train["cluster_sales_kg"].clip(lower=1e-3)
    y_train = np.log1p(train_target)

    X_test = test.drop(columns=["cluster_sales_kg", "orderDate"])
    y_test = test["cluster_sales_kg"]

    X_train['cluster1To1Id'] = X_train['cluster1To1Id'].astype('category')
    X_test['cluster1To1Id'] = X_test['cluster1To1Id'].astype('category')

    logging.info(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def _aggregate_sales_by_cluster(df_orders: pd.DataFrame):
    """
    Aggregates sales data by cluster, calculating total sales and campaign flags.
    Parameters:
        df_orders (pd.DataFrame): DataFrame containing order data with cluster information.
    Returns:
        pd.DataFrame: DataFrame with aggregated sales data by cluster.
    """
    logging.info("Aggregating sales data by cluster...")
    required_columns = [
        'orderDate', 'cluster1To1Id', 'campaigns_ytd', 'salesQuantityKgL',
        'cluster_sales_lag_1', 'cluster_sales_rolling4w', 'weeks_since_campaign',
        'is_campaign_week', 'month', 'week', 'year'
    ]
    if not all(col in df_orders.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    df_cluster = (
        df_orders.groupby(['orderDate', 'cluster1To1Id'])
        .agg(
            cluster_sales_kg=('salesQuantityKgL', 'sum'),
            is_campaign_week=('is_campaign_week', 'max'),
            cluster_sales_lag_1=('cluster_sales_lag_1', 'mean'),
            cluster_sales_rolling4w=('cluster_sales_rolling4w', 'mean'),
            weeks_since_campaign=('weeks_since_campaign', 'mean'),
            campaigns_ytd=('campaigns_ytd', 'max'),
            month=('month', 'first'),
            week=('week', 'first'),
            year=('year', 'first')
        )
        .reset_index()
    )
    logging.info("Aggregated cluster sales data successfully.")
    return df_cluster


def _load_processed_data(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet"):
    """
    Loads processed order and item data from local parquet files.
    
    Returns:
        tuple: DataFrames containing processed order and item data.
    """
    logging.info("Loading processed data from local files...")
    df_orders = pd.read_parquet(orders_path)
    df_items = pd.read_parquet(items_path)
    logging.info(f"Loaded {df_orders.shape[0]} rows of order data and {df_items.shape[0]} rows of item data.")
    return df_orders, df_items


# def train_cluster_sales_model(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet", output_path="data/output/cluster_sales_forecast.parquet", cutoff_date="2025-06-01"):
#     df_orders, df_items = _load_processed_data(orders_path=orders_path, items_path=items_path)

#     df_cluster = _aggregate_sales_by_cluster(df_orders)
#     X_train, y_train, X_test, y_test = _split_data(df_cluster, cutoff_date=cutoff_date)
#     model = _train_model(X_train, y_train)
#     _evaluate_model(X_test, y_test, model)

def train_cluster_sales_models(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet", output_path="data/output/cluster_sales_forecast.parquet", horizons=[2,3,4], cutoff_date="2025-04-01"):
    df_orders, _ = _load_processed_data(orders_path=orders_path, items_path=items_path)
    df_cluster = _aggregate_sales_by_cluster(df_orders)


    for h in horizons:
        logging.info(f"Training cluster model horizon={h} weeks ahead")
        X_train, y_train, X_test, y_test = _prepare_horizon_datasets(df_cluster, horizon=h, cutoff_date=cutoff_date)
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)
        # save model with horizon in name
        model_path = f"data/models/cluster_sales_model_h{h}.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Saved cluster horizon-{h} model to {model_path}")
        # evaluate
        _evaluate_model(X_test, y_test, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and forecast cluster sales.")
    parser.add_argument("--orders_path", default="data/processed/orders_processed.parquet")
    parser.add_argument("--items_path", default="data/processed/items_processed.parquet")
    parser.add_argument("--output_path", default="data/output/cluster_sales_forecast.parquet")
    parser.add_argument("--cutoff_date", default="2025-06-01")
    args = parser.parse_args()
    train_cluster_sales_model(orders_path=args.orders_path, items_path=args.items_path, output_path=args.output_path, cutoff_date=args.cutoff_date)
