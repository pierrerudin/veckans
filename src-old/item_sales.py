import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import logging
import argparse
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet"):
    """Load preprocessed order and item data."""
    df_orders = pd.read_parquet(orders_path)
    df_items = pd.read_parquet(items_path)
    return df_orders, df_items

def _prepare_item_horizon(df: pd.DataFrame, horizon: int, cutoff_date: str):
    df = df.sort_values(["itemId", "orderDate"]).copy()
    target_col = f"target_h{horizon}"
    df[target_col] = df.groupby("itemId")["item_share_of_cluster"].shift(-horizon)

    df = df[df[target_col].notna()]
    df['orderDate'] = pd.to_datetime(df['orderDate'])
    train = df[df["orderDate"] < cutoff_date]
    test = df[df["orderDate"] >= cutoff_date]

    features = [
        "itemId", "cluster1To1Id",
        "is_campaign_week", "weeks_since_campaign", "other_campaign_in_cluster",
        "campaigns_ytd", "month", "week", "item_sales_lag_1", "item_share_lag_1"
    ]

    X_train = train[features].copy()
    y_train = train[target_col]
    X_test = test[features].copy()
    y_test = test[target_col]

    X_train["itemId"] = X_train["itemId"].astype("category")
    X_train["cluster1To1Id"] = X_train["cluster1To1Id"].astype("category")
    X_test["itemId"] = X_test["itemId"].astype("category")
    X_test["cluster1To1Id"] = X_test["cluster1To1Id"].astype("category")

    return X_train, y_train, X_test, y_test


def prepare_data(df_orders, df_items):
    """Generate training data for item share prediction."""

    # Clean sales figures
    df_orders["salesQuantityKgL"] = df_orders["salesQuantityKgL"].clip(lower=0.0)

    # Compute cluster totals and avoid division by zero
    cluster_totals = df_orders.groupby(["orderDate", "cluster1To1Id"])["salesQuantityKgL"].transform("sum")
    cluster_totals = cluster_totals.clip(lower=1e-6)
    df_orders["item_share_of_cluster"] = df_orders["salesQuantityKgL"] / cluster_totals

    # Feature: other article on campaign in same cluster and week
    df_orders["campaign_in_cluster"] = df_orders.groupby(["orderDate", "cluster1To1Id"])["is_campaign_week"].transform("max")
    df_orders["other_campaign_in_cluster"] = (
        (df_orders["campaign_in_cluster"] == 1) & (df_orders["is_campaign_week"] == 0)
    ).astype(int)
    df_orders.drop(columns="campaign_in_cluster", inplace=True)

    # Lag features
    logging.info("Creating item-level lag features...")
    df_orders = df_orders.sort_values(["itemId", "orderDate"])
    df_orders["item_sales_lag_1"] = df_orders.groupby("itemId")["salesQuantityKgL"].shift(1).fillna(0)
    df_orders["item_share_lag_1"] = df_orders.groupby("itemId")["item_share_of_cluster"].shift(1).fillna(0)

    # Drop rows where target is missing
    df_orders = df_orders[df_orders["item_share_of_cluster"].notna()]

    # Convert IDs to categorical
    df_orders["itemId"] = df_orders["itemId"].astype("category")
    df_orders["cluster1To1Id"] = df_orders["cluster1To1Id"].astype("category")

    return df_orders

def prepare_features_and_target(df):
    """Extract model input features and target."""
    features = [
        "itemId", "cluster1To1Id",
        "is_campaign_week", "weeks_since_campaign", "other_campaign_in_cluster",
        "campaigns_ytd", "month", "week", "item_sales_lag_1", "item_share_lag_1"
    ]
    target = "item_share_of_cluster"

    X = df[features].copy()
    y = df[target].copy()

    return X, y

def train_test_split(df, X, y, cutoff_date="2025-06-01"):
    """Split data into time-based train/test sets."""
    df_train = df[df["orderDate"] < cutoff_date]
    df_test = df[df["orderDate"] >= cutoff_date]

    X_train = df_train[X.columns]
    y_train = df_train[y.name]
    X_test = df_test[X.columns]
    y_test = df_test[y.name]

    return X_train, y_train, X_test, y_test, df_test

def _evaluate_model(model, X_test, y_test, df_test, output_path="data/output/item_sales_forecast.parquet"):
    """Evaluate model and save predictions."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    smape = 100 * (2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))).mean()
    logging.info(f"Item share model MAE: {mae:.4f}")
    logging.info(f"Item share model SMAPE: {smape:.4f}")

    logging.info(f"Item-level predictions saved to {output_path}.")

# def _train_model(X_train, y_train):
#     """Train LightGBM model for item share prediction."""
#     logging.info("Training LightGBM model for item share prediction...")
#     model = lgb.LGBMRegressor(random_state=42)
#     model.fit(X_train, y_train, categorical_feature=["itemId", "cluster1To1Id"])
#     # Save the model using pickle
#     with open("data/models/item_share_model.pkl", "wb") as f:
#         pickle.dump(model, f)
#     logging.info("Model training completed successfully.")
#     return model


# def train_item_sales_model(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet", output_path="data/output/item_sales_forecast.parquet"):
#     """Main function to train item sales forecasting model."""
#     logging.info("Starting item sales model training...")
#     df_orders, df_items = load_data(orders_path=orders_path, items_path=items_path)
#     df = prepare_data(df_orders, df_items)
#     X, y = prepare_features_and_target(df)
#     X_train, y_train, X_test, y_test, df_test = train_test_split(df, X, y)
#     model = _train_model(X_train, y_train)
#     _evaluate_model(model, X_test, y_test, df_test, output_path=output_path)
#     logging.info("Item sales model training and evaluation completed successfully.")

def train_item_share_models(orders_path="data/processed/orders_processed.parquet", items_path="data/processed/items_processed.parquet", output_path="data/output/item_sales_forecast.parquet", horizons=[2,3,4,5,6], cutoff_date="2025-04-01"):
    df_orders, df_items = load_data(...)
    df = prepare_data(df_orders, df_items)
    
    for h in horizons:
        logging.info(f"Training item-share model horizon={h}")
        X_train, y_train, X_test, y_test = _prepare_item_horizon(df, horizon=h, cutoff_date=cutoff_date)
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train, categorical_feature=["itemId", "cluster1To1Id"])
        model_path = f"data/models/item_share_model_h{h}.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Saved item-share horizon-{h} model to {model_path}")
        # evaluation
        _evaluate_model(model, X_test, y_test, df_test=None, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/predict item share within cluster.")
    parser.add_argument("--orders_path", default="data/processed/orders_processed.parquet")
    parser.add_argument("--items_path", default="data/processed/items_processed.parquet")
    parser.add_argument("--output_path", default="data/output/item_sales_forecast.parquet")
    args = parser.parse_args()
    train_item_sales_model(orders_path=args.orders_path, items_path=args.items_path, output_path=args.output_path)
