import os
import logging
import argparse
from datetime import timedelta, date
from typing import List, Dict, Any, Sequence
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

from preprocess import preprocess, _add_cluster_features, _add_time_features, _add_cluster_temporal_features
from cluster_sales import _aggregate_sales_by_cluster
from item_sales import prepare_data as prepare_item_data, prepare_features_and_target


# Constants / config
LOGCUT = pd.Timestamp("2025-04-01")
HORIZONS = [2, 3, 4]  # weeks ahead
MODEL_DIR = "data/models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_horizons(campaign_start: date, campaign_end: date):
    global HORIZONS
    # Determine max horizon based on campaign start date
    monday_this_week = date.today() - timedelta(days=date.today().weekday())
    monday_campaign_start = campaign_start - timedelta(days=campaign_start.weekday())
    monday_campaign_end = campaign_end - timedelta(days=campaign_end.weekday())

    weeks_until_campaign_start = (monday_campaign_start - monday_this_week).days // 7
    weeks_until_campaign_end = (monday_campaign_end - monday_this_week).days // 7

    HORIZONS = [h for h in range(weeks_until_campaign_start, weeks_until_campaign_end + 1)]
    
    if not HORIZONS:
        raise ValueError("Campaign start is too soon; no valid horizons available.")
    logging.info(f"Set forecasting horizons to: {HORIZONS}")


# ---------- Model training / caching ----------

def ensure_cluster_models(df_orders, df_cutoff=LOGCUT):
    """Train (if missing) cluster-level models for each horizon."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_cluster = _aggregate_sales_by_cluster(df_orders)


    for h in HORIZONS:
        model_path = f"{MODEL_DIR}/cluster_sales_model_h{h}.joblib"
        if os.path.exists(model_path):
            logging.info(f"[Cluster H{h}] model exists. Skipping retrain.")
            continue

        logging.info(f"[Cluster H{h}] training horizon-{h} model...")
        df_cluster = df_cluster.sort_values(["cluster1To1Id", "orderDate"])
        target_col = f"target_h{h}"
        df_cluster[target_col] = df_cluster.groupby("cluster1To1Id")["cluster_sales_kg"].shift(-h)

        df_trainable = df_cluster[df_cluster[target_col].notna()].copy()
        df_trainable["orderDate"] = pd.to_datetime(df_trainable["orderDate"])

        train = df_trainable[df_trainable["orderDate"] < df_cutoff]
        test = df_trainable[df_trainable["orderDate"] >= df_cutoff]

        drop_cols = ["cluster_sales_kg", "orderDate"] + [c for c in df_cluster.columns if c.startswith("target_h")]
        X_train = train.drop(columns=drop_cols)
        y_train = np.log1p(train[target_col].clip(lower=1e-3))
        X_test = test.drop(columns=drop_cols)
        y_test = test[target_col]

        X_train["cluster1To1Id"] = X_train["cluster1To1Id"].astype("category")
        X_test["cluster1To1Id"] = X_test["cluster1To1Id"].astype("category")

        # train
        model = LGBMRegressor(
                    random_state=42,
                    n_estimators=2000,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_data_in_leaf=200,
                    min_data_per_group=200,
                    cat_smooth=10)
        model.fit(X_train, y_train, categorical_feature=["cluster1To1Id"])
        joblib.dump(model, model_path)

        # eval
        log_preds = model.predict(X_test)
        preds = np.expm1(np.clip(log_preds, a_min=-10, a_max=20))
        preds = np.clip(preds, 0, None)  # <-- ADD
        mae = mean_absolute_error(y_test, preds)
        den = (np.abs(y_test) + np.abs(preds))
        smape = 100 * (2 * np.abs(y_test - preds) / np.where(den == 0, 1e-6, den)).mean()
        logging.info(f"[Cluster H{h}] MAE: {mae:.3f}, SMAPE: {smape:.3f}")
        # efter att du har y_test och preds (kluster)

        w = np.clip(y_test, 0, None)  # vikt = faktisk volym
        den = np.abs(y_test) + np.abs(preds)
        smape_w = 100 * (2 * w * np.abs(y_test - preds) / np.where(den==0, 1e-6, den)).sum() / np.where(w.sum()==0, 1e-6, w.sum())
        rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y_test))**2))
        logging.info(f"[Cluster H{h}] weighted SMAPE: {smape_w:.3f}, RMSLE: {rmsle:.3f}")


def ensure_item_share_models(df_orders, df_items, df_cutoff=LOGCUT):
    """Train (if missing) item-share models for each horizon."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_item_frame = prepare_item_data(df_orders, df_items)

    for h in HORIZONS:
        model_path = f"{MODEL_DIR}/item_share_model_h{h}.joblib"
        if os.path.exists(model_path):
            logging.info(f"[Item-share H{h}] model exists. Skipping retrain.")
            continue

        logging.info(f"[Item-share H{h}] training horizon-{h} model...")
        df_shifted = df_item_frame.sort_values(["itemId", "orderDate"]).copy()
        target_col = f"target_h{h}"
        df_shifted[target_col] = df_shifted.groupby("itemId")["item_share_of_cluster"].shift(-h)
        df_shifted = df_shifted[df_shifted[target_col].notna()].copy()
        df_shifted["orderDate"] = pd.to_datetime(df_shifted["orderDate"])

        train = df_shifted[df_shifted["orderDate"] < df_cutoff]
        test = df_shifted[df_shifted["orderDate"] >= df_cutoff]

        features = [
            "itemId", "cluster1To1Id",
            "is_campaign_week", "weeks_since_campaign", "other_campaign_in_cluster",
            "campaigns_ytd", "month", "week", "item_sales_lag_1", "item_share_lag_1"
        ]

        X_train = train[features].copy()
        y_train = train[target_col]
        X_test = test[features].copy()
        y_test = test[target_col]

        for col in ["itemId", "cluster1To1Id"]:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        eps = 1e-4
        model = LGBMRegressor(
                    random_state=42,
                    n_estimators=2000,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_data_in_leaf=200,
                    min_data_per_group=200,
                    cat_smooth=10
        )
        y_train_t = np.log( np.clip(y_train, eps, 1-eps) / np.clip(1 - y_train, eps, 1-eps) )
        model.fit(X_train, y_train_t, categorical_feature=["itemId", "cluster1To1Id"])
        joblib.dump(model, model_path)

        # eval
        # infer
        logit_pred = model.predict(X_test)
        y_pred = 1 / (1 + np.exp(-logit_pred))
        # valfritt “sunt förnuft”: shares i [0,1]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        mae = mean_absolute_error(y_test, y_pred)
        den = (np.abs(y_test) + np.abs(y_pred))
        smape = 100 * (2 * np.abs(y_test - y_pred) / np.where(den == 0, 1e-6, den)).mean()

        logging.info(f"[Item-share H{h}] MAE: {mae:.3f}, SMAPE: {smape:.3f}")

        y_pred = np.clip(y_pred, 0.0, 1.0)
        den = np.abs(y_test) + np.abs(y_pred)
        smape = 100 * (2 * np.abs(y_test - y_pred) / np.where(den==0, 1e-6, den)).mean()
        logging.info(f"[Item-share H{h}] SMAPE: {smape:.3f}")



# ---------- Scenario construction & feature snapshot ----------

def build_feature_snapshot(df_orders, df_items, as_of_week):
    """
    Build cluster & item feature snapshots as of a given week.
    Returns:
        cluster_snapshot: DataFrame with one row per cluster at as_of_week
        item_snapshot: DataFrame with item-level rows (with features) at as_of_week
        X_item: feature matrix for item-share model
    """
    # keep history up to as_of_week
    df_cut = df_orders[df_orders["orderDate"] <= as_of_week].copy()
    df_cut["orderDate"] = pd.to_datetime(df_cut["orderDate"])

    # ensure cluster1To1Id exists
    if "cluster1To1Id" not in df_cut.columns:
        df_cut = df_cut.merge(df_items[["itemId", "cluster1To1Id"]], on="itemId", how="left")

    # recompute derived features (campaign flag assumed present)
    df_cut = _add_cluster_features(df_cut, df_items)
    df_cut = _add_time_features(df_cut)
    df_cut = _add_cluster_temporal_features(df_cut)

    # Cluster snapshot
    df_cluster = _aggregate_sales_by_cluster(df_cut)
    cluster_snapshot = df_cluster[df_cluster["orderDate"] == as_of_week].copy()

    # Item snapshot
    df_item = prepare_item_data(df_cut, df_items)
    item_snapshot = df_item[df_item["orderDate"] == as_of_week].copy()
    X_item, _ = prepare_features_and_target(item_snapshot)

    return cluster_snapshot, item_snapshot, X_item



def make_two_scenarios(df_orders, item_ids, campaign_start_dt, campaign_end_dt):
    """
    From baseline orders, create two scenario copies:
      - no_campaign: force-off campaign flag for focus items in window
      - with_campaign: force-on campaign flag for focus items in window
    Returns dict of name -> df_orders copy.
    """
    mask_focus = (
        (df_orders["itemId"].isin(item_ids))
        & (df_orders["orderDate"] >= campaign_start_dt)
        & (df_orders["orderDate"] <= campaign_end_dt)
    )
    df_no_campaign = df_orders.copy()
    df_no_campaign.loc[mask_focus, "is_campaign_week"] = 0

    df_with_campaign = df_orders.copy()
    df_with_campaign.loc[mask_focus, "is_campaign_week"] = 1

    return {"no_campaign": df_no_campaign, "with_campaign": df_with_campaign}


# ---------- Forecasting per scenario / horizon ----------

def forecast_scenario(item_ids, df_scenario_orders, df_items, campaign_start_dt, campaign_end_dt, scenario_name):
    """
    For a given scenario df and list of focus item_ids, perform multi-horizon forecasting.
    Returns a concatenated DataFrame with columns including scenario, horizon, and predicted_quantity_kg.
    """
    outputs = []
    for h in HORIZONS:
        as_of_week = campaign_start_dt - pd.Timedelta(weeks=h)
        logging.info(f"Forecasting horizon {h} (as_of_week={as_of_week.date()}) for scenario '{scenario_name}'.")
        cluster_snap, item_snap, X_item = build_feature_snapshot(df_scenario_orders, df_items, as_of_week)

        FEATURES_ITEM = [
            "itemId", "cluster1To1Id",
            "is_campaign_week", "weeks_since_campaign", "other_campaign_in_cluster",
            "campaigns_ytd", "month", "week", "item_sales_lag_1", "item_share_lag_1"
        ]
        X_item = X_item[FEATURES_ITEM].copy()

        if cluster_snap.empty or item_snap.empty:
            logging.warning(f"Missing snapshot for horizon={h}; skipping.")
            continue

        # Load models
        cluster_model_path = f"{MODEL_DIR}/cluster_sales_model_h{h}.joblib"
        item_model_path = f"{MODEL_DIR}/item_share_model_h{h}.joblib"
        cluster_model = joblib.load(cluster_model_path)
        item_model = joblib.load(item_model_path)

        # Prepare cluster features (as during training)
        X_cluster = cluster_snap.drop(columns=["cluster_sales_kg", "orderDate"], errors="ignore").copy()
        X_cluster["cluster1To1Id"] = X_cluster["cluster1To1Id"].astype("category")

        # Predict cluster sales (log1p model)
        log_pred_cluster = cluster_model.predict(X_cluster)
        predicted_cluster_kg = np.expm1(np.clip(log_pred_cluster, a_min=-10, a_max=20))
        predicted_cluster_kg = np.clip(predicted_cluster_kg, 0, None)
        cluster_snap = cluster_snap.copy()
        cluster_snap["pred_cluster_h{}".format(h)] = predicted_cluster_kg

        # Predict item share
        X_item = X_item.copy()
        X_item["itemId"] = X_item["itemId"].astype("category")
        X_item["cluster1To1Id"] = X_item["cluster1To1Id"].astype("category")
        predicted_share = item_model.predict(X_item)

        # Clip till [0,1]
        predicted_share = np.clip(predicted_share, 0.0, 1.0)

        # Gör en Series som följer item_snap-index (en rad per item i snapshot-veckan)
        share_s = pd.Series(predicted_share, index=item_snap.index, name="pred_share")

        # Grupp-normalisera inom (cluster, vecka) baserat på item_snap (har orderDate)
        grp_keys = ["cluster1To1Id", "orderDate"]
        sum_share = share_s.groupby(item_snap[grp_keys].apply(tuple, axis=1)).transform("sum")

        # Skala bara ned om summan > 1 (lämna annars utrymme för "övrigt")
        scaled_share = np.where(sum_share > 1.0, share_s / sum_share, share_s)

        item_snap = item_snap.copy()
        item_snap[f"pred_item_share_h{h}"] = scaled_share


        # Combine: merge cluster into item
        combined = item_snap.merge(
            cluster_snap[["cluster1To1Id", f"pred_cluster_h{h}"]],
            on="cluster1To1Id",
            how="left"
        )
        combined = combined.copy()
        combined["predicted_quantity_kg"] = (
            combined[f"pred_item_share_h{h}"] * combined[f"pred_cluster_h{h}"]
        )
        combined["horizon"] = h
        combined["scenario"] = scenario_name
        combined["feature_week"] = as_of_week
        combined["target_week"] = as_of_week + pd.Timedelta(weeks=h)
        outputs.append(combined[["itemId", "cluster1To1Id", "scenario", "horizon", "predicted_quantity_kg", "feature_week", "target_week"]])

    if outputs:
        return pd.concat(outputs, ignore_index=True)
    else:
        return pd.DataFrame()



# ---------- Aggregation & comparison ----------

def aggregate_comparison(df_no, df_yes):
    # Combine both scenario frames
    combined = pd.concat([df_no, df_yes], ignore_index=True)

    # Sum predicted quantity per item, scenario, horizon
    agg = (
        combined
        .groupby(["itemId", "scenario", "horizon"])["predicted_quantity_kg"]
        .sum()
        .reset_index()
    )

    # Pivot to have side-by-side scenario comparison
    pivot = agg.pivot_table(
        index=["itemId", "horizon"],
        columns="scenario",
        values="predicted_quantity_kg",
        fill_value=0
    ).reset_index()

    # Compute lift
    pivot["abs_diff"] = pivot.get("with_campaign", 0) - pivot.get("no_campaign", 0)
    base = pivot.get("no_campaign", 0).replace(0, 1e-6)
    pivot["rel_diff_pct"] = pivot["abs_diff"] / base * 100

    return pivot


def simulate_campaign(item_ids, campaign_start, campaign_end, output_dir="data/simulation"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting campaign simulation")


    set_horizons(campaign_start, campaign_end)
    # Step 1: Scoped preprocessing (includes campaign items + peers + simulation items)
    preprocess(item_ids=item_ids)  # side effect writes processed parquet

    # Load core data
    df_orders = pd.read_parquet("data/processed/orders_processed.parquet")
    df_items = pd.read_parquet("data/processed/items_processed.parquet")

    # Step 2: Ensure models are trained
    ensure_cluster_models(df_orders)
    ensure_item_share_models(df_orders, df_items)
    
    # Step 3: Identify relevant universe: clusters of focus items and their members
    campaign_start_dt = pd.to_datetime(campaign_start)
    campaign_end_dt = pd.to_datetime(campaign_end)
    clusters = df_items[df_items["itemId"].isin(item_ids)]["cluster1To1Id"].unique()
    cluster_item_ids = df_items[df_items["cluster1To1Id"].isin(clusters)]["itemId"].unique()
    df_orders = df_orders[df_orders["itemId"].isin(cluster_item_ids)].copy()

    # Step 4: Build scenarios
    scenarios = make_two_scenarios(df_orders, item_ids, campaign_start_dt, campaign_end_dt)

    # Step 5: Forecast each scenario
    df_no = forecast_scenario(item_ids, scenarios["no_campaign"], df_items, campaign_start_dt, campaign_end_dt, scenario_name="no_campaign")
    df_yes = forecast_scenario(item_ids, scenarios["with_campaign"], df_items, campaign_start_dt, campaign_end_dt, scenario_name="with_campaign")


    # Step 6: Combine & summarize
    comparison = aggregate_comparison(df_no, df_yes)
    out_path = os.path.join(output_dir, "campaign_multi_horizon_comparison.parquet")
    comparison.to_parquet(out_path, index=False)
    logging.info(f"Saved comparison to {out_path}")
    return comparison



# ---------- Argument validation ----------

def _parse_iso_date(label: str, s: str) -> date:
    try:
        # Accepts only YYYY-MM-DD (zero-padded) per ISO 8601
        return date.fromisoformat(s)
    except Exception as e:
        raise ValueError(f"{label} must be a valid date formatted as YYYY-MM-DD (got {s!r}).") from e

def _validate_item_ids(ids: Sequence[str]) -> List[str]:
    item_id_re = re.compile(r"^[0-9A-Za-z-]+$")
    if ids is None or len(ids) == 0:
        raise ValueError("item_ids must contain at least one ID.")
    # Do NOT coerce numerics to str; require explicit strings
    for i, v in enumerate(ids):
        if not isinstance(v, str):
            raise TypeError(f"item_ids[{i}] must be a string (got {type(v).__name__}).")
        if v == "":
            raise ValueError(f"item_ids[{i}] must not be empty.")
        if not item_id_re.fullmatch(v):
            raise ValueError(
                f"item_ids[{i}] contains invalid characters: {v!r}. "
                "Allowed: 0-9, A-Z, a-z, and '-'."
            )
    return list(ids)

def validate_args(*, campaign_start: str, campaign_end: str, item_ids: Sequence[str], **_: Any) -> Dict[str, Any]:
    """
    Validate CLI args:
      --campaign-start YYYY-MM-DD
      --campaign-end   YYYY-MM-DD  (must be AFTER start)
      --item-ids       one or more strings; each only [0-9A-Za-z-]
    Returns a normalized dict with parsed dates and the original (validated) strings for item_ids.
    Raises ValueError/TypeError with clear messages on invalid input.
    """
    start = _parse_iso_date("campaign_start", campaign_start)
    end = _parse_iso_date("campaign_end", campaign_end)
    if not (end > start):
        raise ValueError(
            f"campaign_end must be after campaign_start (got start={start.isoformat()}, end={end.isoformat()})."
        )

    ids = _validate_item_ids(item_ids)

    return {
        "campaign_start": start,     # datetime.date
        "campaign_end": end,         # datetime.date
        "item_ids": ids,             # List[str]
    }
    
    

# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campaign simulation with multi-horizon forecast.")
    parser.add_argument("--item-ids", nargs='+', required=True, help="List of itemIds to simulate.")
    parser.add_argument("--campaign-start", required=True, help="Campaign start (ISO date).")
    parser.add_argument("--campaign-end", required=True, help="Campaign end (ISO date).")
    parser.add_argument("--output-dir", default="data/simulation", help="Output directory.")
    args = parser.parse_args()

    validated_args = validate_args(**vars(args))
    result_df = simulate_campaign(**validated_args)
    print("=== Simulation result preview ===")
    print(result_df.to_string())
