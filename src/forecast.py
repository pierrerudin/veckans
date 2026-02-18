"""
Campaign sales forecasting using LightGBM.

Simple design:
- Data arrives preprocessed with all features ready
- Training: split data, train models, return models
- Prediction: use models on preprocessed data, return forecasts
- No data manipulation except splitting
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from typing import Dict, List

from config_forecast import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE,
    LGBM_PARAMS, LGBM_PARAMS_CLUSTER, LGBM_QUANTILE_PARAMS, LGBM_QUANTILE_PARAMS_CLUSTER, QUANTILES,
    COMMON_FEATURES, WEEK_SPECIFIC_FEATURES, CAMPAIGN_FEATURES, CLUSTER_SPECIFIC_FEATURES,
    CATEGORY_FEATURES, TARGET_COL, METRIC_FUNCTIONS, CATEGORICAL_FEATURES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def split_data(X, y):
    """Split data into train/val/test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=RANDOM_STATE
    )
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def train_single_model(X_train, y_train, w_train, X_val, y_val, w_val, quantile=None, level='item'):
    """Train a single LightGBM model with sample weights."""
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=CATEGORICAL_FEATURES)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data, categorical_feature=CATEGORICAL_FEATURES)
    
    # Use cluster-specific parameters for cluster models
    if level == 'cluster':
        params = LGBM_QUANTILE_PARAMS_CLUSTER.copy() if quantile else LGBM_PARAMS_CLUSTER.copy()
    else:
        params = LGBM_QUANTILE_PARAMS.copy() if quantile else LGBM_PARAMS.copy()
    
    if quantile:
        params['alpha'] = quantile
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Log feature importance for point estimate models
    if quantile is None:
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logging.info(f"      Top 10 features by importance ({level}):")
        for idx, row in feature_importance.head(10).iterrows():
            logging.info(f"        {row['feature']:<30} {row['importance']:>12,.0f}")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    
    results = {}
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        if metric_name.upper() == 'SMAPE':
            y_test_original = np.expm1(y_test)
            y_pred_original = np.expm1(y_pred)
            score = metric_func(y_test_original, y_pred_original)
        else:
            score = metric_func(y_test, y_pred)
        results[metric_name] = score
    
    logging.info(f"    {model_name}: SMAPE={results['SMAPE']:.1f}%, RMSE={results['RMSE']:.3f}, MAE={results['MAE']:.3f}")
    return results


def train_models(df_weekly, forecast_horizon_weeks, normalizers, level='cluster', target_ids=None):
    """
    Train models for item or cluster level forecasting.
    
    NOTE: Data should already be normalized by preprocess_data().
    This function expects salesQuantityKgL_log to be pre-normalized.
    
    Parameters:
        df_weekly: Preprocessed weekly data (item or cluster level) - ALREADY NORMALIZED
        forecast_horizon_weeks: Weeks between forecast date and campaign start
        normalizers: Dict of normalization factors from preprocessing (for denormalization)
        level: 'item' or 'cluster' - determines which lag features to use
        target_ids: List of target IDs we're trying to predict (optional, for item level)
    
    Returns:
        Dictionary of models: {week_num: {'point': model, 'q10': model, 'q90': model}}
    """
    logging.info(f"\nTraining {level}-level models for {forecast_horizon_weeks}w horizon...")
    
    # Determine ID column based on level
    id_col = 'baseItemId' if level == 'item' else 'cluster1To1Id'
    
    # Data is already normalized by preprocess_data(), just use it directly
    TARGET_COL_NORMALIZED = TARGET_COL  # salesQuantityKgL_log is already normalized
    
    # Adjust lag feature names based on level
    if level == 'cluster':
        week_specific = {
            1: ['cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1'],
            2: ['cluster_sales_lag_week2', 'cluster_sales_rolling_4w_week2'],
            3: ['cluster_sales_lag_week3', 'cluster_sales_rolling_4w_week3']
        }
    else:  # item level - includes both item and cluster context features
        week_specific = {
            1: ['sales_lag_week1', 'sales_rolling_4w_week1', 'cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1'],
            2: ['sales_lag_week2', 'sales_rolling_4w_week2', 'cluster_sales_lag_week2', 'cluster_sales_rolling_4w_week2'],
            3: ['sales_lag_week3', 'sales_rolling_4w_week3', 'cluster_sales_lag_week3', 'cluster_sales_rolling_4w_week3']
        }
    
    models = {}
    
    for week_num in [1, 2, 3]:
        logging.info(f"\n  Campaign Week {week_num}")
        
        # Filter training data based on level
        campaign_col = f'campaign_week_{week_num}'
        # For clusters: campaign flags are continuous (intensity), so check > 0
        # For items: campaign flags are binary, so check == 1
        if level == 'cluster':
            items_with_campaigns = df_weekly[df_weekly[campaign_col] > 0][id_col].unique()
        else:
            items_with_campaigns = df_weekly[df_weekly[campaign_col] == 1][id_col].unique()
        
        if level == 'item':
            # For items: train on items that have had campaigns + target items we're predicting
            # This lets the model learn campaign patterns AND see the specific item's baseline
            training_ids = set(items_with_campaigns)
            if target_ids:
                training_ids.update(target_ids)
            training_ids = list(training_ids)
            logging.info(f"    Training on {len(items_with_campaigns)} items with campaigns + {len(target_ids) if target_ids else 0} target items")
        else:
            # For clusters: train on all clusters that have had campaigns
            # (already filtered in fetch_data to only load relevant clusters)
            training_ids = items_with_campaigns
            logging.info(f"    Training on {len(training_ids)} clusters that have had campaigns")
        
        # Include ALL weeks for these IDs (both campaign and non-campaign)
        # This gives the model contrast between campaign=1 and campaign=0
        df_week = df_weekly[df_weekly[id_col].isin(training_ids)].copy()
        
        # Balance the dataset intelligently:
        # 1. Keep ALL campaign weeks (these are rare and important)
        # 2. For non-campaign weeks, prefer RECENT weeks before the forecast date
        #    (these are most relevant for understanding current patterns)
        # 3. Drop weeks after campaigns (recovery patterns aren't useful)
        
        # For clusters: campaign_col is continuous, so check > 0
        # For items: campaign_col is binary, so check == 1
        if level == 'cluster':
            df_campaign = df_week[df_week[campaign_col] > 0]
            df_no_campaign = df_week[(df_week[campaign_col] == 0) & (df_week['campaign_week_0'] > 0)]
        else:
            df_campaign = df_week[df_week[campaign_col] == 1]
            df_no_campaign = df_week[(df_week[campaign_col] == 0) & (df_week['campaign_week_0'] == 1)]
        
        # Sort non-campaign weeks by recency (most recent first)
        df_no_campaign = df_no_campaign.sort_values('week_start', ascending=False)
        
        # Take 5x campaign samples to maintain good baseline context
        # (e.g., 382 campaign → 1,910 baseline = 2,292 total vs 49,964 original)
        n_campaign = len(df_campaign)
        n_target_baseline = min(n_campaign * 5, len(df_no_campaign))
        df_no_campaign_sampled = df_no_campaign.head(n_target_baseline)  # Take most recent
        
        # Combine (no shuffle — we'll do a temporal split)
        df_week = pd.concat([df_campaign, df_no_campaign_sampled])
        
        # DEBUG: Check if campaign weeks actually have higher sales
        avg_campaign = df_campaign[TARGET_COL_NORMALIZED].mean()
        avg_baseline = df_no_campaign_sampled[TARGET_COL_NORMALIZED].mean()
        logging.info(f"    Normalized avg: campaign={avg_campaign:.2f}, baseline={avg_baseline:.2f}, ratio={avg_campaign/avg_baseline:.2f}x")
        
        logging.info(f"    Balanced: {n_campaign} campaign + {n_target_baseline} baseline = {len(df_week)} total samples")
        
        # Build feature list
        features = COMMON_FEATURES + week_specific[week_num] + CAMPAIGN_FEATURES

        # Add level-specific features
        if level == 'cluster':
            features = features + CLUSTER_SPECIFIC_FEATURES
        else:
            features = features + CATEGORY_FEATURES
        
        # Check which features are actually available
        available_features = [f for f in features if f in df_week.columns]
        missing_features = set(features) - set(available_features)
        
        if missing_features:
            logging.warning(f"    Missing features: {missing_features}")
        
        X = df_week[available_features]
        y = df_week[TARGET_COL_NORMALIZED]  # Use normalized target
        
        # Create sample weights based on campaign intensity
        # For items: binary (0 or 1), so use 1.0 or 3.0
        # For clusters: continuous (0.0 to 1.0), so scale weight linearly
        #   - 0.0 (no items on campaign) → weight 1.0
        #   - 1.0 (all items on campaign) → weight 1.5
        #   - 0.5 (half items on campaign) → weight 1.25
        campaign_col = f'campaign_week_{week_num}'
        if level == 'cluster':
            # Linear scaling: weight = 1.0 + (intensity * 0.5)
            sample_weights = 1.0 + (df_week[campaign_col].values * 0.5)
        else:
            # Binary weighting for items
            sample_weights = df_week[campaign_col].map({0: 1.0, 1: 3.0}).values
        
        # Debug: Check campaign distribution in training data
        if campaign_col in df_week.columns:
            if level == 'cluster':
                n_campaign = (df_week[campaign_col] > 0).sum()
                avg_intensity = df_week[df_week[campaign_col] > 0][campaign_col].mean()
                logging.info(f"    Samples: {len(df_week)} total ({n_campaign} with campaigns, avg intensity={avg_intensity:.3f})")
            else:
                n_campaign = (df_week[campaign_col] == 1).sum()
                n_total = len(df_week)
                logging.info(f"    Samples: {n_total} total ({n_campaign} weighted 3x for campaign)")
        
        # Temporal split: sort by week_start, then slice by position
        df_week_sorted = df_week.sort_values('week_start').reset_index(drop=True)
        n = len(df_week_sorted)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_sorted = df_week_sorted[available_features]
        y_sorted = df_week_sorted[TARGET_COL_NORMALIZED]

        # Recompute sample weights aligned to sorted order
        if level == 'cluster':
            w_sorted = 1.0 + (df_week_sorted[campaign_col].values * 0.5)
        else:
            w_sorted = df_week_sorted[campaign_col].map({0: 1.0, 1: 3.0}).values

        X_train, y_train, w_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end], w_sorted[:train_end]
        X_val, y_val, w_val = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end], w_sorted[train_end:val_end]
        X_test, y_test, w_test = X_sorted.iloc[val_end:], y_sorted.iloc[val_end:], w_sorted[val_end:]

        logging.info(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} (temporal split)")
        
        # Train models with sample weights
        logging.info("    Training point estimate...")
        point_model = train_single_model(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            level=level
        )
        evaluate_model(point_model, X_test, y_test, "Point")

        logging.info("    Training quantile models...")
        q10_model = train_single_model(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            quantile=QUANTILES[0], level=level
        )

        q90_model = train_single_model(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            quantile=QUANTILES[2], level=level
        )
        
        models[week_num] = {
            'point': point_model,
            'q10': q10_model,
            'q90': q90_model,
            'features': available_features,
            'normalizers': normalizers  # Store for prediction phase
        }
    
    logging.info(f"\n✅ {level.capitalize()} models trained for weeks {list(models.keys())}")
    return models


def train_models_single(df_weekly, forecast_horizon_weeks, normalizers, level='cluster', target_ids=None):
    """
    Train a SINGLE model for all 3 campaign weeks (alternative to 3 separate models).

    Uses campaign_week_number as a numeric feature (1, 2, 3) plus interaction
    features (campaign_week_number * lag). The model learns week-specific
    patterns through these interactions instead of separate training runs.

    Returns the same dict format as train_models() for compatibility:
        {1: model_info, 2: model_info, 3: model_info} — all pointing to the same model.
    """
    logging.info(f"\nTraining SINGLE {level}-level model for all campaign weeks...")

    id_col = 'baseItemId' if level == 'item' else 'cluster1To1Id'
    TARGET_COL_NORMALIZED = TARGET_COL

    # Use a unified lag feature set: the "week1" features as base
    if level == 'cluster':
        base_lag_features = ['cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1']
    else:
        base_lag_features = ['sales_lag_week1', 'sales_rolling_4w_week1',
                             'cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1']

    features = COMMON_FEATURES + base_lag_features + CAMPAIGN_FEATURES + ['campaign_week_number']
    if level == 'cluster':
        features = features + CLUSTER_SPECIFIC_FEATURES
    else:
        features = features + CATEGORY_FEATURES

    # Collect training data across all 3 campaign weeks
    all_dfs = []
    for week_num in [1, 2, 3]:
        campaign_col = f'campaign_week_{week_num}'

        if level == 'cluster':
            ids_with_campaigns = df_weekly[df_weekly[campaign_col] > 0][id_col].unique()
        else:
            ids_with_campaigns = df_weekly[df_weekly[campaign_col] == 1][id_col].unique()

        training_ids = set(ids_with_campaigns)
        if target_ids:
            training_ids.update(target_ids)
        training_ids = list(training_ids)

        df_week = df_weekly[df_weekly[id_col].isin(training_ids)].copy()

        # Balance: keep all campaign weeks, sample baseline
        if level == 'cluster':
            df_campaign = df_week[df_week[campaign_col] > 0].copy()
            df_baseline = df_week[(df_week[campaign_col] == 0) & (df_week['campaign_week_0'] > 0)].copy()
        else:
            df_campaign = df_week[df_week[campaign_col] == 1].copy()
            df_baseline = df_week[(df_week[campaign_col] == 0) & (df_week['campaign_week_0'] == 1)].copy()

        n_campaign = len(df_campaign)
        n_target_baseline = min(n_campaign * 5, len(df_baseline))
        df_baseline_sampled = df_baseline.sort_values('week_start', ascending=False).head(n_target_baseline)

        combined = pd.concat([df_campaign, df_baseline_sampled])

        # Add campaign_week_number feature (0 for baseline, 1/2/3 for campaign)
        combined['campaign_week_number'] = 0
        if level == 'cluster':
            combined.loc[combined[campaign_col] > 0, 'campaign_week_number'] = week_num
        else:
            combined.loc[combined[campaign_col] == 1, 'campaign_week_number'] = week_num

        # Map week-specific lag features to unified names
        lag_mapping = {
            f'sales_lag_week{week_num}': 'sales_lag_week1',
            f'sales_rolling_4w_week{week_num}': 'sales_rolling_4w_week1',
            f'cluster_sales_lag_week{week_num}': 'cluster_sales_lag_week1',
            f'cluster_sales_rolling_4w_week{week_num}': 'cluster_sales_rolling_4w_week1',
        }
        for src, dst in lag_mapping.items():
            if src in combined.columns and src != dst:
                combined[dst] = combined[src]

        all_dfs.append(combined)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Add interaction features (use correct lag column name per level)
    lag_col = 'sales_lag_week1' if level == 'item' else 'cluster_sales_lag_week1'
    roll_col = 'sales_rolling_4w_week1' if level == 'item' else 'cluster_sales_rolling_4w_week1'
    df_all['cwn_x_lag'] = df_all['campaign_week_number'] * df_all[lag_col]
    df_all['cwn_x_rolling'] = df_all['campaign_week_number'] * df_all[roll_col]
    interaction_features = ['cwn_x_lag', 'cwn_x_rolling']
    features = features + interaction_features

    available_features = [f for f in features if f in df_all.columns]

    logging.info(f"  Combined training data: {len(df_all)} samples")
    logging.info(f"  Features ({len(available_features)}): {available_features}")

    # Temporal split
    df_all = df_all.sort_values('week_start').reset_index(drop=True)
    n = len(df_all)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_sorted = df_all[available_features]
    y_sorted = df_all[TARGET_COL_NORMALIZED]

    # Sample weights
    if level == 'cluster':
        w_sorted = np.ones(n)
        for wk in [1, 2, 3]:
            col = f'campaign_week_{wk}'
            if col in df_all.columns:
                w_sorted += df_all[col].values * 0.5
    else:
        w_sorted = np.ones(n)
        for wk in [1, 2, 3]:
            col = f'campaign_week_{wk}'
            if col in df_all.columns:
                w_sorted[df_all[col] == 1] = 3.0

    X_train, y_train, w_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end], w_sorted[:train_end]
    X_val, y_val, w_val = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end], w_sorted[train_end:val_end]
    X_test, y_test, w_test = X_sorted.iloc[val_end:], y_sorted.iloc[val_end:], w_sorted[val_end:]

    logging.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Train single point model
    logging.info("  Training point estimate...")
    point_model = train_single_model(X_train, y_train, w_train, X_val, y_val, w_val, level=level)
    evaluate_model(point_model, X_test, y_test, "Single-Point")

    logging.info("  Training quantile models...")
    q10_model = train_single_model(X_train, y_train, w_train, X_val, y_val, w_val, quantile=QUANTILES[0], level=level)
    q90_model = train_single_model(X_train, y_train, w_train, X_val, y_val, w_val, quantile=QUANTILES[2], level=level)

    # Return same format as train_models() — all weeks share the same model
    model_info = {
        'point': point_model,
        'q10': q10_model,
        'q90': q90_model,
        'features': available_features,
        'normalizers': normalizers,
        'single_model': True,  # Flag for predict() to know about interaction features
    }
    models = {1: model_info, 2: model_info, 3: model_info}

    logging.info(f"\n✅ Single {level}-level model trained (shared across weeks 1-3)")
    return models


def predict(models, df_weekly, target_ids, forecast_date, campaign_start, level='cluster', campaign_on=True):
    """
    Generate forecasts using trained models.

    Parameters:
        models: Trained models from train_models()
        df_weekly: Preprocessed weekly data
        target_ids: List of item IDs or cluster IDs to forecast
        forecast_date: Date when forecast is made
        campaign_start: Campaign start date
        level: 'item' or 'cluster'
        campaign_on: If True, predict with campaign flags active. If False, predict baseline (no campaign).

    Returns:
        DataFrame with predictions
    """
    id_col = 'baseItemId' if level == 'item' else 'cluster1To1Id'
    mode = "campaign" if campaign_on else "baseline"
    logging.info(f"\nGenerating {level}-level {mode} forecasts for {len(target_ids)} targets...")
    
    # Filter to target IDs and data up to forecast date
    df_forecast = df_weekly[
        (df_weekly[id_col].isin(target_ids)) &
        (df_weekly['week_start'] <= forecast_date)
    ].copy()
    if len(df_forecast) == 0:
        logging.warning(f"No data available for {level}-level forecasting")
        return pd.DataFrame()
    
    predictions = []

    for target_id in target_ids:
        df_target = df_forecast[df_forecast[id_col] == target_id]
        
        if len(df_target) == 0:
            logging.warning(f"  No data for {id_col} {target_id}")
            continue
        
        # Get most recent data point
        df_latest = df_target.sort_values('week_start').tail(1).copy()
        
        for week_num in [1, 2, 3]:
            if week_num not in models:
                continue
            
            model_info = models[week_num]
            features = model_info['features']
            
            # Set campaign flags for prediction
            if not campaign_on:
                # Counterfactual: predict without campaign
                if level == 'cluster':
                    df_latest['campaign_week_0'] = 1.0
                else:
                    df_latest['campaign_week_0'] = 1
                df_latest['campaign_week_1'] = 0
                df_latest['campaign_week_2'] = 0
                df_latest['campaign_week_3'] = 0
            elif level == 'cluster':
                # Cluster with campaign: intensity = 1/num_items
                num_items = df_latest['num_items_in_cluster'].values[0]
                intensity = 1.0 / max(num_items, 1)
                df_latest['campaign_week_0'] = 1 - intensity
                df_latest['campaign_week_1'] = intensity if week_num == 1 else 0
                df_latest['campaign_week_2'] = intensity if week_num == 2 else 0
                df_latest['campaign_week_3'] = intensity if week_num == 3 else 0
            else:
                # Item with campaign: binary flags
                df_latest['campaign_week_0'] = 0
                df_latest['campaign_week_1'] = 1 if week_num == 1 else 0
                df_latest['campaign_week_2'] = 1 if week_num == 2 else 0
                df_latest['campaign_week_3'] = 1 if week_num == 3 else 0
            
            # Single-model mode: add campaign_week_number and interaction features
            if model_info.get('single_model'):
                df_latest['campaign_week_number'] = week_num if campaign_on else 0
                # Map week-specific lags to unified names
                for src, dst in [
                    (f'sales_lag_week{week_num}', 'sales_lag_week1'),
                    (f'sales_rolling_4w_week{week_num}', 'sales_rolling_4w_week1'),
                    (f'cluster_sales_lag_week{week_num}', 'cluster_sales_lag_week1'),
                    (f'cluster_sales_rolling_4w_week{week_num}', 'cluster_sales_rolling_4w_week1'),
                ]:
                    if src in df_latest.columns and src != dst:
                        df_latest[dst] = df_latest[src]
                # Interaction features (use correct lag column per level)
                lag_col = 'sales_lag_week1' if level == 'item' else 'cluster_sales_lag_week1'
                roll_col = 'sales_rolling_4w_week1' if level == 'item' else 'cluster_sales_rolling_4w_week1'
                lag_val = df_latest[lag_col].values[0] if lag_col in df_latest.columns else 0
                roll_val = df_latest[roll_col].values[0] if roll_col in df_latest.columns else 0
                df_latest['cwn_x_lag'] = df_latest['campaign_week_number'] * lag_val
                df_latest['cwn_x_rolling'] = df_latest['campaign_week_number'] * roll_val

            # Check if we have all required features
            missing = set(features) - set(df_latest.columns)
            if missing:
                logging.warning(f"  {target_id} week {week_num}: missing {missing}")
                continue

            X_pred = df_latest[features]
            
            # Generate predictions (these are in normalized log scale)
            pred_normalized_log = model_info['point'].predict(X_pred)[0]
            q10_normalized_log = model_info['q10'].predict(X_pred)[0]
            q90_normalized_log = model_info['q90'].predict(X_pred)[0]
            
            # Clip to prevent overflow (log scale: e^100 ≈ 2.7e43)
            pred_normalized_log = np.clip(pred_normalized_log, -10, 20)
            q10_normalized_log = np.clip(q10_normalized_log, -10, 20)
            q90_normalized_log = np.clip(q90_normalized_log, -10, 20)
            
            # Transform back to normalized scale (multiplier of baseline)
            pred_multiplier = np.expm1(pred_normalized_log)
            q10_multiplier = np.expm1(q10_normalized_log)
            q90_multiplier = np.expm1(q90_normalized_log)
            
            # Denormalize: multiply by baseline average to get absolute sales
            normalizers = model_info['normalizers']
            baseline_avg = normalizers.get(target_id, 1000)  # Fallback to 1000 kg/week
            
            pred_kgl = pred_multiplier * baseline_avg
            q10_kgl = q10_multiplier * baseline_avg
            q90_kgl = q90_multiplier * baseline_avg
            
            predictions.append({
                id_col: target_id,
                'campaign_week': week_num,
                'prediction_kgl': pred_kgl,
                'q10_kgl': q10_kgl,
                'q90_kgl': q90_kgl
            })
    
    df_predictions = pd.DataFrame(predictions)
    logging.info(f"Generated {len(df_predictions)} {level}-level {mode} forecasts")
    
    return df_predictions


def predict_baseline(df_weekly, target_ids, forecast_date, campaign_start, normalizers, level='cluster'):
    """
    Generate baseline forecasts (no campaign effect).
    
    Uses historical average for the same time period.
    
    Parameters:
        df_weekly: Preprocessed weekly data
        target_ids: List of item IDs or cluster IDs
        forecast_date: Date when forecast is made
        campaign_start: Campaign start date
        normalizers: Dict mapping ID -> baseline value for denormalization
        level: 'item' or 'cluster'
    
    Returns:
        DataFrame with baseline predictions
    """
    id_col = 'baseItemId' if level == 'item' else 'cluster1To1Id'
    logging.info(f"\nGenerating {level}-level baseline forecasts...")
    
    # Calculate the cutoff date for last complete week
    campaign_weekday = campaign_start.weekday()
    forecast_weekday = forecast_date.weekday()
    days_back = (forecast_weekday - campaign_weekday) % 7
    days_back += 7
    cutoff_week_start = forecast_date - pd.Timedelta(days=days_back)
    
    # Filter data up to cutoff
    df_baseline = df_weekly[df_weekly['week_start'] <= cutoff_week_start].copy()
    
    # Get campaign week of year for seasonality matching
    campaign_week_of_year = campaign_start.isocalendar()[1]
    
    baselines = []
    
    for target_id in target_ids:
        df_target = df_baseline[df_baseline[id_col] == target_id]
        
        if len(df_target) == 0:
            continue
        
        # Find similar weeks (same week of year, ±2 weeks)
        week_range = range(campaign_week_of_year - 2, campaign_week_of_year + 3)
        df_similar = df_target[df_target['week_of_year'].isin(week_range)]
        
        if len(df_similar) == 0:
            df_similar = df_target  # Fallback to all data
        
        # Calculate average (denormalize properly!)
        # Data is: log1p(sales / normalizer)
        # Step 1: expm1 to get normalized values (sales / normalizer)
        # Step 2: multiply by normalizer to get actual kg/L
        df_similar_original = df_similar.copy()
        df_similar_original['sales_normalized'] = np.expm1(df_similar_original['salesQuantityKgL_log'])
        
        # Get normalizer for this target
        normalizer = normalizers.get(target_id, 1000)  # Fallback if missing
        df_similar_original['sales_original'] = df_similar_original['sales_normalized'] * normalizer
        
        # Use median for robustness
        median_sales = df_similar_original['sales_original'].median()
        
        # Sanity check
        if median_sales > 0 and median_sales < 1000000:
            avg_sales = median_sales
        else:
            logging.warning(f"  {target_id}: unrealistic baseline (median={median_sales:.0f}), using mean")
            mean_sales = df_similar_original['sales_original'].mean()
            if mean_sales > 0 and mean_sales < 1000000:
                avg_sales = mean_sales
            else:
                logging.warning(f"  {target_id}: mean also unrealistic ({mean_sales:.0f}), using fallback")
                avg_sales = 1000  # Fallback: 1 ton/week
        
        # Same baseline for all 3 weeks
        for week_num in [1, 2, 3]:
            baselines.append({
                id_col: target_id,
                'campaign_week': week_num,
                'baseline_kgl': avg_sales
            })
    
    df_baselines = pd.DataFrame(baselines)
    logging.info(f"✅ Generated {len(df_baselines)} {level}-level baselines")
    
    return df_baselines
