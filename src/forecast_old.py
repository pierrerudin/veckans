"""if __name__ == 
Campaign sales forecasting using LightGBM.

This module trains separate models for each campaign week and generates
forecasts with uncertainty intervals.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from typing import Dict, Tuple, List

from config_forecast import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE, STRATIFY_BY,
    LGBM_PARAMS, LGBM_QUANTILE_PARAMS, QUANTILES,
    COMMON_FEATURES, WEEK_SPECIFIC_FEATURES, CAMPAIGN_FEATURES,
    TARGET_COL, TARGET_COL_ORIGINAL, METRIC_FUNCTIONS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_features_and_target(df: pd.DataFrame, campaign_week: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector for a specific campaign week.
    
    Parameters:
        df: Weekly data with all features
        campaign_week: Which campaign week (1, 2, or 3)
    
    Returns:
        X: Feature matrix
        y: Target vector (log-transformed sales)
    """
    # Select rows for this campaign week
    df_week = df[df[f'campaign_week_{campaign_week}'] == 1].copy()
    
    # Build feature list
    features = COMMON_FEATURES + WEEK_SPECIFIC_FEATURES[campaign_week] + CAMPAIGN_FEATURES
    
    X = df_week[features]
    y = df_week[TARGET_COL]
    
    logging.info(f"  Campaign week {campaign_week}: {len(df_week)} samples, {len(features)} features")
    
    return X, y, df_week


def split_data(X: pd.DataFrame, y: pd.Series, df_week: pd.DataFrame) -> Dict:
    """
    Split data into train/val/test sets.
    
    Parameters:
        X: Feature matrix
        y: Target vector
        df_week: Original dataframe (not used, kept for compatibility)
    
    Returns:
        Dictionary with train/val/test splits
    """
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE
    )
    
    # Second split: train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_STATE
    )
    
    logging.info(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def train_single_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       quantile: float = None) -> lgb.Booster:
    """
    Train a single LightGBM model (point estimate or quantile regression).
    
    Parameters:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        quantile: If provided, train quantile regression model
    
    Returns:
        Trained LightGBM model
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    if quantile is not None:
        params = {**LGBM_QUANTILE_PARAMS, 'alpha': quantile}
        logging.info(f"      Training quantile model (α={quantile})...")
    else:
        params = LGBM_PARAMS
        logging.info(f"      Training point estimate model...")
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(params.get('early_stopping_rounds', 50)), lgb.log_evaluation(50)]
    )
    
    return model


def evaluate_model(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series,
                   df_test_original: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Predictions are inverse-transformed to original scale for evaluation.
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test target (log-transformed)
        df_test_original: Original test data with non-transformed target
    
    Returns:
        Dictionary of metric scores
    """
    # Predict in log space
    y_pred_log = model.predict(X_test)
    
    # Inverse transform to original scale
    y_pred = np.expm1(y_pred_log)  # expm1 is inverse of log1p
    y_true = df_test_original.loc[X_test.index, TARGET_COL_ORIGINAL]
    
    # Calculate metrics
    scores = {}
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        scores[metric_name] = metric_func(y_true, y_pred)
    
    return scores


def train_models(df_cluster_weekly: pd.DataFrame, forecast_horizon_weeks: int = 8) -> Dict:
    """
    Train forecasting models for all 3 campaign weeks.
    
    For each week, trains:
    - Point estimate model (median prediction)
    - Quantile models for 10th and 90th percentiles (uncertainty)
    
    Parameters:
        df_cluster_weekly: Preprocessed cluster-level weekly data
        forecast_horizon_weeks: Forecast horizon used in preprocessing
    
    Returns:
        Dictionary containing:
        - models: Dict with keys 'week1', 'week2', 'week3', each containing
                 'point', 'q10', 'q50', 'q90' models
        - metrics: Performance metrics on test set
        - feature_importance: Feature importance from models
    """
    logging.info("="*80)
    logging.info("TRAINING CAMPAIGN FORECASTING MODELS")
    logging.info("="*80)
    logging.info(f"Forecast horizon: {forecast_horizon_weeks} weeks")
    logging.info(f"Training on cluster-level data: {len(df_cluster_weekly):,} weekly observations")
    
    models = {}
    all_metrics = {}
    feature_importance = {}
    
    for week_num in [1, 2, 3]:
        logging.info(f"\n{'='*80}")
        logging.info(f"Training models for Campaign Week {week_num}")
        logging.info(f"{'='*80}")
        
        # Prepare data
        X, y, df_week = prepare_features_and_target(df_cluster_weekly, week_num)
        
        if len(X) < 50:
            logging.warning(f"  ⚠️  Only {len(X)} samples for week {week_num}. Skipping.")
            continue
        
        # Split data
        splits = split_data(X, y, df_week)
        
        # Train point estimate model
        logging.info(f"  Training point estimate model...")
        point_model = train_single_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )
        
        # Train quantile models for uncertainty
        quantile_models = {}
        for q in QUANTILES:
            quantile_models[f'q{int(q*100)}'] = train_single_model(
                splits['X_train'], splits['y_train'],
                splits['X_val'], splits['y_val'],
                quantile=q
            )
        
        # Evaluate on test set
        df_test_original = df_week.loc[splits['X_test'].index]
        metrics = evaluate_model(point_model, splits['X_test'], splits['y_test'], df_test_original)
        
        logging.info(f"\n  📊 Test Set Performance (Campaign Week {week_num}):")
        for metric_name, score in metrics.items():
            logging.info(f"    {metric_name}: {score:.2f}")
        
        # Store results
        models[f'week{week_num}'] = {
            'point': point_model,
            **quantile_models
        }
        all_metrics[f'week{week_num}'] = metrics
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': point_model.feature_name(),
            'importance': point_model.feature_importance()
        }).sort_values('importance', ascending=False)
        feature_importance[f'week{week_num}'] = importance
        
        logging.info(f"\n  🎯 Top 5 Features:")
        for idx, row in importance.head(5).iterrows():
            logging.info(f"    {row['feature']}: {row['importance']}")
    
    logging.info(f"\n{'='*80}")
    logging.info("✅ MODEL TRAINING COMPLETE")
    logging.info(f"{'='*80}\n")
    
    return {
        'models': models,
        'metrics': all_metrics,
        'feature_importance': feature_importance,
        'forecast_horizon_weeks': forecast_horizon_weeks
    }


def predict(models: Dict, df_cluster_weekly: pd.DataFrame, 
            target_cluster_ids: List[str], forecast_date: pd.Timestamp,
            campaign_start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Generate forecasts for target clusters.
    
    Parameters:
        models: Trained models from train_models()
        df_cluster_weekly: Preprocessed cluster data (to extract recent lags)
        target_cluster_ids: List of cluster IDs to forecast
        forecast_date: Date when forecast is made
        campaign_start_date: Start date of campaign
    
    Returns:
        DataFrame with forecasts for each cluster, week, and quantile
        Columns: cluster1To1Id, campaign_week, prediction_kgl, q10_kgl, q90_kgl
    """
    logging.info("="*80)
    logging.info("GENERATING FORECASTS")
    logging.info("="*80)
    logging.info(f"Forecast date: {forecast_date.strftime('%Y-%m-%d')}")
    logging.info(f"Campaign start: {campaign_start_date.strftime('%Y-%m-%d')}")
    logging.info(f"Forecasting for {len(target_cluster_ids)} clusters")

    
    forecast_horizon_weeks = models['forecast_horizon_weeks']
    
    # Find the last complete week aligned to campaign day-of-week that's before forecast_date
    # This represents the most recent historical week we can use for lag features
    campaign_weekday = campaign_start_date.weekday()  # 0=Monday, 1=Tuesday, etc.
    forecast_weekday = forecast_date.weekday()
    
    # Days from forecast_date back to most recent campaign-aligned week start
    days_back = (forecast_weekday - campaign_weekday) % 7
    
    # Go back one more week to ensure it's complete (not the current partial week)
    days_back += 7
    
    cutoff_week_start = forecast_date - pd.Timedelta(days=days_back)
    
    logging.info(f"  Last complete week: {cutoff_week_start.date()} ({cutoff_week_start.strftime('%A')})")
    
    # Campaign week 1 starts on campaign_start_date itself
    campaign_week1_start = campaign_start_date
    
    # Get most recent data up to cutoff week
    recent_data = df_cluster_weekly[df_cluster_weekly['week_start'] <= cutoff_week_start].copy()
    
    all_predictions = []
    
    for cluster_id in target_cluster_ids:
        cluster_data = recent_data[recent_data['cluster1To1Id'] == cluster_id].sort_values('week_start')
        
        if len(cluster_data) == 0:
            logging.warning(f"  ⚠️  No historical data for cluster {cluster_id}")
            continue
        
        # For each campaign week
        for week_num in [1, 2, 3]:
            if f'week{week_num}' not in models['models']:
                continue
            
            week_start = campaign_week1_start + pd.Timedelta(weeks=week_num-1)
            
            # Calculate lag date (historical data already aligned to campaign day-of-week)
            lag_weeks = forecast_horizon_weeks + (week_num - 1)
            lag_date = week_start - pd.Timedelta(weeks=lag_weeks)
            
            # Extract lag features from historical data
            lag_data = cluster_data[cluster_data['week_start'] == lag_date]
            print(cluster_data)
            if len(lag_data) == 0:
                logging.warning(f"  ⚠️  No lag data for cluster {cluster_id}, week {week_num}")
                continue
            
            # Build feature vector
            features = {}
            
            # Lag features
            features[f'sales_lag_week{week_num}'] = lag_data.iloc[0][f'sales_lag_week{week_num}']
            features[f'sales_rolling_4w_week{week_num}'] = lag_data.iloc[0][f'sales_rolling_4w_week{week_num}']
            features[f'cluster_sales_lag_week{week_num}'] = lag_data.iloc[0][f'cluster_sales_lag_week{week_num}']
            features['sales_lag_52w'] = lag_data.iloc[0]['sales_lag_52w']
            
            # Time features for forecast week
            week_of_year = week_start.isocalendar().week
            features['week_of_year_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            features['week_of_year_cos'] = np.cos(2 * np.pi * week_of_year / 52)
            
            # Campaign features
            features['campaign_week_0'] = 0
            features['campaign_week_1'] = 1 if week_num == 1 else 0
            features['campaign_week_2'] = 1 if week_num == 2 else 0
            features['campaign_week_3'] = 1 if week_num == 3 else 0
            
            # Create feature DataFrame
            X_pred = pd.DataFrame([features])
            logging.info(f"Displaying features for cluster {cluster_id}, week {week_num}:")
            print(X_pred)

            # Get models for this week
            week_models = models['models'][f'week{week_num}']
            
            # Predict with all models
            pred_log = week_models['point'].predict(X_pred)[0]
            q10_log = week_models['q10'].predict(X_pred)[0]
            q50_log = week_models['q50'].predict(X_pred)[0]
            q90_log = week_models['q90'].predict(X_pred)[0]
            
            # Inverse transform
            pred_kgl = np.expm1(pred_log)
            q10_kgl = np.expm1(q10_log)
            q50_kgl = np.expm1(q50_log)
            q90_kgl = np.expm1(q90_log)
            
            all_predictions.append({
                'cluster1To1Id': cluster_id,
                'campaign_week': week_num,
                'week_start': week_start,
                'prediction_kgl': pred_kgl,
                'q10_kgl': q10_kgl,
                'q50_kgl': q50_kgl,
                'q90_kgl': q90_kgl
            })
    
    df_predictions = pd.DataFrame(all_predictions)
    
    logging.info(f"✅ Generated {len(df_predictions)} forecasts")
    logging.info(f"{'='*80}\n")
    
    return df_predictions


def predict_baseline(df_cluster_weekly: pd.DataFrame, target_cluster_ids: List[str],
                      forecast_date: pd.Timestamp, campaign_start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Generate baseline forecasts (no campaign) for comparison.
    
    Uses sales_lag_52w as simple baseline (same week last year).
    
    Parameters:
        df_cluster_weekly: Preprocessed cluster data
        target_cluster_ids: List of cluster IDs
        forecast_date: Forecast date
        campaign_start_date: Campaign start
    
    Returns:
        DataFrame with baseline predictions
    """
    # Find the last complete week aligned to campaign day-of-week that's before forecast_date
    campaign_weekday = campaign_start_date.weekday()
    forecast_weekday = forecast_date.weekday()
    
    days_back = (forecast_weekday - campaign_weekday) % 7
    days_back += 7  # Go back one more week to ensure it's complete
    
    cutoff_week_start = forecast_date - pd.Timedelta(days=days_back)
    
    # Campaign week 1 starts on campaign_start_date itself
    campaign_week1_start = campaign_start_date
    
    recent_data = df_cluster_weekly[df_cluster_weekly['week_start'] <= cutoff_week_start].copy()
    
    baselines = []
    
    for cluster_id in target_cluster_ids:
        cluster_data = recent_data[recent_data['cluster1To1Id'] == cluster_id].sort_values('week_start')
        
        if len(cluster_data) == 0:
            continue
        
        for week_num in [1, 2, 3]:
            week_start = campaign_week1_start + pd.Timedelta(weeks=week_num-1)
            
            # Use sales from same week last year as baseline
            # Historical data already aligned to campaign day-of-week
            lag_52w_date = week_start - pd.Timedelta(weeks=52)
            
            lag_data = cluster_data[cluster_data['week_start'] == lag_52w_date]
            
            if len(lag_data) > 0:
                baseline_kgl = lag_data.iloc[0][TARGET_COL_ORIGINAL]
                
                baselines.append({
                    'cluster1To1Id': cluster_id,
                    'campaign_week': week_num,
                    'week_start': week_start,
                    'baseline_kgl': baseline_kgl
                })
    
    return pd.DataFrame(baselines)
