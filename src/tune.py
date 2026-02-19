"""
Hyperparameter tuning for campaign forecasting models using Optuna.

Usage:
    python tune.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13
    python tune.py --item-ids 528208 --forecast-date 2024-11-18 --campaign-start 2025-01-13 --level cluster --n-trials 100

Performs walk-forward cross-validation with campaign-aware splits:
- Each fold uses a temporal cutoff so that test data is always in the future
- Evaluation focuses on campaign weeks (where accuracy matters most)
- Optimizes SMAPE on campaign weeks across all 3 week models
"""

import argparse
import logging
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner

from fetch_data import fetch_data
from preprocess import preprocess_data
from config_forecast import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    COMMON_FEATURES, CAMPAIGN_FEATURES,
    CLUSTER_SPECIFIC_FEATURES, CATEGORY_FEATURES,
    CAMPAIGN_COMPETITION_FEATURES, TARGET_COL, METRIC_FUNCTIONS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_feature_list(week_num: int, level: str, df_columns: list) -> list:
    """Build the feature list for a given campaign week and level."""
    if level == 'cluster':
        week_specific = {
            1: ['cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1'],
            2: ['cluster_sales_lag_week2', 'cluster_sales_rolling_4w_week2'],
            3: ['cluster_sales_lag_week3', 'cluster_sales_rolling_4w_week3']
        }
    else:
        week_specific = {
            1: ['sales_lag_week1', 'sales_rolling_4w_week1', 'cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1'],
            2: ['sales_lag_week2', 'sales_rolling_4w_week2', 'cluster_sales_lag_week2', 'cluster_sales_rolling_4w_week2'],
            3: ['sales_lag_week3', 'sales_rolling_4w_week3', 'cluster_sales_lag_week3', 'cluster_sales_rolling_4w_week3']
        }

    features = COMMON_FEATURES + week_specific[week_num] + CAMPAIGN_FEATURES
    if level == 'cluster':
        features = features + CLUSTER_SPECIFIC_FEATURES
    else:
        features = features + CATEGORY_FEATURES + CAMPAIGN_COMPETITION_FEATURES

    return [f for f in features if f in df_columns]


def walk_forward_cv(df_weekly, level, params, n_folds=3):
    """
    Walk-forward cross-validation with campaign-aware evaluation.

    Splits data temporally into n_folds+1 segments. For each fold, trains on
    all prior segments and evaluates on the next segment's campaign weeks.

    Returns mean SMAPE across all folds and all 3 campaign week models.
    """
    id_col = 'baseItemId' if level == 'item' else 'cluster1To1Id'
    campaign_col_base = 'campaign_week_'

    all_smapes = []

    for week_num in [1, 2, 3]:
        campaign_col = f'{campaign_col_base}{week_num}'

        # Filter to IDs that have campaign data
        if level == 'cluster':
            ids_with_campaigns = df_weekly[df_weekly[campaign_col] > 0][id_col].unique()
        else:
            ids_with_campaigns = df_weekly[df_weekly[campaign_col] == 1][id_col].unique()

        df_week = df_weekly[df_weekly[id_col].isin(ids_with_campaigns)].copy()
        df_week = df_week.sort_values('week_start').reset_index(drop=True)

        features = build_feature_list(week_num, level, df_week.columns.tolist())

        # Create temporal folds
        unique_weeks = df_week['week_start'].sort_values().unique()
        n_weeks = len(unique_weeks)
        fold_size = n_weeks // (n_folds + 1)

        for fold in range(n_folds):
            train_end_idx = (fold + 1) * fold_size
            val_end_idx = (fold + 2) * fold_size

            train_cutoff = unique_weeks[min(train_end_idx, n_weeks - 1)]
            val_cutoff = unique_weeks[min(val_end_idx, n_weeks - 1)]

            df_train = df_week[df_week['week_start'] <= train_cutoff]
            df_val = df_week[(df_week['week_start'] > train_cutoff) & (df_week['week_start'] <= val_cutoff)]

            if len(df_train) < 50 or len(df_val) < 10:
                continue

            # Balance training data (same logic as train_models)
            if level == 'cluster':
                df_campaign = df_train[df_train[campaign_col] > 0]
                df_baseline = df_train[(df_train[campaign_col] == 0) & (df_train['campaign_week_0'] > 0)]
            else:
                df_campaign = df_train[df_train[campaign_col] == 1]
                df_baseline = df_train[(df_train[campaign_col] == 0) & (df_train['campaign_week_0'] == 1)]

            n_campaign = len(df_campaign)
            n_target_baseline = min(n_campaign * 5, len(df_baseline))
            df_baseline_sampled = df_baseline.sort_values('week_start', ascending=False).head(n_target_baseline)
            df_train_balanced = pd.concat([df_campaign, df_baseline_sampled])

            X_train = df_train_balanced[features]
            y_train = df_train_balanced[TARGET_COL]

            # Sample weights
            if level == 'cluster':
                w_train = 1.0 + (df_train_balanced[campaign_col].values * 0.5)
            else:
                w_train = df_train_balanced[campaign_col].map({0: 1.0, 1: 3.0}).values

            X_val = df_val[features]
            y_val = df_val[TARGET_COL]

            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=50)]
            )

            # Evaluate on campaign weeks only (where accuracy matters most)
            if level == 'cluster':
                df_val_campaign = df_val[df_val[campaign_col] > 0]
            else:
                df_val_campaign = df_val[df_val[campaign_col] == 1]

            if len(df_val_campaign) < 3:
                continue

            y_true = np.expm1(df_val_campaign[TARGET_COL].values)
            y_pred = np.expm1(model.predict(df_val_campaign[features]))

            smape = METRIC_FUNCTIONS['SMAPE'](y_true, y_pred)
            all_smapes.append(smape)

    if not all_smapes:
        return 100.0  # worst case

    return float(np.mean(all_smapes))


def objective(trial, df_weekly, level):
    """Optuna objective function."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'early_stopping_rounds': 50,
    }

    smape = walk_forward_cv(df_weekly, level, params, n_folds=3)
    return smape


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for campaign forecasting')
    parser.add_argument('--item-ids', required=True, type=str, help='Comma-separated item IDs')
    parser.add_argument('--forecast-date', required=True, type=str, help='Forecast date (YYYY-MM-DD)')
    parser.add_argument('--campaign-start', required=True, type=str, help='Campaign start date (YYYY-MM-DD)')
    parser.add_argument('--level', default='item', choices=['item', 'cluster'], help='Model level to tune')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--output', type=str, default=None, help='Save best params to JSON file')
    args = parser.parse_args()

    item_ids = [x.strip() for x in args.item_ids.split(',')]
    forecast_date = pd.Timestamp(args.forecast_date)
    campaign_start = pd.Timestamp(args.campaign_start)
    forecast_horizon_weeks = int((campaign_start - forecast_date).days / 7)

    logging.info(f"Tuning {args.level}-level models ({args.n_trials} trials)")
    logging.info(f"  Items: {item_ids}")
    logging.info(f"  Forecast horizon: {forecast_horizon_weeks} weeks")

    # Fetch and preprocess data
    logging.info("Fetching data...")
    df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=item_ids)

    logging.info("Preprocessing...")
    df_item_weekly, df_cluster_weekly, df_items, item_normalizers, cluster_normalizers = preprocess_data(
        df_items, df_orders, df_campaigns,
        forecast_horizon_weeks=forecast_horizon_weeks,
        campaign_start_date=campaign_start,
        forecast_date=forecast_date,
        forecast_item_ids=item_ids
    )

    df_weekly = df_item_weekly if args.level == 'item' else df_cluster_weekly

    # Run optimization
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name=f'campaign_{args.level}_tuning'
    )

    study.optimize(
        lambda trial: objective(trial, df_weekly, args.level),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Report results
    print("\n" + "=" * 80)
    print(f"TUNING RESULTS ({args.level}-level, {args.n_trials} trials)")
    print("=" * 80)
    print(f"\nBest SMAPE: {study.best_value:.1f}%")
    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Format as config dict
    best_config = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'early_stopping_rounds': 50,
        **study.best_params
    }

    print(f"\nCopy to config_forecast.py as LGBM_PARAMS{'_CLUSTER' if args.level == 'cluster' else ''}:")
    print(json.dumps(best_config, indent=4))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'level': args.level,
                'best_smape': study.best_value,
                'best_params': best_config,
                'n_trials': args.n_trials,
                'forecast_horizon_weeks': forecast_horizon_weeks,
            }, f, indent=2)
        logging.info(f"Results saved to {args.output}")

    print("=" * 80)


if __name__ == '__main__':
    main()
