"""
Configuration for forecasting models and evaluation.
"""

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_STATE = 42
STRATIFY_BY = None  # Don't stratify - most clusters only appear in 1-2 campaigns

# ============================================================================
# LIGHTGBM MODEL PARAMETERS
# ============================================================================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
}

# Cluster-level hyperparameters (balanced capacity with strong regularization)
LGBM_PARAMS_CLUSTER = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 15,  # Back to 15 for more capacity (from 10)
    'learning_rate': 0.02,  # Even slower learning (from 0.03)
    'feature_fraction': 0.7,  # Less feature sampling
    'bagging_fraction': 0.7,  # Less bagging
    'bagging_freq': 5,
    'max_depth': 5,  # Slightly deeper (from 4)
    'min_data_in_leaf': 40,  # Middle ground (from 50)
    'lambda_l1': 0.2,  # Stronger L1 regularization
    'lambda_l2': 0.2,  # Stronger L2 regularization
    'drop_rate': 0.1,  # Dropout: randomly ignore 10% of features per tree
    'verbose': -1,
    'n_estimators': 400,  # More trees (from 300)
    'early_stopping_rounds': 50,  # Force campaign_week_N and lag features to have positive relationship with sales
}

# Quantile regression parameters for uncertainty estimation
LGBM_QUANTILE_PARAMS = {
    **LGBM_PARAMS,
    'objective': 'quantile',
    'alpha': 0.5,  # Will be overridden for 10th and 90th percentiles
}

LGBM_QUANTILE_PARAMS_CLUSTER = {
    **LGBM_PARAMS_CLUSTER,
    'objective': 'quantile',
    'alpha': 0.5,
}

QUANTILES = [0.1, 0.5, 0.9]  # 10th percentile, median, 90th percentile

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# Base features common to all models
COMMON_FEATURES = [
    'week_of_year_sin',
    'week_of_year_cos',
    'sales_lag_12w',  # 3-month lag for recent trends
    #'sales_lag_52w',  # Year-over-year comparison (safe now with normalization)
]

# Categorical features (LightGBM will handle these specially)
CATEGORICAL_FEATURES = []  # Empty - no categorical features helping so far

# Week-specific features (will be selected based on campaign week)
WEEK_SPECIFIC_FEATURES = {
    1: ['sales_lag_week1', 'sales_rolling_4w_week1', 'cluster_sales_lag_week1', 'cluster_sales_rolling_4w_week1'],
    2: ['sales_lag_week2', 'sales_rolling_4w_week2', 'cluster_sales_lag_week2', 'cluster_sales_rolling_4w_week2'],
    3: ['sales_lag_week3', 'sales_rolling_4w_week3', 'cluster_sales_lag_week3', 'cluster_sales_rolling_4w_week3'],
}

# Campaign features
CAMPAIGN_FEATURES = [
    'campaign_week_0',
    'campaign_week_1',
    'campaign_week_2',
    'campaign_week_3',
]

# Cluster-specific features (only for cluster-level models)
CLUSTER_SPECIFIC_FEATURES = [
    'num_items_in_cluster',  # Size of cluster (helps differentiate single-item vs multi-item clusters)
]

# Target variable (log-transformed)
TARGET_COL = 'salesQuantityKgL_log'

# Original scale target (for evaluation)
TARGET_COL_ORIGINAL = 'salesQuantityKgL'

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS = ['SMAPE', 'MAE', 'MAPE', 'RMSE']

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    import numpy as np
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100.0 * np.mean(diff)

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    import numpy as np
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    import numpy as np
    mask = y_true != 0
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    import numpy as np
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

METRIC_FUNCTIONS = {
    'SMAPE': smape,
    'MAE': mae,
    'MAPE': mape,
    'RMSE': rmse,
}

# ============================================================================
# FORECAST CONFIGURATION
# ============================================================================
DEFAULT_FORECAST_HORIZON_WEEKS = 8
MAX_FORECAST_HORIZON_WEEKS = 12
MIN_FORECAST_HORIZON_WEEKS = 4

# Model storage
MODEL_DIR = '../data/models/'
CACHE_DIR = '../data/cache/'

# Cluster filtering
EXCLUDE_SINGLE_ITEM_CLUSTERS_FROM_FORECAST = True  # Skip forecasting single-item clusters separately
INCLUDE_SINGLE_ITEM_CLUSTERS_IN_TRAINING = True  # But use them for training

# Warning thresholds
ITEM_VS_CLUSTER_WARNING_THRESHOLD = 1.0  # Warn if item forecast > cluster forecast
