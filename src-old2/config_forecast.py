"""
Configuration for campaign forecasting models.
Adjust these parameters to tune model behavior.
"""

# Model Parameters
MODEL_CONFIG = {
    # LightGBM parameters
    'n_estimators': 100,        # Number of boosting rounds
    'max_depth': 5,             # Maximum tree depth (lower = more regularized)
    'learning_rate': 0.1,       # Step size shrinkage
    'min_child_samples': 5,     # Minimum samples per leaf
    'reg_alpha': 0.1,           # L1 regularization
    'reg_lambda': 0.1,          # L2 regularization
    'random_state': 42,
    'verbose': -1
}

# Feature Configuration
FEATURE_COLUMNS = [
    'week',                      # Week of year (seasonality)
    'month',                     # Month (seasonality)
    'year',                      # Year (trend)
    'week_sin',                  # Cyclical week encoding (sine)
    'week_cos',                  # Cyclical week encoding (cosine)
    'month_sin',                 # Cyclical month encoding (sine)
    'month_cos',                 # Cyclical month encoding (cosine)
    'is_campaign_week',          # Campaign flag (KEY FEATURE)
    'cluster_sales_lag_1',       # Previous week cluster sales
    'cluster_sales_rolling4w',   # 4-week rolling average cluster sales
    'weeks_since_campaign',      # Time since last campaign
    'campaigns_ytd'              # Cumulative campaigns this year
]

# Data Filtering
DATA_CONFIG = {
    'min_training_weeks': 10,    # Minimum weeks of history required
    'use_recent_years': 2,       # Use last N years more heavily
    'exclude_incomplete_weeks': True
}

# Forecast Configuration  
FORECAST_CONFIG = {
    'confidence_interval': 0.95,  # (Future) For prediction intervals
    'min_forecast_value': 0,      # Floor for predictions
    'round_to_units': True        # Round forecasts to whole units
}

# Logging
LOG_FEATURE_IMPORTANCE = True
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
