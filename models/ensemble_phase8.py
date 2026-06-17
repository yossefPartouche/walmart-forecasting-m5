import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import lightgbm as lgb
import shap
import numpy as np
import pandas as pd
from validation_framework import *

# TEMPORARY JUST FOR DATA SUBMMISSION UNDERSTANDING 

"""submission = pd.read_csv('data/forecast_submission.csv')
print(submission.head(30))
print("Unique id prefixes:", submission['id'].str.split('_').str[0].unique())
print("Total rows:", len(submission))"""

# Temporary END #

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'master_dataset_phase7.csv'))
df['date'] = pd.to_datetime(df['date'])
df['store_id'] = df['store_id'].astype('category').cat.codes
df['state'] = df['state'].astype('category').cat.codes
df['store_x_event'] = df['store_id'] * df['is_event']
df['state_x_event'] = df['state'] * df['is_event']

# TEMPORARY JUST FOR DATA SUBMMISSION UNDERSTANDING 

"""print("store_id unique values in df:", sorted(df['store_id'].unique()))
print("store_id dtype:", df['store_id'].dtype)
print(df[['store_id', 'store_name']].drop_duplicates().sort_values('store_id'))"""

# Temporary END #

print(df.shape)
print(df[['prophet_prediction', 'prophet_trend', 'prophet_weekly', 'prophet_yearly']].isnull().sum())


LAG_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_90",
]

ROLLING_FEATURES = [
    "rolling_mean_7", "rolling_mean_28", "rolling_mean_90",
    "rolling_std_7", "rolling_std_28",
]

CALENDAR_FEATURES = [
    "day_of_week", "month",
]

STORE_FEATURES = [
    "store_id", "state",
]

STATE_LEVEL_FEATURES = [
    "state_revenue_lag_7", "state_revenue_lag_28",
    "state_rolling_mean_7", "state_rolling_mean_28",
    "state_mean_revenue",
]

GLOBAL_LEVEL_FEATURES = [
    "global_revenue_lag_7", "global_revenue_lag_28",
    "global_rolling_mean_7", "global_rolling_mean_28",
]

RELATIVE_FEATURES = [
    "store_mean_revenue", "store_to_state_ratio", "store_to_global_ratio",
]

EVENT_FEATURES = [
    "is_event", "days_before_event", "days_after_event",
    "event_type_sports", "event_type_federal_holiday",
    "event_type_christian", "event_type_cultural",
    "event_type_jewish", "event_type_islamic",
    "store_x_event", "state_x_event",
]

PROPHET_FEATURES = [
    "prophet_prediction", "prophet_trend",
    "prophet_weekly", "prophet_yearly",
]

# Approach 1 - all features including Prophet components
STACKING_FEATURES = (
    LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES +
    STORE_FEATURES + STATE_LEVEL_FEATURES + GLOBAL_LEVEL_FEATURES +
    RELATIVE_FEATURES + EVENT_FEATURES + PROPHET_FEATURES
)

# Approach 2 - Phase 6 features only, no Prophet (LightGBM predicts residual)
RESIDUAL_FEATURES = (
    LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES +
    STORE_FEATURES + STATE_LEVEL_FEATURES + GLOBAL_LEVEL_FEATURES +
    RELATIVE_FEATURES + EVENT_FEATURES
)

TARGET = "revenue"


print("Stacking features:", len(STACKING_FEATURES))
print("Residual features:", len(RESIDUAL_FEATURES))


MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

def train_model(train_df, features, target="revenue"):
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(
        train_df[features],
        train_df[target],
    )
    return model

def evaluate_split(model, valid_df, features, target="revenue"):
    preds = model.predict(valid_df[features])
    metrics = compute_metrics(valid_df[target].values, preds)
    return metrics, preds

def prophet_only_metrics(valid_df, target="revenue"):
    metrics = compute_metrics(
        valid_df[target].values,
        valid_df["prophet_prediction"].values,
    )
    return metrics


# =============================================================================
# APPROACH 1 — FEATURE STACKING
# =============================================================================

train_df, valid_df = run_validation_split(df, VALIDATION_SPLITS[0])
stacking_model_a = train_model(train_df, STACKING_FEATURES)
stacking_metrics_a, stacking_preds_a = evaluate_split(stacking_model_a, valid_df, STACKING_FEATURES)
prophet_metrics_a = prophet_only_metrics(valid_df)
print("=== SPLIT A ===")
print("Prophet only:      ", prophet_metrics_a)
print("Stacking ensemble: ", stacking_metrics_a)

train_df, valid_df = run_validation_split(df, VALIDATION_SPLITS[1])
stacking_model_b = train_model(train_df, STACKING_FEATURES)
stacking_metrics_b, stacking_preds_b = evaluate_split(stacking_model_b, valid_df, STACKING_FEATURES)
prophet_metrics_b = prophet_only_metrics(valid_df)
print("=== SPLIT B ===")
print("Prophet only:      ", prophet_metrics_b)
print("Stacking ensemble: ", stacking_metrics_b)

# =============================================================================
# APPROACH 2 — RESIDUAL MODELING
# =============================================================================

train_df, valid_df = run_validation_split(df, VALIDATION_SPLITS[0])
train_df = train_df.copy()
train_df['residual'] = train_df['revenue'] - train_df['prophet_prediction']
residual_model_a = train_model(train_df, RESIDUAL_FEATURES, target='residual')
residual_preds_a = valid_df['prophet_prediction'].values + residual_model_a.predict(valid_df[RESIDUAL_FEATURES])
residual_metrics_a = compute_metrics(valid_df['revenue'].values, residual_preds_a)
print("=== SPLIT A ===")
print("Residual ensemble: ", residual_metrics_a)

train_df, valid_df = run_validation_split(df, VALIDATION_SPLITS[1])
train_df = train_df.copy()
train_df['residual'] = train_df['revenue'] - train_df['prophet_prediction']
residual_model_b = train_model(train_df, RESIDUAL_FEATURES, target='residual')
residual_preds_b = valid_df['prophet_prediction'].values + residual_model_b.predict(valid_df[RESIDUAL_FEATURES])
residual_metrics_b = compute_metrics(valid_df['revenue'].values, residual_preds_b)
print("=== SPLIT B ===")
print("Residual ensemble: ", residual_metrics_b)

# =============================================================================
# COMPARISON TABLE
# =============================================================================

comparison = pd.DataFrame([
    {"approach": "Prophet Only",          "split": "A", **{k: v for k, v in prophet_metrics_a.items()}},
    {"approach": "Stacking Ensemble",     "split": "A", **{k: v for k, v in stacking_metrics_a.items()}},
    {"approach": "Residual Ensemble",     "split": "A", **{k: v for k, v in residual_metrics_a.items()}},
    {"approach": "Prophet Only",          "split": "B", **{k: v for k, v in prophet_metrics_b.items()}},
    {"approach": "Stacking Ensemble",     "split": "B", **{k: v for k, v in stacking_metrics_b.items()}},
    {"approach": "Residual Ensemble",     "split": "B", **{k: v for k, v in residual_metrics_b.items()}},
])

print(comparison.to_string(index=False))
comparison.to_csv("phase8_comparison_table.csv", index=False)
print("Saved phase8_comparison_table.csv")


# =============================================================================
# SHAP — STACKING ENSEMBLE (trained on Split B — largest training set)
# =============================================================================

explainer = shap.TreeExplainer(stacking_model_b)

sample = train_df[STACKING_FEATURES].sample(10000, random_state=42)

shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample)

shap_importance = pd.DataFrame({
    "feature": sample.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_importance.to_csv("shap_importance_phase8_stacking.csv", index=False)
print(shap_importance.to_string(index=False))
print("Saved shap_importance_phase8_stacking.csv")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": STACKING_FEATURES,
    "importance": stacking_model_b.feature_importances_,
}).sort_values("importance", ascending=False)

feature_importance.to_csv("feature_importance_phase8_stacking.csv", index=False)
print("Saved feature_importance_phase8_stacking.csv")