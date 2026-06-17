import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import lightgbm as lgb
import numpy as np
import pandas as pd
from validation_framework import *

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'master_dataset_phase7.csv'))
df['date'] = pd.to_datetime(df['date'])
df['store_id'] = df['store_id'].astype('category').cat.codes
df['state'] = df['state'].astype('category').cat.codes
df['store_x_event'] = df['store_id'] * df['is_event']
df['state_x_event'] = df['state'] * df['is_event']

# =============================================================================
# BUILD AGGREGATE DAILY SERIES
# =============================================================================

agg_df = df.groupby('date').agg(
    revenue=('revenue', 'sum'),
    lag_1=('lag_1', 'sum'),
    lag_7=('lag_7', 'sum'),
    lag_14=('lag_14', 'sum'),
    lag_28=('lag_28', 'sum'),
    lag_90=('lag_90', 'sum'),
    lag_365=('lag_365', 'sum'),
    lag_728=('lag_728', 'sum'),
    rolling_mean_7=('rolling_mean_7', 'sum'),
    rolling_mean_28=('rolling_mean_28', 'sum'),
    rolling_mean_90=('rolling_mean_90', 'sum'),
    rolling_std_7=('rolling_std_7', 'mean'),
    rolling_std_28=('rolling_std_28', 'mean'),
    is_event=('is_event', 'max'),
    days_before_event=('days_before_event', 'first'),
    days_after_event=('days_after_event', 'first'),
    event_type_sports=('event_type_sports', 'max'),
    event_type_federal_holiday=('event_type_federal_holiday', 'max'),
    event_type_christian=('event_type_christian', 'max'),
    event_type_cultural=('event_type_cultural', 'max'),
    event_type_jewish=('event_type_jewish', 'max'),
    event_type_islamic=('event_type_islamic', 'max'),
    prophet_prediction=('prophet_prediction', 'sum'),
    prophet_trend=('prophet_trend', 'sum'),
    prophet_weekly=('prophet_weekly', 'mean'),
    prophet_yearly=('prophet_yearly', 'sum'),
    store_yoy=('store_yoy', 'mean'),
    store_yoy_monthly=('store_yoy_monthly', 'mean'),
    global_revenue_lag_7=('global_revenue_lag_7', 'first'),
    global_revenue_lag_28=('global_revenue_lag_28', 'first'),
    global_rolling_mean_7=('global_rolling_mean_7', 'first'),
    global_rolling_mean_28=('global_rolling_mean_28', 'first'),
).reset_index()

# Calendar features
agg_df['day_of_week'] = agg_df['date'].dt.dayofweek
agg_df['month'] = agg_df['date'].dt.month

print("Aggregate series shape:", agg_df.shape)
print(agg_df.head())

# =============================================================================
# FEATURE LIST
# =============================================================================

AGG_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_90",
    "lag_365", "lag_728",
    "rolling_mean_7", "rolling_mean_28", "rolling_mean_90",
    "rolling_std_7", "rolling_std_28",
    "day_of_week", "month",
    "is_event", "days_before_event", "days_after_event",
    "event_type_sports", "event_type_federal_holiday",
    "event_type_christian", "event_type_cultural",
    "event_type_jewish", "event_type_islamic",
    "prophet_prediction", "prophet_trend",
    "prophet_weekly", "prophet_yearly",
    "store_yoy", "store_yoy_monthly",
    "global_revenue_lag_7", "global_revenue_lag_28",
    "global_rolling_mean_7", "global_rolling_mean_28",
]

TARGET = "revenue"

MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 500,
    "num_leaves": 16,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# =============================================================================
# TRAIN ON FULL LEADERBOARD TRAINING DATA
# =============================================================================

train_agg = agg_df[agg_df['date'] <= '2015-09-30'].copy()
print(f"Training aggregate model on {len(train_agg)} rows")

model_agg = lgb.LGBMRegressor(**MODEL_PARAMS)
model_agg.fit(
    train_agg[AGG_FEATURES],
    train_agg[TARGET],
)
print("Aggregate model trained.")

# =============================================================================
# GENERATE FORECAST PERIOD
# =============================================================================

future_prophet = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'forecast_period_prophet.csv'))
future_prophet['date'] = pd.to_datetime(future_prophet['date'])

calendar = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'calendar_events.csv'))
calendar['date'] = pd.to_datetime(calendar['date'])
calendar = calendar[(calendar['date'] >= '2015-10-01') & (calendar['date'] <= '2015-12-31')]

forecast_dates = pd.date_range(start='2015-10-01', end='2015-12-31', freq='D')

# Build aggregate revenue buffer
agg_buffer = agg_df.set_index('date')['revenue'].to_dict()

def get_agg_lag(d, n):
    return agg_buffer.get(d - pd.Timedelta(days=n), np.nan)

def get_agg_rolling(d, window, func='mean'):
    vals = [agg_buffer.get(d - pd.Timedelta(days=i), np.nan) for i in range(1, window+1)]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan
    return np.mean(vals) if func == 'mean' else (np.std(vals) if len(vals) > 1 else 0.0)

# YoY at aggregate level
v14 = agg_df[(agg_df['date'].dt.year == 2014) & (agg_df['date'].dt.month <= 9)]['revenue'].mean()
v15 = agg_df[(agg_df['date'].dt.year == 2015) & (agg_df['date'].dt.month <= 9)]['revenue'].mean()
agg_yoy = v15 / v14 if v14 > 0 else 1.0

agg_forecasts = []

for forecast_date in forecast_dates:
    
    event_row = calendar[calendar['date'] == forecast_date]
    is_event = 1 if len(event_row) > 0 else 0
    event_name = event_row['event'].values[0] if is_event else None

    future_events = calendar[calendar['date'] > forecast_date]['date']
    past_events = calendar[calendar['date'] < forecast_date]['date']
    days_before = int((future_events.min() - forecast_date).days) if len(future_events) > 0 else 0
    days_after = int((forecast_date - past_events.max()).days) if len(past_events) > 0 else 0

    event_types = {
        'event_type_sports': ['SuperBowl', 'NBAFinalsStart', 'NBAFinalsEnd'],
        'event_type_federal_holiday': ['PresidentsDay', 'MemorialDay', 'IndependenceDay', 'LaborDay', 'ColumbusDay', 'Thanksgiving', 'Christmas'],
        'event_type_christian': ['LentStart', 'LentWeek2', 'Easter', 'OrthodoxEaster, Easter'],
        'event_type_cultural': ['ValentinesDay', 'StPatricksDay', 'Cinco De Mayo', "Mother's day", "Father's day"],
        'event_type_jewish': ['Purim End', 'Pesach End', 'Chanukah End'],
        'event_type_islamic': ['Ramadan starts', 'Eid al-Fitr'],
    }
    event_type_flags = {k: 0 for k in event_types}
    if event_name:
        for etype, enames in event_types.items():
            if event_name in enames:
                event_type_flags[etype] = 1

    # Prophet aggregate components
    prophet_agg = future_prophet.groupby('date').agg(
        prophet_prediction=('prophet_prediction', 'sum'),
        prophet_trend=('prophet_trend', 'sum'),
        prophet_weekly=('prophet_weekly', 'mean'),
        prophet_yearly=('prophet_yearly', 'sum'),
    ).reset_index()
    prophet_row = prophet_agg[prophet_agg['date'] == forecast_date]

    lag_365 = get_agg_lag(forecast_date, 365)
    lag_728 = get_agg_lag(forecast_date, 728)

    row = {
        "lag_1": get_agg_lag(forecast_date, 1),
        "lag_7": get_agg_lag(forecast_date, 7),
        "lag_14": get_agg_lag(forecast_date, 14),
        "lag_28": get_agg_lag(forecast_date, 28),
        "lag_90": get_agg_lag(forecast_date, 90),
        "lag_365": lag_365,
        "lag_728": lag_728,
        "rolling_mean_7": get_agg_rolling(forecast_date, 7),
        "rolling_mean_28": get_agg_rolling(forecast_date, 28),
        "rolling_mean_90": get_agg_rolling(forecast_date, 90),
        "rolling_std_7": get_agg_rolling(forecast_date, 7, 'std'),
        "rolling_std_28": get_agg_rolling(forecast_date, 28, 'std'),
        "day_of_week": forecast_date.dayofweek,
        "month": forecast_date.month,
        "is_event": is_event,
        "days_before_event": days_before,
        "days_after_event": days_after,
        **event_type_flags,
        "prophet_prediction": prophet_row['prophet_prediction'].values[0] if len(prophet_row) > 0 else np.nan,
        "prophet_trend": prophet_row['prophet_trend'].values[0] if len(prophet_row) > 0 else np.nan,
        "prophet_weekly": prophet_row['prophet_weekly'].values[0] if len(prophet_row) > 0 else np.nan,
        "prophet_yearly": prophet_row['prophet_yearly'].values[0] if len(prophet_row) > 0 else np.nan,
        "store_yoy": agg_yoy,
        "store_yoy_monthly": agg_yoy,
        "global_revenue_lag_7": get_agg_lag(forecast_date, 7),
        "global_revenue_lag_28": get_agg_lag(forecast_date, 28),
        "global_rolling_mean_7": get_agg_rolling(forecast_date, 7),
        "global_rolling_mean_28": get_agg_rolling(forecast_date, 28),
    }

    row_df = pd.DataFrame([row])
    pred = model_agg.predict(row_df[AGG_FEATURES]).clip(0)[0]

    if forecast_date == pd.Timestamp('2015-12-25'):
        pred = 0.0

    agg_buffer[forecast_date] = pred
    agg_forecasts.append({'date': forecast_date, 'agg_prediction': pred})

agg_forecast_df = pd.DataFrame(agg_forecasts)
print("Aggregate forecast shape:", agg_forecast_df.shape)
print(agg_forecast_df.head(10))

agg_forecast_df.to_csv(
    os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'aggregate_forecast.csv'),
    index=False
)
print("Saved aggregate_forecast.csv")