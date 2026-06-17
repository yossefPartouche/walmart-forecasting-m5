import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prophet import Prophet
import shap
import numpy as np
import pandas as pd
from validation_framework import *

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'master_dataset_phase2.csv'))
df['date'] = pd.to_datetime(df['date'])

print(df['date'].max())
print(len(df['store_id'].unique()))

calendar = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'calendar_events.csv'))
calendar['date'] = pd.to_datetime(calendar['date'])

# Map events to types based on Phase 6 SHAP signal
# Keeping: federal_holiday, christian, jewish, sports
# Dropping: islamic (low SHAP), other (zero SHAP)

federal_holidays = [
    'PresidentsDay', 'MemorialDay', 'IndependenceDay', 
    'LaborDay', 'ColumbusDay', 'Thanksgiving', 'Christmas'
]

christian_events = [
    'LentStart', 'LentWeek2', 'Easter', 'OrthodoxEaster, Easter'
]

jewish_events = [
    'Purim End', 'Pesach End', 'Chanukah End'
]

sports_events = [
    'SuperBowl', 'NBAFinalsStart', 'NBAFinalsEnd'
]

keep_events = federal_holidays + christian_events + jewish_events + sports_events

holidays_df = calendar[calendar['event'].isin(keep_events)].copy()
holidays_df = holidays_df.rename(columns={'date': 'ds', 'event': 'holiday'})
holidays_df['lower_window'] = -2   # 2 days before event
holidays_df['upper_window'] = 2    # 2 days after event

print(holidays_df.head(20))
print("Total holidays:", len(holidays_df))

store_ids = df['store_id'].unique()
all_prophet_results = []

for store_id in store_ids:
    print(f"Fitting Prophet for store {store_id}...")
    
    store_df = df[df['store_id'] == store_id][['date', 'revenue']].copy()
    store_df = store_df.rename(columns={'date': 'ds', 'revenue': 'y'})
    store_df = store_df.sort_values('ds').reset_index(drop=True)
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    
    model.fit(store_df)
    
    forecast = model.predict(store_df[['ds']])
    
    result = pd.DataFrame({
        'store_id': store_id,
        'date': forecast['ds'],
        'prophet_prediction': forecast['yhat'],
        'prophet_trend': forecast['trend'],
        'prophet_weekly': forecast['weekly'],
        'prophet_yearly': forecast['yearly'],
    })
    
    all_prophet_results.append(result)

## Predictions - Testing part ##

# =============================================================================
# GENERATE FORECAST PERIOD OCT-DEC 2015
# =============================================================================

future_dates = pd.date_range(start='2015-10-01', end='2015-12-31', freq='D')
all_future_results = []

for store_id in df['store_id'].unique():
    print(f"Forecasting store {store_id}...")
    
    store_df = df[df['store_id'] == store_id][['date', 'revenue']].copy()
    store_df = store_df.rename(columns={'date': 'ds', 'revenue': 'y'})
    store_df = store_df.sort_values('ds').reset_index(drop=True)
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    
    model.fit(store_df)
    
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    
    result = pd.DataFrame({
    'store_id': store_id,
    'date': forecast['ds'],
    'prophet_prediction': forecast['yhat'],
    'prophet_trend': forecast['trend'],
    'prophet_weekly': forecast['weekly'],
    'prophet_yearly': forecast['yearly'],
})
    
    all_future_results.append(result)

future_prophet_df = pd.concat(all_future_results, ignore_index=True)
future_prophet_df['prophet_prediction'] = future_prophet_df['prophet_prediction']
print(future_prophet_df.head(20))
print("Shape:", future_prophet_df.shape)

future_prophet_df.to_csv(
    os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'forecast_period_prophet.csv'),
    index=False
)
print("Saved forecast_period_prophet.csv")

# =============================================================================
# END FORECAST PERIOD OCT-DEC 2015
# =============================================================================

## Predictions - Testing part ##

prophet_df = pd.concat(all_prophet_results, ignore_index=True)

df = df.merge(prophet_df, on=['store_id', 'date'], how='left')

print("Null check:")
print(df[['prophet_prediction', 'prophet_trend', 'prophet_weekly', 'prophet_yearly']].isnull().sum())

df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'eda_generation', 'master_dataset_phase7.csv'), index=False)
print("Phase 7 complete. Saved master_dataset_phase7.csv")