import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('eda_generation/master_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['store_id', 'date']).reset_index(drop=True)

# ---------------------------------------------------------
# 1. State Features 
# ---------------------------------------------------------
state_daily = df.groupby(['state', 'date'])['revenue'].sum().reset_index()
state_daily = state_daily.rename(columns={'revenue': 'state_total_revenue'})

# FIXED: Explicitly sort by state and date to guarantee chronological order before shifts
state_daily = state_daily.sort_values(['state', 'date']).reset_index(drop=True)

state_daily['state_revenue_lag_7'] = state_daily.groupby('state')['state_total_revenue'].shift(7)
state_daily['state_revenue_lag_28'] = state_daily.groupby('state')['state_total_revenue'].shift(28)

state_daily['state_rolling_mean_7'] = state_daily.groupby('state')['state_total_revenue'].transform(lambda x: x.rolling(7).mean().shift(1))
state_daily['state_rolling_mean_28'] = state_daily.groupby('state')['state_total_revenue'].transform(lambda x: x.rolling(28).mean().shift(1))
state_daily['state_mean_revenue'] = state_daily.groupby('state')['state_total_revenue'].transform(lambda x: x.rolling(90).mean().shift(1))

# FIXED: Drop Phase 1 versions before merging Phase 2 versions to prevent column collision
cols_to_override = [
    'state_rolling_mean_7', 'state_rolling_mean_28',
    'global_rolling_mean_7', 'global_rolling_mean_28'
]
df = df.drop(columns=[c for c in cols_to_override if c in df.columns])

# Merge back
expected_rows = len(df)

df = df.merge(state_daily, on=['state', 'date'], how='left')
assert len(df) == expected_rows, f"Row count changed after state merge: {len(df)} vs {expected_rows}"




# ---------------------------------------------------------
# 2. Global Features 
# ---------------------------------------------------------
# Take a clean snapshot before any merges
df_base = df[['store_id', 'date', 'state', 'revenue']].drop_duplicates()

global_daily = df_base.groupby('date')['revenue'].sum().reset_index()
global_daily = global_daily.rename(columns={'revenue': 'global_total_revenue'})
global_daily = global_daily.sort_values('date').reset_index(drop=True)

global_daily['global_revenue_lag_7'] = global_daily['global_total_revenue'].shift(7)
global_daily['global_revenue_lag_28'] = global_daily['global_total_revenue'].shift(28)

global_daily['global_rolling_mean_7'] = global_daily['global_total_revenue'].rolling(7).mean().shift(1)
global_daily['global_rolling_mean_28'] = global_daily['global_total_revenue'].rolling(28).mean().shift(1)

# Helper feature for ratios
global_daily['global_mean_revenue_90d'] = global_daily['global_total_revenue'].rolling(90).mean().shift(1)

# Merge back
df = df.merge(global_daily, on='date', how='left')
assert len(df) == expected_rows, f"Row count changed after state merge: {len(df)} vs {expected_rows}"

# ---------------------------------------------------------
# 3. Relative Ratios & Cleanup
# ---------------------------------------------------------
# Calculate the store's 90-day historical mean (shifted to prevent leakage)
df = df.sort_values(['store_id', 'date']).reset_index(drop=True)
df['store_mean_revenue'] = df.groupby('store_id')['revenue'].transform(lambda x: x.rolling(90).mean().shift(1))

# Calculate ratios strictly using the historical 90-day rolling averages
df['store_to_state_ratio'] = df['store_mean_revenue'] / (df['state_mean_revenue'] + 1e-8)
df['store_to_global_ratio'] = df['store_mean_revenue'] / (df['global_mean_revenue_90d'] + 1e-8)

# FIXED: Drop all internal helpers AND unshifted aggregation totals to prevent target leakage
cols_to_drop = ['global_mean_revenue_90d', 'state_total_revenue', 'global_total_revenue']
df = df.drop(columns=cols_to_drop)


# ---------------------------------------------------------
# 4. NaN Handling Strategy
# ---------------------------------------------------------
# Lags up to 90 days will produce NaNs at the start of each store's series.
#
# For LightGBM/XGBoost: leave NaNs as-is. Tree models handle missing splits natively.
#
# For Linear models / Neural Nets: impute before training using the block below.
#
# WARNING: Do NOT use bfill() here. It fills early NaNs by borrowing from future
# rows within the store series, which is look-ahead leakage. Use ffill() or 0-fill only.
#
# Uncomment if needed:
# cols_to_fill = [
#     'state_revenue_lag_7', 'state_revenue_lag_28',
#     'state_rolling_mean_7', 'state_rolling_mean_28', 'state_mean_revenue',
#     'global_revenue_lag_7', 'global_revenue_lag_28',
#     'global_rolling_mean_7', 'global_rolling_mean_28',
#     'store_mean_revenue', 'store_to_state_ratio', 'store_to_global_ratio'
# ]
# df[cols_to_fill] = df.groupby('store_id')[cols_to_fill].ffill()
# df[cols_to_fill] = df[cols_to_fill].fillna(0)

# ---------------------------------------------------------
# 5. Long Lag Features (lag_365, lag_728)
# ---------------------------------------------------------
df = df.sort_values(['store_id', 'date']).reset_index(drop=True)

df['lag_365'] = df.groupby('store_id')['revenue'].shift(365)
df['lag_728'] = df.groupby('store_id')['revenue'].shift(728)

# ---------------------------------------------------------
# 6. YoY Features
# ---------------------------------------------------------
yoy_store = {}
yoy_store_month = {}

for sid in sorted(df['store_id'].unique()):
    s = df[df['store_id'] == sid]
    v14 = s[(s['date'].dt.year == 2014) & (s['date'].dt.month <= 9)]['revenue'].mean()
    v15 = s[(s['date'].dt.year == 2015) & (s['date'].dt.month <= 9)]['revenue'].mean()
    yoy_store[sid] = v15 / v14 if v14 > 0 else 1.0

    for m in range(1, 13):
        v_14 = s[(s['date'].dt.year == 2014) & (s['date'].dt.month == m)]['revenue'].mean()
        v_15 = s[(s['date'].dt.year == 2015) & (s['date'].dt.month == m)]['revenue'].mean()
        if not np.isnan(v_14) and not np.isnan(v_15) and v_14 > 0:
            yoy_store_month[(sid, m)] = v_15 / v_14
        else:
            yoy_store_month[(sid, m)] = yoy_store[sid]

df['store_yoy'] = df['store_id'].map(yoy_store)
df['store_yoy_monthly'] = df.apply(
    lambda r: yoy_store_month.get((r['store_id'], r['month']), yoy_store[r['store_id']]),
    axis=1
)

df['lag_365_scaled'] = df['lag_365'] * df['store_yoy_monthly']
df['lag_728_scaled'] = df['lag_728'] * (df['store_yoy_monthly'] ** 2)

print("New features added: lag_365, lag_728, lag_365_scaled, lag_728_scaled, store_yoy, store_yoy_monthly")
print("NaN counts:")
print(df[['lag_365', 'lag_728', 'lag_365_scaled', 'lag_728_scaled', 'store_yoy', 'store_yoy_monthly']].isnull().sum())

EXPECTED_PHASE2_COLS = [
    'state_revenue_lag_7', 'state_revenue_lag_28',
    'state_rolling_mean_7', 'state_rolling_mean_28', 'state_mean_revenue',
    'global_revenue_lag_7', 'global_revenue_lag_28',
    'global_rolling_mean_7', 'global_rolling_mean_28',
    'store_mean_revenue', 'store_to_state_ratio', 'store_to_global_ratio',
    'lag_365', 'lag_728', 'lag_365_scaled', 'lag_728_scaled',
    'store_yoy', 'store_yoy_monthly',
]
missing = [c for c in EXPECTED_PHASE2_COLS if c not in df.columns]
assert not missing, f"Missing columns: {missing}"



print("Column validation passed. Final columns:", df.columns.tolist())

df.to_csv('eda_generation/master_dataset_phase2.csv', index=False)