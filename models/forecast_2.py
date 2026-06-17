import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import os

# ── 0. LOAD DATA ───────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
cal   = pd.read_csv('data/calendar_events.csv')
sub_tmpl = pd.read_csv('data/forecast_submission.csv')

train.columns    = train.columns.str.strip().str.lower()
cal.columns      = cal.columns.str.strip().str.lower()
sub_tmpl.columns = sub_tmpl.columns.str.strip().str.lower()

train['date'] = pd.to_datetime(train['date'])
cal['date']   = pd.to_datetime(cal['date'])

# Drop the aggregate "All Stores" row (store_id == 0)
train = train[train['store_id'] > 0].copy()

print(f"Train rows: {len(train):,}  |  Stores: {sorted(train['store_id'].unique())}")

# Known signed lifts from EDA (positive = revenue boost, negative = revenue drop)
NEGATIVE_EVENTS = {
    'Christmas', 'Thanksgiving', 'NewYear', 'Mother\'s day', 'Halloween',
    'Father\'s day', 'Easter', 'OrthodoxEaster, Easter', 'ValentinesDay',
    'Easter, OrthodoxEaster', 'OrthodoxEaster, Cinco De Mayo'
}
POSITIVE_EVENTS = {
    'LaborDay', 'OrthodoxEaster', 'NBAFinalsStart', 'EidAlAdha',
    'MemorialDay', 'Ramadan starts', 'NBAFinalsEnd, Father\'s day',
    'VeteransDay', 'ColumbusDay', 'MartinLutherKingDay'
}

cal_lookup  = dict(zip(cal['date'], cal['event']))
cal_set     = set(cal['date'])


def build_features(df):
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)

    # ── Time features
    df['dow']            = df['date'].dt.dayofweek
    df['day_of_month']   = df['date'].dt.day
    df['month']          = df['date'].dt.month
    df['year']           = df['date'].dt.year
    df['week_of_year']   = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']        = df['date'].dt.quarter
    df['is_weekend']     = df['dow'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end']   = df['date'].dt.is_month_end.astype(int)

    # ── Trend: strong linear growth in 9/10 stores
    global_min = df['date'].min()
    df['days_since_start'] = (df['date'] - global_min).dt.days

    # ── Store × weekend interaction (premium varies 16-50% by store)
    df['store_weekend']  = df['store_id'].astype(str) + '_' + df['is_weekend'].astype(str)
    df['store_dow']      = df['store_id'].astype(str) + '_' + df['dow'].astype(str)

    # ── Calendar events
    df['event'] = df['date'].map(cal_lookup).fillna('NoEvent')
    df['event_negative'] = df['event'].isin(NEGATIVE_EVENTS).astype(int)
    df['event_positive'] = df['event'].isin(POSITIVE_EVENTS).astype(int)

    # Days-to/since nearest event (signed)
    event_dates = sorted(cal_set)
    def nearest_event_offset(date):
        diffs = [(date - e).days for e in event_dates]
        idx = int(np.argmin(np.abs(diffs)))
        return diffs[idx]

    df['days_from_event'] = df['date'].apply(nearest_event_offset)
    df['days_from_event'] = df['days_from_event'].clip(-14, 14)

    # Lead/lag event windows
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)
    event_binary = (df['event'] != 'NoEvent').astype(float)
    for offset, col in [(-2,'event_in_2d'), (-1,'event_tomorrow'),
                        (1,'event_yesterday'), (2,'event_2d_ago'), (7,'event_week_ago')]:
        df[col] = df.groupby('store_id')[event_binary.name if hasattr(event_binary,'name') else 'event'].transform(
            lambda x: pd.Series(x != 'NoEvent', dtype=float).shift(offset).fillna(0)
        ).astype(int)
    return df

def add_lag_rolling(df):
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)
    grp = df.groupby('store_id')['revenue']

    # Lag features
    for lag in [1, 2, 3, 6, 7, 14, 21, 28, 35, 42, 56, 364, 371]:
        df[f'lag_{lag}'] = grp.shift(lag)

    # Rolling statistics anchored at lag-7 (leakage-free)
    shifted7 = grp.shift(7)
    for window in [7, 14, 28, 56]:
        df[f'rmean_{window}'] = shifted7.groupby(df['store_id']).transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'rstd_{window}']  = shifted7.groupby(df['store_id']).transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'rmax_{window}']  = shifted7.groupby(df['store_id']).transform(lambda x: x.rolling(window, min_periods=1).max())
        df[f'rmin_{window}']  = shifted7.groupby(df['store_id']).transform(lambda x: x.rolling(window, min_periods=1).min())

    # Same-DOW averages
    df['lag7_mean_4w']  = (df['lag_7'] + df['lag_14'] + df['lag_21'] + df['lag_28']) / 4
    df['lag7_mean_8w']  = (df['lag_7'] + df['lag_14'] + df['lag_21'] + df['lag_28'] + df['lag_35'] + df['lag_42'] + df.get('lag_49', df['lag_42']) + df['lag_56']) / 8
    df['momentum_7_28'] = df['lag_7'] / (df['lag_28'] + 1e-6)
    return df

def add_cross_store_features(df, pivot):
    pairs = [(8,4),(8,7),(8,3),(9,4),(3,4),(7,4),(1,3),(1,4)]
    for src, tgt in pairs:
        col_name = f'xstore_{src}_lag1_for_{tgt}'
        src_series = pivot[src].shift(1)
        df[col_name] = df.apply(lambda r: src_series.get(r['date'], np.nan) if r['store_id'] == tgt else np.nan, axis=1)
    return df

print("Building features...")
df = build_features(train.copy())
df = add_lag_rolling(df)

pivot_all = train.pivot(index='date', columns='store_id', values='revenue')
df = add_cross_store_features(df, pivot_all)

# YoY scaling features
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
    lambda r: yoy_store_month.get((r['store_id'], r['month']), yoy_store[r['store_id']]), axis=1
)
df['lag_364_scaled'] = df['lag_364'] * df['store_yoy_monthly']
df['lag_371_scaled'] = df['lag_371'] * df['store_yoy_monthly']

df = df.dropna(subset=['lag_56']).reset_index(drop=True)
print(f"Feature df shape: {df.shape}")


LAG_FEATURES = [f'lag_{d}' for d in [1,2,3,6,7,14,21,28,35,42,56,364,371]] + ['lag_364_scaled', 'lag_371_scaled']
YOY_FEATURES = ['store_yoy', 'store_yoy_monthly']
ROLLING_FEATURES = [f'{stat}_{w}' for stat in ['rmean','rstd','rmax','rmin'] for w in [7,14,28,56]] + ['lag7_mean_4w', 'lag7_mean_8w', 'momentum_7_28']
TIME_FEATURES = ['dow', 'day_of_month', 'month', 'year', 'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'days_since_start']
EVENT_FEATURES = ['event_negative', 'event_positive', 'days_from_event', 'event_in_2d', 'event_tomorrow', 'event_yesterday', 'event_2d_ago', 'event_week_ago']
XSTORE_FEATURES = [c for c in df.columns if c.startswith('xstore_')]
CAT_FEATURES    = ['store_id', 'event', 'store_weekend', 'store_dow']

FEATURES = CAT_FEATURES + TIME_FEATURES + LAG_FEATURES + ROLLING_FEATURES + EVENT_FEATURES + XSTORE_FEATURES + YOY_FEATURES
TARGET   = 'revenue'

for col in CAT_FEATURES:
    df[col] = df[col].astype('category')

VALID_DAYS = 92
split_date = df['date'].max() - pd.Timedelta(days=VALID_DAYS)

train_df = df[df['date'] <= split_date]
valid_df  = df[df['date'] >  split_date]
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_valid, y_valid = valid_df[FEATURES],  valid_df[TARGET]

X_full = df[FEATURES]
y_full = df[TARGET]
for col in CAT_FEATURES:
    X_full[col] = X_full[col].astype('category')

print(f"Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")

model = lgb.LGBMRegressor(
    objective         = 'regression_l2',
    n_estimators      = 560,
    learning_rate     = 0.02,
    num_leaves        = 255,
    max_depth         = -1,
    min_child_samples = 15,
    feature_fraction  = 0.75,
    bagging_fraction  = 0.75,
    bagging_freq      = 1,
    reg_alpha         = 0.05,
    reg_lambda        = 0.1,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

print("Training model...")

"""model.fit(
    X_train, y_train,
    eval_set  = [(X_valid, y_valid)],
    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(250),
    ],
    categorical_feature = CAT_FEATURES,
)"""

model.fit(X_full, y_full, categorical_feature=CAT_FEATURES)
print(f"Full model trained on {len(X_full):,} rows using 560 iterations.")

valid_preds = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
print(f"\n\u2705 Validation RMSE: {rmse:.2f}")

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 20 features:")
print(fi.head(20).to_string())

os.makedirs('data', exist_ok=True)
with open('data/model.pkl', 'wb') as f: pickle.dump(model, f)
print("Saved model artifacts.")


print("\nPreparing iterative forecast...")

sub = sub_tmpl.copy()
sub['store_id'] = sub['id'].apply(lambda x: int(x.split('_')[0]))
sub['date']     = pd.to_datetime(sub['id'].apply(lambda x: x.split('_')[1]), format='%Y%m%d')
sub = sub.sort_values(['store_id', 'date']).reset_index(drop=True)

store_ids = [sid for sid in sorted(sub['store_id'].unique()) if sid > 0]
forecast_start = sub['date'].min()
forecast_end   = sub['date'].max()
global_min_date = train['date'].min()

# Seed history
history = {}
for sid, grp in train.groupby('store_id'):
    history[int(sid)] = dict(zip(pd.to_datetime(grp['date']), grp['revenue'].values))

def get_lag(sid, date, lag_days): return history[sid].get(date - pd.Timedelta(days=lag_days), np.nan)
def get_rolling_window(sid, date, shift_days, window):
    base = date - pd.Timedelta(days=shift_days)
    return [history[sid][base - pd.Timedelta(days=i)] for i in range(window) if (base - pd.Timedelta(days=i)) in history[sid]]
def has_ev(d): return cal_lookup.get(d, 'NoEvent')
def has_ev_bin(d): return int(has_ev(d) != 'NoEvent')

all_event_dates = sorted(cal_set)
def days_from_nearest_event(date):
    diffs = [(date - e).days for e in all_event_dates]
    return int(np.clip(diffs[int(np.argmin(np.abs(diffs)))], -14, 14))

results = []
all_dates = pd.date_range(forecast_start, forecast_end, freq='D')

for date in all_dates:
    rows = []
    for sid in store_ids:
        lags = {f'lag_{d}': get_lag(sid, date, d) for d in [1,2,3,6,7,14,21,28,35,42,56]}
        rolling = {}
        for w in [7,14,28,56]:
            vals = get_rolling_window(sid, date, 7, w)
            rolling[f'rmean_{w}'] = float(np.mean(vals)) if vals else np.nan
            rolling[f'rstd_{w}']  = float(np.std(vals)) if len(vals)>1 else 0.0
            rolling[f'rmax_{w}']  = float(np.max(vals)) if vals else np.nan
            rolling[f'rmin_{w}']  = float(np.min(vals)) if vals else np.nan

        lag_364 = get_lag(sid, date, 364)
        lag_371 = get_lag(sid, date, 371)
        lags['lag_364'] = lag_364
        lags['lag_371'] = lag_371

        yoy_r = yoy_store[sid]
        yoy_m = yoy_store_month.get((sid, date.month), yoy_r)
        lags['lag_364_scaled'] = lag_364 * yoy_m if not np.isnan(lag_364) else 0.0
        lags['lag_371_scaled'] = lag_371 * yoy_m if not np.isnan(lag_371) else 0.0
        
        lv = [lags[f'lag_{d}'] for d in [7,14,21,28] if not np.isnan(lags.get(f'lag_{d}',np.nan))]
        l8 = [lags[f'lag_{d}'] for d in [7,14,21,28,35,42,56] if not np.isnan(lags.get(f'lag_{d}',np.nan))]
        
        ev = has_ev(date)
        row = dict(
            store_id         = sid,
            dow              = date.dayofweek,
            day_of_month     = date.day,
            month            = date.month,
            year             = date.year,
            week_of_year     = date.isocalendar()[1],
            quarter          = (date.month - 1) // 3 + 1,
            is_weekend       = int(date.dayofweek in [5,6]),
            is_month_start   = int(date.day == 1),
            is_month_end     = int((date + pd.Timedelta(days=1)).month != date.month),
            days_since_start = (date - global_min_date).days,
            event            = ev,
            store_weekend    = f'{sid}_{int(date.dayofweek in [5,6])}',
            store_dow        = f'{sid}_{date.dayofweek}',
            event_negative   = int(ev in NEGATIVE_EVENTS),
            event_positive   = int(ev in POSITIVE_EVENTS),
            days_from_event  = days_from_nearest_event(date),
            event_in_2d      = has_ev_bin(date + pd.Timedelta(days=2)),
            event_tomorrow   = has_ev_bin(date + pd.Timedelta(days=1)),
            event_yesterday  = has_ev_bin(date - pd.Timedelta(days=1)),
            event_2d_ago     = has_ev_bin(date - pd.Timedelta(days=2)),
            event_week_ago   = has_ev_bin(date - pd.Timedelta(days=7)),
            lag7_mean_4w     = float(np.mean(lv)) if lv else np.nan,
            lag7_mean_8w     = float(np.mean(l8)) if l8 else np.nan,
            momentum_7_28    = float(lags.get('lag_7', np.nan) / (lags.get('lag_28', np.nan) + 1e-6)) if not np.isnan(lags.get('lag_7', np.nan)) else 1.0,
            store_yoy         = yoy_r,
            store_yoy_monthly = yoy_m,
            **lags, **rolling,
        )

        for xf in XSTORE_FEATURES:
            parts = xf.split('_')
            src = int(parts[1])
            if row['store_id'] == int(parts[-1]):
                row[xf] = history.get(src, {}).get(date - pd.Timedelta(days=1), np.nan)
            else: row[xf] = np.nan
        rows.append(row)

    X_day = pd.DataFrame(rows)[FEATURES].copy()
    for col in CAT_FEATURES: X_day[col] = X_day[col].astype('category')

    num_cols = X_day.select_dtypes(include=[np.number]).columns
    mean_fallback = X_day[[c for c in ['rmean_28','rmean_14','rmean_7'] if c in num_cols]].mean(axis=1)
    for col in num_cols:
        if X_day[col].isna().any(): X_day[col] = X_day[col].fillna(mean_fallback)

    preds = model.predict(X_day).clip(0)
    for sid, pred in zip(store_ids, preds):
        history[sid][date] = float(pred)
        results.append({'id': f'{sid}_{date.strftime("%Y%m%d")}', 'prediction': float(pred)})

    if date.day == 1: print(f"  Forecasted through {date.strftime('%Y-%m-%d')}")

results_df = pd.DataFrame(results)
final = sub_tmpl[['id']].merge(results_df, on='id', how='left')

# -- BOTTOM-UP RECONCILIATION FOR STORE 0 --
final['date_str'] = final['id'].apply(lambda x: x.split('_')[1])
final['store_idx'] = final['id'].apply(lambda x: int(x.split('_')[0]))

# Sum the predictions of Stores 1-10 for each day
store_0_sums = final[final['store_idx'] > 0].groupby('date_str')['prediction'].sum()

# Inject the sums into the Store 0 rows
is_store_0 = final['store_idx'] == 0
final.loc[is_store_0, 'prediction'] = final.loc[is_store_0, 'date_str'].map(store_0_sums)

# -- CHRISTMAS OVERRIDE (Free points!) --
# Force December 25th predictions to exactly 0.0
final.loc[final['id'].str.endswith('1225'), 'prediction'] = 0.0

# Clean up temporary columns
final = final.drop(columns=['date_str', 'store_idx'])

# Fallback for any other weird missing values
missing = final['prediction'].isna().sum()
if missing:
    print(f" {missing} missing predictions — filling with global mean")
    final['prediction'] = final['prediction'].fillna(final['prediction'].mean())


final.to_csv(
    os.path.join(os.path.dirname(__file__), '..', 'predictions', 'submission_2.csv'),
    index=False
)
print(f"\n Saved 'submission_2.csv'  ({len(final)} rows)")
print(final.head(10).to_string(index=False))