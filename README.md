# 🔆 M5 Walmart Hierarchical Forecasting

This repository contains the evolution, feature engineering, and modeling pipeline for the M5 Walmart Forecasting Challenge. Our final recursive forecasting approach achieved an **RMSE score of 4733**, driven by dynamic lag updates, robust YoY growth scaling, and deep calendar context.

---

## The Winning Strategy (Score: 4733)

Our best-performing pipeline relies on a heavily engineered recursive LightGBM model.

### Key Files

| File | Purpose |
| --- | --- |
| `eda_generation/phase2_build_dataset.py` | Builds the master feature dataset. |
| `eda_generation/master_dataset_phase2.csv` | The enriched dataset (input to Prophet/LightGBM). |
| `models/aggregate_forecast.py` | Aggregate-level Prophet forecast (experimental, not used in final). |
| `models/forecast_2.py` | **The final model that produced RMSE 4733.** |
| `predictions/submission_2.csv` | The best submission file. |

### What Made `forecast_2.py` Work

#### 1. Feature Engineering

* **Lag Features:** `lag_1`, `2`, `3`, `6`, `7`, `14`, `21`, `28`, `35`, `42`, `56`, `364`, `371`.
* **Annual Mapping:** `lag_364` (same day last year) and `lag_371` (same weekday last year).
* **YoY Scaling:** `lag_364_scaled = lag_364 × store_yoy_monthly` (similarly applied to `lag_371`) to explicitly account for store-level growth.
* **Pet-Store Ratios:** Pet-store and `pet-store × month` YoY ratios (2015 vs. 2014).
* **Rolling Stats:** `rmean`, `rstd`, `rmax`, `rmin` for windows 7, 14, 28, 56 (anchored at lag-7).
* **Same-DOW Averages:** `lag7_mean_4w`, `lag7_mean_8w`.
* **Momentum:** `lag_7 / lag_28`.
* **Calendar:** `dow`, `day_of_month`, `month`, `year`, `week_of_year`, `quarter`, `is_weekend`, `days_since_start`.
* **Events:** `event_negative`, `event_positive`, `days_from_event`, and specific lead/lag event windows.
* **Cross-Store Features:** Correlated store pairs used as exogenous signals.
* **Categorical Features:** `store_id`, `event`, `store_weekend`, `store_dow`.

#### 2. Model Configuration

* **Architecture:** LightGBM with `n_estimators=8000`, `learning_rate=0.02`, `num_leaves=255`.
* **Training Strategy:** Early stopping triggered at 150 rounds on a 920-row validation holdout (found the optimal iteration to be **560**). The model was then strictly retrained on the *full* 16,500 rows for exactly 560 iterations.

#### 3. Inference & Forecasting

* **Recursive Day-to-Day:** Forecasts the Oct–Dec 2015 window iteratively. Each day's prediction is written back into the history buffer to generate dynamic lag features for the next day.
* **Business Overrides:** Christmas explicitly zeroed out.
* **Hierarchy:** Store 0 (Aggregate) is computed purely as a bottom-up sum of Stores 1–10.

### The "Top 3" Additions

Implementing these three specific features took the model from a baseline score of ~5700 down to **4733**:

1. Adding deep annual lags (`lag_364` and `lag_371`).
2. Adding YoY scaling (`lag_364_scaled`) to capture growth velocities instead of just static volumes.
3. Retraining on the *entire* dataset at the exact optimal iteration found via early stopping.

---

## Experimental & Less Optimal Approaches

During development, several architectures were tested and ultimately discarded:

| Approach | Score | Why it Underperformed |
| --- | --- | --- |
| Prophet Only | ~9000 | Native Prophet lacked auto-regressive lag signals. |
| Phase 8 Prophet + LightGBM | ~9600 | Forecast period had no underlying features to support the meta-learner. |
| Recursive Stacking Ensemble | ~8000 | Severe error accumulation during the recursive loop. |
| Hierarchical Reconciliation | ~6100 | The top-level aggregate model consistently underestimated totals. |
| Raw `forecast_2` (No YoY) | ~5700 | Missing crucial Year-over-Year growth signals. |

---

## Model Evolution: Overcoming Baseline Flaws

### Why the Initial Baseline Underperformed

The original baseline model produced poor forecasting results due to three fundamental flaws:

1. **Train/Inference Mismatch (The Biggest Flaw):** The model was trained on real, dynamic lag values but was fed a static "recent average" for future lags during inference. *Example:* When predicting Day 8, the baseline used a flat historical average for `lag_7` instead of the model's actual Day 1 prediction.
2. **Feature Poverty:** * *Missing Volatility:* Completely omitted rolling statistics (means, maxes, standard deviations).
* *Shallow Seasonality:* Missed critical deeper signals like 4-week same-day averages.
* *Naive Event Handling:* Only flagged if an event happened *today*, missing pre-event surges and post-event drop-offs.
* *Ignored Calendar Effects:* Failed to account for realistic retail cycles like month-end paycheck effects.


3. **Suboptimal Tuning:** The setup was undertrained (1,000 trees), lacked regularization, and evaluated against a noisy, short 30-day window.

### EDA Insights Driving Version Upgrades

* **The "All Stores" Skew:** The aggregate store (`store_id: 0`) generated revenue so high it distorted gradient learning. (Fix: Handled separately via bottom-up summation).
* **Weekend Surge:** Clear weekend spending spikes ($286k vs $206k weekday averages) required explicit `is_weekend` and `store_weekend` mapping.
* **Event Polarity:** Specific events cause outlier spikes (pre-Christmas shopping), while others cause outlier troughs (store closures on Eid al-Fitr/Christmas).

### Version 2 vs. Version 3: Structural Upgrades

Moving from Model v2 to v3 refined our approach significantly, dropping validation RMSE from **2533.28** to **2491.53**.

* **Trend Modeling:** Replaced a basic global linear trend (`days_since_start`) with **Store-Level YoY Growth Ratios**.
* **Cyclical Encoding:** Added Sine/Cosine transformations for time fields (`dow_sin`, `month_cos`, etc.) to remove abrupt boundary discontinuities (e.g., Dec 31st to Jan 1st).
* **Momentum Ratios:** Added scale-independent metrics (`momentum_7_28`, etc.) to capture velocity.
* **Calendar Context:** Hard-coded granular event lifts based on EDA, including injecting a localized `PreIndependenceDay` holiday on July 3rd (+50% sales surge).
* **Pipeline Robustness:** Replaced index-based transforms that caused `KeyError` crashes with a clean, native `.shift()` loop, and added hard overrides for predictable closures.

---

## How to Replicate the Results

**1. Install Dependencies**

```bash
pip install lightgbm pandas numpy scikit-learn

```

**2. Build the Feature Dataset**
Run this from the project root to generate the master dataset:

```bash
python eda_generation/phase2_build_dataset.py

```

*(Output: `eda_generation/master_dataset_phase2.csv`)*

**3. Run the Forecasting Model**
Navigate to the models directory and execute the pipeline:

```bash
cd models
python forecast_2.py

```

**Execution Flow:**

1. Loads `train.csv` and `calendar_events.csv`.
2. Builds all features (lags, rolling stats, YoY scaling, cross-store).
3. Trains LightGBM with early stopping to find the optimal iteration (560).
4. Retrains the model on all 16,500 rows at exactly 560 iterations.
5. Recursively forecasts Oct–Dec 2015 day-by-day.
6. Exports `submission_2.csv` to the `predictions/` folder.

---

## Predicting on the 80% Final Test Set

Our recursive model is dynamically designed to handle any future date range (e.g., the 92-day Kaggle horizon). It works by using real historical revenue values for the initial lag features, and then seamlessly transitioning to using its own predictions as lag features for subsequent days.

**To predict for a new test period:**
In `forecast_2.py`, the date range is dynamically extracted from the submission template:

```python
sub['date'] = pd.to_datetime(sub['id'].apply(lambda x: x.split('_')[1]), format='%Y%m%d')
all_dates = pd.date_range(forecast_start, forecast_end, freq='D')

```

**No code changes are needed.** Simply replace `data/forecast_submission.csv` with your final test submission template, run `forecast_2.py`, and the script will automatically adjust the forecast horizon to match the target dates.