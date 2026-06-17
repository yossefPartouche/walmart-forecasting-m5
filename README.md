## Why the Baseline Underperformed

The baseline model produced poor forecasting results primarily due to a fundamental flaw in its inference logic, coupled with a severe lack of descriptive features and suboptimal model configuration. 

Here is a breakdown of the core issues:

* **Train/Inference Mismatch (The Biggest Flaw):** The model was trained on real, dynamic lag values but was fed a static "recent average" for future lags during inference. This massive distribution shift meant the model was making predictions based on data it didn't understand.
    * *Example:* When predicting sales for Day 8, the baseline used a flat historical average for `lag_7` instead of dynamically using the model's actual prediction for Day 1. 
* **Feature Poverty:** The baseline lacked the necessary historical and temporal context to capture real-world retail behavior.
    * *Missing Trends & Volatility:* It completely omitted rolling statistics (means, maxes, standard deviations), which are essential for understanding if a product's sales are currently trending up or wildly fluctuating.
    * *Shallow Seasonality:* It only looked at 7-day and 28-day lags, missing critical deeper signals like 4-week same-day averages (e.g., averaging the last 4 Mondays).
    * *Naive Event Handling:* It only flagged if an event was happening *today*. It completely missed pre-event shopping surges (`event_tomorrow`) and post-event drop-offs (`event_yesterday`).
    * *Ignored Calendar Effects:* It failed to account for realistic retail cycles, such as the month-end paycheck effect.
* **Suboptimal Tuning and Validation:** * *Weak Configuration:* The LightGBM setup was undertrained (only 1,000 trees) and lacked proper regularization to prevent overfitting.
    * *Unreliable Evaluation:* Validating on a short 30-day window produced noisy, unstable RMSE scores that didn't accurately reflect how the model would perform over the actual competition period.


### Remarks on the training data, 
* The all stores entry is likely affecting the results as it's extremely high in revenue compared to the rest, though it's useful for general pattern recognition across all stores so maybe should be dealt with seperately.
* Weekend Surge, people spend more over the weekend and this is noticable in the data ($286,000) compared to ($206,000)
* Certain events lead to outlier spikes and certain events lead outlier trofts (christmass vs Eid al-Fitr) either buy a lot before or the store closing earlier. 

M5 Walmart Forecasting — Model v2
Key findings from deep EDA driving this version:

M5 Walmart Forecasting — Model v3
Key findings from deep EDA driving this version:
 
# Model Version Comparison: Version 2 vs. Version 3

Below is a concise summary of the engineering differences and performance shifts between the two model versions.

## 1. Performance Summary
* **Version 2 Validation RMSE:** `2533.28` (Stopped early around ~400 iterations at a `0.02` learning rate).
* **Version 3 Validation RMSE:** `2491.53` (Converged methodically at iteration `1340` using a lower `0.01` learning rate).

---

## 2. Structural & Architectural Upgrades

### Trend & Long-Term Modeling
* **Version 2:** Relied on a basic global linear trend column (`days_since_start`) across all stores uniformly. 
* **Version 3:** Introduced **Store-Level Year-over-Year (YoY) Growth Ratios** computed over clean historical windows (Jan–Sep 2014 vs. Jan–Sep 2015). This allowed the model to scale long-term annual lags natively based on individual store trajectories (e.g., handles the fact that Store 10 had zero trend while others grew rapidly).

### Advanced Feature Engineering
* **Cyclical Encoding:** Version 3 introduced Sine and Cosine transformations for time fields (`dow_sin`, `dow_cos`, `month_sin`, etc.). This removed abrupt boundary discontinuities (such as the jump between December and January, or Sunday and Monday) which confused trees in Version 2.
* **Momentum Ratios:** Added scale-independent metrics (`momentum_7_28`, `momentum_7_56`, `momentum_28_365`) to capture growth velocity and acceleration instead of relying purely on absolute raw volumes.
* **Deeper Lags:** Integrated annual look-backs (`lag_364`, `lag_365`, `lag_371`, `lag_728`) to explicitly hook into multi-year matching days of the week.

### Enhanced Calendar Context
* **Granular Event Lifting:** Expanded the calendar profile by hard-coding precise event weights derived from exploratory data analysis (EDA).
* **High-Impact Spike Capture:** Automatically injected a localized `PreIndependenceDay` holiday on July 3rd across all years after data diagnostics showed a consistent unmapped +50% sales surge.

### Pipeline Robustness & Post-Processing
* **Bug-Free Shifting:** Replaced the index-based index grouping transforms in Version 2 (which caused `KeyError` failures on large dataframes) with a clean, optimized temporary binary flag buffer and a native `.shift()` loop.
* **Bottom-Up Hierarchy Reconciliation:** Version 3 safely bypasses the scale-distorting "All Stores" aggregate (`store_id: 0`) during training, but automatically computes its daily values at submission time by grouping and summing the individual child forecasts.
* **Hard Business Overrides:** Implemented absolute limits for predictable closures (such as forcing Christmas Day revenue to zero), preventing the tree estimators from assigning float noise to closed dates.

# Summary of How We Reached Score 4733

**Key Files**

| File | Purpose |
|------|---------|
|```eda_generation/phase2_build_dataset.py``` | Builds the master feature dataset |
|```eda_generation/master_dataset_phase2.csv``` | The enriched dataset (input to Prophet) |
|```models/aggregate_forecast.py``` | Aggregate-level Prophet forecast (not used in final) |
| ```models/forecast_2.py``` | The model that produced RMSE 4733 |
| `predictions/submission_2.csv` | The Best submission | 

### What made ``forecast_2.py`` Work

**Feature Engineering:**

1. Lag features: `lag_1, 2, 3, 6, 7, 14, 21, 28, 35, 42, 56, 364, 371`
2. `lag_364` same day last year, `lag_371` same weekday year
3. YoY scaling: `lag_364_scaled` = `lag_364` × `store_yoy_monthly`, same for `lag_371`
4. Pet-store and `pet-store`x`month` YoY ratios (2015 vs 2014)
5. Rolling stats: `rmean`, `rstd`, `rmax`, `rmin` for windows 7, 14, 28, 56 (anchored at lag-7)
6. Same-DOW averages: `lag7_mean_4w`, `lag7_mean_8w`
7. Momentum: `lag_7` / `lag_28`
8. Calendar: `dow`, `day_of_month`, `month`, `year`, `week_of_year`, `quarter`, `is_weekend`, `days_since_start`
9. Event features: `event_negative`, `event_positive`, `days_from_event`, lead/lag event windows
10. Cross-store features: correlated store pairs as exogenous signals
11. Categorical: `store_id`, `event`, `store_weekend`, `store_dow`

**Model:**

- LightGBM with `n_estimators=8000`, `learning_rate=0.02`, `num_leaves=255`
- Early stopping at 150 rounds → found best iteration at 560
- Retrained on full 16,500 rows at exactly 560 iterations (model)

**Forecast**

- Recursive day-to-day forecast for Oct-Dec 2015
- Each day's prediction fed back into history buffer for next day's lags
- Christmas explicitly zeroed out
- Store 0 (aggregate) computed as a sume 1-10

### Less Optimal Solutions

| Approach| Score | Why  |
|------|---------|-------|
|Prophet Only | ~9000 | No Lag Signal|
|Recursive Stacking Ensemble | ~8000 | Error Accumulation |
|Hierarchical reconciliation | ~6100 | Aggregate Model underestimated |
|Phase 8 Prophet + LightGBM | ~ 9600 | Forecast Period had no Features |
| Raw `forecast_2` without YoY | ~5700 | Missing Year-over-Year Signal| 

### Three Additions taking the model ~5700 $\rightarrow$ 4733

1. Added `lag_364` and `lag_371` 
2. Added YoY scaling `lag_364_scaled` = `lag_364` × `store_yoy_monthly, to account for growth
3. Retrained on full dataset, at the best iteration found by early stopping.

---
---

## How to replicate result:
Run on terminal: `pip install lightgbm pandas numpy scikit-learn`

1. Build the first feature dataset 
Run from the project root: `python eda_generation/phase2_build_dataset.py`
Output: `eda_generation/master_dataset_phase2.csv`

2. Run the forecasting model 
`cd models`
`python forecast_2.py`

This will (in order):
- Loads `data/train.csv` and `data/calendar_events.csv`
- Builds all features including lags, rolling stats, YoY scaling, cross-store features
- Trains LightGBM with early stopping on a 920-row validation holdout
- Records best iteration (560)
- Retrains `model` on all 16,500 rows at exactly 560 iterations
- Forecasts Oct-Dec 2015 day by day using `model`
- Saves `submission_2.csv` to the `predictions/` folder

Output: predictions/submission_2.csv


---
---

## Predicting on the 80% Final Test

The model forecasts any future date range by:
1. Using real historical revenue values for lag features on the first days
2. Using its own predictions as lag features for subsequent days
3. This works for any horizon 92 days (Kaggle) or longer

**If the final test preiod is different from Oct-Dec 2015**

In `forecast_2.py` locate the following line:

`all_dates = pd.date_range(forecast_start, forecast_end, freq='D')`

where, `forecast_start` and `forecast_end` originate from the submission template:

`sub['date'] = pd.to_datetime(sub['id'].apply(lambda x: x.split('_')[1]), format='%Y%m%d')`

So as long as you swap in the correct submission template for the final test, `forecast_2.py` automatically forecasts the right period. 
No code changes needed, just replace `data/forecast_submission.csv `with the final test submission template.