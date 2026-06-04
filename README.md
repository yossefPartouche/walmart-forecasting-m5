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
* 