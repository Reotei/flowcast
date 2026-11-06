# FlowCast: Final Technical Report

PROJ-FLOW-260711, Internmo · Nov 6, 2026 · @Zoveio

## Summary

FlowCast forecasts traffic on the 25-segment Northline Corridor 30 to 120 minutes ahead. On test weeks the models never saw (6–31 May 2025) it met four of the six PRD success criteria.

| Criterion (PRD 3.3) | Target | Result on test | Status |
| --- | --- | --- | --- |
| Volume forecast accuracy | MAPE ≤ 12% | 8.93% (XGBoost) | Met |
| DL vs classical benchmark | Sequence model beats best classical RMSE | GRU 55.73 vs XGBoost 57.03 vehicles | Met |
| Congestion classification | Macro-F1 ≥ 0.80 | 0.766 (Random Forest) | Not met |
| Accident-risk ranking | ROC-AUC ≥ 0.75 | 0.618 (Logistic Regression) | Not met |
| Reproducibility | One command from raw data | `python -m src.pipeline --data-dir data/raw`, 18.5 min | Met |
| Dashboard | All nine views on real outputs | Five pages cover all nine views | Met |

The volume and travel-time forecasts are accurate enough for operational use. They cut the error of a naive "same as now" forecast by 47% and 37% respectively.

The two misses have clear causes in the data, set out in the congestion and accident-risk sections. Congestion errors are almost all one level off, concentrated in the narrow Heavy band. Accidents occur in 0.97% of windows and are only weakly predictable from the available inputs.

All models were selected on the validation weeks, and every test metric uses only originally recorded targets.

## Problem and data

The corridor operations team needs to see congestion and accident risk before they form, not after. FlowCast predicts four targets for each segment and each 30-minute window, 1 to 4 windows ahead:

- traffic volume (vehicles per window, regression)
- travel time (minutes, regression)
- congestion level (Free-flow, Moderate, Heavy, Severe; 4-class classification)
- accident risk (any accident in the window; binary classification)

The data covers 25 segments from 1 January to 31 May 2025 (151 days, 48 windows a day).

| Source table | Grain | Rows | Notes |
| --- | --- | --: | --- |
| traffic\_sensor\_log.csv | segment × 30 min | 178,468 | Dates as YYYY-MM-DD; gaps, duplicates and sensor faults |
| weather\_observations.csv | station × hour | 10,872 | Dates as DD/MM/YYYY; 13 spellings of 4 conditions |
| calendar\_events.csv | date | 151 | 6 holidays, 6 events, 2 roadwork periods |

Each segment maps to one of three weather stations (North 9 segments, Central 8, South 8). Hourly weather is joined to both 30-minute windows in its hour, on station and floored hour. Calendar flags are joined on date.

## Data cleaning

Cleaning turned 178,468 raw traffic rows into a complete grid of 181,200 rows (25 segments × 7,248 windows). No row was dropped silently: 2,471 rejected rows are kept in `data/interim/quarantine.csv`, and every rule is logged with its count in `reports/data_quality.md`.

The steps run in this order, because each depends on the one before:

1. **Deduplicate first.** 1,767 rows repeated a road\_id + date + time key. The most complete record per key was kept and the rest quarantined.
2. **Reindex to the full grid.** 4,499 windows were missing entirely. Adding them before any lag is computed means "one window back" always means 30 minutes back.
3. **Sensor faults to missing.** Physically impossible values were set to NaN: 239 negative volumes, 236 speeds above 200 km/h and 229 occupancies above 100%. Statistical outliers (|z| > 4 within segment × window) were flagged but kept.
4. **Repair volume from vehicle\_count.** vehicle\_count equals traffic\_volume in 100% of rows where both exist, so 4,587 missing or faulty volumes were repaired from it. These rows are excluded from all test metrics.
5. **Impute the rest.** Gaps of up to 2 windows use time interpolation within the segment (4,498 volume values). Longer gaps use the segment × time-of-day median from training dates only (1 volume value).
6. **Harmonise weather.** 13 raw spellings were stripped, lowercased and mapped to Clear, Cloudy, Rain and Fog. The 358 "Overcast" readings are not in the documented vocabulary and were mapped to Cloudy.
7. **Fill blank congestion labels.** 31,123 labels were blank. They were derived with the V/C rule; the 5,180 derived from imputed volume are excluded from evaluation.

After cleaning, mean volume is 431 vehicles per window (median 362), mean speed is 42.1 km/h and mean travel time is 3.6 minutes.

## Features and leakage controls

Every feature for a forecast made at window t uses only windows up to and including t. The feature table has 172,900 rows per horizon after the first 7 days are dropped as warm-up, because the one-week lag needs a full week of history.

| Group | Features |
| --- | --- |
| Lags (from window t back) | Volume at t, t−1, t−2 and the change t−1 to t; speed and travel time at t and t−1; occupancy, signal timing and vehicle mix at t |
| Seasonal lags (from the target window back) | Volume in the same window yesterday (48 back) and last week (336 back) |
| Rolling statistics | Mean and standard deviation of volume, and mean speed, over the 4 and 8 windows ending at t |
| Capacity | V/C ratio at t, headroom to capacity at t, congestion level at t, road capacity |
| Time (of the target window) | Time of day and day of week as sine/cosine pairs, weekend flag, peak flag (07:30–10:30, 17:00–20:30) |
| Weather (at t, not forecast) | Temperature, rainfall, visibility, rain flag (≥ 0.1 mm), low-visibility flag (< 1,000 m), Cloudy/Rain/Fog indicators |
| Calendar | Holiday, event and roadwork flags; holiday × peak; event × peak; event within one day |
| Accident history | Accidents on the segment in the past week and past 30 days, up to t |
| Location | Segment indicators, latitude, longitude |

The strongest single predictor of next-window volume is current volume (Pearson r = 0.932), followed by the same window last week (0.896) and the current V/C ratio (0.879).

The data contains three traps that would make scores look far better than they really are. Each is closed off:

- **Congestion is a function of volume.** The V/C rule reproduces 100% of the 146,171 existing labels. So the target window's volume, V/C, occupancy and speed are never inputs when predicting that window's congestion.
- **vehicle\_count is the volume target.** It equals volume in every row, so it is used only to repair volume during cleaning, never as a feature.
- **Interpolation looks ahead.** PRD 9.1 asks for time interpolation, but an interpolated value at t uses the value at t+1. Lag features therefore use a causal fill instead: forward fill for up to 2 windows, then the training median. The interpolated column is used only for EDA and the dashboard.

In addition, lags and rolling statistics are computed per segment in time order, shifting before rolling. Scalers, imputation medians and encoders are fitted on training dates only. A perturbation test in `tests/test_features_leakage.py` changes every value from a cut time onward and checks that no feature before that time moves. The pipeline also raises an alarm if any congestion model scores macro-F1 above 0.97.

## Evaluation protocol

The data is split by date, never shuffled, so every model predicts a future it has not seen.

| Split | Dates (2025) | Days | Grid rows | Used for |
| --- | --- | --: | --: | --- |
| Train | 1 Jan – 10 Apr | 100 | 120,000 | Fitting models, scalers and imputation medians |
| Validation | 11 Apr – 5 May | 25 | 30,000 | Choosing the model, the accident threshold and the interval widths |
| Test | 6 May – 31 May | 26 | 31,200 | Reported once, after all choices are fixed |

- **Tuning:** hyperparameters are searched with 3-fold `TimeSeriesSplit` inside the training dates. Each fold trains on earlier weeks and scores on the weeks after them.
- **Selection:** the model in use for each target is the one with the best validation score, never the best test score. For example, XGBoost was chosen for volume on validation RMSE (61.64 vs Random Forest 62.59).
- **Scoring only real values:** test metrics count only targets that were recorded. Repaired, interpolated and derived values are excluded, which leaves 29,606 volume windows and 30,296 congestion windows in the test set.
- **Baselines:** every target has a naive forecast to beat. These are "same as now" (persistence), "same window last week" and, for accidents, "accidents on this segment in the past 30 days".

| Task | Primary metric | Supporting metrics |
| --- | --- | --- |
| Volume, travel time | RMSE | MAE, MAPE, R² |
| Congestion level | Macro-F1 | Precision, recall, accuracy, confusion matrix |
| Accident risk | ROC-AUC | PR-AUC, precision and recall at the validation-chosen threshold |

Accuracy is not a headline metric for accidents: predicting "no accident" everywhere would score 99%.

## Mathematics used

Each mathematical idea below has a concrete place in the pipeline and was implemented or checked explicitly, not left to a library default.

**Linear regression by gradient descent (built once in NumPy).** The model is ŷ = Xw + b, trained on z-scored features with learning rate α = 0.1 for 600 epochs. The cost and its gradients are:

```latex
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
```

```latex
\frac{\partial J}{\partial w} = \frac{1}{m} X^{\top} (\hat{y} - y), \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i), \qquad w \leftarrow w - \alpha \frac{\partial J}{\partial w}
```

Its test RMSE (70.89) is within 0.7% of the closed-form least-squares solution from scikit-learn (70.39), which confirms the descent converged. The cost curve is in `reports/figures/gd_cost_curve.png`.

**Congestion bands (V/C ratio).** A 30-minute window carries half the hourly design capacity, so:

```latex
\text{V/C} = \frac{\text{volume}}{\text{capacity} / 2}
```

The bands are: below 0.50 Free-flow; 0.50–0.79 Moderate; 0.80–0.99 Heavy; 1.00 and above Severe.

**Cyclical time.** Time of day w (0–47) is encoded as sin(2πw/48) and cos(2πw/48). This places 23:30 next to 00:00 instead of 47 steps away. Day of week is encoded the same way with period 7.

**Error metrics.** RMSE = √(mean of squared errors) penalises large misses; MAPE = mean of |error| ÷ actual, as a percentage. Macro-F1 averages the F1 of the four congestion classes equally, so the rare Heavy and Severe classes count as much as Free-flow. ROC-AUC is the probability that a random accident window is ranked above a random accident-free window (0.5 = chance).

**Likely ranges (90% intervals).** On the validation weeks, for each hour of the day h, q\_h is the 90th percentile of the absolute residual |y − ŷ|. A forecast's range is then ŷ − q\_h to ŷ + q\_h. This is a split-conformal style interval: on the test weeks it held 91% of recorded counts against a 90% target.

**Accident probabilities (prior correction).** Class weights make a model act as if accidents were as common as non-accidents. Bayes' rule with the true training rate π restores real probabilities:

```latex
\text{odds}_{\text{true}} = \text{odds}_{\text{weighted}} \times \frac{\pi}{1 - \pi}, \qquad p = \frac{\text{odds}_{\text{true}}}{1 + \text{odds}_{\text{true}}}
```

The correction is monotonic, so ROC-AUC and PR-AUC are unchanged; only the displayed probability and the threshold value move.

**GRU cell (as in PyTorch's nn.GRU).** For input x\_t and previous hidden state h\_{t−1}, the reset gate r\_t decides how much of the old state feeds the candidate, and the update gate z\_t decides how much of the old state is kept:

```latex
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z), \quad r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
```

```latex
n_t = \tanh(W_n x_t + b_n + r_t \odot (U_n h_{t-1} + c_n)), \quad h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
```

Each gate is a set of matrix–vector products Wx + b. The network is trained by backpropagation through time with the Adam optimiser on mean squared error, with gradient norms clipped at 1.0.

## Results: traffic volume

Volume is the strongest result. The XGBoost model in use misses by 8.93% on a typical window (MAPE), against a target of 12% or less. Its RMSE of 57.0 vehicles is 47% below the naive "same as now" forecast.

&#91;embedded content: reports/scoreboard.csv · test split, 30 minutes ahead, 29,606 recorded windows\]

| Model | MAE (vehicles) | MAPE | R² |
| --- | --: | --: | --: |
| GRU | 38.5 | 9.70% | 0.963 |
| LSTM | 38.9 | 9.17% | 0.961 |
| XGBoost (in use) | 38.6 | 8.93% | 0.961 |
| Random Forest | 40.1 | 9.25% | 0.958 |
| Decision Tree | 46.0 | 10.50% | 0.943 |
| Linear regression (closed form) | 49.1 | 12.63% | 0.941 |
| Linear regression (NumPy GD) | 49.6 | 12.85% | 0.940 |
| Baseline: same as now | 73.7 | 16.60% | 0.863 |
| Baseline: same window last week | 75.1 | 18.50% | 0.837 |

The GRU has the lowest RMSE but a higher MAPE than XGBoost. RMSE weighs the large misses in busy windows, where the GRU is better; MAPE weighs relative misses, which are largest in quiet night-time windows. The volume target (MAPE) is reported for XGBoost, the model chosen on validation.

&#91;embedded content: reports/scoreboard.csv · test split, horizons 1 to 4\]

The learned model stays accurate up to 2 hours ahead because the same-window-yesterday and same-window-last-week lags carry the daily and weekly pattern. The naive forecast has no such memory.

## Results: travel time

The XGBoost model in use predicts the next window's travel time to within 1.08 minutes RMSE (9.5% MAPE), 37% better than the naive forecast. The PRD sets no numeric target for travel time.

| Model | RMSE (min) | MAE (min) | MAPE | R² |
| --- | --: | --: | --: | --: |
| XGBoost (in use) | 1.081 | 0.447 | 9.49% | 0.828 |
| Random Forest | 1.093 | 0.442 | 9.19% | 0.824 |
| Decision Tree | 1.245 | 0.536 | 11.65% | 0.772 |
| Linear regression | 1.341 | 0.634 | 16.81% | 0.735 |
| Baseline: same as now | 1.711 | 0.690 | 14.28% | 0.569 |

XGBoost and Random Forest are within 1% of each other on RMSE; XGBoost was kept because it scored better on validation (1.083 vs 1.101). Linear regression beats the baseline on RMSE but not on MAPE. Travel time is length divided by speed, a curve that a straight-line model can only approximate, and the tree models capture it better.

## Results: congestion level

Congestion macro-F1 is 0.766, short of the 0.80 target. The model in use is right on 86.7% of windows, and almost every error is exactly one level off.

&#91;embedded content: reports/scoreboard.csv · test split, congestion, horizon 1\]

Two kinds of method were compared. Classifiers predict the level directly. The other route forecasts volume and applies the V/C bands to it. Both land in the same narrow range, so the limit comes from the task itself, not from the choice of model.

Confusion matrix for the Random Forest in use (rows are actual levels, columns are forecasts):

| Actual \\ Forecast | Free-flow | Moderate | Heavy | Severe |
| --- | --: | --: | --: | --: |
| Free-flow | 16,923 | 1,145 | 0 | 0 |
| Moderate | 452 | 6,107 | 711 | 29 |
| Heavy | 0 | 437 | 1,910 | 733 |
| Severe | 0 | 4 | 507 | 1,338 |

| Level | Windows | Recall | Precision | F1 |
| --- | --: | --: | --: | --: |
| Free-flow | 18,068 | 93.7% | 97.4% | 0.955 |
| Moderate | 7,299 | 83.7% | 79.4% | 0.815 |
| Heavy | 3,080 | 62.0% | 61.1% | 0.615 |
| Severe | 1,849 | 72.4% | 63.7% | 0.678 |

Why the target is missed:

- **Errors are one level off.** Only 33 of 30,296 windows (0.1%) are two levels off. Free-flow is never confused with Heavy or Severe.
- **Heavy is the weak class.** Its band is the narrowest (V/C 0.80 to 0.99) and it has neighbours on both sides. A small volume error moves it into Moderate (437 windows) or Severe (733 windows).
- **Macro-F1 weighs every class equally.** Free-flow and Moderate score 0.955 and 0.815, but the two rarest classes, Heavy and Severe, pull the average down to 0.766.

The model still adds a clear gain over the naive "same level as now" forecast, which scores 0.607 macro-F1 and 77.1% accuracy.

**A note on model selection.** The Random Forest was chosen among the classical models on validation macro-F1 (0.764), before the sequence models existed. The GRU route scored higher on validation (0.771). Under the same rule it would have been chosen, but its test score (0.768) is also below 0.80, so the conclusion does not change.

## Results: accident risk

Accident-risk ROC-AUC is 0.618, well short of the 0.75 target. Accidents are rare (0.97% of test windows) and only weakly predictable from the inputs available. The model still ranks risk about twice as well as chance.

| Model | ROC-AUC | PR-AUC | Precision at threshold | Recall at threshold |
| --- | --: | --: | --: | --: |
| Logistic regression (in use) | 0.618 | 0.0196 | 2.7% | 27.4% |
| SVM (LinearSVC) | 0.605 | 0.0169 | 2.2% | 19.6% |
| Decision Tree | 0.592 | 0.0171 | 2.2% | 31.8% |
| Random Forest | 0.584 | 0.0159 | 2.2% | 9.8% |
| XGBoost | 0.525 | 0.0149 | 2.5% | 6.8% |
| Baseline: past-30-day accident count | 0.535 | 0.0113 | 1.2% | 18.9% |

How to read these numbers:

- **PR-AUC against chance.** A random ranking has a PR-AUC equal to the accident rate, 0.0097. The model's 0.0196 is a 2.0× lift.
- **At the operating threshold.** The threshold (0.023) was chosen on validation to maximise F1. On the test weeks it flags 3,041 of 30,412 windows (10%) and catches 81 of the 296 accident windows (27%). Its precision is 2.7%, about 2.7 times the base rate.
- **Uncertainty.** With only 296 accident windows in the test set, the ROC-AUC standard error is about 0.018 (Hanley–McNeil). The 95% interval is roughly 0.58 to 0.65, still well below 0.75.

Why accidents are hard to predict here:

- **The strongest signal covers few windows.** Rain and fog raise the accident rate about fourfold: 3.46% and 3.53% of windows, against 0.83% in clear weather. But rain and fog cover only 4.6% of windows.
- **Most accidents happen in ordinary conditions.** About two thirds of accident windows are in clear weather (roughly 1,100 of 1,650), where the rate is low and shows no strong pattern.
- **Flexible models overfit the rare cases.** XGBoost scores 0.525, close to chance. With roughly 1,100 accident windows in training, the simple logistic regression generalises best.

No tuning was done against the test set to close the gap; doing so would be leakage. What would genuinely help is listed under limitations.

## Sequence models

The GRU, built and trained from scratch in PyTorch with no pre-trained weights, beats the best classical model on volume RMSE by 2.3% (55.73 vs 57.03). This meets the deep-learning benchmark criterion.

**Architecture.** Each sample is the last 24 windows (12 hours) of one segment, with 15 features per step. A 2-layer recurrent encoder with 64 hidden units reads the sequence. Its final hidden state is joined with the known time and calendar features of the target window and a 4-dimensional learned segment embedding. A dense layer (64 units, ReLU) then outputs the forecast:

```latex
\hat{y} = W_2 \, \text{ReLU}(W_1 [h_N, x_T, e_{\text{seg}}] + b_1) + b_2
```

**Training.** Sequences are built per segment in time order: 106,058 for training, 28,482 for validation and 29,606 for test. Inputs are scaled with training statistics only.

| Setting | Value |
| --- | --- |
| Optimiser | Adam, learning rate 0.002, weight decay 0.00001 |
| Batch size, maximum epochs | 512, 30 |
| Regularisation | Dropout 0.2 between layers and in the head; gradient norm clipped at 1.0 |
| Early stopping | Stop after 4 epochs without validation improvement; keep the best epoch |
| Learning-rate schedule | Halve the rate after 2 epochs without validation improvement |

| Model | Epochs run | Best validation loss | Validation RMSE | Test RMSE | Test MAPE |
| --- | --: | --: | --: | --: | --: |
| GRU (headline) | 30 | 0.0373 | 54.47 | 55.73 | 9.70% |
| LSTM | 14 (early stop) | 0.0386 | 55.43 | 56.95 | 9.17% |

The GRU was chosen as the headline sequence model on validation RMSE. Its learning rate was halved three times (0.002 to 0.00025) as validation loss levelled off. The LSTM stopped early at epoch 14, after 4 epochs without beating its epoch-10 best.

Training loss stays slightly above validation loss throughout. This is expected: dropout is active during training and switched off during validation, so the training pass is deliberately harder. The loss curves are in `reports/figures/dl_loss_curves.png` and on the dashboard's Accuracy page.

The GRU's advantage is modest. The engineered lags already give the tree models most of the daily and weekly pattern that a sequence model would otherwise have to learn.

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) covers all nine required views and four supporting modules on five pages, each one or two clicks from any other. It only reads saved pipeline outputs and never retrains when a page opens.

| PRD view or module (14.1, 14.2) | Where it is |
| --- | --- |
| Live prediction | Now: corridor map, three headline numbers against a usual week, needs-attention list, selected road |
| Historical trends | Roads › Trends |
| Congestion heatmap | Today (every road, every half-hour) and Roads › Busy times (date range) |
| Road comparison | Roads › Compare roads |
| Model performance | Accuracy › Targets and Compare models |
| Feature importance | Accuracy › What the models use |
| Forecast visualisation | Accuracy › Forecast vs recorded; Roads › One road |
| Prediction confidence | Accuracy › Likely ranges; a likely range on every forecast |
| Weather vs traffic | Roads › Weather |
| Data upload | Data › Upload new data (cleans and forecasts without retraining) |
| Model training | Data › Data checks (background retraining run) |
| Prediction interface | Roads › One road (any segment, 30 to 120 minutes ahead) |
| Reports and insights | Data › Download a report (Markdown and CSV) |

The design follows PRD 14.3:

- Every number shown is a model output or an aggregate of recorded data.
- Every forecast shows its likely range.
- Grey is the default colour; amber, orange and red mean Moderate, Heavy and Severe on every page.
- "Usual" means the median of recorded values for the same segment, weekday and half-hour. It turns raw numbers into comparisons an operator can act on.

## Limitations and recommendations

FlowCast is ready to support decisions on volume and travel time. Congestion levels should be read with their one-level uncertainty, and accident risk as a relative ranking only, not a probability to act on alone.

Limitations:

- **Short history.** Five months of one corridor, tested on 26 days in May. Seasonal effects such as the monsoon are not in the data.
- **Few calendar examples.** Six holidays and six events give the models little to learn calendar effects from.
- **Weather is observed, not forecast.** Features use the weather at the time of the forecast. A live system 2 hours ahead would need weather forecasts.
- **Batch, not live.** "Now" forecasts start from the last window in the data file. Real-time use needs a feed that adds readings every 30 minutes.
- **Rare accidents.** About 1,650 accident windows in five months, with no incident context, cap what any model can learn.

Recommendations, in order of expected value:

1. **Congestion:** tune per-class decision thresholds on validation to favour Heavy and Severe. Alternatively, forecast V/C with its likely range and show the probability of each band.
2. **Accident risk:** add incident, CCTV or police-report feeds and road-geometry data, and collect more months. Until then, use the model to rank the riskiest 10% of windows for patrols.
3. **Live operation:** stream sensor readings, use weather forecasts as inputs, and schedule retraining with drift monitoring (PRD 19).
4. **Model scope:** include the sequence models in congestion model selection, and test graph-based models that share information between neighbouring segments (PRD 19).

## Reproducing the results

Every number in this report is rebuilt from the raw files by one command. All randomness is seeded (seed 42), and the full run took 18.5 minutes on a MacBook Air.

```bash
pip install -r requirements.txt
python -m src.pipeline --data-dir data/raw
streamlit run dashboard/app.py
```

| Output | Path |
| --- | --- |
| Scoreboard (every model, target, split and horizon) | `reports/scoreboard.csv` |
| Data-quality report and cleaning log | `reports/data_quality.md` |
| Quarantined rows | `data/interim/quarantine.csv` |
| Model cards (training window, features, hyperparameters, test metrics) | `models/*.md` |
| EDA, loss-curve and gradient-descent figures | `reports/figures/` |
| Step-by-step walkthroughs | `notebooks/01_eda`, `02_cleaning`, `03_modelling` |

The test suite (`python -m pytest`, 31 tests) covers ingestion, cleaning rules, metrics and the leakage perturbation test. Code layout follows PRD 16: `src/` holds the pipeline, `dashboard/` the app, and `config.yaml` every rule, split date and hyperparameter.
