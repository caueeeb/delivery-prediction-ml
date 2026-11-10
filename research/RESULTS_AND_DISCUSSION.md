# Results and Discussion

## Temporal Distribution Shift and Its Impact on E-commerce Delivery Delay Prediction

### 1. Study scope

This study investigates temporal distribution changes and their relationship with the performance of a Logistic Regression model for late-delivery prediction using the Brazilian E-Commerce Public Dataset by Olist.

The analytical base contains **96,470 delivered orders**, represented by 33 columns, including identifiers, timestamps, the target, and engineered variables. The model uses 26 predictors. The target `is_late` indicates whether the actual delivery date was later than the estimated delivery date.

The data were partitioned chronologically:

| Partition | Orders | Date range | Late-delivery rate |
|---|---:|---|---:|
| Fit / training | 61,740 | 2016-09-15 to 2018-03-20 | 7.82% |
| Validation | 15,436 | 2018-03-20 to 2018-05-26 | 6.69% |
| Test | 19,294 | 2018-05-26 to 2018-08-29 | 3.49% |

The model was trained on the fit partition, the decision threshold of **0.15** was selected using validation data, and the final model was evaluated on the held-out test partition. The test set was not used to select the threshold.

### 2. Feature-distribution changes

The numerical feature analysis identified the largest PSI values between fit and test for:

| Feature | PSI (fit vs. test) | KS statistic |
|---|---:|---:|
| `purchase_month` | 10.19742 | 0.43137 |
| `estimated_delivery_window_days` | 0.42668 | 0.25442 |
| `total_freight` | 0.30196 | 0.20079 |

For fit versus validation, `purchase_month` had PSI 10.29670, `total_freight` 0.26742, and `estimated_delivery_window_days` 0.11007. The categorical feature `customer_state` had Jensen-Shannon distances of 0.05627 (fit vs. validation) and 0.06261 (fit vs. test).

**Interpretation and methodological caution.** The very high PSI values for `purchase_month` should not be interpreted using the same heuristic thresholds as continuous variables. `purchase_month` is a discrete, cyclical calendar variable and the chronological partitions cover different portions of the timeline. A separate categorical comparison produced Jensen-Shannon distances of 0.8034 (fit vs. validation) and 0.7712 (fit vs. test), confirming that the month composition differs substantially across partitions. The calendar-month variable also omits the year, so it cannot represent the complete temporal sequence by itself.

The PSI values for `estimated_delivery_window_days` and `total_freight` provide additional evidence of changes in the input-feature distributions. KS p-values should be interpreted cautiously because the large sample sizes make small distributional differences statistically detectable. Effect sizes and visual diagnostics are more informative than p-values alone.

### 3. Target distribution and chronological pattern

The overall late-delivery rate declined from **7.82% in fit** to **6.69% in validation** and **3.49% in test**. This is evidence of a change in the target prevalence across the chronological partitions.

Combining the partitions by calendar month (so a month split across partitions is reunited) shows that the pattern was not a monotonic decline:

| Month | Orders | Late orders | Late-delivery rate |
|---|---:|---:|---:|
| 2017-11 | 7,288 | 904 | 12.40% |
| 2017-12 | 5,513 | 411 | 7.46% |
| 2018-01 | 7,069 | 403 | 5.70% |
| 2018-02 | 6,555 | 926 | 14.13% |
| 2018-03 | 7,003 | 1,328 | 18.96% |
| 2018-04 | 6,798 | 306 | 4.50% |
| 2018-05 | 6,749 | 443 | 6.56% |
| 2018-06 | 6,096 | 71 | 1.16% |
| 2018-07 | 6,156 | 208 | 3.38% |
| 2018-08 | 6,351 | 393 | 6.19% |

The rate peaked in March 2018 among months with substantial order volumes, dropped in April, and then fluctuated in the test period. September 2016 and December 2016 each contain only one order and are not informative for trend interpretation.

The chronological partitions cut through March 2018 and May 2018. Consequently, partition-specific rates for those months refer to partial-month intervals and should not be compared as if they represented complete calendar months. The consolidated monthly table avoids this particular issue by combining all partitions.

### 4. Monthly test performance

The model was evaluated on the test partition using the validation-selected threshold of 0.15.

| Test month | Orders | Actual late rate | Precision | Recall | F1-score | Average Precision | AP lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018-05 | 691 | 0.29% | 0.00% | 0.00% | 0.00% | 0.0288 | 9.96 |
| 2018-06 | 6,096 | 1.17% | 9.09% | 30.99% | 14.06% | 0.0794 | 6.82 |
| 2018-07 | 6,156 | 3.38% | 6.11% | 30.77% | 10.20% | 0.0691 | 2.04 |
| 2018-08 | 6,351 | 6.19% | 6.79% | 23.92% | 10.57% | 0.0807 | 1.30 |

May contains only two positive cases, so its precision, recall, F1-score, and AP estimate are highly unstable. It should not be used to draw strong conclusions about monthly model behavior.

From June to August, AP lift declined from 6.82 to 1.30. AP lift is calculated as Average Precision divided by the positive-class prevalence for that month. It is a descriptive normalization, not a guarantee of statistical significance. The decline suggests that the model's ranking advantage over a prevalence-level reference weakened over these test months.

### 5. Classification errors and majority-class baseline

The fixed-threshold predictions generated substantially more late-delivery alerts than the observed number of late orders:

| Month | Actual late rate | Predicted late rate | True positives | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| 2018-05 | 0.29% | 0.29% | 0 | 2 | 2 |
| 2018-06 | 1.17% | 3.97% | 22 | 220 | 49 |
| 2018-07 | 3.38% | 17.01% | 64 | 983 | 144 |
| 2018-08 | 6.19% | 21.81% | 94 | 1,291 | 299 |

In August, only 94 of the 1,385 predicted late orders were true positives, giving precision of 6.79%. Thus, most alerts were false positives.

The classifier's accuracy was also below a simple majority-class baseline that predicts every test order as on-time:

| Month | Model accuracy | All-on-time baseline accuracy | Difference (percentage points) |
|---|---:|---:|---:|
| 2018-05 | 99.42% | 99.71% | -0.29 |
| 2018-06 | 95.59% | 98.84% | -3.25 |
| 2018-07 | 81.69% | 96.62% | -14.93 |
| 2018-08 | 74.96% | 93.81% | -18.85 |

This baseline is useful for contextualizing accuracy under class imbalance, but it is not a useful alerting system because it never identifies a late order. The model's ranking metrics and business costs must therefore also be considered. Nevertheless, the comparison shows that the chosen threshold did not yield competitive classification accuracy on the test period.

### 6. Probability calibration

The mean predicted late-delivery probability exceeded the observed late-delivery rate in every test month:

| Month | Actual late rate | Mean predicted probability | Difference |
|---|---:|---:|---:|
| 2018-05 | 0.29% | 2.66% | +2.37 pp |
| 2018-06 | 1.17% | 5.36% | +4.20 pp |
| 2018-07 | 3.38% | 9.70% | +6.33 pp |
| 2018-08 | 6.19% | 11.38% | +5.19 pp |

The reliability curve placed the observed event frequencies below the perfect-calibration diagonal, indicating systematic overprediction over the displayed probability bins.

The monthly Brier scores were:

| Month | Model Brier | Training-prevalence baseline Brier |
|---|---:|---:|
| 2018-05 | 0.00364 | 0.00856 |
| 2018-06 | 0.01354 | 0.01594 |
| 2018-07 | 0.03918 | 0.03462 |
| 2018-08 | 0.06408 | 0.05832 |

The model beat the training-prevalence baseline in May and June, but not in July or August. May has only two positive cases, so its score is especially unstable.

The overall Brier score was **0.03800**. A constant-probability baseline using the fit-partition prevalence achieved **0.03558**, yielding a Brier Skill Score of **-0.068**. A hindsight reference using the test prevalence achieved a Brier score of **0.03371**; this test-prevalence reference is descriptive only and cannot be used as a deployable baseline because it uses the evaluation outcomes to set its probability.

The negative Brier Skill Score means that the model's probability forecasts were worse than the constant training-prevalence reference on the test set. This finding complements the low precision and high false-positive counts at threshold 0.15.

### 7. Main findings

The experiments provide evidence of temporal changes in both input-feature distributions and target prevalence. The largest input-distribution changes were observed for the month composition, estimated delivery window, and freight amount. The late-delivery prevalence fell across the aggregate fit, validation, and test partitions, while the complete monthly timeline showed substantial fluctuations rather than a smooth downward trend.

On the test set, the model retained some ranking signal, as Average Precision exceeded the positive-class prevalence, but its relative ranking lift weakened from June to August. At the fixed threshold selected on validation, it generated many false-positive alerts and achieved lower accuracy than the all-on-time baseline. Its probability estimates were also poorly calibrated and had a negative Brier Skill Score relative to the training-prevalence baseline.

Taken together, these results show that the original model did not generalize reliably to the later test period under the chosen threshold and probability interpretation. They are consistent with the presence of temporal distribution shift and a deterioration in predictive utility. They do **not**, by themselves, establish that covariate shift caused the performance decline or prove concept drift in the conditional relationship between features and late delivery.

### 8. Limitations

1. **Delivered-order selection:** The analytical dataset contains orders with observed delivery outcomes. It does not directly establish performance on in-flight orders whose final delivery outcome is not yet known. This limits claims about real-time operational deployment.
2. **Observational analysis:** The study identifies temporal associations, not causal explanations. Operational changes, carrier mix, geography, seasonality, data collection, and other factors may contribute.
3. **Single historical dataset:** The analysis uses one e-commerce dataset and one chronological split. Results may not generalize to other platforms or periods.
4. **Small monthly positive counts:** In particular, May 2018 has only two late orders in the test segment, making its metrics unstable.
5. **Threshold transfer:** The threshold was selected on validation and held fixed on test. This is appropriate for an untouched test evaluation, but a single threshold may not represent all operational cost trade-offs.
6. **Calibration diagnostics:** Monthly mean probabilities and a quantile-binned reliability curve are useful diagnostics, but further analysis with confidence intervals, calibration intercept/slope, and larger independent periods would strengthen the assessment.
7. **Multiple comparisons:** Many features were examined. Statistical significance should be interpreted with caution, and future work should consider multiple-testing control or emphasize effect sizes.

### 9. Conclusion

The study found clear temporal changes in the Olist data and a marked gap between validation and test performance for the Logistic Regression delivery-delay model. In the later test period, the model overestimated late-delivery probabilities, produced many false-positive alerts at the fixed 0.15 threshold, and underperformed a simple all-on-time strategy in accuracy. Although Average Precision indicates some residual ranking information, the AP lift declined over the test months and the Brier Skill Score was negative against the training-prevalence reference.

The findings support the practical conclusion that model evaluation should preserve chronology and monitor both predictive ranking and probability calibration under changing data distributions. Before operational use, the model would require further investigation, recalibration or retraining using an appropriate time-based development strategy, and validation on a newer, representative cohort. No corrective model was trained on the test data in this study.

### 10. Reproducibility artifacts

The analysis generated the following files under `research/reports/`:

- `feature_drift_numeric.csv`
- `feature_drift_categorical.csv`
- `target_drift_year_month.csv`
- `complete_monthly_timeline.csv`
- `monthly_model_performance_test.csv`
- `monthly_baseline_comparison.csv`
- `monthly_confusion_analysis.csv`
- `monthly_calibration.csv`
- `monthly_brier_scores.csv`

The analysis notebook is `research/01_temporal_distribution_shift.ipynb`.
