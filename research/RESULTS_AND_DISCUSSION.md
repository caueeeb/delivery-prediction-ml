# Results and Discussion

## Temporal Distribution Shift and Its Impact on E-commerce Delivery Delay Prediction

### 1. Study overview

This study investigates how chronological changes in an e-commerce order dataset relate to the performance of a Logistic Regression model designed to predict late deliveries. The analysis reuses the feature-engineering and temporal-splitting pipeline from the companion Olist project.

The objective is diagnostic rather than causal: identify changes in the input and target distributions, measure model performance on a later time window, and assess whether the predicted probabilities remain useful. The experiments do not establish why delivery outcomes changed, nor do they demonstrate that a particular operational event caused the observed differences.

### 2. Dataset and temporal design

The processed analytical base contains **96,470 delivered orders** and **33 columns**, including identifiers, timestamps, the target, and engineered attributes. The model uses **26 input features**. The purchase timestamps range from **2016-09-15 12:16:38** to **2018-08-29 15:00:37**.

The original chronological split was reproduced:

| Partition | Orders | Period (purchase timestamp) | Late orders | Late rate |
|---|---:|---|---:|---:|
| Fit / training | 61,740 | 2016-09-15 to 2018-03-20 | 4,828 | 7.820% |
| Validation | 15,436 | 2018-03-20 to 2018-05-26 | 1,032 | 6.686% |
| Test | 19,294 | 2018-05-26 to 2018-08-29 | 674 | 3.493% |

The exact split boundaries occur within March and May 2018. Consequently, those calendar months appear in more than one partition and should not be compared as if each partition contained a complete calendar month.

The analytical cohort contains delivered orders with usable delivery outcomes. Results therefore describe this cohort and should not be generalized to canceled, unavailable, or still-open orders without further study.

### 3. Methods

The study performed the following analyses:

1. **Feature distribution comparisons:** numerical features were compared using the Kolmogorov-Smirnov (KS) statistic and a Population Stability Index (PSI) implementation using bins derived from the fit partition. The categorical `customer_state` feature was compared using Jensen-Shannon distance.
2. **Target distribution analysis:** late-delivery prevalence was calculated in each chronological partition and by calendar month.
3. **Monthly model evaluation:** the saved Logistic Regression pipeline was evaluated on the untouched test partition using the previously selected probability threshold of **0.15**. Metrics included accuracy, precision, recall, F1-score, Average Precision (AP), and AP lift relative to the monthly positive-class prevalence.
4. **Baseline comparison:** model accuracy was compared with a naive classifier that predicts every order as on-time.
5. **Probability diagnostics:** monthly mean predicted probability was compared with observed late-delivery prevalence. Brier scores and a quantile-based calibration curve were also calculated.

The test partition was not used to select the threshold or change the model.

### 4. Feature distribution findings

The largest interpretable numerical shifts from fit to test were observed in:

| Feature | PSI | KS statistic |
|---|---:|---:|
| `estimated_delivery_window_days` | 0.42668 | 0.25442 |
| `total_freight` | 0.30196 | 0.20079 |

For fit versus validation, `total_freight` had PSI **0.26742** and KS statistic **0.14765**, while `estimated_delivery_window_days` had PSI **0.11007** and KS statistic **0.15288**.

The Jensen-Shannon distance for `customer_state` was **0.05627** for fit versus validation and **0.06261** for fit versus test, indicating a comparatively smaller change in this categorical distribution.

**Important methodological caveat:** `purchase_month` is a discrete calendar variable with a chronological structure. The quantile-binned PSI calculation produced very large values (approximately 10.2) for this feature. That value is not interpreted as a conventional, directly comparable PSI result: quantile binning is unsuitable for this discrete month variable in this setup. The KS statistic and the monthly distribution plots still show that the time composition differs across chronological partitions, which is expected given the temporal split.

KS p-values were often very small. Because the partitions contain many observations, statistical significance should not be confused with practical effect size. The KS statistic and distribution plots are therefore reported alongside p-values rather than using p-values alone to rank drift severity.

### 5. Target distribution over time

The late-delivery rate decreased across the three chronological partitions, from **7.820%** in fit to **6.686%** in validation and **3.493%** in test. The complete monthly timeline also shows non-monotonic variation rather than a steady decline.

Among months with substantial order volume, notable monthly rates included:

| Month | Orders | Late orders | Late rate |
|---|---:|---:|---:|
| 2017-11 | 7,288 | 904 | 12.404% |
| 2018-02 | 6,555 | 926 | 14.127% |
| 2018-03 | 7,003 | 1,328 | 18.963% |
| 2018-04 | 6,798 | 306 | 4.501% |
| 2018-05 | 6,749 | 443 | 6.564% |
| 2018-06 | 6,096 | 71 | 1.165% |
| 2018-07 | 6,156 | 208 | 3.379% |
| 2018-08 | 6,351 | 393 | 6.188% |

Very early months with only one or a few orders are not used to infer trends. The change in target prevalence is evidence of a change in the marginal target distribution, but it does not on its own prove a change in the conditional relationship between features and the target.

### 6. Overall test performance

The final Logistic Regression model was evaluated on the 19,294-order test partition at threshold 0.15:

| Metric | Test result |
|---|---:|
| Accuracy | 0.84503 |
| Precision | 0.06727 |
| Recall | 0.26706 |
| F1-score | 0.10746 |
| Average Precision | 0.07164 |

The confusion matrix was:

| | Predicted on-time | Predicted late |
|---|---:|---:|
| Actually on-time | 16,124 | 2,496 |
| Actually late | 494 | 180 |

A naive all-on-time classifier would achieve **96.51% accuracy** on this test set because only 3.493% of test orders were late. The Logistic Regression classifier therefore had lower accuracy than this naive baseline at the selected threshold. However, the naive classifier has zero recall for late orders, so accuracy alone does not describe the operational trade-off.

The model's Average Precision was approximately **2.05 times** the test-set late-order prevalence (0.07164 / 0.03493). This indicates some ranking signal overall, but does not make the thresholded predictions operationally useful by itself.

### 7. Monthly performance and error analysis

The monthly results show a widening gap between actual late-delivery prevalence and the share of orders flagged by the model:

| Month | Actual late rate | Predicted late rate | Precision | Recall | F1-score | AP | AP lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018-05* | 0.289% | 0.289% | 0.0000 | 0.0000 | 0.0000 | 0.0288 | 9.9553 |
| 2018-06 | 1.165% | 3.970% | 0.0909 | 0.3099 | 0.1406 | 0.0794 | 6.8177 |
| 2018-07 | 3.379% | 17.008% | 0.0611 | 0.3077 | 0.1020 | 0.0691 | 2.0448 |
| 2018-08 | 6.188% | 21.808% | 0.0679 | 0.2392 | 0.1057 | 0.0807 | 1.3037 |

*The May test segment contains only two late orders, so its precision, recall, F1, and AP estimates are highly unstable and should not be used for strong month-to-month conclusions.*

In August, 1,385 orders were flagged as late, but only 94 of those were true positives; 1,291 were false positives. This corresponds to a precision of approximately **6.79%**. The model substantially over-flagged late deliveries in the later test months.

The AP lift decreased from **6.82** in June to **1.30** in August. Because AP lift is normalized by each month's late-order prevalence, it helps contextualize ranking performance under changing class balance. It is descriptive and should be interpreted with sample-size and prevalence differences in mind.

### 8. Probability calibration

The monthly mean predicted probabilities exceeded observed late-order rates in every test month:

| Month | Actual late rate | Mean predicted probability | Difference |
|---|---:|---:|---:|
| 2018-05 | 0.289% | 2.655% | +2.365 percentage points |
| 2018-06 | 1.165% | 5.362% | +4.197 percentage points |
| 2018-07 | 3.379% | 9.704% | +6.325 percentage points |
| 2018-08 | 6.188% | 11.376% | +5.188 percentage points |

The overall calibration curve placed the observed frequencies below the perfect-calibration diagonal, consistent with overestimation of risk in the plotted probability bins.

The overall Brier score was **0.03800** for the model. A constant predictor using the fit-set late-order prevalence (7.82%) had Brier score **0.03558**, producing a Brier Skill Score of **-0.068** relative to that baseline. A retrospective constant predictor using the test prevalence had Brier score **0.03371**; this uses information unavailable before test labels are observed and is shown only as a descriptive reference.

Brier score is a proper scoring rule that reflects calibration, resolution, and outcome uncertainty together. Therefore, the higher Brier score demonstrates worse overall probabilistic performance than the stated baseline, but it should not be interpreted as a pure calibration-only measure. The reliability diagram provides a separate visual diagnostic of calibration.

Monthly model and fit-prevalence baseline Brier scores were:

| Month | Model Brier | Fit-prevalence baseline Brier |
|---|---:|---:|
| 2018-05 | 0.00364 | 0.00856 |
| 2018-06 | 0.01354 | 0.01594 |
| 2018-07 | 0.03918 | 0.03462 |
| 2018-08 | 0.06408 | 0.05832 |

The model performed better than this baseline in May and June, but worse in July and August.

### 9. Discussion

The experiments identify several concurrent changes between the fit and later evaluation periods:

- The late-delivery prevalence decreased across the broad chronological partitions, although the monthly timeline was non-monotonic.
- The input distributions of `estimated_delivery_window_days` and `total_freight` changed materially according to PSI and KS effect sizes.
- At the fixed threshold of 0.15, the model generated a growing proportion of late-delivery alerts relative to the observed late rate in June, July, and August.
- The model's precision was low, and its accuracy was below the all-on-time baseline on the overall test set.
- The model's overall Brier score was worse than the fit-prevalence baseline, while the calibration curve showed systematic overestimation in the displayed bins.

These observations are consistent with temporal dataset shift and degraded out-of-time predictive utility. They do **not** establish that feature drift caused the performance degradation. In particular, a change in the marginal distribution of features, P(X), or of the target, P(Y), does not by itself demonstrate that the conditional relationship P(Y|X) changed. Establishing real concept drift would require further conditional analysis, controlled experiments, or additional labeled temporal windows. The distinction between changing input distributions and changing feature-target relationships is discussed in the concept-drift literature (Gama et al., 2014).

### 10. Limitations

1. **Historical and domain-specific data:** the dataset covers one Brazilian e-commerce platform and a historical period from 2016 to 2018.
2. **Cohort selection:** the analytical base includes delivered orders with observed delivery outcomes. Canceled, unavailable, and still-open orders are outside this analysis.
3. **Temporal boundary months:** March and May 2018 are split across partitions. Month-level comparisons need to respect the exact timestamp boundaries.
4. **Small positive counts in May test:** only two late orders occur in the May test segment, making its classification and ranking metrics unstable.
5. **Drift metric caveat:** PSI for the discrete `purchase_month` feature is not interpreted because the quantile-binning setup produces an extreme, unsuitable result for this discrete calendar variable.
6. **Large-sample significance:** KS p-values can be very small for modest distribution differences; effect sizes and visual comparisons are also needed.
7. **No causal identification:** the observed changes cannot be attributed to a specific logistics, marketplace, seasonal, or operational cause using this analysis alone.
8. **No adaptation experiment:** no retraining, recalibration, threshold re-selection, or drift-response strategy was evaluated on a new future holdout.
9. **Operational readiness:** this research is retrospective and does not establish that the model is safe or useful for live delivery decisions.

### 11. Conclusion

This case study found substantial temporal differences in the Olist delivery dataset and lower predictive utility for a Logistic Regression model evaluated on a later time window. The late-order prevalence fell from 7.82% in the fit partition to 3.49% in the test partition, while the distributions of delivery-window estimates and freight costs also changed.

At the previously selected threshold of 0.15, the classifier achieved 84.50% accuracy, 6.73% precision, 26.71% recall, 0.1075 F1-score, and 0.0716 Average Precision on the test set. Its accuracy was below an all-on-time baseline, and the model's probability estimates were worse than a constant fit-prevalence baseline according to Brier score. Monthly results also showed substantial over-flagging of late orders in July and August 2018.

The findings support the practical importance of chronological validation, monitoring changes in both input and target distributions, and evaluating ranking, thresholded classification, and probability quality separately. They do not prove a causal mechanism or confirm real concept drift in P(Y|X). Future work should investigate rolling-origin validation, more stable month-level estimates, conditional drift analysis, probability recalibration using validation-only data, threshold selection based on explicit operational costs, and adaptation strategies assessed on a strictly future holdout.

### 12. Supplementary diagnostic: random versus chronological evaluation (H3)

Hypothesis H3 of the research protocol asks whether a random stratified split and a chronological split produce different estimates of generalization performance. This diagnostic was run after the main analysis, with `random_vs_chronological.py`, using the same preprocessing, model definition, validation-only threshold selection (maximum F1 on a grid from 0.05 to 0.95) and final refit on training plus validation as the main experiment. Random partitions used the same proportions (64% / 16% / 20%), stratified by the target, over ten random seeds. Random splitting mixes observations from different dates, so it does not represent a future-prediction scenario; it is a diagnostic only.

| Metric (test partition) | Chronological | Random stratified (mean ± std, 10 seeds) |
|---|---:|---:|
| Late-order prevalence | 3.49% | 6.77% |
| Average Precision | 0.0716 | 0.1517 ± 0.0077 |
| AP lift over prevalence | 2.05× | 2.24× ± 0.11 |
| ROC-AUC | 0.699 | 0.696 ± 0.007 |
| Precision (validation-selected threshold) | 6.73% | 14.71% ± 0.32 |
| Recall | 26.71% | 40.46% ± 0.81 |
| F1-score | 0.1075 | 0.2157 ± 0.0040 |
| Threshold selected on validation | 0.15 | 0.10 |

**Reading the result.** The random split gives much higher absolute Average Precision, precision, recall and F1-score than the chronological split, so random evaluation would have reported a considerably more optimistic picture. However, the two rank-based measures that are normalized for class balance are very similar: ROC-AUC is about 0.70 under both protocols, and Average Precision is roughly two times the prevalence under both. In this experiment, most of the drop in absolute Average Precision and F1-score between the validation period and the test period is therefore consistent with the lower late-order prevalence (a change in P(Y)) rather than with a clear loss of ranking ability. The threshold chosen on validation data (0.15) was also tuned for a period with a higher late-delivery rate, which helps explain the large number of false alerts reported in sections 7 and 9.

This does not contradict the earlier findings, and it does not show that P(Y|X) stayed constant: ROC-AUC and AP lift are summary measures over the whole test period, and the monthly analysis shows AP lift weakening in later months. It does suggest that, for this dataset, reporting threshold-dependent and prevalence-dependent metrics alone can overstate how much the model's discriminative signal degraded.

### 13. References

- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). [A survey on concept drift adaptation](https://doi.org/10.1145/2523813). *ACM Computing Surveys, 46*(4), Article 44.
- scikit-learn developers. [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). scikit-learn documentation.
- scikit-learn developers. [Brier score loss](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss). scikit-learn documentation.

### Reproducibility

The analysis notebook and generated CSV reports are stored in the `research/` directory of the companion project repository. The original raw CSV dataset is not included in the repository; readers must obtain it from the original dataset source and comply with its access and usage terms.
