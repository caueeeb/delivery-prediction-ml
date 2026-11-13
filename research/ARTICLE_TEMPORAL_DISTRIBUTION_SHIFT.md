# When Machine Learning Models Meet Temporal Distribution Shift: A Brazilian E-commerce Case Study

*An empirical study of delivery-delay prediction using the Olist e-commerce dataset*

## Abstract

A machine learning model can perform reasonably well during validation and still become unreliable when applied to later observations. This article investigates that problem through a temporal evaluation of a delivery-delay classifier built with the Brazilian Olist e-commerce dataset. We reconstructed 96,470 delivered orders, engineered 26 predictors, and evaluated a Logistic Regression pipeline using chronological training, validation, and test partitions. The late-delivery prevalence decreased from 7.82% in the training period to 3.49% in the test period. On the test set, the model achieved 84.50% accuracy, 6.73% precision, 26.71% recall, 10.75% F1-score, and 0.0716 Average Precision. A simple all-on-time classifier reached 96.51% accuracy, while the model's Average Precision remained above the test-set positive-class prevalence. Monthly analysis also revealed substantial overprediction of late deliveries and weaker relative ranking performance in the later test months. These findings demonstrate why chronological evaluation, class-aware metrics, probability diagnostics, and explicit limitations matter when assessing models for operational use.

**Keywords:** temporal distribution shift, data drift, delivery prediction, imbalanced classification, probability calibration, e-commerce, machine learning

---

## 1. Introduction

Machine learning evaluations often assume that future observations will resemble the data used during model development. In operational settings, that assumption can fail: customer behavior, product mix, logistics capacity, policies, or seasonal patterns may change over time. Research on temporal distribution shift studies how model assessment should account for changing environments rather than relying only on historical random splits (Han, Huang, & Wang, 2024).

This case study asks:

> How does a chronological change in e-commerce order data relate to the measured performance of a delivery-delay classifier?

The objective is not to claim that one specific cause explains the observed performance changes. Instead, we examine feature distributions, target prevalence, monthly model performance, classification errors, and probability estimates. We also distinguish observable distribution changes from *concept drift* in the strict sense. A change in the feature distribution or target prevalence alone does not establish that the conditional relationship between features and the target has changed (Lukats et al., 2025).

## 2. Dataset and prediction task

The study uses the Brazilian E-Commerce Public Dataset by Olist, obtained through the [Olist/Kaggle dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The analysis base contains **96,470 delivered orders** with usable purchase and delivery timestamps, spanning **September 15, 2016 to August 29, 2018**.

The target, `is_late`, is binary:

- `1`: the order's delivery date was later than its estimated delivery date;
- `0`: the order was delivered on or before the estimated date.

The resulting task is an imbalanced binary classification problem. Across the full analysis base, 6.77% of orders were late.

An important scope limitation is that the analysis base contains delivered orders with observed delivery outcomes. It does not establish that the same model is suitable for live prediction over every newly placed order, including orders that may later be cancelled, unavailable, or otherwise lack an observed delivery outcome.

## 3. Features and model development

The feature-engineering process produced **26 predictors**, including customer state, item and seller counts, item price and freight totals, payment summaries, seller-location relationships, purchase-time features, the estimated delivery window, and a payment-record indicator.

The original identifiers and outcome-derived fields were excluded from the model inputs. Engineered features included:

- item and seller aggregates at order level;
- `same_state_seller_share`, representing the share of sellers in the customer's state;
- purchase hour, day of week, month, and weekend indicator;
- `estimated_delivery_window_days`, calculated from purchase time and the estimated delivery date;
- `has_payment_record`, an indicator for the presence of a payment record.

Several candidate algorithms were compared during development. Logistic Regression was selected after validation experiments. The final pipeline included preprocessing for numerical and categorical variables and used a classification threshold of **0.15**, selected using validation data. The final model was refitted on the combined development data before evaluation on the untouched chronological test partition.

## 4. Chronological evaluation design

Rather than randomly mixing dates across partitions, the data was split in chronological order:

| Partition | Orders | Date range | Late-delivery rate |
|---|---:|---|---:|
| Training (`fit`) | 61,740 | 2016-09-15 to 2018-03-20 | 7.82% |
| Validation | 15,436 | 2018-03-20 to 2018-05-26 | 6.69% |
| Test | 19,294 | 2018-05-26 to 2018-08-29 | 3.49% |

The positive-class prevalence declined by more than half between the training and test partitions. This is an observed change in the marginal distribution of the target, often called *label shift* or *prior-probability shift*. The partitions also differ in their calendar composition, as expected from a chronological split.

Chronological evaluation provides a more realistic estimate of performance on later observations than a random split when time is relevant. It does not, by itself, identify the cause of any performance change.

## 5. Results

### 5.1 Overall test performance

The final model was evaluated on 19,294 test orders using the validation-selected threshold of 0.15.

| Metric | Test result |
|---|---:|
| Accuracy | 84.50% |
| Precision | 6.73% |
| Recall | 26.71% |
| F1-score | 10.75% |
| Average Precision | 0.0716 |

The test-set late-delivery prevalence was 3.49%. A naive classifier that predicts every order as on-time would therefore achieve **96.51% accuracy**. The Logistic Regression model did not beat this simple accuracy baseline at the chosen threshold.

Accuracy is not sufficient for this imbalanced task, however. The model's Average Precision of 0.0716 was approximately 2.05 times the test positive-class prevalence. This indicates some ranking signal, although its practical value was limited and declined in relative terms across the later test months.

### 5.2 Feature-distribution diagnostics

The numerical feature comparison highlighted two notable differences between training and test:

| Feature | PSI | KS statistic |
|---|---:|---:|
| `estimated_delivery_window_days` | 0.4267 | 0.2544 |
| `total_freight` | 0.3020 | 0.2008 |

The categorical distribution of `customer_state`, measured using Jensen-Shannon distance, changed less substantially (0.0626 for training versus test).

These statistics are diagnostic summaries, not proof of a causal mechanism. The PSI thresholds sometimes used in practice are heuristics, and statistical tests such as Kolmogorov-Smirnov can flag small differences when sample sizes are large. A separate analysis of `purchase_month` showed that calendar composition differed strongly across chronological partitions; month-of-year alone must be interpreted carefully because it does not encode the year.

### 5.3 Monthly target behavior

When all partitions are recombined and months are represented as year-month values, the 2018 late-delivery rates include:

| Month | Orders | Late orders | Late rate |
|---|---:|---:|---:|
| January 2018 | 7,069 | 403 | 5.70% |
| February 2018 | 6,555 | 926 | 14.13% |
| March 2018 | 7,003 | 1,328 | 18.96% |
| April 2018 | 6,798 | 306 | 4.50% |
| May 2018 | 6,749 | 443 | 6.56% |
| June 2018 | 6,096 | 71 | 1.16% |
| July 2018 | 6,156 | 208 | 3.38% |
| August 2018 | 6,351 | 393 | 6.19% |

The target rate did not decline monotonically. It peaked in March 2018, fell in April and May, reached a low in June, and rose again through August. This pattern cautions against describing the observations as a simple, steady downward trend.

The chronological partition boundaries fall within March and May 2018. Consequently, monthly metrics calculated within each partition can represent only part of a calendar month. For example, the May test slice contains 691 orders and only two late deliveries; its month-specific metrics are highly unstable and should not be compared as if they represented a full month.

### 5.4 Monthly model behavior and classification errors

At the fixed threshold of 0.15, the model substantially overpredicted the share of late deliveries in the test period:

| Test month | Actual late rate | Predicted late rate | False positives | False negatives |
|---|---:|---:|---:|---:|
| May 2018 (partial) | 0.29% | 0.29% | 2 | 2 |
| June 2018 | 1.17% | 3.97% | 220 | 49 |
| July 2018 | 3.38% | 17.01% | 983 | 144 |
| August 2018 | 6.19% | 21.81% | 1,291 | 299 |

In August, only 94 of 1,385 orders predicted as late were actually late, corresponding to approximately 6.79% precision. The model generated many false alerts. Monthly Average Precision remained above each month's positive-class prevalence, but its lift over that prevalence declined from 6.82 in June to 1.30 in August. These monthly estimates, especially the May result with only two positive cases, should be interpreted cautiously.

### 5.5 Probability estimates and Brier score

For each test month, the average predicted probability exceeded the observed late-delivery rate:

| Month | Actual late rate | Mean predicted probability |
|---|---:|---:|
| May 2018 (partial) | 0.29% | 2.66% |
| June 2018 | 1.17% | 5.36% |
| July 2018 | 3.38% | 9.70% |
| August 2018 | 6.19% | 11.38% |

The reliability diagram also placed the observed event frequencies below the perfect-calibration line across the displayed bins, consistent with probability overestimation in this test sample.

The overall Brier score was **0.03800**. A constant-probability baseline using the training prevalence (7.82%) achieved **0.03558**, yielding a Brier Skill Score of **-0.068** relative to that baseline. A retrospective constant baseline using the test prevalence achieved 0.03371, but this test prevalence would not have been known in advance in a real deployment.

The Brier score is a proper scoring rule that reflects calibration, resolution/discrimination, and outcome uncertainty together; it should not be interpreted as a pure calibration measure (scikit-learn documentation, *Probability calibration*).

### 5.6 Supplementary diagnostic: random versus chronological evaluation

To check how much the evaluation protocol itself matters, the same pipeline was also evaluated with random stratified partitions (64% / 16% / 20%) over ten random seeds. Random evaluation reported Average Precision of 0.152 ± 0.008 and an F1-score of 0.216 ± 0.004, roughly double the chronological values (0.072 and 0.108). Rank-based measures normalized for class balance, however, were nearly the same: ROC-AUC was 0.70 under both protocols, and Average Precision was about 2.0–2.2 times the prevalence.

In this dataset, a large part of the drop in absolute Average Precision and F1-score is therefore consistent with the lower late-delivery prevalence in the test period (3.49% versus 6.77% overall), rather than with a clear loss of ranking ability. Random evaluation would still have been misleading, because it overstated the precision and recall a team could expect for a fixed threshold in a later period. The full table, method and caveats are in `RESULTS_AND_DISCUSSION.md` (section 12) and `random_vs_chronological.py`.

## 6. Discussion

The experiments reveal several connected issues:

1. **The target prevalence changed over time.** The late-delivery rate fell from 7.82% in training to 3.49% in test, with substantial month-to-month variation.
2. **Some input distributions also changed.** Freight and estimated delivery-window features showed notable differences between training and test.
3. **The fixed decision threshold produced too many alerts.** Precision was low, and the majority-class baseline achieved higher accuracy in every evaluated test month.
4. **Probability estimates were not reliable enough for operational interpretation.** The mean predicted risk was higher than the observed event frequency, and the model did not outperform a training-prevalence Brier baseline.
5. **Some ranking information remained.** Average Precision exceeded the positive-class prevalence overall and in the later test months, although the relative lift weakened. The supplementary diagnostic in section 5.6 suggests that much of the drop in absolute Average Precision and F1-score is associated with the lower prevalence, because ROC-AUC (about 0.70) was similar under random and chronological evaluation.

These results are consistent with temporal distribution shift affecting model evaluation and usefulness. They do **not** establish strict concept drift, which would require evidence that the conditional relationship between features and the target, P(Y|X), changed over time. Establishing that would require further analysis, for example rolling temporal evaluation, subgroup/conditional error analysis, and sufficiently supported comparisons of P(Y|X) across periods.

## 7. Limitations

- **Historical and selected population:** the analysis includes delivered orders with observed outcomes, not every order entering a live e-commerce workflow.
- **Single chronological split:** the results come from one training/validation/test design. Rolling-origin evaluation would provide stronger evidence about stability across multiple cutoffs.
- **No causal identification:** the analyses identify distributional and performance differences but do not explain why those differences occurred.
- **No uncertainty intervals:** monthly metrics, especially those based on few late deliveries, can be noisy. Confidence intervals or bootstrap estimates would improve interpretation.
- **Threshold dependence:** precision, recall, F1, and accuracy depend on the 0.15 threshold. It was selected on validation data and intentionally kept fixed for the test evaluation.
- **Drift diagnostics are not definitive tests of concept drift:** marginal feature and target changes do not prove that the conditional relationship P(Y|X) changed.
- **Brier score interpretation:** the Brier score combines multiple aspects of probabilistic performance and is not a calibration-only measure.

## 8. Next steps

A stronger follow-up study could:

1. Run rolling-origin evaluation across multiple historical cutoffs.
2. Report confidence intervals for Average Precision, recall, precision, and Brier score.
3. Investigate feature-conditional error rates and whether P(Y|X) changes across time windows.
4. Compare fixed-threshold decisions with cost-sensitive thresholds selected using validation data only.
5. Evaluate probability calibration on a separate, temporally appropriate calibration set before considering recalibration.
6. Monitor both feature distributions and delayed ground-truth performance in a production-like simulation.

No test-period labels should be used to tune the final threshold or calibrator if the test set is intended to remain an unbiased final evaluation.

## 9. Conclusion

This Olist case study shows why a single validation score is not enough to assess a model intended for future data. A Logistic Regression classifier that appeared useful during development showed weak precision, lower-than-baseline accuracy, excessive late-delivery alerts, and poor probabilistic performance on a later chronological test set. At the same time, Average Precision remained above the test positive-class prevalence, indicating that the model retained limited ranking signal.

The main lesson is practical: **evaluate models over time, compare them with simple baselines, inspect their errors, and assess their probabilities—not just their class labels.** The observed results justify further investigation and monitoring, but they do not prove a causal explanation or strict concept drift.

## References

- Han, E., Huang, C., & Wang, K. (2024). *Model Assessment and Selection under Temporal Distribution Shift*. Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, 235, 17374–17392. https://proceedings.mlr.press/v235/han24b.html
- Lukats, D., Zielinski, O., Hahn, A., et al. (2025). *A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams*. International Journal of Data Science and Analytics, 19, 1–31. https://doi.org/10.1007/s41060-024-00620-y
- scikit-learn developers. *Probability calibration*. https://scikit-learn.org/stable/modules/calibration.html
- Olist. *Brazilian E-Commerce Public Dataset*. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Project links

- Code and experiments: https://github.com/caueeeb/delivery-prediction-ml
- Related Titanic classification project: https://github.com/caueeeb/kaggle-titanic
