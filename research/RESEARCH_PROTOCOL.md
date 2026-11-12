# Temporal Distribution Shift and Its Impact on E-commerce Delivery Delay Prediction

**Study type:** Reproducible, observational machine-learning study  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist  
**Task:** Binary classification of delivery delay among delivered orders  
**Primary model:** Logistic Regression  
**Status:** Research protocol and preliminary motivation; the planned analyses are not yet complete.

## 1. Abstract

Machine-learning models are often evaluated under the assumption that future observations will resemble the data used during training. In operational settings, however, customer behavior, product mix, seller composition, logistics conditions, and other factors may change over time. This study investigates whether temporal distribution changes in the Olist e-commerce dataset are associated with reduced generalization performance in a model that predicts whether an order is delivered after its estimated delivery date.

The existing project uses chronological training, validation, and test partitions. Preliminary results show a decrease in the prevalence of late orders from 7.82% in training to 3.49% in the test period, accompanied by lower Average Precision and F1-score on the test set. The proposed study will quantify changes in target prevalence and feature distributions, compare random and chronological evaluation, and examine performance across time. These observations will be treated as evidence of temporal distribution shift, not as proof of a specific causal mechanism or of concept drift by themselves.

## 2. Research question

**How does temporal change in the Olist data affect the generalization performance of a machine-learning model that predicts delivery delays?**

### Supporting questions

1. Does the prevalence of late deliveries change across chronological partitions?
2. Which input features show the largest distribution differences between the training and later periods?
3. Does a random stratified evaluation produce a different performance estimate from a chronological evaluation?
4. How do ranking and classification metrics vary across later time windows?

## 3. Motivation and preliminary evidence

The current Olist project defines the target for delivered orders by comparing the calendar date of actual delivery with the estimated delivery date. The modeling base contains 96,470 delivered orders with valid delivery-date information.

The existing chronological partitions are:

| Partition | Rows | Date range | Late-order prevalence |
|---|---:|---|---:|
| Training | 61,740 | 2016-09-15 to 2018-03-20 | 7.82% |
| Validation | 15,436 | 2018-03-20 to 2018-05-26 | 6.69% |
| Test | 19,294 | 2018-05-26 to 2018-08-29 | 3.49% |

Previously reported model results:

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.1751 | 0.0716 |
| F1-score | 0.2630 | 0.1075 |

These figures raise a research question: to what extent is the performance decline associated with temporal changes in the target and input distributions? Additional analyses are required before drawing conclusions.

## 4. Hypotheses

- **H1 — Target prevalence shift:** the proportion of late orders differs across chronological partitions.
- **H2 — Feature distribution shift:** at least some input features have materially different distributions in the training and test periods.
- **H3 — Evaluation sensitivity:** a random stratified split and a chronological split produce different estimates of generalization performance.
- **H4 — Temporal performance variation:** model performance is not constant across later time windows.

The study may find evidence supporting some hypotheses and not others. Results must be reported as observed, including null or inconclusive findings.

## 5. Conceptual distinctions

This study will use the following terms carefully:

- **Temporal distribution shift:** a broad description of differences in data distributions across time.
- **Covariate shift / feature drift:** a change in the distribution of input features, commonly written as a change in `P(X)`.
- **Label/prior shift:** a change in the target prevalence or marginal distribution `P(Y)`.
- **Concept drift (strict sense):** a change in the relationship between inputs and target, `P(Y | X)`.

A decrease in late-order prevalence is evidence of a change in `P(Y)`, but it does not establish that `P(Y | X)` changed. Feature-distribution comparisons can reveal covariate shift, but even those do not prove a change in the conditional relationship. The analysis must avoid using “concept drift” as a synonym for every observed difference over time.

## 6. Methodology

### 6.1 Data and target

Use the existing Olist raw tables and the project's established preprocessing pipeline. Do not create a second, subtly different target definition. Keep the analysis restricted to delivered orders for which actual and estimated delivery dates are available, consistent with the original project.

Raw CSV files must remain local and must not be committed to GitHub.

### 6.2 Chronological evaluation

Reuse the project's existing chronological training, validation, and test partitions. Preserve the established date boundaries and the original test set as a final holdout. Do not tune features, thresholds, or model parameters based on test outcomes.

### 6.3 Random evaluation comparison

As a diagnostic experiment only, create stratified random partitions with the same approximate proportions as the chronological partitions (64% training, 16% validation, 20% test) and a fixed random seed. Compare their estimates with chronological evaluation. This keeps partition sizes more comparable while deliberately mixing dates in the random experiment. Clearly state that random splitting mixes observations from different times and therefore does not represent a strict future-prediction scenario.

### 6.4 Target distribution analysis

For each partition and time window, calculate:

- total number of orders;
- number and proportion of late orders;
- confidence interval for late-order prevalence;
- absolute and relative prevalence differences.

Because the positive class is relatively uncommon, do not rely on accuracy alone.

### 6.5 Feature distribution analysis

Compare the training period with validation and test periods.

- For numeric features, report descriptive statistics and a two-sample Kolmogorov–Smirnov statistic. Include an effect-size measure or distribution-distance summary so large samples do not make tiny differences look operationally important merely because a p-value is small.
- For categorical features, compare category proportions and use a chi-square test when its assumptions are satisfied. Report Cramér's V as an effect size.
- Handle missing values explicitly and report category/binning choices. Do not silently discard observations.

The tests are descriptive evidence of distribution differences; statistical significance does not imply causality or practical importance.

### 6.6 Model performance over time

Use the same preprocessing and model definition from the original project. Report at minimum:

- Average Precision (primary ranking metric for the imbalanced target);
- AP lift over the prevalence baseline, calculated descriptively as `Average Precision / positive-class prevalence` (a random-ranking baseline has AP approximately equal to prevalence);
- precision, recall, and F1-score at the threshold selected using validation data;
- ROC-AUC as a complementary ranking metric;
- confusion matrix and sample size for each evaluation window.

Choose the classification threshold using validation data only. Keep it fixed for test evaluation. Where later time windows are analyzed, label those comparisons as diagnostic unless they were specified before model evaluation.

## 7. Expected outputs

The completed research contribution should contain:

1. `notebooks/02_temporal_distribution_shift.ipynb` — fully reproducible analysis.
2. `research/RESEARCH_PROTOCOL.md` — this protocol, updated with completed methods and findings.
3. `research/results/` — CSV tables and figures generated by the notebook.
4. `research/REPORT.md` — a concise technical report with question, methodology, results, limitations, and references.

Do not fabricate missing results. Record library versions, random seeds, split dates, feature lists, and model parameters.

## 8. Limitations

- The study uses historical observational data and cannot establish causal explanations for delivery delays.
- The target is defined only for delivered orders with usable delivery-date information; findings do not directly describe canceled or unavailable orders.
- A change in target prevalence can affect precision, F1-score, and Average Precision even without a change in the ranking relationship `P(Y | X)`.
- Olist is one historical Brazilian marketplace dataset. Results may not generalize to another marketplace, period, carrier network, or operational process.
- A temporal performance decline can have multiple explanations, including feature shift, label shift, conditional concept drift, sample composition, or data-quality changes. The study should not claim one cause without supporting analysis.

## 9. References

1. Han, E., Huang, C., & Wang, K. (2024). *Model Assessment and Selection under Temporal Distribution Shift*. Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, 235, 17374–17392. https://proceedings.mlr.press/v235/han24b.html
2. Lukats, D., Zielinski, O., Hahn, A., et al. (2025). *A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams*. International Journal of Data Science and Analytics, 19, 1–31. https://doi.org/10.1007/s41060-024-00620-y
3. Gunasekara, N., Pfahringer, B., Gomes, H. M., Bifet, A., & Koh, Y. S. (2024). *Recurrent Concept Drifts on Data Streams*. Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 8029–8037. https://doi.org/10.24963/ijcai.2024/888
