# Research Project — Olist Temporal Distribution Shift

This folder contains the research protocol for a reproducible study of temporal distribution shift in e-commerce delivery-delay prediction.

## Research question

How does temporal change in the Olist e-commerce data affect the generalization performance of a machine-learning model that predicts delivery delays?

## Current status

The repository's existing Olist experiment provides preliminary motivation:

| Split | Late-order prevalence |
|---|---:|
| Training | 7.82% |
| Validation | 6.69% |
| Test | 3.49% |

The Logistic Regression model's previously reported validation and test metrics also differ:

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.1751 | 0.0716 |
| F1-score | 0.2630 | 0.1075 |

These values motivate further investigation. They do **not**, on their own, prove concept drift or identify its cause.

## Contents

- `RESEARCH_PROTOCOL.md`: question, hypotheses, methodology, metrics, limitations, and references.

## Planned notebook

The next research step is to create `notebooks/02_temporal_distribution_shift.ipynb` inside the main `delivery-prediction-ml` project. It will reuse that project's preprocessing and modeling code, analyze differences among chronological splits, compare random and temporal evaluation, and save tables/figures under `research/results/`.

Raw Olist CSV files must remain local and must not be committed to GitHub.

## Research Results

The complete experimental findings, discussion,
limitations, and conclusions are available in:

[Results and Discussion](RESULTS_AND_DISCUSSION.md)