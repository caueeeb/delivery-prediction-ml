
# Delivery Delay Prediction

Machine Learning project focused on predicting delivery delays in Brazilian e-commerce using the Brazilian E-Commerce Public Dataset by Olist.

## Project Overview

This project explores historical e-commerce data to identify patterns associated with late deliveries.

A supervised binary classification approach was developed to classify delivered orders as:

- `0` - Delivered on time
- `1` - Delivered after the estimated delivery date

The project covers data analysis, feature engineering, model training, hyperparameter experiments, evaluation and software engineering practices.

**Important:** This is a retrospective classification experiment using completed orders. It is not yet a validated production system for predicting delays across all ongoing orders.

## Dataset

Source: Brazilian E-Commerce Public Dataset by Olist.

The dataset contains approximately 100,000 orders from the Brazilian e-commerce market between 2016 and 2018.

The analysis uses information from customers, orders, products, sellers, payments and delivery dates.

The original CSV files are not included in this repository. Users must obtain the dataset from its original source and comply with its license and terms.

## Project Workflow

1. Exploratory Data Analysis
2. Data Cleaning and Validation
3. Feature Engineering
4. Relational Data Integration
5. Temporal Train, Validation and Test Split
6. Machine Learning Experiments
7. Hyperparameter Optimization
8. Threshold Optimization
9. Model Interpretation
10. Final Evaluation

## Feature Engineering

The project creates features such as:

- Number of items per order
- Total product price
- Total freight value
- Average item price
- Total product weight
- Number of sellers
- Customer and seller location
- Same-state seller proportion
- Purchase hour, weekday and month
- Estimated delivery window
- Payment information

Permutation Importance was used to investigate feature relevance.

The most relevant features in the validation experiments included:

- Customer state
- Estimated delivery window
- Same-state seller proportion
- Average item price

These results indicate predictive relevance, not causal relationships.

## Machine Learning Models

Three classification algorithms were evaluated:

| Model | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 61.03% | 11.72% | 73.93% | 20.24% | 16.78% |
| Random Forest | 87.92% | 17.18% | 21.12% | 18.95% | 14.52% |
| HistGradientBoosting | 72.51% | 11.71% | 47.58% | 18.79% | 14.09% |

The Logistic Regression model achieved the best validation F1-score after threshold optimization and initial hyperparameter experiments.

### Selected Model Configuration

- Algorithm: Logistic Regression
- C: 10.0
- Class Weight: None
- Maximum Iterations: 1500
- Classification Threshold: 0.15

## Temporal Evaluation

The dataset was divided chronologically:

| Dataset | Records |
|---|---:|
| Training | 61,740 |
| Validation | 15,436 |
| Final Test | 19,294 |

The final test set contains more recent orders than the development data.

## Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 84.50% |
| Precision | 6.73% |
| Recall | 26.71% |
| F1-score | 10.75% |
| Average Precision | 7.16% |

The final evaluation showed a performance reduction compared with validation.

This behavior may be associated with temporal distribution changes, since the late-delivery rate decreased from 6.69% in validation to 3.49% in the test set.

This highlights the importance of temporal validation and monitoring distribution changes in machine learning applications.

## Limitations

- The target is available only for orders with completed deliveries.
- The experiment does not establish performance on ongoing or canceled orders.
- Temporal distribution changes affected generalization.
- Further feature engineering, probability calibration and model experiments are needed.
- Feature availability must be verified for any proposed real-time prediction scenario.

The current model is an educational and experimental portfolio project, not a production-ready delivery prediction system.

## Project Structure

```text
delivery-prediction-ml/
│
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
├── reports/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/caueeeb/delivery-prediction-ml.git
cd delivery-prediction-ml
```

Create a virtual environment:

```bash
py -3.13 -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the original Olist dataset and place the CSV files inside:

```text
data/raw/
```

Run the automated tests:

```bash
python -m pytest -v
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git and GitHub
- Pytest
- Joblib

## Author

**Cauê Buzolin**

Analysis and Systems Development student at Fatec Taquaritinga.

Interested in Artificial Intelligence, Machine Learning, Data Science and Software Engineering.

GitHub: https://github.com/caueeeb

---

This project was developed as part of my learning journey in Artificial Intelligence and Machine Learning.
