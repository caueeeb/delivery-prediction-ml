# Delivery Delay Prediction

Machine Learning project focused on predicting delivery delays in Brazilian e-commerce using the Brazilian E-Commerce Public Dataset by Olist.

## Project overview

This project explores historical e-commerce data to identify patterns associated with late deliveries.

A supervised binary classification approach was developed to classify delivered orders as:

- `0` - Delivered on time
- `1` - Delivered after the estimated delivery date

The project covers exploratory data analysis, feature engineering, relational data integration, temporal model evaluation, model interpretation, automated tests, and a REST API built with FastAPI.

**Important:** this is a retrospective classification experiment using completed orders. It is not yet a validated production system for predicting delays across all ongoing orders.

## Dataset

**Source:** Brazilian E-Commerce Public Dataset by Olist.

The dataset contains approximately 100,000 orders from the Brazilian e-commerce market between 2016 and 2018. The analysis uses information from customers, orders, products, sellers, payments, and delivery dates.

The original CSV files are not included in this repository. Obtain the dataset from its original source and comply with its license and terms. Place the files in `data/raw/` locally; the `data/` directory is excluded from Git.

## Project workflow

1. Exploratory Data Analysis
2. Data cleaning and validation
3. Feature engineering
4. Relational data integration
5. Temporal train, validation, and test split
6. Machine Learning experiments
7. Hyperparameter optimization
8. Threshold optimization
9. Model interpretation with permutation importance
10. Final temporal evaluation
11. Automated tests
12. REST API development and local testing

## Feature engineering

The project creates order-level features such as:

- Number of items, distinct products, and distinct sellers
- Total product price, freight value, and average item price
- Total product weight and product-category information
- Customer and seller location features
- Same-state seller proportion
- Purchase hour, weekday, month, and weekend indicator
- Estimated delivery window
- Payment totals, installments, methods, and record counts

Permutation importance was used to investigate feature relevance. The most relevant features in validation experiments included customer state, estimated delivery window, same-state seller proportion, and average item price. These results indicate predictive relevance, not causal relationships.

## Machine Learning experiments

Several classifiers were evaluated, including a Dummy baseline, Logistic Regression, Random Forest, and HistGradientBoosting. Logistic Regression achieved the best validation F1-score after threshold optimization and hyperparameter experiments.

### Selected model configuration

- Algorithm: Logistic Regression
- `C`: 10.0
- `class_weight`: `None`
- Maximum iterations: 1500
- Classification threshold: 0.15
- Input features: 26

### Temporal split

The dataset was divided chronologically:

| Dataset | Records |
|---|---:|
| Training | 61,740 |
| Validation | 15,436 |
| Final test | 19,294 |

The final test set contains more recent orders than the development data.

### Final test results

| Metric | Result |
|---|---:|
| Accuracy | 84.50% |
| Precision | 6.73% |
| Recall | 26.71% |
| F1-score | 10.75% |
| Average Precision | 7.16% |

The final evaluation showed a performance reduction compared with validation. This behavior may be associated with temporal distribution changes: the late-delivery rate decreased from 6.69% in validation to 3.49% in the test set. This highlights the importance of temporal validation and monitoring distribution changes in Machine Learning applications.

## REST API (FastAPI)

The trained model is exposed through a local REST API. The API loads the serialized pipeline from `models/delivery_delay_logistic_regression.joblib` and uses the selected classification threshold (`0.15`).

### Start the API

From the repository root, activate the virtual environment, install the requirements, and run:

```powershell
uvicorn src.api:app --reload
```

The local API is available at `http://127.0.0.1:8000`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Checks API availability and whether the model was loaded. |
| `GET` | `/model-info` | Returns model name, selected configuration, threshold, target, and feature count. |
| `POST` | `/predict` | Receives one order's features and returns the predicted late-delivery probability and class. |

### Prediction input

The `/predict` endpoint expects the 26 model features in the request body:

`customer_state`, `item_count`, `distinct_products`, `distinct_sellers`, `total_item_price`, `total_freight`, `avg_item_price`, `total_product_weight_g`, `distinct_categories`, `missing_product_weights`, `missing_product_categories`, `payment_total`, `payment_installments_max`, `payment_methods_count`, `payment_records_count`, `seller_count`, `seller_state_count`, `seller_city_count`, `same_state_seller_count`, `same_state_seller_share`, `purchase_hour`, `purchase_day_of_week`, `purchase_month`, `purchase_is_weekend`, `estimated_delivery_window_days`, and `has_payment_record`.

Use the interactive `/docs` page to inspect the exact request schema and try a prediction. The response includes the estimated probability, the applied threshold, the predicted class, and a human-readable label.

Example response shape:

```json
{
  "late_probability": 0.050264,
  "predicted_is_late": 0,
  "threshold": 0.15,
  "label": "not_predicted_late"
}
```

The probability is a model output, not a guarantee about an individual delivery. The endpoint is a local demonstration and has not been deployed or validated for production use.

## Tests

Run the automated test suite from the repository root:

```powershell
python -m pytest -v
```

The current local test suite contains 13 tests covering preprocessing, model training, prediction, model serialization, and API behavior (health, model information, valid prediction, invalid state, and unexpected input fields).

## Limitations

- The target is available only for orders with completed deliveries.
- The experiment does not establish performance on ongoing or canceled orders.
- Temporal distribution changes affected generalization.
- Further feature engineering, probability calibration, and model experiments are needed.
- Feature availability must be verified for any proposed real-time prediction scenario.
- The API is a local prototype, not a production deployment.

The current model is an educational and experimental portfolio project, not a production-ready delivery prediction system.

## Project structure

```text
delivery-prediction-ml/
├── data/                         # Local dataset; excluded from Git
├── models/
│   └── delivery_delay_logistic_regression.joblib
├── notebooks/
│   └── 01_entendimento_dados.ipynb
├── reports/                      # Evaluation metrics and experiment outputs
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── requirements.txt
├── requirements-api.txt
├── README.md
└── .gitignore
```

## Installation and usage

The commands below use Windows PowerShell and Python 3.13.

```powershell
git clone https://github.com/caueeeb/delivery-prediction-ml.git
cd delivery-prediction-ml

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-api.txt
```

`requirements-api.txt` records the API and API-test dependencies used by this project.

Download the original Olist dataset and place its CSV files in `data/raw/`. Run the tests with `python -m pytest -v`, or start the API with `uvicorn src.api:app --reload`.

## Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, FastAPI, Uvicorn, Pydantic, HTTPX, Pytest, Joblib, Git, and GitHub.

## Author

**Cauê Buzolin**  
Analysis and Systems Development student at Fatec Taquaritinga.  
Interested in Artificial Intelligence, Machine Learning, Data Science, and Software Engineering.  
GitHub: [@caueeeb](https://github.com/caueeeb)

---

This project was developed as part of my learning journey in Artificial Intelligence and Machine Learning.
