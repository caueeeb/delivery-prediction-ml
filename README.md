# Delivery Delay Prediction (Olist)

[![CI](https://github.com/caueeeb/delivery-prediction-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/caueeeb/delivery-prediction-ml/actions/workflows/ci.yml)

Predicting late deliveries in Brazilian e-commerce with the [Olist public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): data integration across six tables, chronological evaluation, a tested FastAPI service, and a study of what happens to the model when the data changes over time.

> **Resumo (PT-BR):** projeto de classificação para prever pedidos entregues após a data estimada, com base de 96.470 pedidos, 26 atributos, separação cronológica treino/validação/teste, API FastAPI com testes automatizados, Docker e CI. O ponto central é a avaliação honesta: no teste cronológico o modelo tem sinal fraco (Average Precision 0,072, cerca de 2× a prevalência de atrasos de 3,5%), e o estudo em [`research/`](research/) investiga por que as métricas caíram entre validação e teste.

**This is a retrospective, educational experiment on completed orders. It is not a validated production system.**

## What this project shows

- **Evaluation discipline.** Chronological train / validation / test split; threshold chosen on validation only; test set used once; results always reported next to a chance baseline.
- **A model with modest signal, reported as such.** See [Results in context](#results-in-context).
- **Reproducibility.** `python -m src.train` rebuilds the committed model from the raw CSVs; a test suite checks that the model, its metadata and the API stay consistent.
- **Software engineering around the model.** Typed request validation, a metadata-driven API, 22 automated tests, linting, a Dockerfile and a CI workflow.
- **Research.** A [temporal distribution shift study](research/README.md) and a [technical article](research/ARTICLE_TEMPORAL_DISTRIBUTION_SHIFT.md).

## Task and data

Binary classification on delivered orders:

- `0`: delivered on or before the estimated delivery date
- `1`: delivered after the estimated delivery date (calendar-day comparison)

The analysis base has **96,470 delivered orders** (2016-09-15 to 2018-08-29) built from the orders, customers, order items, products, payments and sellers tables, with **26 input features** (order size and value, freight, weight, payment, customer and seller location, purchase time, and the estimated delivery window).

No input feature is computed from carrier or delivery timestamps; those dates are used only to build the target. Whether every field would be available at prediction time in a real system (for example payment records) still needs to be confirmed.

The original CSV files are **not** included. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), follow its license and terms, and place the files in `data/raw/` (excluded from Git).

## Results in context

Chronological partitions (by purchase date):

| Partition | Orders | Period | Late rate |
|---|---:|---|---:|
| Training | 61,740 | 2016-09-15 to 2018-03-20 | 7.82% |
| Validation | 15,436 | 2018-03-20 to 2018-05-26 | 6.69% |
| Test | 19,294 | 2018-05-26 to 2018-08-29 | 3.49% |

Selected model: Logistic Regression (`C=10`, `class_weight=None`), threshold **0.15** (maximum F1 on validation). The model used for the test results was refit on training + validation (77,176 orders); the validation column below comes from the model fit on the training partition only.

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.1751 | 0.0716 |
| Chance level for AP (= late rate) | 0.0669 | 0.0349 |
| AP lift over chance | 2.62× | 2.05× |
| ROC-AUC | 0.735 | 0.699 |
| Precision | 19.8% | 6.7% |
| Recall | 39.3% | 26.7% |
| F1-score | 0.263 | 0.1075 |
| Accuracy | 85.3% | 84.5% |

How to read this honestly:

- At the 0.15 threshold the model flags about 14% of test orders as late and is right 6.7% of the time. A classifier that always predicts "on time" would have 96.5% accuracy on the test set, so **accuracy is not a useful metric here**.
- The model ranks late orders about **twice as well as chance** (AP lift 2.05×, ROC-AUC 0.70). That is real but limited signal.
- A diagnostic comparing random and chronological splits ([research, section 12](research/RESULTS_AND_DISCUSSION.md)) shows that random evaluation would have reported roughly double the absolute AP and F1 (0.152 and 0.216 versus 0.072 and 0.108), while ROC-AUC was about 0.70 under both. Much of the validation-to-test drop is therefore consistent with the lower late rate (6.7% to 3.5%), not necessarily with a collapse in ranking ability. This does not prove that the feature–target relationship stayed constant.
- Permutation importance on validation put `customer_state`, `estimated_delivery_window_days`, `same_state_seller_share` and `avg_item_price` first. These are predictive associations, not causes.

## Quick start

Requires Python 3.11+ (developed on 3.13).

```bash
git clone https://github.com/caueeeb/delivery-prediction-ml.git
cd delivery-prediction-ml

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Run the tests (they do not need the dataset; they use synthetic Olist-shaped tables):

```bash
pytest
```

Retrain from the raw CSVs in `data/raw/` and write the model and metadata:

```bash
python -m src.train                          # overwrites models/ and reports/model_metadata.json
python -m src.train --output-dir artifacts   # or write somewhere else
```

Retraining on the original dataset reproduces the committed model (same coefficients to about 1e-11, same test metrics).

## REST API

```bash
uvicorn src.api:app --reload
```

Interactive documentation: `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service status and whether the model loaded (`degraded` if the artifact is missing). |
| `GET` | `/model-info` | Algorithm, parameters, threshold, feature list, training size and test metrics, read from the loaded model and `reports/model_metadata.json`. |
| `POST` | `/predict` | Takes one order's 26 features and returns the late probability, threshold and class. |

Example request and response:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "customer_state": "SP", "item_count": 1, "distinct_products": 1, "distinct_sellers": 1,
  "total_item_price": 100.0, "total_freight": 15.0, "avg_item_price": 100.0,
  "total_product_weight_g": 1500.0, "distinct_categories": 1,
  "missing_product_weights": 0, "missing_product_categories": 0,
  "payment_total": 115.0, "payment_installments_max": 1.0, "payment_methods_count": 1.0,
  "payment_records_count": 1.0, "seller_count": 1, "seller_state_count": 1,
  "seller_city_count": 1, "same_state_seller_count": 1, "same_state_seller_share": 1.0,
  "purchase_hour": 14, "purchase_day_of_week": 2, "purchase_month": 6, "purchase_is_weekend": 0,
  "estimated_delivery_window_days": 15, "has_payment_record": 1
}'
```

```json
{
  "late_probability": 0.050264,
  "predicted_is_late": 0,
  "threshold": 0.15,
  "prediction_label": "not_predicted_late"
}
```

Invalid input (unknown fields, a state that is not two capital letters, out-of-range values) is rejected with HTTP 422. The probability is a model output, not a guarantee about a delivery. Research shows these probabilities are over-confident in later periods (see limitations).

### Docker

```bash
docker build -t delivery-prediction-api .
docker run --rm -p 8000:8000 delivery-prediction-api
```

The image contains only the code, the serialized model and its metadata, and runs as a non-root user.

## Limitations

- Orders without an observed delivery date (canceled, unavailable or still open) are outside the analysis.
- Late-delivery prevalence changed from 7.82% (training) to 3.49% (test). Thresholded metrics and Average Precision depend on prevalence, so they are not comparable across periods without a chance baseline.
- Probabilities are not calibrated: the study found mean predicted risk above the observed late rate in every test month, and a Brier score worse than a constant training-prevalence predictor. Treat them as a ranking score, not as probabilities.
- A single chronological split was used; rolling-origin evaluation and confidence intervals are future work.
- The model is a local prototype with no authentication, monitoring or deployment.

## Possible next steps

- Seller–customer distance from the geolocation table, and seller history computed only from deliveries finished before each purchase.
- Probability calibration fitted on validation data only, and a threshold based on an explicit operational cost.
- Rolling-origin evaluation with confidence intervals; gradient boosting with time-aware tuning.

## Project structure

```text
delivery-prediction-ml/
├── .github/workflows/ci.yml      # lint + tests + Docker smoke test
├── data/raw/                     # Olist CSVs (not versioned)
├── models/                       # serialized scikit-learn pipeline
├── notebooks/01_entendimento_dados.ipynb
├── reports/                      # experiment tables and model_metadata.json
├── research/                     # temporal distribution shift study and article
├── src/
│   ├── config.py                 # paths, parameters, threshold (single source of truth)
│   ├── preprocessing.py          # base construction and chronological splits
│   ├── train.py                  # training CLI: python -m src.train
│   ├── predict.py                # batch prediction helper
│   └── api.py                    # FastAPI application
├── tests/                        # 22 tests (preprocessing, training CLI, API, consistency)
├── Dockerfile
├── pyproject.toml                # pytest and ruff configuration
├── requirements.txt              # runtime dependencies (pinned)
└── requirements-dev.txt          # tests, lint and notebooks
```

## Technologies

Python, pandas, NumPy, scikit-learn, FastAPI, Pydantic, Uvicorn, pytest, Ruff, Docker, GitHub Actions, Jupyter, Matplotlib and Seaborn.

## Author

**Cauê Buzolin** — Analysis and Systems Development student at Fatec Taquaritinga, focused on machine learning and AI engineering.
[GitHub](https://github.com/caueeeb) · [LinkedIn](https://www.linkedin.com/in/cau%C3%AA-buzolin-76a4283b3/) · [Hugging Face](https://huggingface.co/caueeb)

Released under the [MIT License](LICENSE). The Olist dataset has its own license and terms.
