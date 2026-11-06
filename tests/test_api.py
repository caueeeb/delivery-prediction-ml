
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


SAMPLE_ORDER = {
    "customer_state": "SP",
    "item_count": 1,
    "distinct_products": 1,
    "distinct_sellers": 1,
    "total_item_price": 100.0,
    "total_freight": 15.0,
    "avg_item_price": 100.0,
    "total_product_weight_g": 1500.0,
    "distinct_categories": 1,
    "missing_product_weights": 0,
    "missing_product_categories": 0,
    "payment_total": 115.0,
    "payment_installments_max": 1.0,
    "payment_methods_count": 1.0,
    "payment_records_count": 1.0,
    "seller_count": 1,
    "seller_state_count": 1,
    "seller_city_count": 1,
    "same_state_seller_count": 1,
    "same_state_seller_share": 1.0,
    "purchase_hour": 14,
    "purchase_day_of_week": 2,
    "purchase_month": 6,
    "purchase_is_weekend": 0,
    "estimated_delivery_window_days": 15,
    "has_payment_record": 1
}


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint():

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model"] == "Logistic Regression"
    assert response.json()["threshold"] == 0.15


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json=SAMPLE_ORDER
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["late_probability"] <= 1
    assert result["predicted_is_late"] in [0, 1]
    assert result["threshold"] == 0.15


def test_invalid_customer_state():

    invalid_order = SAMPLE_ORDER.copy()
    invalid_order["customer_state"] = "SP1"

    response = client.post(
        "/predict",
        json=invalid_order
    )

    assert response.status_code == 422


def test_unexpected_input_field():

    invalid_order = SAMPLE_ORDER.copy()
    invalid_order["unknown_feature"] = 123

    response = client.post(
        "/predict",
        json=invalid_order
    )

    assert response.status_code == 422
