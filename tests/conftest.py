
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():

    rng = np.random.default_rng(42)

    total = 100

    X = pd.DataFrame({
        "customer_state": np.tile(
            ["SP", "RJ", "MG", "BA"], 25
        ),
        "total_freight": rng.uniform(5, 100, total),
        "item_count": rng.integers(1, 5, total),
        "purchase_hour": rng.integers(0, 24, total)
    })

    y = pd.Series(
        [0, 1] * 50,
        name="is_late"
    )

    return X, y


@pytest.fixture
def olist_dir(tmp_path):
    """Cria tabelas sintéticas no formato do Olist para testar o fluxo
    completo (preprocessamento -> treino -> artefatos) sem o dataset real.

    Os números NÃO representam o Olist: servem apenas para validar o
    código. A chance de atraso depende de duas variáveis para que o modelo
    tenha algo a aprender e as três partições tenham as duas classes.
    """

    rng = np.random.default_rng(7)

    total = 1200

    states = np.array(["SP", "RJ", "MG", "BA", "RS"])

    purchase = pd.Timestamp("2017-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 500, total)), unit="D"
    )

    customer_state = rng.choice(states, total)
    freight = rng.uniform(5, 80, total)
    window_days = rng.integers(8, 30, total)

    # Atraso mais provável com frete alto e prazo curto
    late_logit = -2.2 + 0.03 * freight - 0.12 * (window_days - 15)
    is_late = rng.random(total) < 1 / (1 + np.exp(-late_logit))

    estimated = purchase.normalize() + pd.to_timedelta(window_days, unit="D")
    delivered = estimated + pd.to_timedelta(
        np.where(is_late, 3, -3), unit="D"
    )

    order_ids = [f"o{i}" for i in range(total)]
    customer_ids = [f"c{i}" for i in range(total)]

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "order_status": "delivered",
        "order_purchase_timestamp": purchase,
        "order_approved_at": purchase,
        "order_delivered_carrier_date": purchase,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": estimated,
    })

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_unique_id": customer_ids,
        "customer_city": "cidade",
        "customer_state": customer_state,
    })

    items = pd.DataFrame({
        "order_id": order_ids,
        "order_item_id": 1,
        "product_id": [f"p{i % 50}" for i in range(total)],
        "seller_id": [f"s{i % 20}" for i in range(total)],
        "price": rng.uniform(20, 300, total),
        "freight_value": freight,
    })

    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(50)],
        "product_category_name": ["cat_a", "cat_b"] * 25,
        "product_weight_g": rng.integers(100, 5000, 50),
    })

    # Cada vendedor pertence a um único estado
    seller_states = rng.choice(states, 20)

    sellers = pd.DataFrame({
        "seller_id": [f"s{i}" for i in range(20)],
        "seller_city": "cidade",
        "seller_state": seller_states,
    })

    payments = pd.DataFrame({
        "order_id": order_ids,
        "payment_sequential": 1,
        "payment_type": "credit_card",
        "payment_installments": rng.integers(1, 6, total),
        "payment_value": items["price"] + items["freight_value"],
    })

    tables = {
        "olist_orders_dataset.csv": orders,
        "olist_customers_dataset.csv": customers,
        "olist_order_items_dataset.csv": items,
        "olist_order_payments_dataset.csv": payments,
        "olist_products_dataset.csv": products,
        "olist_sellers_dataset.csv": sellers,
    }

    for filename, table in tables.items():
        table.to_csv(tmp_path / filename, index=False)

    return tmp_path
