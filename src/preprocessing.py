
from pathlib import Path

import pandas as pd

# Colunas utilizadas para identificar as datas
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

# Colunas que não serão utilizadas como entrada do modelo
EXCLUDED_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_city",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "delivery_delay_days",
    "is_late"
]

# Colunas relacionadas aos pagamentos
PAYMENT_COLUMNS = [
    "payment_total",
    "payment_installments_max",
    "payment_methods_count",
    "payment_records_count"
]


def load_raw_tables(data_dir: Path) -> dict:
    """Carrega as tabelas necessárias do dataset Olist."""

    filenames = {
        "orders": "olist_orders_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv"
    }

    data_dir = Path(data_dir)

    missing_files = [
        filename
        for filename in filenames.values()
        if not (data_dir / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Arquivos não encontrados: {missing_files}"
        )

    tables = {
        name: pd.read_csv(data_dir / filename)
        for name, filename in filenames.items()
    }

    # Converter as datas da tabela de pedidos
    for column in DATE_COLUMNS:
        tables["orders"][column] = pd.to_datetime(
            tables["orders"][column],
            errors="coerce"
        )

    return tables


def build_feature_base(data_dir: Path) -> pd.DataFrame:
    """Constrói uma base com uma linha por pedido entregue."""

    tables = load_raw_tables(data_dir)

    orders = tables["orders"]
    customers = tables["customers"]
    order_items = tables["order_items"]
    payments = tables["payments"]
    products = tables["products"]
    sellers = tables["sellers"]

    # Selecionar pedidos entregues com datas disponíveis
    delivered_orders = orders[
        orders["order_status"] == "delivered"
    ].copy()

    delivered_orders = delivered_orders.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    # Criar a variável-alvo
    delivered_orders["delivery_delay_days"] = (
        delivered_orders["order_delivered_customer_date"].dt.normalize()
        - delivered_orders["order_estimated_delivery_date"].dt.normalize()
    ).dt.days

    delivered_orders["is_late"] = (
        delivered_orders["delivery_delay_days"] > 0
    ).astype(int)

    # Integrar itens e produtos
    items_with_products = order_items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Agregar informações dos produtos por pedido
    item_features = (
        items_with_products
        .groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            distinct_products=("product_id", "nunique"),
            distinct_sellers=("seller_id", "nunique"),
            total_item_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_item_price=("price", "mean"),
            total_product_weight_g=("product_weight_g", "sum"),
            distinct_categories=("product_category_name", "nunique"),
            missing_product_weights=(
                "product_weight_g",
                lambda x: x.isna().sum()
            ),
            missing_product_categories=(
                "product_category_name",
                lambda x: x.isna().sum()
            )
        )
        .reset_index()
    )

    # Agregar pagamentos por pedido
    payment_features = (
        payments
        .groupby("order_id")
        .agg(
            payment_total=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_methods_count=("payment_type", "nunique"),
            payment_records_count=("payment_sequential", "count")
        )
        .reset_index()
    )

    # Relacionar vendedores, pedidos e clientes
    seller_order = (
        order_items[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(
            sellers[["seller_id", "seller_state", "seller_city"]],
            on="seller_id",
            how="left",
            validate="many_to_one"
        )
        .merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
            validate="many_to_one"
        )
        .merge(
            customers[["customer_id", "customer_state"]],
            on="customer_id",
            how="left",
            validate="many_to_one"
        )
    )

    seller_order["same_state"] = (
        seller_order["seller_state"]
        == seller_order["customer_state"]
    ).astype(int)

    # Agregar características dos vendedores
    seller_features = (
        seller_order
        .groupby("order_id")
        .agg(
            seller_count=("seller_id", "nunique"),
            seller_state_count=("seller_state", "nunique"),
            seller_city_count=("seller_city", "nunique"),
            same_state_seller_count=("same_state", "sum")
        )
        .reset_index()
    )

    # Criar a base principal
    base = delivered_orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "delivery_delay_days",
            "is_late"
        ]
    ].copy()

    # Integrar clientes
    base = base.merge(
        customers[
            ["customer_id", "customer_city", "customer_state"]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # Integrar as características dos produtos
    base = base.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Integrar pagamentos
    base = base.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Integrar vendedores
    base = base.merge(
        seller_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Identificar pedidos sem registro de pagamento
    base["has_payment_record"] = (
        base["payment_records_count"].notna()
    ).astype(int)

    base[PAYMENT_COLUMNS] = (
        base[PAYMENT_COLUMNS].fillna(0)
    )

    # Criar características temporais
    base["purchase_hour"] = (
        base["order_purchase_timestamp"].dt.hour
    )

    base["purchase_day_of_week"] = (
        base["order_purchase_timestamp"].dt.day_of_week
    )

    base["purchase_month"] = (
        base["order_purchase_timestamp"].dt.month
    )

    base["purchase_is_weekend"] = (
        base["purchase_day_of_week"] >= 5
    ).astype(int)

    base["estimated_delivery_window_days"] = (
        base["order_estimated_delivery_date"].dt.normalize()
        - base["order_purchase_timestamp"].dt.normalize()
    ).dt.days

    # Calcular a proporção de vendedores do mesmo estado
    base["same_state_seller_share"] = (
        base["same_state_seller_count"]
        / base["seller_count"]
    )

    # Padronizar a ordem das colunas da base
    column_order = [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "delivery_delay_days",
        "is_late",
        "customer_city",
        "customer_state",
        "item_count",
        "distinct_products",
        "distinct_sellers",
        "total_item_price",
        "total_freight",
        "avg_item_price",
        "total_product_weight_g",
        "distinct_categories",
        "missing_product_weights",
        "missing_product_categories",
        "payment_total",
        "payment_installments_max",
        "payment_methods_count",
        "payment_records_count",
        "seller_count",
        "seller_state_count",
        "seller_city_count",
        "same_state_seller_count",
        "same_state_seller_share",
        "purchase_hour",
        "purchase_day_of_week",
        "purchase_month",
        "purchase_is_weekend",
        "estimated_delivery_window_days",
        "has_payment_record"
    ]

    base = base[column_order]

    # Garantir uma linha por pedido
    if base["order_id"].duplicated().any():
        raise ValueError("A base contém pedidos duplicados.")

    if base[PAYMENT_COLUMNS].isnull().any().any():
        raise ValueError("Existem valores ausentes nos pagamentos.")

    return base


def make_temporal_splits(base: pd.DataFrame) -> dict:
    """Divide os dados cronologicamente em treino, validação e teste."""

    ordered = (
        base
        .sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )

    # Separar os últimos 20% para teste
    test_index = int(len(ordered) * 0.80)

    development_data = ordered.iloc[:test_index].copy()
    test_data = ordered.iloc[test_index:].copy()

    # Separar treino e validação
    validation_index = int(len(development_data) * 0.80)

    fit_data = development_data.iloc[:validation_index].copy()
    validation_data = development_data.iloc[validation_index:].copy()

    def prepare_xy(dataset):
        X = dataset.drop(columns=EXCLUDED_COLUMNS)
        y = dataset["is_late"]

        return X, y

    X_fit, y_fit = prepare_xy(fit_data)
    X_val, y_val = prepare_xy(validation_data)
    X_test, y_test = prepare_xy(test_data)

    return {
        "fit_data": fit_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }
