"""API REST (FastAPI) para estimar o risco de atraso de entrega.

Execução local::

    uvicorn src.api:app --reload

O threshold, os parâmetros e as métricas apresentados pela API vêm do
modelo carregado e de ``reports/model_metadata.json``; nada é repetido
manualmente neste arquivo.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.predict import load_model, predict_late

logger = logging.getLogger("delivery_api")

# Uvicorn só configura os próprios loggers; este garante logs da aplicação
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@lru_cache(maxsize=1)
def get_model():
    """Carrega o modelo uma única vez (e só quando necessário)."""

    model = load_model(config.MODEL_PATH)

    logger.info("Modelo carregado de %s", config.MODEL_PATH)

    return model


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    """Carrega os metadados do modelo uma única vez."""

    return config.load_metadata()


def get_threshold() -> float:
    """Threshold de classificação salvo junto com o modelo."""

    return float(
        get_metadata().get("threshold", config.DEFAULT_THRESHOLD)
    )


# Criar a aplicação
app = FastAPI(
    title="Delivery Delay Prediction API",
    description=(
        "API for estimating delivery delay risk "
        "using a Logistic Regression model trained on the "
        "Olist e-commerce dataset. Educational prototype."
    ),
    version="1.1.0"
)


# Estrutura das informações recebidas
class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    customer_state: str = Field(
        min_length=2,
        max_length=2,
        pattern="^[A-Z]{2}$"
    )

    item_count: int = Field(ge=1)
    distinct_products: int = Field(ge=1)
    distinct_sellers: int = Field(ge=1)

    total_item_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    avg_item_price: float = Field(ge=0)

    total_product_weight_g: float = Field(ge=0)
    distinct_categories: int = Field(ge=0)

    missing_product_weights: int = Field(ge=0)
    missing_product_categories: int = Field(ge=0)

    payment_total: float = Field(ge=0)
    payment_installments_max: float = Field(ge=0)
    payment_methods_count: float = Field(ge=0)
    payment_records_count: float = Field(ge=0)

    seller_count: int = Field(ge=1)
    seller_state_count: int = Field(ge=1)
    seller_city_count: int = Field(ge=1)

    same_state_seller_count: int = Field(ge=0)
    same_state_seller_share: float = Field(ge=0, le=1)

    purchase_hour: int = Field(ge=0, le=23)
    purchase_day_of_week: int = Field(ge=0, le=6)
    purchase_month: int = Field(ge=1, le=12)
    purchase_is_weekend: int = Field(ge=0, le=1)

    # Dias entre a compra e a data estimada de entrega
    estimated_delivery_window_days: int = Field(ge=1)

    has_payment_record: int = Field(ge=0, le=1)


# Estrutura da resposta
class PredictionResponse(BaseModel):

    late_probability: float
    predicted_is_late: int
    threshold: float
    prediction_label: str


@app.get("/health")
def health_check():
    """Verifica se a API está funcionando e se o modelo carrega."""

    try:
        get_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.get("/model-info")
def model_info():
    """Apresenta informações sobre o modelo carregado."""

    model = _require_model()
    metadata = get_metadata()

    classifier = model.named_steps["classifier"]
    parameters = classifier.get_params()

    features = [
        str(name) for name in getattr(model, "feature_names_in_", [])
    ]

    return {
        "model": metadata.get("algorithm", "Logistic Regression"),
        "C": parameters.get("C"),
        "class_weight": parameters.get("class_weight"),
        "threshold": get_threshold(),
        "features_count": len(features),
        "features": features,
        "target": metadata.get("target", "is_late"),
        "target_definition": metadata.get("target_definition"),
        "training_records": metadata.get("training_records"),
        "test_metrics": metadata.get("metrics"),
        "test_context": metadata.get("test_context"),
        "sklearn_version": metadata.get("sklearn_version"),
        "note": (
            "Retrospective experiment on historical Olist data; "
            "not validated for production use."
        )
    }


def _require_model():
    """Devolve o modelo ou responde 503 se o artefato não existir."""

    try:
        return get_model()
    except FileNotFoundError as error:
        logger.error("Modelo indisponível: %s", error)

        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `python -m src.train`."
        ) from error


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    """Recebe os dados de um pedido e calcula o risco."""

    model = _require_model()
    threshold = get_threshold()

    # Converter os dados recebidos em DataFrame
    input_data = pd.DataFrame([
        request.model_dump()
    ])

    # Executar previsão
    result = predict_late(
        model,
        input_data,
        threshold=threshold
    )

    probability = float(
        result["late_probability"].iloc[0]
    )

    prediction = int(
        result["predicted_is_late"].iloc[0]
    )

    logger.info(
        "prediction state=%s probability=%.4f predicted_is_late=%d",
        request.customer_state,
        probability,
        prediction
    )

    return PredictionResponse(
        late_probability=round(probability, 6),
        predicted_is_late=prediction,
        threshold=threshold,
        prediction_label=(
            "late" if prediction == 1 else "not_predicted_late"
        )
    )
