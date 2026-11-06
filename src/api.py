
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_late


# Localizar a raiz do projeto
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Localizar o modelo treinado
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "delivery_delay_logistic_regression.joblib"
)

# Configuração do modelo
THRESHOLD = 0.15

# Carregar o modelo
model = load_model(MODEL_PATH)


# Criar a aplicação
app = FastAPI(
    title="Delivery Delay Prediction API",
    description=(
        "API for estimating delivery delay risk "
        "using a Logistic Regression model."
    ),
    version="1.0.0"
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

    estimated_delivery_window_days: int

    has_payment_record: int = Field(ge=0, le=1)


# Estrutura da resposta
class PredictionResponse(BaseModel):

    late_probability: float
    predicted_is_late: int
    threshold: float
    prediction_label: str


@app.get("/health")
def health_check():
    """Verifica se a API está funcionando."""

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    """Apresenta informações sobre o modelo."""

    return {
        "model": "Logistic Regression",
        "C": 10.0,
        "class_weight": None,
        "threshold": THRESHOLD,
        "features_count": 26,
        "target": "is_late"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    """Recebe os dados de um pedido e calcula o risco."""

    # Converter os dados recebidos em DataFrame
    input_data = pd.DataFrame([
        request.model_dump()
    ])

    # Executar previsão
    result = predict_late(
        model,
        input_data,
        threshold=THRESHOLD
    )

    probability = float(
        result["late_probability"].iloc[0]
    )

    prediction = int(
        result["predicted_is_late"].iloc[0]
    )

    return PredictionResponse(
        late_probability=round(probability, 6),
        predicted_is_late=prediction,
        threshold=THRESHOLD,
        prediction_label=(
            "late" if prediction == 1 else "not_predicted_late"
        )
    )
    