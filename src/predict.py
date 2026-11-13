
from pathlib import Path

import joblib
import pandas as pd

from src import config


def load_model(filepath):
    """Carrega um modelo de Machine Learning salvo."""

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {filepath}"
        )

    return joblib.load(filepath)


def predict_late(model, X, threshold=config.DEFAULT_THRESHOLD):
    """
    Realiza previsões de atraso.

    Retorna a probabilidade de atraso e a classificação
    final de cada pedido.
    """

    # Validar o threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            "O threshold deve estar entre 0 e 1."
        )

    # Verificar se o modelo oferece probabilidades
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "O modelo precisa oferecer predict_proba()."
        )

    # Recuperar as variáveis utilizadas no treinamento
    expected_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if expected_features is not None:

        expected_features = list(expected_features)

        missing_features = [
            column
            for column in expected_features
            if column not in X.columns
        ]

        if missing_features:
            raise ValueError(
                f"Variáveis ausentes: {missing_features}"
            )

        # Manter a mesma ordem das variáveis do treinamento
        X = X.loc[:, expected_features]

    # Calcular probabilidades
    probabilities = model.predict_proba(X)[:, 1]

    # Converter probabilidades em classificações
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Organizar os resultados
    results = pd.DataFrame({
        "late_probability": probabilities,
        "predicted_is_late": predictions
    }, index=X.index)

    return results
