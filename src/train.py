
from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_pipeline(X):
    """Cria o pipeline de pré-processamento e Regressão Logística."""

    # Identificar as variáveis numéricas
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    # Identificar as variáveis categóricas
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    # Pré-processamento numérico
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pré-processamento categórico
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combinar os tratamentos
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Criar o modelo otimizado
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=10.0,
            class_weight=None,
            max_iter=1500,
            random_state=42
        ))
    ])

    return model


def train_model(X_train, y_train):
    """Treina o modelo de Regressão Logística."""

    model = build_logistic_pipeline(X_train)

    model.fit(X_train, y_train)

    return model


def evaluate_at_threshold(model, X, y, threshold=0.15):
    """Avalia o modelo utilizando um threshold personalizado."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y, predictions, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y, probabilities
        )
    }

    matrix = confusion_matrix(y, predictions)

    return metrics, matrix, predictions, probabilities


def save_model(model, filepath):
    """Salva o pipeline treinado em um arquivo."""

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, filepath)

    print(f"Modelo salvo em: {filepath}")
