"""Treino, avaliação e persistência do modelo de atraso de entrega.

Uso (a partir da raiz do repositório)::

    python -m src.train                     # usa data/raw/ e grava em models/ e reports/
    python -m src.train --data-dir D:\\dados\\olist --output-dir artefatos

O fluxo reproduz o experimento descrito no README:

1. constrói a base com uma linha por pedido entregue;
2. separa os dados cronologicamente (treino / validação / teste);
3. treina com o treino, escolhe o threshold na validação (maior F1);
4. reajusta o modelo com treino + validação;
5. avalia **uma única vez** no teste e grava modelo + metadados.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


def build_logistic_pipeline(X, model_params=None):
    """Cria o pipeline de pré-processamento e Regressão Logística."""

    params = dict(config.MODEL_PARAMS)

    if model_params:
        params.update(model_params)

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

    # Criar o modelo
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(**params))
    ])

    return model


def train_model(X_train, y_train, model_params=None):
    """Treina o modelo de Regressão Logística."""

    model = build_logistic_pipeline(X_train, model_params)

    model.fit(X_train, y_train)

    return model


def evaluate_at_threshold(
    model,
    X,
    y,
    threshold=config.DEFAULT_THRESHOLD
):
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


def select_threshold(y_true, probabilities, grid=None):
    """Escolhe o threshold com maior F1-score em um conjunto de validação.

    Deve ser chamado apenas com dados de validação: o conjunto de teste
    nunca participa da escolha do ponto de corte.
    """

    if grid is None:
        grid = np.arange(
            config.THRESHOLD_GRID_START,
            config.THRESHOLD_GRID_STOP + 1e-9,
            config.THRESHOLD_GRID_STEP
        )

    scores = [
        f1_score(
            y_true,
            (probabilities >= value).astype(int),
            zero_division=0
        )
        for value in grid
    ]

    return round(float(grid[int(np.argmax(scores))]), 2)


def summarize_metrics(model, X, y, threshold):
    """Reúne métricas de classificação, ranking e o baseline de acaso."""

    metrics, matrix, _, probabilities = evaluate_at_threshold(
        model, X, y, threshold
    )

    prevalence = float(np.mean(y))

    summary = {key: float(value) for key, value in metrics.items()}
    summary["ROC-AUC"] = float(roc_auc_score(y, probabilities))
    summary["Late rate (prevalence)"] = prevalence
    # Um ranking aleatório tem AP aproximadamente igual à prevalência
    summary["AP lift over prevalence"] = (
        summary["Average Precision"] / prevalence
        if prevalence > 0 else float("nan")
    )
    summary["Confusion matrix [[TN, FP], [FN, TP]]"] = (
        matrix.astype(int).tolist()
    )

    return summary


def save_model(model, filepath):
    """Salva o pipeline treinado em um arquivo."""

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, filepath)

    logger.info("Modelo salvo em: %s", filepath)


def _describe_split(dataset):
    """Resumo de uma partição cronológica para os metadados."""

    timestamps = dataset["order_purchase_timestamp"]

    return {
        "rows": int(len(dataset)),
        "start": str(timestamps.min().date()),
        "end": str(timestamps.max().date()),
        "late_orders": int(dataset["is_late"].sum()),
        "late_rate": float(dataset["is_late"].mean()),
    }


def run_training(
    data_dir=config.DATA_DIR,
    model_path=config.MODEL_PATH,
    metadata_path=config.METADATA_PATH,
    threshold=None,
    base=None
):
    """Executa o fluxo completo de treino e grava modelo + metadados.

    Se ``threshold`` for None, ele é escolhido na validação (maior F1).
    ``base`` permite injetar uma base já construída (útil em testes).
    Retorna o dicionário de metadados gravado.
    """

    # Importação local: o pré-processamento só é necessário no treino
    import pandas as pd

    from src.preprocessing import build_feature_base, make_temporal_splits

    if base is None:
        logger.info("Construindo a base a partir de %s", data_dir)
        base = build_feature_base(Path(data_dir))

    splits = make_temporal_splits(base)

    # 1) Treina somente com o conjunto de treino e avalia na validação
    validation_model = train_model(splits["X_fit"], splits["y_fit"])

    validation_probabilities = validation_model.predict_proba(
        splits["X_val"]
    )[:, 1]

    selected_threshold = (
        float(threshold) if threshold is not None
        else select_threshold(splits["y_val"], validation_probabilities)
    )

    validation_metrics = summarize_metrics(
        validation_model,
        splits["X_val"],
        splits["y_val"],
        selected_threshold
    )

    # 2) Reajusta com treino + validação (dados de desenvolvimento)
    X_dev = pd.concat([splits["X_fit"], splits["X_val"]])
    y_dev = pd.concat([splits["y_fit"], splits["y_val"]])

    final_model = train_model(X_dev, y_dev)

    # 3) Avalia uma única vez no teste, com o threshold já fixado
    test_metrics = summarize_metrics(
        final_model,
        splits["X_test"],
        splits["y_test"],
        selected_threshold
    )

    save_model(final_model, model_path)

    classifier = final_model.named_steps["classifier"]

    metadata = {
        "project": "Delivery Delay Prediction",
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "algorithm": "Logistic Regression",
        "parameters": {
            key: classifier.get_params()[key]
            for key in config.MODEL_PARAMS
        },
        "threshold": selected_threshold,
        "threshold_selection": (
            "maior F1-score na validação"
            if threshold is None else "definido manualmente"
        ),
        "training_records": int(len(X_dev)),
        "test_records": int(len(splits["X_test"])),
        "splits": {
            "fit": _describe_split(splits["fit_data"]),
            "validation": _describe_split(splits["validation_data"]),
            "test": _describe_split(splits["test_data"]),
        },
        "features": list(X_dev.columns),
        "metrics": {
            key: test_metrics[key]
            for key in (
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Average Precision",
            )
        },
        "test_context": {
            key: test_metrics[key]
            for key in (
                "ROC-AUC",
                "Late rate (prevalence)",
                "AP lift over prevalence",
                "Confusion matrix [[TN, FP], [FN, TP]]",
            )
        },
        "validation_metrics": validation_metrics,
        "target": "is_late",
        "target_definition": "Delivery after estimated calendar date",
        "sklearn_version": sklearn.__version__,
    }

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)
        file.write("\n")

    logger.info("Metadados salvos em: %s", metadata_path)

    return metadata


def main(argv=None):
    """Ponto de entrada de linha de comando (python -m src.train)."""

    parser = argparse.ArgumentParser(
        description="Treina o modelo de atraso de entrega (Olist)."
    )
    parser.add_argument(
        "--data-dir",
        default=str(config.DATA_DIR),
        help="Pasta com os CSVs originais do Olist.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Pasta de saída. Grava <output-dir>/models/ e "
            "<output-dir>/reports/. Padrão: pastas do projeto."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold fixo. Se omitido, é escolhido na validação.",
    )
    args = parser.parse_args(argv)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        model_path = output_dir / "models" / config.MODEL_PATH.name
        metadata_path = output_dir / "reports" / config.METADATA_PATH.name
    else:
        model_path = config.MODEL_PATH
        metadata_path = config.METADATA_PATH

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    metadata = run_training(
        data_dir=args.data_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        threshold=args.threshold,
    )

    test = metadata["metrics"]
    context = metadata["test_context"]

    print(
        "\nTeste cronológico (threshold "
        f"{metadata['threshold']:.2f}): "
        f"AP={test['Average Precision']:.4f} "
        f"(prevalência {context['Late rate (prevalence)']:.4f}, "
        f"lift {context['AP lift over prevalence']:.2f}x) | "
        f"Precision={test['Precision']:.4f} "
        f"Recall={test['Recall']:.4f} F1={test['F1-score']:.4f}"
    )


if __name__ == "__main__":
    main()
