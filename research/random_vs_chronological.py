"""Diagnóstico H3 do protocolo: avaliação aleatória vs. cronológica.

Compara a estimativa de desempenho obtida com partições aleatórias
estratificadas (64% / 16% / 20%, mesmas proporções do split cronológico)
com a obtida na partição cronológica original. A avaliação aleatória mistura
datas e, portanto, NÃO representa um cenário real de previsão do futuro: é
usada apenas como diagnóstico.

Uso (a partir da raiz do repositório)::

    python research/random_vs_chronological.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.preprocessing import (  # noqa: E402
    EXCLUDED_COLUMNS,
    build_feature_base,
    make_temporal_splits,
)
from src.train import select_threshold, summarize_metrics, train_model  # noqa: E402

SEEDS = range(10)
OUTPUT = Path(__file__).resolve().parent / "reports" / "random_vs_chronological.csv"

METRICS = ["Average Precision", "AP lift over prevalence", "ROC-AUC",
           "Precision", "Recall", "F1-score", "Late rate (prevalence)"]


def evaluate_partition(X_fit, y_fit, X_val, y_val, X_test, y_test):
    """Mesmo protocolo do treino: threshold na validação, teste uma vez."""

    model = train_model(X_fit, y_fit)
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])

    final = train_model(pd.concat([X_fit, X_val]), pd.concat([y_fit, y_val]))
    result = summarize_metrics(final, X_test, y_test, threshold)
    result["threshold"] = threshold

    return result


def main():
    base = build_feature_base(config.DATA_DIR)

    rows = []

    # Cronológico (a partição original do projeto)
    splits = make_temporal_splits(base)
    chrono = evaluate_partition(
        splits["X_fit"], splits["y_fit"], splits["X_val"], splits["y_val"],
        splits["X_test"], splits["y_test"],
    )
    rows.append({"evaluation": "chronological", "seed": "-", **chrono})

    # Aleatório estratificado, várias sementes
    X = base.drop(columns=EXCLUDED_COLUMNS)
    y = base["is_late"]

    for seed in SEEDS:
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=seed)

        result = evaluate_partition(X_fit, y_fit, X_val, y_val, X_test, y_test)
        rows.append({"evaluation": "random_stratified", "seed": seed, **result})

    table = pd.DataFrame(rows)
    keep = ["evaluation", "seed", "threshold", *METRICS]
    table[keep].to_csv(OUTPUT, index=False)

    summary = (
        table.groupby("evaluation")[METRICS + ["threshold"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary.T.to_string())
    print(f"\nSalvo em {OUTPUT}")


if __name__ == "__main__":
    main()
