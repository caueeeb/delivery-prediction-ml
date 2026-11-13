"""Configurações centrais do projeto.

Este módulo é a única fonte de verdade para caminhos, parâmetros do modelo
e threshold de classificação. A API, o script de treino e os testes leem
os valores daqui (ou do arquivo de metadados gerado no treino), em vez de
repetir constantes em vários arquivos.
"""

from __future__ import annotations

import json
from pathlib import Path

# Raiz do projeto (pasta que contém src/, models/, reports/ ...)
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Diretórios padrão
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# Artefatos do modelo
MODEL_PATH = MODELS_DIR / "delivery_delay_logistic_regression.joblib"
METADATA_PATH = REPORTS_DIR / "model_metadata.json"

# Reprodutibilidade
RANDOM_STATE = 42

# Hiperparâmetros do modelo selecionado (Regressão Logística)
MODEL_PARAMS = {
    "C": 10.0,
    "class_weight": None,
    "max_iter": 1500,
    "random_state": RANDOM_STATE,
}

# Threshold usado quando nenhum metadado está disponível. Em produção local
# (API) o valor vem de reports/model_metadata.json, gerado pelo treino.
DEFAULT_THRESHOLD = 0.15

# Grade de thresholds avaliada na validação para escolher o ponto de corte
THRESHOLD_GRID_START = 0.05
THRESHOLD_GRID_STOP = 0.95
THRESHOLD_GRID_STEP = 0.05


def load_metadata(path: Path | str = METADATA_PATH) -> dict:
    """Lê o arquivo de metadados do modelo.

    Retorna um dicionário vazio se o arquivo não existir, para que
    ferramentas auxiliares (por exemplo, testes de unidade com modelos
    sintéticos) continuem funcionando sem o artefato real.
    """

    path = Path(path)

    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_threshold(path: Path | str = METADATA_PATH) -> float:
    """Retorna o threshold salvo nos metadados ou o padrão do projeto."""

    metadata = load_metadata(path)

    return float(metadata.get("threshold", DEFAULT_THRESHOLD))
