import json

import numpy as np

from src import config
from src.train import main, run_training, select_threshold


def test_select_threshold_picks_best_f1():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array(
        [0.05, 0.10, 0.20, 0.30, 0.60, 0.70, 0.80, 0.90]
    )

    threshold = select_threshold(y, probabilities)

    # Qualquer corte entre 0.30 e 0.60 separa as classes perfeitamente
    assert 0.30 < threshold <= 0.60


def test_training_cli_writes_artifacts(olist_dir, tmp_path):
    output_dir = tmp_path / "artifacts"

    main([
        "--data-dir", str(olist_dir),
        "--output-dir", str(output_dir),
    ])

    model_path = output_dir / "models" / config.MODEL_PATH.name
    metadata_path = output_dir / "reports" / config.METADATA_PATH.name

    assert model_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["algorithm"] == "Logistic Regression"
    assert 0 < metadata["threshold"] < 1
    assert metadata["target"] == "is_late"

    # O treino final usa treino + validação; o teste fica de fora
    splits = metadata["splits"]

    assert (
        metadata["training_records"]
        == splits["fit"]["rows"] + splits["validation"]["rows"]
    )
    assert metadata["test_records"] == splits["test"]["rows"]

    # Partições cronológicas, sem sobreposição de datas
    assert splits["fit"]["end"] <= splits["validation"]["start"]
    assert splits["validation"]["end"] <= splits["test"]["start"]

    for key in ("Precision", "Recall", "F1-score", "Average Precision"):
        assert 0 <= metadata["metrics"][key] <= 1

    # O baseline de acaso (prevalência) acompanha as métricas
    context = metadata["test_context"]

    assert 0 < context["Late rate (prevalence)"] < 1
    assert context["AP lift over prevalence"] > 0


def test_fixed_threshold_is_respected(olist_dir, tmp_path):
    metadata = run_training(
        data_dir=olist_dir,
        model_path=tmp_path / "m.joblib",
        metadata_path=tmp_path / "meta.json",
        threshold=0.3,
    )

    assert metadata["threshold"] == 0.3
    assert metadata["threshold_selection"] == "definido manualmente"
