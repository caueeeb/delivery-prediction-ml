"""Garante que o modelo versionado, os metadados e a API estão coerentes."""

import joblib
import pytest
from fastapi.testclient import TestClient
from test_api import SAMPLE_ORDER

from src import api, config

client = TestClient(api.app)


@pytest.fixture(scope="module")
def metadata():
    return config.load_metadata()


@pytest.fixture(scope="module")
def model():
    return joblib.load(config.MODEL_PATH)


def test_metadata_features_match_model(metadata, model):
    assert metadata["features"] == list(model.feature_names_in_)


def test_metadata_parameters_match_model(metadata, model):
    parameters = model.named_steps["classifier"].get_params()

    for key, value in metadata["parameters"].items():
        assert parameters[key] == value


def test_threshold_has_single_source_of_truth(metadata):
    assert api.get_threshold() == metadata["threshold"]
    assert config.get_threshold() == metadata["threshold"]


def test_model_info_is_read_from_model_not_hardcoded(metadata, model):
    info = client.get("/model-info").json()

    assert info["features_count"] == len(model.feature_names_in_)
    assert info["features"] == list(model.feature_names_in_)
    assert info["C"] == model.named_steps["classifier"].C
    assert info["threshold"] == metadata["threshold"]
    assert info["training_records"] == metadata["training_records"]


def test_reported_test_metrics_are_internally_consistent(metadata):
    metrics = metadata["metrics"]
    context = metadata["test_context"]

    (tn, fp), (fn, tp) = context["Confusion matrix [[TN, FP], [FN, TP]]"]

    assert tp / (tp + fp) == pytest.approx(metrics["Precision"])
    assert tp / (tp + fn) == pytest.approx(metrics["Recall"])
    assert (tp + tn) / (tp + tn + fp + fn) == pytest.approx(
        metrics["Accuracy"]
    )
    assert context["Late rate (prevalence)"] == pytest.approx(
        (tp + fn) / (tp + tn + fp + fn)
    )
    assert context["AP lift over prevalence"] == pytest.approx(
        metrics["Average Precision"] / context["Late rate (prevalence)"]
    )


def test_health_degrades_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config, "MODEL_PATH", tmp_path / "does_not_exist.joblib"
    )
    api.get_model.cache_clear()

    try:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "degraded",
            "model_loaded": False,
        }

        prediction = client.post("/predict", json=SAMPLE_ORDER)

        assert prediction.status_code == 503
    finally:
        api.get_model.cache_clear()
