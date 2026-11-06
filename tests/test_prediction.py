
import pytest

from src.train import train_model, save_model
from src.predict import predict_late, load_model


def test_prediction(sample_data):

    X, y = sample_data

    model = train_model(X, y)

    results = predict_late(
        model,
        X.head(10),
        threshold=0.15
    )

    assert len(results) == 10

    assert "late_probability" in results.columns
    assert "predicted_is_late" in results.columns

    assert results["predicted_is_late"].isin([0, 1]).all()


def test_invalid_threshold(sample_data):

    X, y = sample_data

    model = train_model(X, y)

    with pytest.raises(ValueError):
        predict_late(
            model,
            X.head(10),
            threshold=1.5
        )


def test_missing_features(sample_data):

    X, y = sample_data

    model = train_model(X, y)

    X_incomplete = X.drop(columns=["purchase_hour"])

    with pytest.raises(ValueError):
        predict_late(
            model,
            X_incomplete,
            threshold=0.15
        )


def test_save_and_load_model(sample_data, tmp_path):

    X, y = sample_data

    model = train_model(X, y)

    model_path = tmp_path / "test_model.joblib"

    save_model(model, model_path)

    loaded_model = load_model(model_path)

    assert hasattr(loaded_model, "predict_proba")
