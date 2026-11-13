
from src.train import evaluate_at_threshold, train_model


def test_model_training(sample_data):

    X, y = sample_data

    model = train_model(X, y)

    assert hasattr(model, "predict_proba")

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)


def test_model_evaluation(sample_data):

    X, y = sample_data

    model = train_model(X, y)

    metrics, matrix, predictions, probabilities = (
        evaluate_at_threshold(
            model,
            X,
            y,
            threshold=0.15
        )
    )

    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-score" in metrics
    assert "Average Precision" in metrics

    assert matrix.shape == (2, 2)

    assert len(predictions) == len(y)
    assert len(probabilities) == len(y)

    assert all(value >= 0 for value in probabilities)
    assert all(value <= 1 for value in probabilities)
