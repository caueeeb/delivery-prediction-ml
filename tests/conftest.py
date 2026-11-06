
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():

    rng = np.random.default_rng(42)

    total = 100

    X = pd.DataFrame({
        "customer_state": np.tile(
            ["SP", "RJ", "MG", "BA"], 25
        ),
        "total_freight": rng.uniform(5, 100, total),
        "item_count": rng.integers(1, 5, total),
        "purchase_hour": rng.integers(0, 24, total)
    })

    y = pd.Series(
        [0, 1] * 50,
        name="is_late"
    )

    return X, y
