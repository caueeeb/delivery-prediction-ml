
import pandas as pd
import pytest

from src.preprocessing import load_raw_tables, make_temporal_splits


def test_temporal_splits():

    total = 20

    dates = pd.date_range(
        "2020-01-01",
        periods=total,
        freq="h"
    )

    base = pd.DataFrame({
        "order_id": [f"order_{i}" for i in range(total)],
        "customer_id": [f"customer_{i}" for i in range(total)],
        "order_purchase_timestamp": dates,
        "order_estimated_delivery_date": dates + pd.Timedelta(days=5),
        "delivery_delay_days": [0] * total,
        "is_late": [i % 2 for i in range(total)],
        "customer_city": ["sao paulo"] * total,
        "customer_state": ["SP"] * total,
        "purchase_hour": [i % 24 for i in range(total)]
    })

    splits = make_temporal_splits(base)

    assert splits["X_fit"].shape[0] == 12
    assert splits["X_val"].shape[0] == 4
    assert splits["X_test"].shape[0] == 4

    assert (
        splits["fit_data"]["order_purchase_timestamp"].max()
        <
        splits["validation_data"]["order_purchase_timestamp"].min()
    )

    assert (
        splits["validation_data"]["order_purchase_timestamp"].max()
        <
        splits["test_data"]["order_purchase_timestamp"].min()
    )


def test_missing_dataset_files(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_raw_tables(tmp_path)
