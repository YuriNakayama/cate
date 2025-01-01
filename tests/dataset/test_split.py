import polars as pl
import pytest

import cate.dataset as cds


@pytest.fixture
def sample_dataset() -> cds.Dataset:
    df = pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    x_columns = ["feature1", "feature2"]
    y_columns = ["target"]
    w_columns = ["weight"]
    return cds.Dataset(df, x_columns, y_columns, w_columns)


def test_split_with_test_frac(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_frac=0.33, random_state=42)

    assert len(train_ds) == 2
    assert len(test_ds) == 1

    assert set(train_ds.X.columns) == set(sample_dataset.X.columns)
    assert set(test_ds.X.columns) == set(sample_dataset.X.columns)
    assert set(train_ds.y.columns) == set(sample_dataset.y.columns)
    assert set(test_ds.y.columns) == set(sample_dataset.y.columns)
    assert set(train_ds.w.columns) == set(sample_dataset.w.columns)
    assert set(test_ds.w.columns) == set(sample_dataset.w.columns)


def test_split_with_test_n(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_n=1, random_state=42)

    assert len(train_ds) == 2
    assert len(test_ds) == 1

    assert set(train_ds.X.columns) == set(sample_dataset.X.columns)
    assert set(test_ds.X.columns) == set(sample_dataset.X.columns)
    assert set(train_ds.y.columns) == set(sample_dataset.y.columns)
    assert set(test_ds.y.columns) == set(sample_dataset.y.columns)
    assert set(train_ds.w.columns) == set(sample_dataset.w.columns)
    assert set(test_ds.w.columns) == set(sample_dataset.w.columns)


def test_split_with_test_frac_zero(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_frac=0, random_state=42)

    assert len(train_ds) == 3
    assert len(test_ds) == 0


def test_split_with_test_n_zero(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_n=0, random_state=42)

    assert len(train_ds) == 3
    assert len(test_ds) == 0


def test_split_with_test_frac_one(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_frac=1, random_state=42)

    assert len(train_ds) == 0
    assert len(test_ds) == 3


def test_split_with_test_n_one(sample_dataset: cds.Dataset) -> None:
    train_ds, test_ds = cds.split(sample_dataset, test_n=3, random_state=42)

    assert len(train_ds) == 0
    assert len(test_ds) == 3
