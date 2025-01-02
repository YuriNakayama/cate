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


def test_sample_with_n(sample_dataset: cds.Dataset) -> None:
    sampled_dataset = cds.sample(sample_dataset, n=2, random_state=42)
    assert len(sampled_dataset) == 2
    assert set(sampled_dataset.x_columns) == set(sample_dataset.x_columns)
    assert set(sampled_dataset.y_columns) == set(sample_dataset.y_columns)
    assert set(sampled_dataset.w_columns) == set(sample_dataset.w_columns)


def test_sample_with_frac(sample_dataset: cds.Dataset) -> None:
    sampled_dataset = cds.sample(sample_dataset, frac=0.5, random_state=42)
    assert len(sampled_dataset) == 1
    assert set(sampled_dataset.x_columns) == set(sample_dataset.x_columns)
    assert set(sampled_dataset.y_columns) == set(sample_dataset.y_columns)
    assert set(sampled_dataset.w_columns) == set(sample_dataset.w_columns)


def test_sample_with_n_zero(sample_dataset: cds.Dataset) -> None:
    sampled_dataset = cds.sample(sample_dataset, n=0)
    assert len(sampled_dataset) == 0
    assert set(sampled_dataset.x_columns) == set(sample_dataset.x_columns)
    assert set(sampled_dataset.y_columns) == set(sample_dataset.y_columns)
    assert set(sampled_dataset.w_columns) == set(sample_dataset.w_columns)


def test_sample_with_frac_zero(sample_dataset: cds.Dataset) -> None:
    sampled_dataset = cds.sample(sample_dataset, frac=0.0)
    assert len(sampled_dataset) == 0
    assert set(sampled_dataset.x_columns) == set(sample_dataset.x_columns)
    assert set(sampled_dataset.y_columns) == set(sample_dataset.y_columns)
    assert set(sampled_dataset.w_columns) == set(sample_dataset.w_columns)
