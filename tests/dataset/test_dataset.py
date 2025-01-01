import os
import shelve
from pathlib import Path

import polars as pl
import pytest

from cate.dataset import Dataset


def test_dataset_init_valid_columns() -> None:
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

    dataset = Dataset(df, x_columns, y_columns, w_columns)

    assert dataset.x_columns == x_columns
    assert dataset.y_columns == y_columns
    assert dataset.w_columns == w_columns
    assert dataset.X.shape == (3, 2)
    assert dataset.y.shape == (3, 1)
    assert dataset.w.shape == (3, 1)


def test_dataset_init_invalid_x_columns() -> None:
    df = pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    x_columns = ["feature1", "feature3"]  # feature3 does not exist
    y_columns = ["target"]
    w_columns = ["weight"]

    with pytest.raises(ValueError, match="x columns {'feature3'} do not exist in df."):
        Dataset(df, x_columns, y_columns, w_columns)

    x_columns = ["feature1", "feature2"]
    y_columns = ["target2"]  # target2 does not exist
    w_columns = ["weight"]

    with pytest.raises(ValueError, match="x columns {'target2'} do not exist in df."):
        Dataset(df, x_columns, y_columns, w_columns)

    x_columns = ["feature1", "feature2"]
    y_columns = ["target"]
    w_columns = ["weight2"]  # weight2 does not exist

    with pytest.raises(ValueError, match="x columns {'weight2'} do not exist in df."):
        Dataset(df, x_columns, y_columns, w_columns)


