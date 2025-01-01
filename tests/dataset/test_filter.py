import polars as pl
import polars.testing as pt
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


def test_filter(sample_dataset: cds.Dataset) -> None:
    flgs = [
        pl.Series([True, False, True]),
        pl.Series([True, True, False]),
    ]
    filtered_dataset = cds.filter(sample_dataset, flgs)

    expected_df = pl.DataFrame(
        {
            "feature1": [1],
            "feature2": [4],
            "target": [0],
            "weight": [0.1],
        }
    )
    pt.assert_frame_equal(filtered_dataset.to_frame(), expected_df)
    assert filtered_dataset.x_columns == sample_dataset.x_columns
    assert filtered_dataset.y_columns == sample_dataset.y_columns
    assert filtered_dataset.w_columns == sample_dataset.w_columns


def test_filter_all_false(sample_dataset: cds.Dataset) -> None:
    flgs = [
        pl.Series([False, False, False]),
    ]
    filtered_dataset = cds.filter(sample_dataset, flgs)

    expected_df = pl.DataFrame(
        {
            "feature1": [],
            "feature2": [],
            "target": [],
            "weight": [],
        },
        schema=sample_dataset.to_frame().schema,
    )
    pt.assert_frame_equal(filtered_dataset.to_frame(), expected_df)
    assert filtered_dataset.x_columns == sample_dataset.x_columns
    assert filtered_dataset.y_columns == sample_dataset.y_columns
    assert filtered_dataset.w_columns == sample_dataset.w_columns
