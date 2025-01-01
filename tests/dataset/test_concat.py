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


def test_concat_datasets(sample_dataset: cds.Dataset) -> None:
    df1 = pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    df2 = pl.DataFrame(
        {
            "feature1": [7, 8, 9],
            "feature2": [10, 11, 12],
            "target": [1, 0, 1],
            "weight": [0.4, 0.5, 0.6],
        }
    )
    dataset1 = cds.Dataset(
        df1,
        sample_dataset.x_columns,
        sample_dataset.y_columns,
        sample_dataset.w_columns,
    )
    dataset2 = cds.Dataset(
        df2,
        sample_dataset.x_columns,
        sample_dataset.y_columns,
        sample_dataset.w_columns,
    )

    concatenated_dataset = cds.concat([dataset1, dataset2])

    expected_df = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 7, 8, 9],
            "feature2": [4, 5, 6, 10, 11, 12],
            "target": [0, 1, 0, 1, 0, 1],
            "weight": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    pt.assert_frame_equal(concatenated_dataset.to_frame(), expected_df)
    assert concatenated_dataset.x_columns == sample_dataset.x_columns
    assert concatenated_dataset.y_columns == sample_dataset.y_columns
    assert concatenated_dataset.w_columns == sample_dataset.w_columns


def test_concat_empty_list() -> None:
    with pytest.raises(ValueError):
        cds.concat([])


def test_concat_single_dataset(sample_dataset: cds.Dataset) -> None:
    concatenated_dataset = cds.concat([sample_dataset])

    pt.assert_frame_equal(concatenated_dataset.to_frame(), sample_dataset.to_frame())
    assert concatenated_dataset.x_columns == sample_dataset.x_columns
    assert concatenated_dataset.y_columns == sample_dataset.y_columns
    assert concatenated_dataset.w_columns == sample_dataset.w_columns
