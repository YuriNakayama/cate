import shelve
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pt
import pytest
from numpy.testing import assert_array_equal

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

    dataset = cds.Dataset(df, x_columns, y_columns, w_columns)

    assert dataset.x_columns == x_columns
    assert dataset.y_columns == y_columns
    assert dataset.w_columns == w_columns
    assert dataset.X.shape == (3, 2)
    assert dataset.y.shape == (3,)
    assert dataset.w.shape == (3,)


def test_dataset_init_invalid_columns() -> None:
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

    with pytest.raises(ValueError, match="X columns {'feature3'} do not exist in df."):
        cds.Dataset(df, x_columns, y_columns, w_columns)

    x_columns = ["feature1", "feature2"]
    y_columns = ["target2"]  # target2 does not exist
    w_columns = ["weight"]

    with pytest.raises(ValueError, match="y columns {'target2'} do not exist in df."):
        cds.Dataset(df, x_columns, y_columns, w_columns)

    x_columns = ["feature1", "feature2"]
    y_columns = ["target"]
    w_columns = ["weight2"]  # weight2 does not exist

    with pytest.raises(ValueError, match="z columns {'weight2'} do not exist in df."):
        cds.Dataset(df, x_columns, y_columns, w_columns)


def test_save_creates_files(sample_dataset: cds.Dataset, tmp_path: Path) -> None:
    save_path = tmp_path / "dataset"
    sample_dataset.save(save_path)

    assert (save_path / "data.parquet").exists()
    assert list(save_path.glob("meta*"))


def test_save_overwrites_existing_directory(
    sample_dataset: cds.Dataset, tmp_path: Path
) -> None:
    save_path = tmp_path / "dataset"
    save_path.mkdir()
    (save_path / "dummy_file.txt").write_text("This is a dummy file.")

    sample_dataset.save(save_path)

    assert not (save_path / "dummy_file.txt").exists()
    assert (save_path / "data.parquet").exists()
    assert list(save_path.glob("meta*"))


def test_save_meta_content(sample_dataset: cds.Dataset, tmp_path: Path) -> None:
    save_path = tmp_path / "dataset"
    sample_dataset.save(save_path)

    with shelve.open(save_path / "meta") as shelf:
        assert shelf["x_columns"] == sample_dataset.x_columns
        assert shelf["y_columns"] == sample_dataset.y_columns
        assert shelf["w_columns"] == sample_dataset.w_columns


def test_load_valid_dataset(sample_dataset: cds.Dataset, tmp_path: Path) -> None:
    save_path = tmp_path / "dataset"
    sample_dataset.save(save_path)

    loaded_dataset = cds.Dataset.load(save_path)

    assert loaded_dataset.x_columns == sample_dataset.x_columns
    assert loaded_dataset.y_columns == sample_dataset.y_columns
    assert_array_equal(loaded_dataset.X, sample_dataset.X)
    assert_array_equal(loaded_dataset.y, sample_dataset.y)
    assert_array_equal(loaded_dataset.w, sample_dataset.w)


def test_load_missing_data_file(tmp_path: Path) -> None:
    save_path = tmp_path / "dataset"
    save_path.mkdir()
    with shelve.open(save_path / "meta") as shelf:
        shelf["x_columns"] = ["feature1", "feature2"]
        shelf["y_columns"] = ["target"]
        shelf["w_columns"] = ["weight"]

    with pytest.raises(FileNotFoundError):
        cds.Dataset.load(save_path)


def test_load_missing_meta_file(tmp_path: Path) -> None:
    save_path = tmp_path / "dataset"
    save_path.mkdir()
    df = pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    df.write_parquet(save_path / "data.parquet")

    with pytest.raises(FileNotFoundError):
        cds.Dataset.load(save_path)


def test_dataset_len(sample_dataset: cds.Dataset) -> None:
    assert len(sample_dataset) == 3

    empty_df = pl.DataFrame(
        {
            "feature1": [],
            "feature2": [],
            "target": [],
            "weight": [],
        }
    )
    empty_dataset = cds.Dataset(
        empty_df, ["feature1", "feature2"], ["target"], ["weight"]
    )
    assert len(empty_dataset) == 0


def test_to_frame(sample_dataset: cds.Dataset) -> None:
    df = sample_dataset.to_frame()
    expected_df = pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    pt.assert_frame_equal(df, expected_df)


def test_getitem_with_valid_indices(sample_dataset: cds.Dataset) -> None:
    subset = sample_dataset[[0, 2]]
    expected_df = pl.DataFrame(
        {
            "feature1": [1, 3],
            "feature2": [4, 6],
            "target": [0, 0],
            "weight": [0.1, 0.3],
        }
    )
    pt.assert_frame_equal(subset.to_frame(), expected_df)
    assert subset.x_columns == sample_dataset.x_columns
    assert subset.y_columns == sample_dataset.y_columns
    assert subset.w_columns == sample_dataset.w_columns


def test_getitem_with_empty_indices(sample_dataset: cds.Dataset) -> None:
    subset = sample_dataset[[]]
    expected_df = pl.DataFrame(
        {
            "feature1": [],
            "feature2": [],
            "target": [],
            "weight": [],
        },
        schema=sample_dataset.to_frame().schema,
    )
    pt.assert_frame_equal(subset.to_frame(), expected_df)
    assert subset.x_columns == sample_dataset.x_columns
    assert subset.y_columns == sample_dataset.y_columns
    assert subset.w_columns == sample_dataset.w_columns


def test_getitem_with_numpy_array(sample_dataset: cds.Dataset) -> None:
    indices = np.array([1])
    subset = sample_dataset[indices]
    expected_df = pl.DataFrame(
        {
            "feature1": [2],
            "feature2": [5],
            "target": [1],
            "weight": [0.2],
        }
    )
    pt.assert_frame_equal(subset.to_frame(), expected_df)
    assert subset.x_columns == sample_dataset.x_columns
    assert subset.y_columns == sample_dataset.y_columns
    assert subset.w_columns == sample_dataset.w_columns
