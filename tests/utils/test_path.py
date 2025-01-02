from pathlib import Path

import pytest

from cate.utils.path import PathLink, dataset_type, path_linker


@pytest.mark.parametrize(
    "dataset", ["lenta", "criteo", "hillstorm", "megafon", "x5", "test"]
)
def test_path_linker(dataset: dataset_type) -> None:
    path_link = path_linker(dataset)

    assert isinstance(path_link, PathLink)
    assert path_link.dataset == dataset

    base_path = Path("/workspace")
    assert path_link.lake == base_path / "data/lake" / f"{dataset}.parquet"
    assert path_link.cleansing == base_path / "data/processed" / dataset
    assert path_link.mart == base_path / "data/mart" / dataset
    assert path_link.prediction == base_path / "data/prediction" / dataset
    assert path_link.output == base_path / "output" / dataset

    # Check if the directories are created
    for path in [
        path_link.cleansing,
        path_link.mart,
        path_link.prediction,
        path_link.output,
    ]:
        assert path.exists()
        assert path.is_dir()

    # Check if the lake file is created
    assert path_link.lake.exists()
    assert path_link.lake.is_file()
