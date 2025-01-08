from pathlib import Path

import pytest

from cate.utils.path import PathLink, dataset_type, path_linker


@pytest.mark.parametrize(
    "dataset",
    [
        "lenta",
        "criteo",
        "hillstorm",
        "megafon",
        "x5",
        "test",
    ],
)
def test_path_linker(tmp_path: Path, dataset: dataset_type) -> None:
    path_link = path_linker(dataset, tmp_path)

    assert isinstance(path_link, PathLink)
    assert path_link.dataset == dataset

    if dataset == "test":
        assert path_link.lake == tmp_path / "data/lake" / "criteo.parquet"
    else:
        assert path_link.lake == tmp_path / "data/lake" / f"{dataset}.parquet"
    assert path_link.cleansing == tmp_path / "data/processed" / dataset
    assert path_link.mart == tmp_path / "data/mart" / dataset
    assert path_link.prediction == tmp_path / "data/prediction" / dataset

    # Check if the directories are created
    for _path in [
        path_link.cleansing,
        path_link.mart,
        path_link.prediction,
    ]:
        assert _path.exists()
        assert _path.is_dir()

    # Check if the lake file is created
    assert path_link.lake.parent.exists()
    assert path_link.lake.parent.is_dir()
