import shutil
from pathlib import Path
from typing import Any, Generator

import pytest

from cate.utils.path import PathLink, dataset_type, path_linker


@pytest.fixture
def path() -> Generator[Path, Any, None]:
    yield Path("/workspace/pytest")
    shutil.rmtree("/workspace/pytest")


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
def test_path_linker(path: Path, dataset: dataset_type) -> None:
    path_link = path_linker(dataset, path)

    assert isinstance(path_link, PathLink)
    assert path_link.dataset == dataset

    assert path_link.lake == path / "data/lake" / f"{dataset}.parquet"
    assert path_link.cleansing == path / "data/processed" / dataset
    assert path_link.mart == path / "data/mart" / dataset
    assert path_link.prediction == path / "data/prediction" / dataset
    assert path_link.output == path / "output" / dataset

    # Check if the directories are created
    for _path in [
        path_link.cleansing,
        path_link.mart,
        path_link.prediction,
        path_link.output,
    ]:
        assert _path.exists()
        assert _path.is_dir()

    # Check if the lake file is created
    assert path_link.lake.parent.exists()
    assert path_link.lake.parent.is_dir()
