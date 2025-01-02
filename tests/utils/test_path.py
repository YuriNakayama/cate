from pathlib import Path

import pytest

from cate.utils.path import DataLayer, dataset_type, path_linker


@pytest.mark.parametrize(
    "dataset, layer, expected",
    [
        ("lenta", DataLayer.lake, Path("data/lake/lenta.parquet")),
        ("criteo", DataLayer.cleansing, Path("data/processed/criteo")),
        ("hillstorm", DataLayer.mart, Path("data/mart/hillstorm")),
        ("megafon", DataLayer.prediction, Path("data/prediction/megafon")),
        ("x5", DataLayer.output, Path("output/x5")),
    ],
)
def test_path_linker(
    dataset: dataset_type,
    layer: Path,
    expected: Path,
):
    result = path_linker(dataset, layer)
    assert result == expected
