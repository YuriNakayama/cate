from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from attrs import field

dataset_type = Literal["lenta", "criteo", "hillstorm", "megafon", "x5", "test"]


@dataclass
class PathLink:
    dataset: dataset_type

    _base: Path = Path("/workspace")

    _lake: Path = Path("data/lake")
    _cleansing: Path = Path("data/processed")
    _mart: Path = Path("data/mart")
    _prediction: Path = Path("data/prediction")
    _output: Path = Path("output")

    lake: Path = field(init=False)
    cleansing: Path = field(init=False)
    mart: Path = field(init=False)
    prediction: Path = field(init=False)
    output: Path = field(init=False)

    @staticmethod
    def _make_path(path: Path) -> None:
        if not path.exists():
            if path.suffix:
                target_dir = path.parent
            else:
                target_dir = path
            target_dir.mkdir(exist_ok=True, parents=True)

    def __post_init__(self) -> None:
        self.lake = self._base / self._lake / f"{self.dataset}.parquet"
        self.cleansing = self._base / self._cleansing / self.dataset
        self.mart = self._base / self._mart / self.dataset
        self.prediction = self._base / self._prediction / self.dataset
        self.output = self._base / self._output / self.dataset

        for path in (
            self.lake,
            self.cleansing,
            self.mart,
            self.prediction,
            self.output,
        ):
            self._make_path(path)


def path_linker(dataset: dataset_type) -> PathLink:
    path_link = PathLink(dataset)
    return path_link
