from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class DataLayer:
    lake: Path = Path("data/lake")
    cleansing: Path = Path("data/processed")
    mart: Path = Path("data/mart")
    prediction: Path = Path("data/prediction")
    output: Path = Path("output")


dataset_type = Literal["lenta", "criteo", "hillstorm", "megafon", "x5"]


@dataclass
class PathLink:
    dataset: dataset_type
    layer: Path

    @staticmethod
    def _make_paths(path: Path) -> None:
        if not path.exists():
            if path.suffix:
                target_dir = path.parent
            else:
                target_dir = path
            target_dir.mkdir(exist_ok=True, parents=True)

    def __post_init__(self) -> None:
        if self.layer == Path("data/lake"):
            self.link = self.layer / f"{self.dataset}.parquet"
        else:
            self.link = self.layer / self.dataset
        self._make_paths(self.link)


def path_linker(dataset: dataset_type, layer: Path) -> Path:
    path_link = PathLink(dataset, layer)
    return path_link.link
