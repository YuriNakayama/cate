from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AbstractFlow(ABC):
    origin: Path
    base: Path
    prediction: Path

    def make(self) -> None:
        for _, path in asdict(self).items():
            path.parent.mkdir(exist_ok=True, parents=True)

    def __post_init__(self) -> None:
        self.make()


@dataclass(frozen=True)
class Lenta(AbstractFlow):
    origin: Path = Path("/workspace/data/origin/lenta.csv")
    base: Path = Path("/workspace/data/base/lenta.csv")
    prediction: Path = Path("/workspace/data/prediction/lenta.csv")


@dataclass(frozen=True)
class Criteo(AbstractFlow):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/criteo.csv")
    prediction: Path = Path("/workspace/data/prediction/criteo.csv")


@dataclass(frozen=True)
class DataPath:
    lenta: Lenta = Lenta()
    criteo: Criteo = Criteo()


@dataclass(frozen=True)
class PathLinker:
    data: DataPath = DataPath()
