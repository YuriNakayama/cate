from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AbstractFlow(ABC):
    raw: Path
    shaped: Path

    def make(self) -> None:
        for _, path in asdict(self).items():
            path.parent.mkdir(exist_ok=True, parents=True)

    def __post_init__(self) -> None:
        self.make()


@dataclass(frozen=True)
class Lenta(AbstractFlow):
    raw: Path = Path("/workspace/data/Raw/lenta.csv")
    shaped: Path = Path("/workspace/data/Shaped/lenta.csv")


@dataclass(frozen=True)
class Criteo(AbstractFlow):
    raw: Path = Path("/workspace/data/Raw/criteo.csv")
    shaped: Path = Path("/workspace/data/Shaped/criteo.csv")


@dataclass
class DataPath:
    lenta: Lenta = Lenta()
    criteo: Criteo = Criteo()


@dataclass
class PathLinker:
    data: DataPath = DataPath()
