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
    raw: Path = Path("/workspace/data/raw/lenta.csv")
    shaped: Path = Path("/workspace/data/shaped/lenta.csv")


@dataclass(frozen=True)
class Criteo(AbstractFlow):
    raw: Path = Path("/workspace/data/raw/criteo.csv")
    shaped: Path = Path("/workspace/data/shaped/criteo.csv")


@dataclass(frozen=True)
class DataPath:
    lenta: Lenta = Lenta()
    criteo: Criteo = Criteo()


@dataclass(frozen=True)
class PathLinker:
    data: DataPath = DataPath()
