from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AbstractFlow(ABC):
    origin: Path
    base: Path
    prediction: Path

    def _make_paths(self) -> None:
        for attr, path in asdict(self).items():
            if isinstance(path, Path):
                try:
                    if path.suffix:
                        target_dir = path.parent
                    else:
                        target_dir = path
                    target_dir.mkdir(exist_ok=True, parents=True)

                except Exception as e:
                    raise RuntimeError(
                        f"Error creating directories for {attr} ({path}): {e}"
                    )

    def __post_init__(self) -> None:
        self._make_paths()


@dataclass(frozen=True)
class Lenta(AbstractFlow):
    origin: Path = Path("/workspace/data/origin/lenta.csv")
    base: Path = Path("/workspace/data/base/lenta")
    prediction: Path = Path("/workspace/data/prediction/lenta")


@dataclass(frozen=True)
class Criteo(AbstractFlow):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/criteo")
    prediction: Path = Path("/workspace/data/prediction/criteo")


@dataclass(frozen=True)
class Test(AbstractFlow):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/test")
    prediction: Path = Path("/workspace/data/prediction/test")


@dataclass(frozen=True)
class DataPath:
    lenta: Lenta = Lenta()
    criteo: Criteo = Criteo()
    test: Test = Test()


@dataclass(frozen=True)
class PathLinker:
    data: DataPath = DataPath()
