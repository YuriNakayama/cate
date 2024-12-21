from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AbstractLink(ABC):
    origin: Path
    base: Path
    prediction: Path
    output: Path

    def _make_paths(self) -> None:
        for _, path in asdict(self).items():
            if isinstance(path, Path):
                if path.suffix:
                    target_dir = path.parent
                else:
                    target_dir = path
                target_dir.mkdir(exist_ok=True, parents=True)

    def __post_init__(self) -> None:
        self._make_paths()


@dataclass(frozen=True)
class Test(AbstractLink):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/test")
    prediction: Path = Path("/workspace/data/prediction/test")
    output: Path = Path("/workspace/data/output/test")


@dataclass(frozen=True)
class Lenta(AbstractLink):
    origin: Path = Path("/workspace/data/origin/lenta.csv")
    base: Path = Path("/workspace/data/base/lenta")
    prediction: Path = Path("/workspace/data/prediction/lenta")
    output: Path = Path("/workspace/data/output/lenta")


@dataclass(frozen=True)
class Criteo(AbstractLink):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/criteo")
    prediction: Path = Path("/workspace/data/prediction/criteo")
    output: Path = Path("/workspace/data/output/criteo")


@dataclass(frozen=True)
class Hillstorm(AbstractLink):
    origin: Path = Path("/workspace/data/origin/hillstorm.csv")
    base: Path = Path("/workspace/data/base/hillstorm")
    prediction: Path = Path("/workspace/data/prediction/hillstorm")
    output: Path = Path("/workspace/data/output/hillstorm")


@dataclass(frozen=True)
class Megafon(AbstractLink):
    origin: Path = Path("/workspace/data/origin/megafon.csv")
    base: Path = Path("/workspace/data/base/megafon")
    prediction: Path = Path("/workspace/data/prediction/megafon")
    output: Path = Path("/workspace/data/output/megafon")


@dataclass(frozen=True)
class X5(AbstractLink):
    origin: Path = Path("/workspace/data/origin/x5.csv")
    base: Path = Path("/workspace/data/base/x5")
    prediction: Path = Path("/workspace/data/prediction/x5")
    output: Path = Path("/workspace/data/output/x5")


def path_linker(link_name: str) -> AbstractLink:
    if link_name == "test":
        return Test()
    elif link_name == "lenta":
        return Lenta()
    elif link_name == "criteo":
        return Criteo()
    elif link_name == "hillstorm":
        return Hillstorm()
    elif link_name == "megafon":
        return Megafon()
    elif link_name == "x5":
        return X5()
    else:
        raise ValueError(f"Unknown link name: {link_name}")
