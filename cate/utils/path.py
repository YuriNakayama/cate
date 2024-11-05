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
class Test(AbstractLink):
    origin: Path = Path("/workspace/data/origin/criteo.csv")
    base: Path = Path("/workspace/data/base/test")
    prediction: Path = Path("/workspace/data/prediction/test")
    output: Path = Path("/workspace/data/output/test")


def path_linker(link_name: str) -> AbstractLink:
    if link_name == "lenta":
        return Lenta()
    elif link_name == "criteo":
        return Criteo()
    elif link_name == "test":
        return Test()
    else:
        raise ValueError(f"Unknown link name: {link_name}")
