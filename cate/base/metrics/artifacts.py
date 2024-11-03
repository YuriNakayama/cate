from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Image:
    name: str
    data: Figure

    def save(self, path: Path) -> tuple[str, Path]:
        self.data.savefig(path / self.name)
        return self.name, path / self.name


@dataclass(frozen=True)
class Table:
    name: str
    data: pd.DataFrame

    def save(self, path: Path) -> tuple[str, Path]:
        self.data.to_csv(path / self.name)
        return self.name, path / self.name


class AbstractImageArtifact(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(self, pred, y, w) -> Figure:
        """
        artifactsの計算ロジック
        """
        raise NotImplementedError

    def __call__(self, pred, y, w) -> Image:
        return Image(self.name, self._calculate(pred, y, w))


class AbstractTableArtifact(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(self, pred, y, w) -> pd.DataFrame:
        """
        artifactsの計算ロジック
        """
        raise NotImplementedError

    def __call__(self, pred, y, w) -> Table:
        return Table(self.name, self._calculate(pred, y, w))
