from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
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
        self.data.to_json(path / self.name)
        return self.name, path / self.name


class AbstractImageArtifact(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> pd.DataFrame:
        """
        artifactsの計算ロジック
        """
        raise NotImplementedError

    @abstractmethod
    def _plot(self, data: pd.DataFrame) -> Figure:
        """
        dataをplotするロジック
        """
        raise NotImplementedError

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Image:
        data = self._calculate(pred, y, w)
        return Image(self.name, self._plot(data))


class AbstractTableArtifact(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> pd.DataFrame:
        """
        artifactsの計算ロジック
        """
        raise NotImplementedError

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Table:
        return Table(self.name, self._calculate(pred, y, w))
