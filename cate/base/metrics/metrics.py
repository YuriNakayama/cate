from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Value:
    name: str
    data: float


class AbstractMetric(ABC):
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
    ) -> float:
        raise NotImplementedError

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Value:
        return Value(self.name, self._calculate(pred, y, w))

