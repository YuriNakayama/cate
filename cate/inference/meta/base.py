from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger("causalml")


class BaseLearner(ABC):
    @abstractmethod
    def fit(
        self,
        X: npt.NDArray[Any],
        treatment: npt.NDArray[np.int_],
        y: npt.NDArray[np.float_ | np.int_],
        p: npt.NDArray[np.float_] | None = None,
        verbose: int = 1,
    ) -> BaseLearner:
        pass

    @abstractmethod
    def predict(
        self,
        X: npt.NDArray[Any],
        p: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float64]:
        pass
