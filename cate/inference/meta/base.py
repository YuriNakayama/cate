from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger("causalml")


class BaseLearner(ABC):
    @abstractmethod
    def fit(self, X, treatment, y, p=None) -> BaseLearner:
        pass

    @abstractmethod
    def predict(
        self, X, treatment=None, y=None, p=None, verbose=True
    ) -> npt.NDArray[np.float64]:
        pass
