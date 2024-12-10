from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class Classifier(Protocol):
    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[np.int_]) -> Classifier: ...
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int_]: ...
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float_]: ...


class BaseLearner(ABC):
    @abstractmethod
    def fit(
        self,
        X: npt.NDArray[Any],
        treatment: npt.NDArray[np.int_],
        y: npt.NDArray[np.float_ | np.int_],
        p: npt.NDArray[np.float_] | None = None,
        eval_set: list[
            tuple[
                npt.NDArray[Any],
                npt.NDArray[np.int_],
                npt.NDArray[np.float_ | np.int_],
                npt.NDArray[np.float_] | None,
            ]
        ]
        | None = None,
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
