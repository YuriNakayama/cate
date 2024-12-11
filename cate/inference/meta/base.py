from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class MetaLearnerException(Exception):
    pass


class Classifier(Protocol):
    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        eval_set: list[
            tuple[
                npt.NDArray[Any],
                npt.NDArray[np.int_],
                npt.NDArray[np.float_ | np.int_],
                npt.NDArray[np.float_] | None,
            ]
        ]
        | None = None,
    ) -> Classifier: ...
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int_]: ...
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float_]: ...
    def __repr__(self) -> str: ...


class Regressor(Protocol):
    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[np.float_]) -> Regressor: ...
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float_]: ...
    def __repr__(self) -> str: ...


class AbstractMetaLearner(ABC):
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
        params: dict[str, Any] | None = None,
    ) -> AbstractMetaLearner:
        pass

    @abstractmethod
    def predict(
        self,
        X: npt.NDArray[Any],
        p: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float64]:
        pass
