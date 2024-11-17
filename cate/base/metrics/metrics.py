from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


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

    def shape_data(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> pd.DataFrame:
        """
        Transforms prediction, actual values, and group data into a sorted DataFrame.

        Parameters
        ----------
        pred : npt.NDArray[np.float_]
            Array of predicted scores.
        y : npt.NDArray[np.float_ | np.int_]
            Array of actual conversion values.
        w : npt.NDArray[np.float_ | np.int_]
            Array of group identifiers.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the columns 'score', 'group', and 'conversion',
            sorted by 'score' in descending order and with specified data types.
        """
        return (
            pd.DataFrame({"score": pred, "group": w, "conversion": y})
            .sort_values(by="score", ascending=False)
            .astype(
                {
                    "score": float,
                    "group": int,
                    "conversion": int,
                }
            )
        )

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Value:
        return Value(self.name, self._calculate(pred, y, w))
