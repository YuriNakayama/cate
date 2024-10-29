from abc import ABC, abstractmethod

import pandas as pd
from matplotlib.figure import Figure


class AbstractMetrics(ABC):
    @abstractmethod
    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        raise NotImplementedError

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        return self._calculate(score, group, conversion)


class AbstraceArtifats(ABC):
    @abstractmethod
    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> Figure | pd.DataFrame:
        raise NotImplementedError

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> Figure | pd.DataFrame:
        return self._calculate(score, group, conversion)
