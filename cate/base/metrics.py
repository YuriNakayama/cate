from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


class AbstractMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        raise NotImplementedError

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        return self._calculate(score, group, conversion)


class AbstraceImageArtifat(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> Figure:
        raise NotImplementedError

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series, path: Path
    ) -> tuple[str, Path]:
        figure = self._calculate(score, group, conversion)
        figure.savefig(path / self.name)
        return self.name, path / self.name


class AbstractChartArtifat(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> pd.DataFrame:
        raise NotImplementedError

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series, path: Path
    ) -> tuple[str, Path]:
        df = self._calculate(score, group, conversion)
        df.to_csv(path / self.name)
        return self.name, path / self.name


class Metrics:
    def __init__(
        self,
        metrics: list[AbstractMetric],
        score: pd.Series,
        group: pd.Series,
        conversion: pd.Series,
    ) -> None:
        self.metrics = metrics
        self.results = self._calculate(score, group, conversion)

    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> dict[str, float]:
        return {
            metric.name: metric(score, group, conversion) for metric in self.metrics
        }

    def to_dict(self) -> dict[str, float]:
        return self.results


class Artifacts:
    def __init__(
        self,
        artifacts: list[AbstraceImageArtifat | AbstractChartArtifat],
        score: pd.Series,
        group: pd.Series,
        conversion: pd.Series,
        path: Path,
    ) -> None:
        self.artifacts = artifacts
        self.results = self._calculate(score, group, conversion, path)

    def _calculate(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series, path: Path
    ) -> dict[str, Path]:
        return dict(
            artifact(score, group, conversion, path) for artifact in self.artifacts
        )

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.results.items()}
