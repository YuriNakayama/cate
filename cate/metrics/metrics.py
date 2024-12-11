from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cate.base.metrics.evaluate import (
    AbstractImageArtifact,
    AbstractMetric,
    AbstractTableArtifact,
    Image,
    Table,
    Value,
)


class Metrics:
    def __init__(self, metrics: list[AbstractMetric]) -> None:
        self.metrics = metrics
        self.results: list[Value] = []

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Metrics:
        self.results = [metric(pred, y, w) for metric in self.metrics]
        return self

    @property
    def result(self) -> list[Value]:
        return self.results

    def clear(self) -> Metrics:
        self.results = []
        return self


class Artifacts:
    def __init__(
        self, artifacts: list[AbstractImageArtifact | AbstractTableArtifact]
    ) -> None:
        self.artifacts = artifacts
        self.results: list[Image | Table] = []

    def __call__(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> Artifacts:
        self.results = [artifact(pred, y, w) for artifact in self.artifacts]
        return self

    @property
    def result(self) -> list[Image | Table]:
        return self.results

    def clear(self) -> Artifacts:
        self.results = []
        return self
