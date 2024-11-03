from __future__ import annotations


import numpy.typing as npt

from cate.base.metrics import (
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
        pred: npt.NDArray,
        y: npt.NDArray,
        w: npt.NDArray,
    ) -> Metrics:
        self.results = [metrics(pred, y, w) for metrics in self.metrics]
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
        pred: npt.NDArray,
        y: npt.NDArray,
        w: npt.NDArray,
    ) -> Artifacts:
        self.results = [artifact(pred, y, w) for artifact in self.artifacts]
        return self

    @property
    def result(self) -> list[Image | Table]:
        return self.results

    def clear(self) -> Artifacts:
        self.results = []
        return self
