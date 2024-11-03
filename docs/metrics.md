# Evaluation Index

## Usage

```python
from cate.mlflow import MlflowClient
from cate.metrics import Auuc, Metrics, UpliftCurve, Artifacts

client = MlflowClient("example")
client.start_run()

metrics = Metrics([Auuc()])
artifacts = Artifacts([UpliftCurve()])
for model in models:
    for epoch in epochs:
        artifacts(score, group, conversion)
        metrics(score, group, conversion)
    client.log_metrics(metrics)
    artifacts.clear()
    client.log_artifacts(artifacts)
    metrics.clear()

client.end_run()
```

## Classes

### Metrics

```python
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
        self, pred, y, w
    ) -> float:
        """
        metricsの計算ロジック
        """

    def __call__(
        self, pred, y, w
    ) -> Value:
        return Value(self.name, self._calculate(pred, y, w))
    

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
        self.results = [
            metrics(pred, y, w) for metrics in self.metrics
        ]
        return self
    
    @property
    def result(self) -> list[Value]:
        return self.results

    def clear(self) -> Metrics:
        self.results = []
        return self
    
class MlflowClient:
    def log_metrics(self, metrics: Metrics) -> None:
        self.client.log_metrics(
            {value.name: value.data for value in metrics.results}
        )

```

### Artifacts

```python
@dataclass(frozen=True)
class Image:
    name: str
    data: plt.Figure

    def save(self, path: Path) -> tuple[str, Path]:
        self.data.savefig(path / self.name)
        return self.name, path / self.name

class AbstractImageArtifact(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _calculate(
        self, pred, y, w
    ) -> plt.Figure:
        """
        artifactsの計算ロジック
        """

    def __call__(
        self, pred, y, w
    ) -> Image:
        return Image(self.name, self._calculate(pred, y, w))
    

class Artifacts:
    def __init__(self, artifacts: list[AbstractImageArtifact]) -> None:
        self.artifacts = artifacts
        self.results: list[Image | Table] = []
    
    def __call__(
        self, 
        pred: npt.NDArray, 
        y: npt.NDArray,
        w: npt.NDArray, 
    ) -> Artifacts:
        self.results = [
            artifact(pred, y, w) for artifact in self.artifacts
        ]
    
    @property
    def result(self) -> list[Image | Table]:
        return self.results

    def clear(self) -> Artifacts:
        self.results = []
        return self
    
class MlflowClient:
    def log_artifacts(self, artifacts: Artifacts) -> None:
        with TemporaryDirectory() as tmpdir:
            for artifact in artifacts.results:
                self.client.log_artifact(artifact.save(tmpdir))
```