from typing import Any

import mlflow
import mlflow.system_metrics
from mlflow.entities.experiment import Experiment

from cate.base.metrics import Artifacts, Metrics


def initialize(experiment_name: str) -> Experiment:
    mlflow.set_tracking_uri("http://ec2-44-217-145-52.compute-1.amazonaws.com:5000")
    mlflow.system_metrics.enable_system_metrics_logging()  # type: ignore
    experiment = mlflow.set_experiment(experiment_name)
    return experiment


class MlflowClient:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Metrics) -> None:
        mlflow.log_metrics(metrics.to_dict())

    def log_artifacts(self, artifact_path: Artifacts) -> None:
        for name, local_path in artifact_path.to_dict().items():
            mlflow.log_artifact(local_path, name)
