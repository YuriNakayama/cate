from typing import Any

import mlflow
import mlflow.system_metrics
from mlflow.entities.experiment import Experiment

from cate.base.metrics import AbstraceArtifats, AbstractMetrics


def initialize(experiment_name: str) -> Experiment:
    mlflow.set_tracking_uri("http://ec2-44-217-145-52.compute-1.amazonaws.com:5000")
    mlflow.system_metrics.enable_system_metrics_logging()  # type: ignore
    experiment = mlflow.set_experiment(experiment_name)
    return experiment


class MLflowClient:
    def __init__(self, experiment_name: str) -> None:
        mlflow.set_tracking_uri("http://ec2-44-217-145-52.compute-1.amazonaws.com:5000")
        mlflow.system_metrics.enable_system_metrics_logging()  # type: ignore
        self.experiment = mlflow.set_experiment(experiment_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        mlflow.log_metrics(metrics)

    def log_custom_metrics(self, metrics: list[AbstractMetrics]) -> None:
        for metric in metrics:
            mlflow.log_metrics({"name": metric})

    def log_artifacts(self, artifact_path: AbstraceArtifats) -> None:
        mlflow.log_artifacts(artifact_path)

    def end(self) -> None:
        mlflow.end_run()
