from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import mlflow.system_metrics

from cate.base.metrics import Artifacts, Metrics


def initialize(experiment_name: str) -> str:
    mlflow.set_tracking_uri("http://ec2-44-217-145-52.compute-1.amazonaws.com:5000")
    mlflow.system_metrics.enable_system_metrics_logging()  # type: ignore

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)
        return experiment_id
    else:
        mlflow.set_experiment(experiment.experiment_id)
        experiment_id: str = experiment.experiment_id
        return experiment_id


class MlflowClient:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_id = initialize(experiment_name)

    def start_run(
        self,
        run_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        parent_run_id: str | None = None,
        tags: dict[str, Any] | None = None,
        description: str | None = None,
        log_system_metrics: bool | None = None,
    ) -> None:
        mlflow.start_run(
            run_id=run_id,
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            parent_run_id=parent_run_id,
            tags=tags,
            description=description,
            log_system_metrics=log_system_metrics,
        )

    def end_run(self) -> None:
        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Metrics) -> None:
        with TemporaryDirectory() as tempdir:
            metrics.log(tempdir)
            mlflow.log_metrics(metrics.to_dict())

    def log_artifacts(self, artifact_path: Artifacts) -> None:
        with TemporaryDirectory() as tempdir:
            artifact_path.log(tempdir)
            for name, local_path in artifact_path.to_dict().items():
                mlflow.log_artifacts(local_path, name)
