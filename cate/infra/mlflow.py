from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow import ActiveRun, system_metrics

from cate.metrics.metrics import Artifacts, Metrics

REMOTE_TRACKING_URI = "http://ec2-44-217-145-52.compute-1.amazonaws.com:5000"


# TODO: MlflowClientを使用するように変更
class MlflowClient:
    def __init__(
        self, experiment_name: str, tracking_uri: str = REMOTE_TRACKING_URI
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_id = self.initialize(experiment_name, tracking_uri)

    @staticmethod
    def initialize(experiment_name: str, tracking_uri: str) -> str:
        mlflow.set_tracking_uri(tracking_uri)
        system_metrics.enable_system_metrics_logging()  # type: ignore

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            return experiment_id

        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        return str(experiment.experiment_id)

    def start_run(
        self,
        run_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        parent_run_id: str | None = None,
        tags: dict[str, Any] | None = None,
        description: str | None = None,
        log_system_metrics: bool | None = None,
    ) -> ActiveRun:
        return mlflow.start_run(
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

    def log_metrics(self, metrics: Metrics, step: int | None) -> None:
        mlflow.log_metrics(
            {value.name: value.data for value in metrics.results}, step=step
        )

    # TODO: client.log_figure()により実装
    # TODO: log_artifact
    def log_artifacts(self, artifacts: Artifacts) -> None:
        with TemporaryDirectory() as tmpdir:
            for artifact in artifacts.results:
                _, _ = artifact.save(Path(tmpdir))
            mlflow.log_artifacts(local_dir=tmpdir)

    def search_runs_by_tags(self, tags: dict[str, str]) -> list[str]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=" and ".join([f"tags.{k} = '{v}'" for k, v in tags.items()]),
            output_format="list",
        )

        run_ids = []
        for run in runs:
            run_id = run.info.run_id
            run_ids.append(run_id)
        return run_ids
