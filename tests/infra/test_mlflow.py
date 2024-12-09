from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import mlflow
import pytest

from cate.infra.mlflow import MlflowClient


@pytest.fixture
def mlflow_client() -> Generator[MlflowClient, Any, None]:
    tmpdir = TemporaryDirectory(dir=Path.cwd())
    tracking_uri = f"file://{tmpdir.name}"
    yield MlflowClient("test_experiment", tracking_uri)
    tmpdir.cleanup()


def test_initialize_new_experiment(
    mlflow_client: MlflowClient,
) -> None:
    experiment_name = "new_experiment"
    tracking_uri = mlflow_client.tracking_uri
    actual = mlflow_client.initialize(experiment_name, tracking_uri)
    expect = mlflow.get_experiment_by_name(experiment_name)

    assert expect is not None and actual == expect.experiment_id


def test_initialize_existing_experiment(
    mlflow_client: MlflowClient,
) -> None:
    experiment_name = "existing_experiment"
    tracking_uri = mlflow_client.tracking_uri
    mlflow.create_experiment(experiment_name)
    actual = mlflow_client.initialize(experiment_name, tracking_uri)
    expect = mlflow.get_experiment_by_name(experiment_name)

    assert expect is not None and actual == expect.experiment_id
