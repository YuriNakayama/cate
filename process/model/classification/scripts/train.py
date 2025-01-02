from logging import Logger

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics, evaluate
from cate.utils.path import AbstractLink


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    link: AbstractLink,
    parent_run_id: str,
) -> None:
    logger.info(f"start train in {cfg.data.name} dataset with {cfg.model.name} model")
    logger.info("load dataset")
    ds = cds.Dataset.load(link.base)

    models = {"lightgbm": lgb.LGBMClassifier(**cfg.training.classifier)}
    model = models[cfg.model.name]
    np.int = int  # type: ignore

    client.start_run(
        run_name=f"{cfg.data.name}-{cfg.model.name}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "mlflow.parentRunId": parent_run_id,
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
    )
    client.log_params(dict(cfg.training) | dict(cfg.model) | dict(cfg.data))

    base_dfs: list[pl.DataFrame] = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for epoch, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(ds)), ds.y)):
        logger.info(f"start {epoch} fold")
        train_X = ds.X[train_idx]
        train_y = ds.y[train_idx]
        valid_X = ds.X[valid_idx]
        valid_w = ds.w[valid_idx]
        valid_y = ds.y[valid_idx]

        model.fit(train_X, train_y)
        pred = np.array(model.predict_proba(valid_X))[:, 1].reshape(-1)

        metrics = Metrics(
            list(
                [evaluate.Auuc(10)]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
            )
        )
        metrics(pred, valid_y, valid_w)
        client.log_metrics(metrics, epoch)
        base_dfs.append(pl.DataFrame({"pred": pred, "y": valid_y, "w": valid_w}))
    base_df = pl.concat(base_dfs)
    _metrics = {
        "roc_auc": roc_auc_score(base_df["y"], base_df["pred"]),
        "accuracy": accuracy_score(base_df["y"], base_df["pred"] > 0.5),
        "precision": precision_score(base_df["y"], base_df["pred"] > 0.5),
        "recall": recall_score(base_df["y"], base_df["pred"] > 0.5),
        "f1": f1_score(base_df["y"], base_df["pred"] > 0.5),
        # "confusion_matrix": confusion_matrix(ds.y, base_df.pred > 0.5),
        # "classification_report": classification_report(ds.y, base_df.pred > 0.5),
    }

    mlflow.log_metrics(_metrics)

    artifacts = Artifacts([evaluate.UpliftCurve(10), evaluate.Outputs()])
    artifacts(
        base_df["pred"].to_numpy(), base_df["y"].to_numpy(), base_df["w"].to_numpy()
    )
    client.log_artifacts(artifacts)

    client.end_run()
