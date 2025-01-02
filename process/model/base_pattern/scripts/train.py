from logging import Logger

import lightgbm as lgb
import numpy as np
import polars as pl
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cate.dataset import Dataset, sample
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics, evaluate
from cate.utils import PathLink, dict_flatten


def train(
    cfg: DictConfig,
    pathlink: PathLink,
    client: MlflowClient,
    logger: Logger,
    parent_run_id: str,
) -> None:
    logger.info("start train")
    logger.info("load dataset")
    ds = Dataset.load(pathlink.mart)
    ds = sample(ds, frac=cfg.data.sample_ratio, random_state=42)
    base_classifier = lgb.LGBMClassifier(**cfg.training.classifier)
    base_regressor = lgb.LGBMRegressor(**cfg.training.regressor)

    models = {
        "drlearner": meta.BaseDRLearner(base_regressor),
        "xlearner": meta.BaseXClassifier(base_classifier, base_regressor),
        "rlearner": meta.BaseRClassifier(base_classifier, base_regressor),
        "slearner": meta.BaseSClassifier(base_classifier),
        "tlearner": meta.BaseTClassifier(base_classifier),
        # "cevae": CEVAE(),
    }

    model = models[cfg.model.name]
    np.int = int  # type: ignore

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    client.start_run(
        run_name=f"{cfg.data.name}-{cfg.model.name}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "sample_ratio": str(cfg.data.sample_ratio),
            "mlflow.parentRunId": parent_run_id,
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
    )
    client.log_params(dict_flatten(cfg))
    pred_dfs: list[pl.DataFrame] = []
    for epoch, (train_idx, valid_idx) in tqdm(
        enumerate(skf.split(np.zeros(len(ds)), ds.y))
    ):
        logger.info(f"epoch {epoch}")
        train_X = ds.X[train_idx]
        train_y = ds.y[train_idx]
        train_w = ds.w[train_idx]
        valid_X = ds.X[valid_idx]
        valid_y = ds.y[valid_idx]
        valid_w = ds.w[valid_idx]

        model.fit(
            train_X,
            train_w,
            train_y,
            p=np.full(train_w.shape, train_w.mean()),
        )

        pred = model.predict(valid_X, p=np.full(valid_X.shape[0], train_w.mean()))

        metrics = Metrics(
            list(
                [evaluate.Auuc(20)]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1, 0.1)]
            )
        )
        metrics(pred.reshape(-1), valid_y, valid_w)
        client.log_metrics(metrics, epoch)

        pred_dfs.append(pl.DataFrame({"pred": pred, "y": valid_y, "w": valid_w}))

    pred_df = pl.concat(pred_dfs)

    artifacts = Artifacts([evaluate.UpliftCurve(40), evaluate.Outputs()])
    artifacts(
        pred_df["pred"].to_numpy(), pred_df["y"].to_numpy(), pred_df["w"].to_numpy()
    )
    client.log_artifacts(artifacts)
    client.end_run()
