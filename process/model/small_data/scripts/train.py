from logging import Logger

import lightgbm as lgb
import numpy as np
import polars as pl
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics, evaluate
from cate.utils.path import PathLink


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    link: PathLink,
    *,
    sample_ratio: float,
    random_state: int,
) -> None:
    logger.info(
        f"start train in sample_ratio {sample_ratio}, random_state {random_state}"
    )
    logger.info("load dataset")
    ds = cds.Dataset.load(link.mart)

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

    client.start_run(
        run_name=f"{cfg.data.name}-{cfg.model.name}-sample_ratio_{sample_ratio}-random_state_{random_state}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "sample_ratio": str(sample_ratio),
            "random_state": str(random_state),
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
    )
    client.log_params(
        dict(cfg.training)
        | dict(cfg.model)
        | {"random_state": random_state, "sample_ratio": sample_ratio}
    )

    base_dfs: list[pl.DataFrame] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for epoch, (train_index, test_index) in enumerate(skf.split(ds.X, ds.y)):
        logger.info(f"epoch {epoch}")
        _train_ds = cds.Dataset(
            df=ds.to_frame()
            .with_row_index()
            .filter(pl.col("index").is_in(train_index))
            .drop("index"),
            x_columns=ds.x_columns,
            y_columns=ds.y_columns,
            w_columns=ds.w_columns,
        )
        train_ds = cds.sample(_train_ds, frac=sample_ratio, random_state=random_state)
        test_ds = cds.Dataset(
            df=ds.to_frame()
            .with_row_index()
            .filter(pl.col("index").is_in(test_index))
            .drop("index"),
            x_columns=ds.x_columns,
            y_columns=ds.y_columns,
            w_columns=ds.w_columns,
        )
        model.fit(
            train_ds.X,
            train_ds.w,
            train_ds.y,
            p=np.full(train_ds.w.shape, train_ds.w.mean()),
        )

        pred = model.predict(
            test_ds.X, p=np.full(test_ds.X.shape[0], train_ds.w.mean())
        )
        metrics = Metrics(
            list(
                [evaluate.Auuc(10)]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
            )
        )
        metrics(pred.reshape(-1), test_ds.y, test_ds.w)
        client.log_metrics(metrics, epoch)

        _base_df = pl.DataFrame(
            {
                "pred": pred.reshape(-1),
                "y": test_ds.y,
                "W": test_ds.w,
            }
        )
        base_dfs.append(_base_df)
    base_df = pl.concat(base_dfs)

    artifacts = Artifacts([evaluate.UpliftCurve(10), evaluate.Outputs()])
    artifacts(
        base_df["pred"].to_numpy(), base_df["y"].to_numpy(), base_df["w"].to_numpy()
    )
    client.log_artifacts(artifacts)
    client.end_run()
