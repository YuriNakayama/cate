from logging import Logger

import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

import cate.dataset as cds
from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics
from cate.utils.path import AbstractLink


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    link: AbstractLink,
    *,
    sample_ratio: float,
    random_state: int,
) -> None:
    logger.info(
        f"start train in sample_ratio {sample_ratio}, random_state {random_state}"
    )
    logger.info("load dataset")
    ds = cds.Dataset.load(link.base)

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
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(
        dict(cfg.training)
        | dict(cfg.model)
        | {"random_state": random_state, "sample_ratio": sample_ratio}
    )

    base_dfs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for epoch, (train_index, test_index) in enumerate(skf.split(ds.X, ds.y)):
        logger.info(f"epoch {epoch}")
        _train_ds = cds.Dataset(
            df=ds.to_pandas().iloc[train_index],
            x_columns=ds.x_columns,
            y_columns=ds.y_columns,
            w_columns=ds.w_columns,
        )
        train_ds = cds.sample(_train_ds, frac=sample_ratio, random_state=random_state)
        test_ds = cds.Dataset(
            df=ds.to_pandas().iloc[test_index],
            x_columns=ds.x_columns,
            y_columns=ds.y_columns,
            w_columns=ds.w_columns,
        )
        train_X = train_ds.X
        train_y = train_ds.y.to_numpy().reshape(-1)
        train_w = train_ds.w.to_numpy().reshape(-1)
        test_X = test_ds.X
        test_y = test_ds.y.to_numpy().reshape(-1)
        test_w = test_ds.w.to_numpy().reshape(-1)

        model.fit(
            train_X,
            train_w,
            train_y,
            p=np.full(train_w.shape, train_w.mean()),
        )

        pred = model.predict(test_X, p=np.full(test_X.shape[0], train_w.mean()))
        metrics = Metrics(
            list(
                [evaluate.Auuc(10)]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
            )
        )
        metrics(pred.reshape(-1), test_y, test_w)
        client.log_metrics(metrics, epoch)
        
        _pred_df = pd.DataFrame(
            {"index": test_ds.y.index, "pred": pred.reshape(-1)}
        ).set_index("index")
        _base_df = pd.merge(
            test_ds.y.rename(columns={test_ds.y_columns[0]: "y"}),
            test_ds.w.rename(columns={test_ds.w_columns[0]: "w"}),
            left_index=True,
            right_index=True,
        )
        _base_df = pd.merge(
            _base_df,
            _pred_df,
            left_index=True,
            right_index=True,
        )
        base_dfs.append(_base_df)
    base_df = pd.concat(base_dfs)

    artifacts = Artifacts([evaluate.UpliftCurve(10), evaluate.Outputs()])
    artifacts(base_df.pred.to_numpy(), base_df.y.to_numpy(), base_df.w.to_numpy())
    client.log_artifacts(artifacts)
    client.end_run()
