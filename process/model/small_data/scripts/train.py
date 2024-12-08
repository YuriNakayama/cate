from logging import Logger

import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from omegaconf import DictConfig


import cate.dataset as cds
from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics
from cate.utils.path import AbstractLink

def setup_dataset(
    cfg: DictConfig,
    logger: Logger,
    link: AbstractLink,
    sample_ratio: float,
    random_state: int,
) -> tuple[cds.Dataset, cds.Dataset]:
    logger.info("load dataset")
    ds = cds.Dataset.load(link.base)
    train_ds, test_ds = cds.split(ds, 1 - sample_ratio, random_state=random_state)
    return train_ds, test_ds


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    *,
    sample_ratio: float,
    random_state: int,
    train_ds: cds.Dataset,
    test_ds: cds.Dataset,
) -> None:
    logger.info(
        f"start train in sample_ratio {sample_ratio}, random_state {random_state}"
    )
    base_classifier = lgb.LGBMClassifier(
    importance_type="gain", random_state=42, force_col_wise=True, n_jobs=-1
    )
    base_regressor = lgb.LGBMRegressor(
        importance_type="gain", random_state=42, force_col_wise=True, n_jobs=-1
    )
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
    
    logger.info("split dataseet")
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

    pred_df = pd.DataFrame(
        {"index": test_ds.y.index, "pred": pred.reshape(-1)}
    ).set_index("index")
    base_df = pd.merge(
        test_ds.y.rename(columns={test_ds.y_columns[0]: "y"}),
        test_ds.w.rename(columns={test_ds.w_columns[0]: "w"}),
        left_index=True,
        right_index=True,
    )
    output_df = pd.merge(base_df, pred_df, left_index=True, right_index=True)

    artifacts = Artifacts([evaluate.UpliftCurve(10), evaluate.Outputs()])
    artifacts(output_df.pred.to_numpy(), output_df.y.to_numpy(), output_df.w.to_numpy())
    client.log_artifacts(artifacts)
    client.end_run()
