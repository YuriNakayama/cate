from logging import Logger

import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cate import evaluate
from cate.dataset import Dataset, sample
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics
from cate.utils.path import AbstractLink


def train(
    cfg: DictConfig,
    pathlink: AbstractLink,
    client: MlflowClient,
    logger: Logger,
    sample_ratio: float = 1.0,
) -> None:
    logger.info(f"start train in sample_ratio {sample_ratio}")
    logger.info("load dataset")
    ds = Dataset.load(pathlink.base)
    ds = sample(ds, frac=sample_ratio, random_state=42)
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
        run_name=f"{cfg.data.name}-{cfg.model.name}-sample_ratio_{sample_ratio}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "sample_ratio": str(sample_ratio),
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(
        dict(cfg.training)
        | dict(cfg.model)
        | {"sample_ratio": sample_ratio}
    )
    _pred_dfs = []
    for epoch, (train_idx, valid_idx) in tqdm(
        enumerate(skf.split(np.zeros(len(ds)), ds.y))
    ):
        logger.info(f"epoch {epoch}")
        train_X = ds.X.iloc[train_idx]
        train_y = ds.y.iloc[train_idx].to_numpy().reshape(-1)
        train_w = ds.w.iloc[train_idx].to_numpy().reshape(-1)
        valid_X = ds.X.iloc[valid_idx]
        valid_y = ds.y.iloc[valid_idx].to_numpy().reshape(-1)
        valid_w = ds.w.iloc[valid_idx].to_numpy().reshape(-1)

        model.fit(
            train_X,
            train_w,
            train_y,
            p=np.full(train_w.shape, train_w.mean()),
        )

        pred = model.predict(valid_X, p=np.full(valid_X.shape[0], train_w.mean()))

        metrics = Metrics(
            list(
                [evaluate.Auuc()]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1, 0.1)]
            )
        )
        metrics(pred.reshape(-1), valid_y, valid_w)
        client.log_metrics(metrics, epoch)

        _pred_dfs.append(
            pd.DataFrame(
                {"index": ds.y.index[valid_idx], "pred": pred.reshape(-1)}
            ).set_index("index")
        )

    pred_df = pd.concat(_pred_dfs, axis=0)
    base_df = pd.merge(
        ds.y.rename(columns={ds.y_columns[0]: "y"}),
        ds.w.rename(columns={ds.w_columns[0]: "w"}),
        left_index=True,
        right_index=True,
    )
    output_df = pd.merge(base_df, pred_df, left_index=True, right_index=True)

    artifacts = Artifacts([evaluate.UpliftCurve(), evaluate.Outputs()])
    artifacts(
        output_df.pred.to_numpy(), output_df.y.to_numpy(), output_df.w.to_numpy()
    )
    client.log_artifacts(artifacts)
    client.end_run()
