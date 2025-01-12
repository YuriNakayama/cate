from __future__ import annotations

from copy import deepcopy
from logging import Logger
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.metrics import Artifacts, Metrics, evaluate
from cate.utils import PathLink, dict_flatten


class BaggingModel:
    def __init__(self, classifiers: list[Any]) -> None:
        self.classifiers = classifiers

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[np.int_]) -> BaggingModel:
        for classifier in self.classifiers:
            classifier.fit(X, y)
        return self

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray[np.float_]:
        return np.mean(
            [classifier.predict_proba(X) for classifier in self.classifiers], axis=0
        )


def create_cv_models(
    X: npt.NDArray[Any],
    y: npt.NDArray[np.int_],
    base_classifier: Any,
    random_state: int = 42,
) -> tuple[BaggingModel, npt.NDArray[np.float_]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = []
    pred = np.zeros_like(y, dtype=float)
    for train_idx, valid_idx in skf.split(X, y):
        model = deepcopy(base_classifier)
        model.fit(X[train_idx], y[train_idx])
        models.append(model)
        pred[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
    return BaggingModel(models), pred


def get_biased_ds(
    ds: cds.Dataset,
    rank_flg: pl.Series,
) -> cds.Dataset:
    tg_flg = pl.Series("group", ds.w) == 1
    tg_ds = cds.filter(ds, [tg_flg, rank_flg])
    cg_ds = cds.filter(ds, [~tg_flg, ~rank_flg])
    biased_ds = cds.concat([tg_ds, cg_ds])
    return cds.sample(biased_ds, frac=1)


def tg_cg_split(
    ds: cds.Dataset,
    rank_flg: pl.Series,
    random_ratio: float = 0.0,
    random_state: int = 42,
) -> cds.Dataset:
    if random_ratio == 0:
        return get_biased_ds(ds, rank_flg)

    sample_bias_ds = get_biased_ds(ds, rank_flg)
    biased_ds_ratio = len(sample_bias_ds) / len(ds)
    random_ds_ratio = random_ratio * biased_ds_ratio
    if random_ratio == 1:
        return cds.sample(ds, frac=random_ds_ratio, random_state=random_state)

    ds = cds.Dataset(
        ds.to_frame().with_row_index(), ds.x_columns, ds.y_columns, ds.w_columns
    )
    _ds, random_ds = cds.split(ds, test_frac=random_ds_ratio, random_state=random_state)
    rank_flg = (
        rank_flg.to_frame()
        .with_row_index()
        .filter(pl.col("index").is_in(_ds.to_frame()["index"]))
        .drop("index")
        .to_series()
    )
    biased_ds = get_biased_ds(_ds, rank_flg)
    return cds.sample(
        cds.concat([biased_ds, random_ds]), frac=1, random_state=random_state
    )


def setup_dataset(
    cfg: DictConfig, logger: Logger, link: PathLink
) -> tuple[cds.Dataset, cds.Dataset, pl.Series]:
    logger.info("load dataset")
    ds = cds.Dataset.load(link.mart)
    train_ds, test_ds = cds.split(ds, 1 / 3, random_state=42)

    # Add Bias To Train Dataset Using LightGBM
    logger.info("start training bias model")
    pred_dfs: list[pl.DataFrame] = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for i, (train_idx, valid_idx) in enumerate(
        skf.split(np.zeros(len(train_ds)), train_ds.y)
    ):
        logger.info(f"start {i} fold")
        train_X = train_ds.X[train_idx]
        train_y = train_ds.y[train_idx]
        valid_X = train_ds.X[valid_idx]
        valid_y = train_ds.y[valid_idx]
        valid_w = train_ds.w[valid_idx]

        base_classifier = lgb.LGBMClassifier(
            verbosity=-1,
            n_jobs=-1,
            importance_type="gain",
            force_col_wise=True,
            random_state=42,
        )
        base_classifier.fit(train_X, train_y)
        pred = np.array(base_classifier.predict_proba(valid_X))
        _pred_df = pl.DataFrame(
            {
                "pred": pred[:, 1].reshape(-1),
                train_ds.y_columns[0]: valid_y,
                train_ds.w_columns[0]: valid_w,
            }
        )
        _pred_df = pl.concat(
            [
                _pred_df,
                pl.from_numpy(valid_X, schema=train_ds.x_columns),
            ],
            how="horizontal",
        )
        pred_dfs.append(_pred_df)
    pred_df = pl.concat(pred_dfs)
    rank = cds.to_rank(
        pl.Series("index", range(len(pred_df))), pred_df["pred"], k=cfg.model.num_rank
    )

    return (
        cds.Dataset(
            pred_df,
            train_ds.x_columns,
            train_ds.y_columns,
            train_ds.w_columns,
        ),
        test_ds,
        rank,
    )


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    *,
    train_ds: cds.Dataset,
    test_ds: cds.Dataset,
    rank: pl.Series,
    parent_run_id: str | None = None,
) -> None:
    logger.info(
        f"start train in rank {cfg.data.rank}, random_ratio {cfg.data.random_ratio}"
    )
    # Fit Metalearner
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
        run_name=f"{cfg.data.name}-{cfg.model.name}-rank_{cfg.data.rank}-random_ratio_{cfg.data.random_ratio}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "rank": str(cfg.data.rank),
            "random_ratio": str(cfg.data.random_ratio),
            "sample_ratio": str(cfg.data.sample_ratio),
            "mlflow.parentRunId": parent_run_id,
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
    )
    client.log_params(dict_flatten(cfg))

    logger.info("split dataseet")
    rank_flg = rank <= cfg.data.rank
    train_ds = tg_cg_split(
        train_ds, rank_flg, random_ratio=cfg.data.random_ratio, random_state=42
    )
    train_ds = cds.sample(train_ds, frac=cfg.data.sample_ratio, random_state=42)

    train_X = train_ds.X
    train_y = train_ds.y
    train_w = train_ds.w
    test_X = test_ds.X
    test_y = test_ds.y
    test_w = test_ds.w
    logger.info(f"train X shape{train_X.shape}, test X shape{test_X.shape}")
    del train_ds

    logger.info("strat train propensity score model")
    propensity_base_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**cfg.training.classifier),
        method="sigmoid",
    )
    propensity_model, train_p = create_cv_models(
        train_X, train_y, propensity_base_model
    )

    logger.info(f"strart train {cfg.model.name}")
    model.fit(train_X, train_w, train_y, p=train_p)

    logger.info(f"strart prediction {cfg.model.name}")
    test_p = propensity_model.predict_proba(test_X)[:, 1]
    pred = model.predict(test_X, p=test_p).reshape(-1)

    metrics = Metrics(
        list(
            [evaluate.Auuc()]
            + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1, 0.1)]
            + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1, 0.1)]
        )
    )
    metrics(pred, test_y, test_w)
    client.log_metrics(metrics, cfg.data.rank)

    artifacts = Artifacts([evaluate.UpliftCurve(), evaluate.Outputs()])
    artifacts(
        pred,
        test_ds.y,
        test_ds.w,
    )
    client.log_artifacts(artifacts)
    client.end_run()
