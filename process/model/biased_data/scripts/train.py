from __future__ import annotations

from logging import Logger

import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.model.dataset import Dataset, concat, filter, sample, split, to_rank
from cate.model.metrics import Artifacts, Metrics
from cate.utils import AbstractLink


def get_biased_ds(
    ds: Dataset,
    rank_flg: pd.Series[bool],
) -> Dataset:
    tg_flg = (ds.w == 1)[ds.w.columns[0]]
    tg_ds = filter(ds, [tg_flg, rank_flg])
    cg_ds = filter(ds, [~tg_flg, ~rank_flg])
    biased_ds = concat([tg_ds, cg_ds])
    return sample(biased_ds, frac=1)


def tg_cg_split(
    ds: Dataset,
    rank_flg: pd.Series[bool],
    random_ratio: float = 0.0,
    random_state: int = 42,
) -> Dataset:
    if random_ratio == 0:
        return get_biased_ds(ds, rank_flg)

    sample_bias_ds = get_biased_ds(ds, rank_flg)
    biased_ds_ratio = len(sample_bias_ds) / len(ds)
    random_ds_ratio = random_ratio * biased_ds_ratio
    if random_ratio == 1:
        return sample(ds, frac=random_ds_ratio, random_state=random_state)

    _ds, random_ds = split(ds, test_frac=random_ds_ratio, random_state=random_state)
    biased_ds = get_biased_ds(_ds, rank_flg)
    return sample(concat([biased_ds, random_ds]), frac=1, random_state=random_state)


def setup_dataset(
    cfg: DictConfig, logger: Logger, link: AbstractLink
) -> tuple[Dataset, Dataset, pd.DataFrame]:
    logger.info("load dataset")
    ds = Dataset.load(link.base)
    train_ds, test_ds = split(ds, 1 / 3, random_state=42)

    # Add Bias To Train Dataset Using LightGBM
    logger.info("start training bias model")
    _pred_dfs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for i, (train_idx, valid_idx) in enumerate(
        skf.split(np.zeros(len(train_ds)), train_ds.y)
    ):
        logger.info(f"start {i} fold")
        train_X = train_ds.X.iloc[train_idx]
        train_y = train_ds.y.iloc[train_idx].to_numpy().reshape(-1)
        valid_X = train_ds.X.iloc[valid_idx]

        # base_classifier = LogisticRegression(max_iter=5)
        base_classifier = lgb.LGBMClassifier(
            verbosity=-1,
            n_jobs=-1,
            importance_type="gain",
            force_col_wise=True,
            random_state=42,
        )
        base_classifier.fit(train_X, train_y)
        pred = np.array(base_classifier.predict_proba(valid_X))

        _pred_dfs.append(
            pd.DataFrame(
                {"index": train_ds.y.index[valid_idx], "pred": pred[:, 1].reshape(-1)}
            ).set_index("index")
        )

    pred_df = pd.concat(_pred_dfs)
    rank = to_rank(
        pred_df.index.to_series(), pred_df["pred"], k=cfg.model.num_rank
    ).to_frame()
    train_df = pd.merge(train_ds.to_pandas(), rank, left_index=True, right_index=True)

    return (
        Dataset(
            train_df,
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
    rank: int,
    random_ratio: float,
    train_ds: Dataset,
    test_ds: Dataset,
    rank_df: pd.DataFrame,
) -> None:
    logger.info(f"start train in rank {rank}, random_ratio {random_ratio}")
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
        run_name=f"{cfg.data.name}_{cfg.model.name}_rank-{rank}_random_ratio-{random_ratio}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
            "rank": rank,
            "random_ratio": random_ratio,
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(
        dict(cfg.training)
        | dict(cfg.model)
        | {"rank": rank, "random_ratio": random_ratio}
    )

    logger.info("split dataseet")
    rank_flg = rank_df <= rank
    train_ds = tg_cg_split(
        train_ds, rank_flg["rank"], random_ratio=random_ratio, random_state=42
    )

    train_X = train_ds.X
    train_y = train_ds.y.to_numpy().reshape(-1)
    train_w = train_ds.w.to_numpy().reshape(-1)
    test_X = test_ds.X
    test_y = test_ds.y.to_numpy().reshape(-1)
    test_w = test_ds.w.to_numpy().reshape(-1)

    del train_ds

    logger.info(f"strart train {cfg.model.name}")
    model.fit(
        train_X,
        train_w,
        train_y,
    )

    logger.info(f"strart prediction {cfg.model.name}")
    pred = model.predict(test_X)

    metrics = Metrics(
        list(
            [evaluate.Auuc()]
            + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1, 0.1)]
            + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1, 0.1)]
        )
    )
    metrics(pred.reshape(-1), test_y, test_w)
    client.log_metrics(metrics, rank)

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

    artifacts = Artifacts([evaluate.UpliftCurve(), evaluate.Outputs()])
    artifacts(
        output_df.pred.to_numpy(),
        output_df.y.to_numpy(),
        output_df.w.to_numpy(),
    )
    client.log_artifacts(artifacts)
    client.end_run()
