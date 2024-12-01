from logging import Logger

import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.model.dataset import Dataset, sample, split, to_rank
from cate.model.metrics import Artifacts, Metrics
from cate.utils import AbstractLink


def tg_cg_split(ds: Dataset, rank_flg: pd.Series) -> Dataset:
    df = ds.to_pandas()
    tg_flg = df[ds.w_columns] == 1
    tg_df = df[rank_flg & tg_flg]
    cg_df = df[~rank_flg & ~tg_flg]
    localized_ds = Dataset(
        pd.concat([tg_df, cg_df]),
        ds.x_columns,
        ds.y_columns,
        ds.w_columns,
    )
    return sample(localized_ds, frac=1)


def setup_dataset(
    cfg: DictConfig, client: MlflowClient, logger: Logger, link: AbstractLink
) -> tuple[Dataset, Dataset]:
    logger.info("load dataset")
    ds = Dataset.load(link.base)
    train_ds, test_ds = split(ds, 1 / 3, random_state=42)

    # Add Bias To Train Dataset Using LightGBM
    _pred_dfs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for train_idx, valid_idx in skf.split(np.zeros(len(train_ds)), train_ds.y):
        train_X = train_ds.X.iloc[train_idx]
        train_y = train_ds.y.iloc[train_idx].to_numpy().reshape(-1)
        valid_X = train_ds.X.iloc[valid_idx]
        valid_y = train_ds.y.iloc[valid_idx].to_numpy().reshape(-1)

        base_classifier = lgb.LGBMClassifier(
            importance_type="gain",
            random_state=42,
            force_col_wise=True,
            n_jobs=-1,
            verbosity=0,
        )
        base_classifier.fit(
            train_X, train_y, eval_set=[(valid_X, valid_y)], eval_metric="auc"
        )
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

    return Dataset(
        train_df,
        train_ds.x_columns,
        train_ds.y_columns,
        train_ds.w_columns,
    ), test_ds


def train(
    cfg: DictConfig,
    client: MlflowClient,
    logger: Logger,
    link: AbstractLink,
    *,
    rank: int,
    train_ds: Dataset,
    test_ds: Dataset,
) -> None:
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

    logger.info(f"start {cfg.model.name}")
    client.start_run(
        run_name=f"{cfg.data.name}_{cfg.model.name}_{rank}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "causalml",
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(cfg.trainig | cfg.model)

    logger.info(f"rank {rank}")
    train_df = train_ds.to_pandas()
    rank_flg = train_df["rank"] <= rank
    train_ds = tg_cg_split(train_ds, rank_flg)
    localized_train_ds = Dataset(
        train_df.loc[rank_flg],
        train_ds.x_columns,
        train_ds.y_columns,
        train_ds.w_columns,
    )

    train_X = localized_train_ds.X
    train_y = localized_train_ds.y.to_numpy().reshape(-1)
    train_w = localized_train_ds.w.to_numpy().reshape(-1)
    test_X = test_ds.X
    test_y = test_ds.y.to_numpy().reshape(-1)
    test_w = test_ds.w.to_numpy().reshape(-1)

    model.fit(
        train_X,
        train_w,
        train_y,
    )

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
