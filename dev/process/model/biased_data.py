import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.model.dataset import Dataset, split, to_rank
from cate.model.metrics import Artifacts, Metrics
from cate.utils import get_logger, path_linker

dataset_name = "test"
client = MlflowClient("biased_data")
pathlinker = path_linker(dataset_name)
logger = get_logger("causalml")
logger.info("load dataset")

ds = Dataset.load(pathlinker.base)
train_ds, test_ds = split(ds, 1 / 3, random_state=42)

dataset_name = "test"
pathlinker = path_linker(dataset_name)
logger = get_logger("causalml")
logger.info("load dataset")

ds = Dataset.load(pathlinker.base)
train_ds, test_ds = split(ds, 1 / 3, random_state=42)

# Add Bias To Train Dataset Using LightGBM
_pred_dfs = []
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for i, (train_idx, valid_idx) in enumerate(
    skf.split(np.zeros(len(train_ds)), train_ds.y)
):
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
rank = to_rank(pred_df.index.to_series(), pred_df["pred"]).to_frame()
train_df = pd.merge(train_ds.to_pandas(), rank, left_index=True, right_index=True)
train_ds_list: list[Dataset] = []
for rank in range(1, 101):
    rank_flg = train_df["rank"] <= rank
    localized_train_df = train_df.loc[rank_flg]
    localized_train_ds = Dataset(
        localized_train_df, train_ds.x_columns, train_ds.y_columns, train_ds.w_columns
    )
    train_ds_list.append(localized_train_ds)


# Fit Metalearner
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

np.int = int  # type: ignore

skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, model in models.items():
    logger.info(f"start {name}")
    client.start_run(
        run_name=f"{dataset_name}_{name}",
        tags={
            "model": name,
            "dataset": dataset_name,
            "package": "causalml",
            # "sample": "0.1",
        },
        description=f"base_pattern: {name} training and evaluation using {dataset_name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(
        {
            "importance_type": "gain",
            "random_state": 42,
            "n_jobs": -1,
            "force_col_wise": True,
            "model_name": name,
        }
    )
    _pred_dfs = []
    for i, (train_idx, valid_idx) in tqdm(
        enumerate(skf.split(np.zeros(len(ds)), ds.y))
    ):
        logger.info(f"epoch {i}")
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
        client.log_metrics(metrics, i)

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
    artifacts(output_df.pred.to_numpy(), output_df.y.to_numpy(), output_df.w.to_numpy())
    client.log_artifacts(artifacts)
    client.end_run()
