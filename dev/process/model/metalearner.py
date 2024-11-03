import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference.meta import (
    BaseDRLearner,
    BaseRClassifier,
    BaseSClassifier,
    BaseTClassifier,
    BaseXClassifier,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cate.model.dataset import Dataset, to_rank
from cate.model.evaluate import Auuc, QiniByPercentile, UpliftByPercentile, UpliftCurve
from cate.model.metrics import Artifacts, Metrics
from cate.model.mlflow import MlflowClient
from cate.utils import Timer, get_logger, path_linker

dataset_name = "test"
pathlinker = path_linker(dataset_name)
client = MlflowClient(dataset_name)
timer = Timer()
logger = get_logger("causalml")

client.start_run(tags={"models": "metalearner", "dataset": dataset_name})

ds = Dataset.load(pathlinker.base)
base_classifier = lgb.LGBMClassifier(
    importance_type="gain", random_state=42, force_col_wise=True, n_jobs=-1
)
base_regressor = lgb.LGBMRegressor(
    importance_type="gain", random_state=42, force_col_wise=True, n_jobs=-1
)

models = {
    "drlearner": BaseDRLearner(base_regressor),
    "xlearner": BaseXClassifier(base_classifier, base_regressor),
    "rlearner": BaseRClassifier(base_classifier, base_regressor),
    "slearner": BaseSClassifier(base_classifier),
    "tlearner": BaseTClassifier(base_classifier),
    # "cevae": CEVAE(),
}

np.int = int  # type: ignore

metrics = Metrics([Auuc(), UpliftByPercentile(0.1), QiniByPercentile(0.1)])
artifacts = Artifacts([UpliftCurve()])

pred_dfs = {}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, model in models.items():
    _pred_dfs = []
    logger.info(f"start {name}")
    for i, (train_idx, valid_idx) in tqdm(
        enumerate(skf.split(np.zeros(len(ds)), ds.y))
    ):
        train_X = ds.X.iloc[train_idx]
        train_y = ds.y.iloc[train_idx].to_numpy().reshape(-1)
        train_w = ds.w.iloc[train_idx].to_numpy().reshape(-1)
        valid_X = ds.X.iloc[valid_idx]
        valid_y = ds.y.iloc[valid_idx].to_numpy().reshape(-1)
        valid_w = ds.w.iloc[valid_idx].to_numpy().reshape(-1)

        timer.start(name, "train", i)
        model.fit(
            train_X,
            train_w,
            train_y,
            p=np.full(train_w.shape, train_w.mean()),
        )
        timer.stop(name, "train", i)

        timer.start(name, "predict", i)
        pred = model.predict(valid_X, p=np.full(valid_X.shape[0], train_w.mean()))
        timer.stop(name, "predict", i)

        artifacts(pred, valid_y, valid_w)
        metrics(pred, valid_y, valid_w)

        _pred_dfs.append(
            pd.DataFrame({"index": ds.y.index[valid_idx], "pred": pred.reshape(-1)})
        )
    client.log_metrics(metrics)
    artifacts.clear()
    client.log_artifacts(artifacts)
    metrics.clear()
    pred_dfs[name] = _pred_dfs

output_df = pd.merge(ds.y.copy(), ds.w.copy(), left_index=True, right_index=True)
for name, pred_df_list in pred_dfs.items():
    pred_df = pd.concat(pred_df_list, axis=0)
    rank = to_rank(pred_df["index"], pred_df["pred"], ascending=False)
    pred_df = pd.merge(pred_df, rank, left_on="index", right_index=True).set_index(
        "index", drop=True
    )
    pred_df = pred_df.rename(columns={"pred": f"{name}_pred", "rank": f"{name}_rank"})
    output_df = pd.merge(output_df, pred_df, left_index=True, right_index=True)

output_df.to_csv(pathlinker.prediction / "metaleaner.csv")
timer.to_csv(pathlinker.prediction / "metalearner_duration.csv")

client.end_run()
