import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cate import evaluate
from cate.infra.mlflow import MlflowClient
from cate.model.dataset import Dataset
from cate.model.metrics import Artifacts, Metrics
from cate.utils import Timer, get_logger, path_linker

dataset_name = "lenta"
logger = get_logger("causalml")
pathlinker = path_linker(dataset_name)
client = MlflowClient("base_pattern")
timer = Timer()

logger.info("load dataset")
ds = Dataset.load(pathlinker.base)
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
        tags={"model": name, "dataset": dataset_name, "package": "causalml"},
        description=f"base_pattern: {name} training and evaluation using {dataset_name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",
    )
    client.log_params(
        {
            "importance_type": "gain",
            "random_state": 42,
            "n_jobs": -1,
            "force_col_wise": True,
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
            pd.DataFrame({"index": ds.y.index[valid_idx], "pred": pred.reshape(-1)})
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

timer.to_csv(pathlinker.prediction / "metalearner_duration.csv")
