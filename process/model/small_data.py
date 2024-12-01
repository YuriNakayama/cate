import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference import meta
from tqdm import tqdm

from cate.infra.mlflow import MlflowClient
from cate.model.dataset import Dataset, split
from cate import evaluate
from cate.model.metrics import Artifacts, Metrics
from cate.utils import Timer, get_logger, path_linker

dataset_name = "criteo"
sample_num = 100
sample_size = 10_000
logger = get_logger("causalml")
pathlinker = path_linker(dataset_name)
client = MlflowClient("small_data")
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
    # "drlearner": meta.BaseDRLearner(base_regressor),
    # "xlearner": meta.BaseXClassifier(base_classifier, base_regressor),
    "rlearner": meta.BaseRClassifier(base_classifier, base_regressor),
    "slearner": meta.BaseSClassifier(base_classifier),
    "tlearner": meta.BaseTClassifier(base_classifier),
    # "cevae": CEVAE(),
}

np.int = int  # type: ignore

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
            "model_name": name,
        }
    )
    for i in tqdm(range(sample_num)):
        logger.info(f"epoch {i}")
        train_ds, test_ds = split(ds, sample_size, random_state=i)
        train_X = train_ds.X
        train_y = train_ds.y.to_numpy().reshape(-1)
        train_w = train_ds.w.to_numpy().reshape(-1)
        valid_X = test_ds.X
        valid_y = test_ds.y.to_numpy().reshape(-1)
        valid_w = test_ds.w.to_numpy().reshape(-1)

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
                [evaluate.Auuc(10)]
                + [evaluate.UpliftByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
                + [evaluate.QiniByPercentile(k) for k in np.arange(0, 1.1, 0.1)]
            )
        )
        metrics(pred.reshape(-1), valid_y, valid_w)
        client.log_metrics(metrics, i)

    pred_df = pd.DataFrame(
        {"index": test_ds.y.index, "pred": pred.reshape(-1)}
    ).set_index("index")
    base_df = pd.merge(
        ds.y.rename(columns={ds.y_columns[0]: "y"}),
        ds.w.rename(columns={ds.w_columns[0]: "w"}),
        left_index=True,
        right_index=True,
    )
    output_df = pd.merge(base_df, pred_df, left_index=True, right_index=True)

    artifacts = Artifacts([evaluate.UpliftCurve(10), evaluate.Outputs()])
    artifacts(output_df.pred.to_numpy(), output_df.y.to_numpy(), output_df.w.to_numpy())
    client.log_artifacts(artifacts)
    client.end_run()

timer.to_csv(pathlinker.prediction / "metalearner_duration.csv")
