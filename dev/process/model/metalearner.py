import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from causalml.inference.meta import (
    BaseDRLearner,
    BaseRLearner,
    BaseXLearner,
    BaseSLearner,
    BaseTLearner,
    BaseRClassifier,
    BaseXClassifier,
    BaseTClassifier,
    BaseSClassifier,
    BaseDRRegressor,
)
from causalml.inference.torch import CEVAE

from cate.dataset import Dataset
from cate.utils import PathLinker, Timer

pathlinker = PathLinker().data.criteo
timer = Timer()

def to_rank(
    primary_key: pd.Series, score: pd.Series, ascending: bool = True
) -> pd.Series:
    df = pd.DataFrame({primary_key.name: primary_key, score.name: score}).set_index(
        primary_key.name, drop=True
    )
    df = df.sort_values(by=score.name, ascending=ascending)  # type: ignore
    df["rank"] = np.ceil(np.arange(len(df)) / len(df) * 100).astype(int)
    return df["rank"]

ds = Dataset.load(pathlinker.base)
base_classifier = lgb.LGBMClassifier(importance_type="gain")
base_regressor = lgb.LGBMRegressor(importance_type="gain")
names = [
    "drlearner",
    "xlearner",
    "rlearner",
    "slearner",
    "tlearner",
    # "cevae",
]

models = [
    BaseDRLearner(base_classifier),
    BaseXClassifier(base_classifier, base_regressor),
    BaseRClassifier(base_classifier, base_regressor),
    BaseSClassifier(base_classifier),
    BaseTClassifier(base_classifier),
    # CEVAE(),
]

np.int = int

pred_dfs = {}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, model in zip(names, models):
    _pred_dfs = []
    for i, (train_idx, valid_idx) in tqdm(
        enumerate(skf.split(np.zeros(len(ds)), ds.y))
    ):
        train_X = ds.X.iloc[train_idx]
        train_y = ds.y.iloc[train_idx].to_numpy().reshape(-1)
        train_w = ds.w.iloc[train_idx].to_numpy().reshape(-1)
        valid_X = ds.X.iloc[valid_idx]
        valid_y = ds.y.iloc[valid_idx].to_numpy().reshape(-1)
        valid_w = ds.w.iloc[valid_idx].to_numpy().reshape(-1)

        timer.start(f"fit_{name}_{i}")
        model.fit(train_X, train_w, train_y)
        timer.stop(f"fit_{name}_{i}")

        timer.start(f"predict_{name}_{i}")
        pred = model.predict(valid_X)
        timer.stop(f"predict_{name}_{i}")

        _pred_dfs.append(
            pd.DataFrame({"index": ds.y.index[valid_idx], "pred": pred.reshape(-1)})
        )  # type: ignore
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

cvs = {}
for name in names:
    cv_list = []
    for rank in range(100):
        rank_flg = output_df[f"{name}_rank"] <= rank
        tg_flg = output_df["treatment"] == 1
        cv = (
            output_df.loc[rank_flg & tg_flg, "conversion"].mean()
            - output_df.loc[rank_flg & ~tg_flg, "conversion"].mean()
        )
        cv_list.append(cv)
    cvs[name] = cv_list

cv_df = pd.DataFrame(cvs)
cv_df.to_csv("/workspace/outputs/meta_learner.csv", index=False)
timer.to_csv(pathlinker.prediction / "metalearner_duration.csv")