import pandas as pd
from sklift.datasets.datasets import fetch_criteo, fetch_lenta

from cate.utils import PathLinker

pathlinker = PathLinker()


def merge_Xyt(
    X: pd.DataFrame | pd.Series,
    y: pd.DataFrame | pd.Series,
    t: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    df = pd.merge(X, y, left_index=True, right_index=True)
    df = pd.merge(df, t, left_index=True, right_index=True)
    return df


X, y, t = fetch_lenta()
df = merge_Xyt(X, y, t)
df.to_csv(pathlinker.data.lenta.origin, index=False)

X, y, t = fetch_criteo(return_X_y_t=True)
df = merge_Xyt(X, y, t)
df.to_csv(pathlinker.data.criteo.origin, index=False)
