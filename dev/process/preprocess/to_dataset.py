import pandas as pd

from cate.dataset import Dataset
from cate.utils import PathLinker

pathlinker = PathLinker()


# lenta
df = pd.read_csv(pathlinker.data.lenta.origin)
y_columns = ["response_att"]
w_column = "group"
x_columns = [column for column in df.columns if column not in y_columns + [w_column]]
df = pd.merge(
    df.drop("gender", axis=1),
    pd.get_dummies(df["gender"], prefix="gender", dtype=int),
    left_index=True,
    right_index=True,
)
df[w_column] = df[w_column].apply(lambda x: {"test": 1, "control": 0}.get(x))
ds = Dataset(df, x_columns, y_columns, [w_column])
ds.save(pathlinker.data.lenta.base)

# criteo
df = pd.read_csv(pathlinker.data.criteo.origin)
ds = Dataset(
    df,
    ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
    ["conversion"],
    ["treatment"],
)
ds.save(pathlinker.data.criteo.base)

# test data
df = pd.read_csv(pathlinker.data.criteo.origin).sample(n=100_000, random_state=42)
ds = Dataset(
    df,
    ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
    ["conversion"],
    ["treatment"],
)
ds.save(pathlinker.data.criteo.base)
