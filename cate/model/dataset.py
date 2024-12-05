from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def to_rank(
    primary_key: pd.Series, score: pd.Series, ascending: bool = True, k: int = 100
) -> pd.Series:
    df = pd.DataFrame({primary_key.name: primary_key, score.name: score}).set_index(
        primary_key.name, drop=True
    )
    df = df.sort_values(by=str(score.name), ascending=ascending)
    df["rank"] = np.ceil(np.arange(1, len(df) + 1) / len(df) * k).astype(int)
    return df["rank"]


class Dataset:
    def __init__(
        self,
        df: pd.DataFrame,
        x_columns: list[str],
        y_columns: list[str],
        w_columns: list[str],
    ) -> None:
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.w_columns = w_columns
        self.__df = df.copy()
        self._validate(
            self.__df.columns.to_list(), self.x_columns, self.y_columns, self.w_columns
        )

    def _validate(
        self,
        columns: list[str],
        x_columns: list[str],
        y_columns: list[str],
        w_columns: list[str],
    ) -> None:
        x_diff_columns = set(x_columns) - set(columns)
        y_diff_columns = set(y_columns) - set(columns)
        w_diff_columns = set(w_columns) - set(columns)
        if x_diff_columns:
            raise ValueError(f"x columns {x_diff_columns} do not exist in df.")
        if y_diff_columns:
            raise ValueError(f"x columns {y_diff_columns} do not exist in df.")
        if w_diff_columns:
            raise ValueError(f"x columns {w_diff_columns} do not exist in df.")

    @property
    def X(self) -> pd.DataFrame:
        return self.__df.loc[:, self.x_columns].copy()

    @property
    def y(self) -> pd.DataFrame:
        return self.__df.loc[:, self.y_columns].copy()

    @property
    def w(self) -> pd.DataFrame:
        return self.__df.loc[:, self.w_columns].copy()

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True, parents=True)
        self.__df.to_csv(path / "data.csv", index=False)
        json.dump(
            {
                "x_columns": self.x_columns,
                "y_columns": self.y_columns,
                "w_columns": self.w_columns,
            },
            (path / "meta.json").open("w"),
        )

    @classmethod
    def load(cls, path: Path) -> Dataset:
        data_path = path / "data.csv"
        meta_path = path / "meta.json"
        if (not data_path.exists()) or (not meta_path.exists()):
            raise FileNotFoundError()

        df = pd.read_csv(data_path)
        property = json.load(meta_path.open(mode="r"))
        return cls(df, **property)

    def __len__(self) -> int:
        return len(self.__df)

    def to_pandas(self) -> pd.DataFrame:
        return self.__df.copy()


def filter(ds: Dataset, flgs: list[pd.Series[bool]]) -> Dataset:
    flg = pd.concat(flgs, axis=1).all(axis=1)
    df = ds.to_pandas().loc[flg]
    return Dataset(df, ds.x_columns, ds.y_columns, ds.w_columns)


def concat(ds_list: list[Dataset]) -> Dataset:
    df = pd.concat([ds.to_pandas() for ds in ds_list])
    return Dataset(df, ds_list[0].x_columns, ds_list[0].y_columns, ds_list[0].w_columns)


def sample(
    ds: Dataset, n: int | None = None, frac: float | None = None, random_state: int = 42
) -> Dataset:
    if n == 0 or frac == 0:
        return Dataset(
            pd.DataFrame(ds.to_pandas().columns),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        )

    df = ds.to_pandas().sample(n=n, frac=frac, random_state=random_state)
    return Dataset(df, ds.x_columns, ds.y_columns, ds.w_columns)


def split(
    ds: Dataset,
    test_frac: float | None = None,
    test_n: float | None = None,
    random_state: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Splits a Dataset into training and testing sets.

    Parameters:
    ds (Dataset): The dataset to be split.
    test_frac (float, optional): The fraction of the dataset to be used as the test set. Defaults to None.
    test_n (float, optional): The number of samples to be used as the test set. Defaults to None.
    random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
    tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    if test_frac == 0 or test_n == 0:
        return ds, Dataset(
            pd.DataFrame(columns=ds.to_pandas().columns),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        )
    if test_frac == 1 or test_n == 1:
        return Dataset(
            pd.DataFrame(columns=ds.to_pandas().columns),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        ), ds

    test_size = test_frac if test_frac is not None else test_n
    train_df, test_df = train_test_split(
        ds.to_pandas(), test_size=test_size, random_state=random_state
    )
    return (
        Dataset(train_df, ds.x_columns, ds.y_columns, ds.w_columns),
        Dataset(test_df, ds.x_columns, ds.y_columns, ds.w_columns),
    )
