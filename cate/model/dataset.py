from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_rank(
    primary_key: pd.Series, score: pd.Series, ascending: bool = True
) -> pd.Series:
    df = pd.DataFrame({primary_key.name: primary_key, score.name: score}).set_index(
        primary_key.name, drop=True
    )
    df = df.sort_values(by=str(score.name), ascending=ascending)
    df["rank"] = np.ceil(np.arange(len(df)) / len(df) * 100).astype(int)
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

    def sample(self, n: int, frac: float, random_state: int) -> Dataset:
        df = self.__df.sample(n=n, frac=frac, random_state=random_state)
        return Dataset(df, self.x_columns, self.y_columns, self.w_columns)

    def to_pandas(self) -> pd.DataFrame:
        return self.__df.copy()
