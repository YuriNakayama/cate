from __future__ import annotations

import shelve
from pathlib import Path
from shutil import rmtree

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split


def to_rank(
    primary_key: pl.Series, score: pl.Series, descending: bool = False, k: int = 100
) -> pl.Series:
    df = pl.DataFrame(
        {primary_key.name: primary_key.clone(), score.name: score.clone()}
    )
    df = df.sort(by=str(score.name), descending=descending)
    df = df.with_columns(
        pl.Series(
            name="rank",
            values=np.ceil(np.arange(1, len(df) + 1) / len(df) * k),
            dtype=pl.Int64,
        )
    )
    df = primary_key.to_frame().join(df, on=primary_key.name, how="left")
    return df["rank"]


class Dataset:
    def __init__(
        self,
        df: pl.DataFrame,
        x_columns: list[str],
        y_columns: list[str],
        w_columns: list[str],
    ) -> None:
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.w_columns = w_columns
        self.__df = df.clone()
        self._validate(
            self.__df.columns, self.x_columns, self.y_columns, self.w_columns
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
    def X(self) -> pl.DataFrame:
        return self.__df.select(self.x_columns).clone()

    @property
    def y(self) -> pl.DataFrame:
        return self.__df.select(self.y_columns).clone()

    @property
    def w(self) -> pl.DataFrame:
        return self.__df.select(self.w_columns).clone()

    def save(self, path: Path) -> None:
        if path.exists():
            rmtree(path)
        path.mkdir(exist_ok=True, parents=True)
        self.__df.write_parquet(path / "data.parquet")
        with shelve.open(path / "meta") as shelf:
            shelf["x_columns"] = self.x_columns
            shelf["y_columns"] = self.y_columns
            shelf["w_columns"] = self.w_columns

    @classmethod
    def load(cls, path: Path) -> Dataset:
        if (not (path / "data.parquet").exists()) or (not (path / "meta.db").exists()):
            raise FileNotFoundError("Data or meta file not found.")

        df = pl.read_parquet(path / "data.parquet")
        meta = shelve.open(path / "meta")
        return cls(df, **meta)

    def __len__(self) -> int:
        return len(self.__df)

    def to_frame(self) -> pl.DataFrame:
        return self.__df.clone()


def filter(ds: Dataset, flgs: list[pl.Series]) -> Dataset:
    flg = pl.all_horizontal(flgs)
    df = ds.to_frame().filter(flg)
    return Dataset(df, ds.x_columns, ds.y_columns, ds.w_columns)


def concat(ds_list: list[Dataset]) -> Dataset:
    df = pl.concat([ds.to_frame() for ds in ds_list])
    return Dataset(df, ds_list[0].x_columns, ds_list[0].y_columns, ds_list[0].w_columns)


def sample(
    ds: Dataset, n: int | None = None, frac: float | None = None, random_state: int = 42
) -> Dataset:
    if n == 0 or frac == 0:
        return Dataset(
            pl.DataFrame(schema=ds.to_frame().schema),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        )

    df = ds.to_frame().sample(n=n, fraction=frac, seed=random_state)
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
    """  # noqa: E501

    if test_frac == 0 or test_n == 0:
        return ds, Dataset(
            pl.DataFrame(schema=ds.to_frame().schema),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        )
    if test_frac == 1 or test_n == len(ds):
        return Dataset(
            pl.DataFrame(schema=ds.to_frame().schema),
            ds.x_columns,
            ds.y_columns,
            ds.w_columns,
        ), ds

    test_size = test_frac if test_frac is not None else test_n
    train_df, test_df = train_test_split(
        ds.to_frame(), test_size=test_size, random_state=random_state
    )
    return (
        Dataset(train_df, ds.x_columns, ds.y_columns, ds.w_columns),
        Dataset(test_df, ds.x_columns, ds.y_columns, ds.w_columns),
    )


def synthetic_data(
    n: int = 1000, p: int = 5, random_state: int = 42
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Generate synthetic data for testing purposes.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate. Default is 1000.
    p : int, optional
        Number of features for each sample. Default is 5.
    random_state : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - X : numpy.ndarray of shape (n, p)
            The generated feature matrix with `n` samples and `p` features.
        - w : numpy.ndarray of shape (n,)
            The generated weights (binary values).
        - y : numpy.ndarray of shape (n,)
            The generated target values (binary values).
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    w = rng.integers(2, size=(n, 2))[:, 0].reshape(-1)
    y = rng.integers(2, size=(n, 2))[:, 1].reshape(-1)
    return X, w, y
