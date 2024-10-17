import json
from pathlib import Path

import pandas as pd


class Dataset:
    def __init__(
        self,
        df: pd.DataFrame,
        x_columns: list[str | int],
        y_columns: list[str | int],
        w_columns: list[str | int],
    ) -> None:
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.w_columns = w_columns
        self.__df = df.copy()

    @property
    def X(self) -> pd.DataFrame:
        return self.__df.loc[:, self.x_columns]

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
                "z_columns": self.w_columns,
            },
            (path / "property.json").open("w"),
        )

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        data_path = path / "data.csv"
        property_path = path / "property.json"
        if (not data_path.exists()) or (not property_path.exists()):
            raise FileNotFoundError()

        df = pd.read_csv(data_path)
        property = json.load(property_path.open(mode="r"))
        return cls(df, **property)
    
    def __len__(self) -> int:
        return len(self.__df)
