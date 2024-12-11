import click
import pandas as pd

from cate.dataset import Dataset
from cate.utils import path_linker


def make_lenta() -> None:
    pathlinker = path_linker("lenta")
    df = pd.read_csv(pathlinker.origin)
    df = pd.merge(
        df.drop("gender", axis=1),
        pd.get_dummies(df["gender"], prefix="gender", dtype=int),
        left_index=True,
        right_index=True,
    )
    df = df.fillna(0)
    df["group"] = df["group"].apply(lambda x: {"test": 1, "control": 0}.get(x))
    y_columns = ["response_att"]
    w_columns = ["group"]
    x_columns = [column for column in df.columns if column not in y_columns + w_columns]
    ds = Dataset(df, x_columns, y_columns, w_columns)
    ds.save(pathlinker.base)


def make_criteo() -> None:
    pathlinker = path_linker("criteo")
    df = pd.read_csv(pathlinker.origin)
    ds = Dataset(
        df,
        ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        ["conversion"],
        ["treatment"],
    )
    ds.save(pathlinker.base)


def make_test() -> None:
    pathlinker = path_linker("test")
    df = pd.read_csv(pathlinker.origin).sample(n=100_000, random_state=42)
    ds = Dataset(
        df,
        ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        ["conversion"],
        ["treatment"],
    )
    ds.save(pathlinker.base)


@click.command()
@click.argument(
    "name",
    required=True,
)
def main(name: str) -> None:
    if "lenta" == name:
        click.echo("create lenta")
        make_lenta()
    elif "criteo" == name:
        click.echo("create criteo")
        make_criteo()
    elif "test" == name:
        click.echo("create test")
        make_test()
    elif "all" == name:
        click.echo("create lenta")
        make_lenta()
        click.echo("create criteo")
        make_criteo()
        click.echo("create test")
        make_test()


if __name__ == "__main__":
    main()
