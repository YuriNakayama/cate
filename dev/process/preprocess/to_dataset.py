import click
import pandas as pd

from cate.dataset import Dataset
from cate.utils import PathLinker

pathlinker = PathLinker()


def make_lenta() -> None:
    df = pd.read_csv(pathlinker.data.lenta.origin)
    df = pd.merge(
        df.drop("gender", axis=1),
        pd.get_dummies(df["gender"], prefix="gender", dtype=int),
        left_index=True,
        right_index=True,
    )
    df["group"] = df["group"].apply(lambda x: {"test": 1, "control": 0}.get(x))
    y_columns = ["response_att"]
    w_columns = ["group"]
    x_columns = [column for column in df.columns if column not in y_columns + w_columns]
    ds = Dataset(df, x_columns, y_columns, w_columns)
    ds.save(pathlinker.data.lenta.base)


def make_criteo() -> None:
    df = pd.read_csv(pathlinker.data.criteo.origin)
    ds = Dataset(
        df,
        ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        ["conversion"],
        ["treatment"],
    )
    ds.save(pathlinker.data.criteo.base)


def make_test() -> None:
    df = pd.read_csv(pathlinker.data.test.origin).sample(n=100_000, random_state=42)
    ds = Dataset(
        df,
        ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        ["conversion"],
        ["treatment"],
    )
    ds.save(pathlinker.data.test.base)


@click.command()
@click.argument(
    "name",
    required=True,
)
def main(name: str) -> None:
    if "lenta" == name:
        click.echo("download lenta")
        make_lenta()
    elif "criteo" == name:
        click.echo("download criteo")
        make_criteo()
    elif "test" == name:
        click.echo("download test")
        make_test()
    elif "all" == name:
        click.echo("download lenta")
        make_lenta()
        click.echo("download criteo")
        make_criteo()
        click.echo("download test")
        make_test()


if __name__ == "__main__":
    main()
