import click
import pandas as pd

from cate.dataset import Dataset
from cate.utils import path_linker, send_messages


def onehot_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    _df = df.copy()
    for column in columns:
        _df = pd.merge(
            _df.drop(column, axis=1),
            pd.get_dummies(df[column], prefix=column, dtype=int),
            left_index=True,
            right_index=True,
        )
    return _df


def label_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    _df = df.copy()
    for column in columns:
        _df[column] = _df[column].astype("category").cat.codes
    return _df


def make_lenta() -> None:
    pathlinker = path_linker("lenta")
    df = pd.read_csv(pathlinker.origin)
    df = onehot_encoding(df, ["gender"])
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


def make_hillstorm() -> None:
    pathlinker = path_linker("hillstorm")
    df = pd.read_csv(pathlinker.origin)
    df = label_encoding(df, ["history_segment"])
    df = onehot_encoding(df, ["zip_code", "channel"])
    df = df.fillna(0)
    df["segment"] = df["segment"].apply(
        lambda x: {"Mens E-Mail": 1, "Womens E-Mail": 1, "No E-Mail": 0}.get(x)
    )
    y_columns = ["conversion"]
    w_columns = ["segment"]
    x_columns = [column for column in df.columns if column not in y_columns + w_columns]
    ds = Dataset(df, x_columns, y_columns, w_columns)
    ds.save(pathlinker.base)


def make_megafon() -> None:
    pathlinker = path_linker("megafon")
    df = pd.read_csv(pathlinker.origin)
    df["treatment_group"] = df["treatment_group"].apply(
        lambda x: {"treatment": 1, "control": 0}.get(x)
    )
    y_columns = ["conversion"]
    w_columns = ["treatment_group"]
    x_columns = [column for column in df.columns if column not in y_columns + w_columns]
    ds = Dataset(df, x_columns, y_columns, w_columns)
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
    match name:
        case "lenta":
            click.echo("create lenta")
            make_lenta()
        case "criteo":
            click.echo("create criteo")
            make_criteo()
        case "hillstorm":
            click.echo("create hillstorm")
            make_hillstorm()
        case "megafon":
            click.echo("create megafon")
            make_megafon()
        case "test":
            click.echo("create test")
            make_test()
        case "all":
            click.echo("create lenta")
            make_lenta()
            click.echo("create criteo")
            make_criteo()
            click.echo("create hillstorm")
            make_hillstorm()
            click.echo("create megafon")
            make_megafon()
            click.echo("create test")
            make_test()


if __name__ == "__main__":
    main()
    send_messages(["dataset createds"])
