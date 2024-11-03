import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklift.datasets.datasets import fetch_lenta

from cate.utils import path_linker


def download_from_kaggle(owner_slug: str, dataset_slug: str) -> pd.DataFrame:
    with TemporaryDirectory() as tmpdir:
        _ = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                f"{owner_slug}/{dataset_slug}",
                "--quiet",
                "--path",
                tmpdir,
            ]
        )
        return pd.read_csv(Path(tmpdir) / f"{dataset_slug}.zip")


def merge_Xyt(
    X: pd.DataFrame | pd.Series,
    y: pd.DataFrame | pd.Series,
    t: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    df = pd.merge(X, y, left_index=True, right_index=True)
    df = pd.merge(df, t, left_index=True, right_index=True)
    return df


# lenta
pathlinker = path_linker("lenta")
X, y, t = fetch_lenta(return_X_y_t=True)
df = merge_Xyt(X, y, t)
df.to_csv(pathlinker.origin, index=False)

# criteo
pathlinker = path_linker("criteo")
df = download_from_kaggle("arashnic", "uplift-modeling")
df.to_csv(pathlinker.origin, index=False)
