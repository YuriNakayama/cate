import multiprocessing as mp
import random

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset

mp.set_start_method("spawn", force=True)


def fix_seed(seed) -> None:
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0].item() + worker_id)


class BinaryClassificationDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, x_columns: list[str], y_columns: list[str]
    ) -> None:
        self._df = df.clone()
        self.x_columns = x_columns
        self.y_columns = y_columns

    def __len__(self) -> int:
        return self._df.select(pl.len()).to_numpy()[0][0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        features = np.array(self._df.select(self.x_columns).row(idx))
        target = np.array(self._df.select(self.y_columns).row(idx))
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )

    @property
    def y(self) -> Tensor:
        return self._df.select(self.y_columns).to_torch()
