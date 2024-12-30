from logging import Logger
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

from cate.infra.mlflow import MlflowClient
from cate.utils.path import AbstractLink

from .dataset import BinaryClassificationDataset, fix_seed, worker_init_fn
from .model import FullConnectedModel


def create_dataset() -> BinaryClassificationDataset:
    df = pl.read_csv("/workspace/data/origin/criteo.csv").head(100_000)
    y_columns = ["visit"]
    other_columns = ["treatment", "exposure", "conversion"]
    X_columns = [col for col in df.columns if col not in y_columns + other_columns]
    dataset = BinaryClassificationDataset(df, X_columns, y_columns)
    return dataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, np.floating[Any], np.floating[Any]]:
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader:
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label.squeeze().to(torch.int64))
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label.squeeze().to(torch.int64))
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


def train(
    cfg: DictConfig,
    pathlink: AbstractLink,
    client: MlflowClient,
    logger: Logger,
    parent_run_id: str,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    fix_seed(cfg.training.seed)
    dataset = create_dataset()

    model = FullConnectedModel(len(dataset.x_columns), 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr)

    # 訓練の実行
    train_loss = []
    test_loss = []

    for epoch in tqdm(range(cfg.training.epochs)):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, valid_index in skf.split(range(len(dataset)), dataset.y):
            train_dataset = Subset(dataset, train_index)
            valid_dataset = Subset(dataset, valid_index)

            # データローダーの作成
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.train_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=cfg.training.valid_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )

            model, train_l, test_l = train_model(
                model, train_loader, valid_loader, optimizer, criterion, device
            )
            train_loss.append(train_l)
            test_loss.append(test_l)
            # 10エポックごとにロスを表示
            if epoch % 10 == 0:
                print(
                    "Train loss: {a:.3f}, Test loss: {b:.3f}".format(
                        a=train_loss[-1], b=test_loss[-1]
                    )
                )
