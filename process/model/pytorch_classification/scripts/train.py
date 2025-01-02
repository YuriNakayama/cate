from logging import Logger
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.utils import PathLink, dict_flatten

from .dataset import BinaryClassificationDataset, fix_seed, worker_init_fn
from .model import FullConnectedModel


def create_dataset(
    logger: Logger,
    link: PathLink,
) -> BinaryClassificationDataset:
    logger.info("load dataset")
    ds = cds.Dataset.load(link.mart)

    dataset = BinaryClassificationDataset(ds.to_frame(), ds.x_columns, ds.y_columns)
    return dataset


def train_model(
    model: Any,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: optim.Optimizer,
    criterion: Any,
    device: torch.device,
) -> tuple[Any, np.floating[Any], np.floating[Any]]:
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
    model.eval()
    test_batch_loss = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label.squeeze().to(torch.int64))
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


def train(
    cfg: DictConfig,
    pathlink: PathLink,
    client: MlflowClient,
    logger: Logger,
    parent_run_id: str,
) -> None:
    logger.info("Start training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    fix_seed(cfg.training.seed)
    logger.info("Start dataset creation")
    dataset = create_dataset(logger, pathlink)

    logger.info("Start model creation")
    model = FullConnectedModel(len(dataset.x_columns), 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr)

    client.start_run(
        run_name=f"{cfg.data.name}-{cfg.model.name}-rank_{cfg.data.rank}-random_ratio_{cfg.data.random_ratio}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "package": "pytorch",
            "mlflow.parentRunId": parent_run_id,
        },
        description=f"base_pattern: {cfg.model.name} training and evaluation using {cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
    )
    client.log_params(
        dict_flatten(cfg),
    )

    # 訓練の実行
    logger.info("Start training loop")
    for epoch in tqdm(range(cfg.training.epochs)):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, valid_index in skf.split(np.zeros(len(dataset)), dataset.y):
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

            model, train_loss, test_loss = train_model(
                model, train_loader, valid_loader, optimizer, criterion, device
            )

        # 10エポックごとにロスを表示
        if epoch % 2 == 0:
            logger.info(f"Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}")
