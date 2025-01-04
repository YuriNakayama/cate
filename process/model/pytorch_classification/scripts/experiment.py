from logging import Logger
from typing import Any, Generator

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.utils import PathLink, dict_flatten

from .dataset import BinaryClassificationDataset, fix_seed, worker_init_fn
from .model import FullConnectedModel


class DatasetGenerator:
    def __init__(self, link: PathLink, logger: Logger) -> None:
        self.link = link
        self.logger = logger
        self.logger.info("load dataset")
        ds = cds.Dataset.load(link.mart)
        self.dataset = BinaryClassificationDataset(
            ds.to_frame(), ds.x_columns, ds.y_columns
        )

    def __call__(
        self, seed: int = 42
    ) -> Generator[tuple[Subset[Any], Subset[Any]], Any, None]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, valid_index in skf.split(
            np.zeros(len(self.dataset)), self.dataset.y
        ):
            train_dataset = Subset(self.dataset, train_index)
            valid_dataset = Subset(self.dataset, valid_index)
            yield train_dataset, valid_dataset


class Experiment:
    def __init__(
        self,
        cfg: DictConfig,
        link: PathLink,
        client: MlflowClient,
        logger: Logger,
        parent_run_id: str,
    ) -> None:
        self.cfg = cfg
        self.link = link
        self.client = client
        self.logger = logger
        self.parent_run_id = parent_run_id

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        fix_seed(self.cfg.training.seed)

        self.dataset_generator = DatasetGenerator(self.link, self.logger)
        self.trainer = Trainer(self.client, self.logger, self.cfg.train, self.device)

    def __call__(self) -> None:
        self.logger.info("Start training")

        self.client.start_run(
            run_name=f"{self.cfg.data.name}-{self.cfg.model.name}",
            tags={
                "model": self.cfg.model.name,
                "dataset": self.cfg.data.name,
                "package": "pytorch",
                "mlflow.parentRunId": self.parent_run_id,
            },
            description=f"base_pattern: {self.cfg.model.name} training and evaluation using {self.cfg.data.name} dataset with causalml package and lightgbm model with 5-fold cross validation and stratified sampling.",  # noqa: E501
        )
        self.client.log_params(dict_flatten(self.cfg))
        for fold, (train_ds, test_ds) in enumerate(self.dataset_generator()):
            self.logger.info(f"Start model creation (fold: {fold})")
            model = FullConnectedModel(len(train_ds), 2).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.training.lr)

            model = self.trainer(model, criterion, optimizer, train_ds)
            y_pred = model.predict(test_ds)
            metrics(y_pred, test_ds.y, test_ds.w)


class Trainer:
    def __init__(
        self,
        client: MlflowClient,
        logger: Logger,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.client = client
        self.logger = logger
        self.cfg = cfg
        self.device = device

    def _train(
        self,
        model: Any,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        valid_loader: DataLoader[tuple[Tensor, Tensor]],
        optimizer: optim.Optimizer,
        criterion: Any,
    ) -> Any:
        # Train loop ----------------------------
        model.train()
        train_batch_loss = []
        for data, label in train_loader:
            data, label = data.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label.squeeze().to(torch.int64))
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
        # Test(val) loop ----------------------------
        model.eval()
        valid_batch_loss = []
        with torch.no_grad():
            for data, label in valid_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = model(data)
                loss = criterion(output, label.squeeze().to(torch.int64))
                valid_batch_loss.append(loss.item())

        return model

    def __call__(
        self,
        model: Any,
        criterion: Any,
        optimizer: optim.Optimizer,
        ds: BinaryClassificationDataset,
        seed: int = 42,
    ) -> Any:
        self.logger.info("Start training loop")

        for epoch in tqdm(range(self.cfg.training.epochs)):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_index, valid_index in skf.split(np.zeros(len(ds)), ds.y):
                train_dataset = Subset(ds, train_index)
                valid_dataset = Subset(ds, valid_index)

                # データローダーの作成
                train_loader = DataLoader(
                    train_dataset,
                    **self.cfg.training.train_loader,
                    worker_init_fn=worker_init_fn,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    **self.cfg.training.valid_loader,
                    worker_init_fn=worker_init_fn,
                )

                model, train_loss, test_loss = self._train(
                    model, train_loader, valid_loader, optimizer, criterion
                )
                mlflow.log_metrics(
                    {"train_loss": float(train_loss), "test_loss": float(test_loss)},
                    step=epoch,
                )

            # 1エポックごとにロスを表示
            if epoch % 1 == 0:
                self.logger.info(
                    f"Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}"
                )
