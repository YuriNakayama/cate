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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cate.dataset as cds
from cate.infra.mlflow import MlflowClient
from cate.utils import PathLink, dict_flatten

from .dataset import BinaryClassificationDataset, fix_seed, worker_init_fn
from .model import FullConnectedModel


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
        self.dataset_generator = DatasetGenerator(link, logger)
        self.model_fetcher = ModelFetcher()
        self.trainer = Trainer(client, logger, optimizer, criterion, device)

    def __call__(self) -> None:
        for train_ds, test_ds in self.dataset_generator(self.cfg.training):
            model = self.model_fetcher("model_name")
            model = self.trainer(model, train_ds, cfg.train)
            y_pred = model.predict(test_ds.X)
            metrics(y_pred, test_ds.y, test_ds.w)


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
        self, config: DictConfig, seed: int = 42
    ) -> Generator[tuple[DataLoader[Any], DataLoader[Any]], Any, None]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, valid_index in skf.split(
            np.zeros(len(self.dataset)), self.dataset.y
        ):
            train_dataset = Subset(self.dataset, train_index)
            valid_dataset = Subset(self.dataset, valid_index)
            train_loader = DataLoader(
                train_dataset,
                **config.train_loader,
                worker_init_fn=worker_init_fn,
            )
            valid_loader = DataLoader(
                valid_dataset,
                **config.valid_loader,
                worker_init_fn=worker_init_fn,
            )
            yield train_loader, valid_loader


class ModelFetcher:
    def __call__(name: str) -> Classifier:
        pass


class Trainer:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        criterion: Any,
        device: torch.device,
        client: MlflowClient,
        logger: Logger,
    ) -> None:
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.client = client
        self.logger = logger

    def _train(
        self,
        model: Any,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        test_loader: DataLoader[tuple[Tensor, Tensor]],
    ):
        pass

    def __call__(model: Classifier, ds: Dataset, client: MlflowClient) -> Classifier:
        pass
