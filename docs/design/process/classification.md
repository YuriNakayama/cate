# classification task

## Usecase

```python
experiment = Experiment(cfg, client, logger, parent_run_id)
experiment()
```

## Interface

```python
class Experiment:
    def __init__(
        cfg: DictConfig, client: MlflowClient, logger: Logger, parent_run_id: str
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.logger = logger
        self.parent_run_id = parent_run_id
        self.dataset_creator = DatasetCreator()
        self.model_fetcher = ModelFetcher()
        self.trainer = Trainer(client, logger, optimizer, criterion, device)

    def __call__(self) -> None:
        datasets = self.dataset_creator(name: str, seed: int)
        for train_ds, test_ds in datasets:
            model = self.model_fetcher("model_name")
            model = self.trainer(model, train_ds, cfg.train)
            y_pred = model.predict(test_ds.X)
            metrics(y_pred, test_ds.y, test_ds.w)
```

## Class

```python
class DatasetCreator:
    def __call__(name: str, seed: int) -> list[tuple[Dataset, Dataset]]:
        pass

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
        logger: Logger
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
```
