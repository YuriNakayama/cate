# classification task

## usecase

```python
def experiment(
    cfg: DictConfig, client: MlflowClient, logger: Logger, parent_run_id: str
) -> None:
    datasets = create_datasets(name, seed)
    for train_ds, test_ds in datasets:
        model = fetch_model("model_name")
        model = train(model, train_ds)
        y_pred = model.predict(test_ds.X)
        metrics(y_pred, test_ds.y, test_ds.w)
```

## interface

```python
def create_datasets(name: str, seed: int) -> list[tuple[Dataset, Dataset]]:
    pass

def fetch_model(name: str) -> Classifier:
    pass

def train(model: Classifier, ds: Dataset) -> Classifier:
    pass
```
