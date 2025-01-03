# classification task

## usecase

```python
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    train_ds = ds[train_index]
    test_ds = ds[test_index]
    model = model_getter("model_name")
    model = train(model, train_ds)
    y_pred = model.predict(test_ds.X)
    metrics(y_pred, test_ds.y, test_ds.w)
```
