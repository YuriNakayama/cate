# Metrics

## Usage

```python
from cate.mlflow import MlflowClient
from cate.metrics import Auuc, Metrics, UpliftCurve, Artifacts

client = MlflowClient("example")
client.start_run()

metrics = Metrics(Auuc())
artifacts = Artifacts(UpliftCurve())
for model in models:
    for epoch in epochs:
        artifacts.log(score, group, conversion)
        metrics.log(score, group, conversion)
    client.log_metrics(metrics.to_dict())
    artifacts.clear()
    client.log_artifacts(artifacts.to_dict())
    metrics.clear()

client.end_run()
```
