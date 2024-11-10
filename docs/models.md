# Uplift Model

## UseCase

```python
from cate.model import Tlearner


client = MlflowClient("example")


metrics = Metrics([Auuc()])
artifacts = Artifacts([UpliftCurve()])
client.start_run()
for epoch in epochs:
    
    metrics(score, group, conversion)
    client.log_metrics(metrics)
artifacts(score, group, conversion)
client.log_artifacts(artifacts)
client.end_run()
```
