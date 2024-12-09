# Uplift Model

## UseCase

```python
from cate.model import Tlearner

xlearner = Xlearner(learner=lgbm_classifier)
for epoch in epochs:
    xlearner.fit(
        X, 
        y, 
        w, 
        eval_set=[(X_eval, y_eval, w_eval)], 
        p=None, 
        sample_weight=None, 
        verbose=1
    )
    xlearner.predict(X, p=None)
    
```

## Reference

- [causalml](https://github.com/uber/causalml)
- [lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)
