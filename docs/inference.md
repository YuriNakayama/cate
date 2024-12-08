# Uplift Model

## UseCase

```python
from cate.model import Tlearner

tlearner = Tlearner(learner=lgbm_classifier)
for epoch in epochs:
    tlearner.fit(
        X, 
        y, 
        w, 
        eval_set=[(X_eval, y_eval, w_eval)], 
        p=None, 
        sample_weight=None, 
        verbose=1
    )
    tlearner.predict(X, p=None)
    
```
