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

```python
class BaseLearner(ABC):
    @abstractmethod
    def fit(
        self,
        X: npt.NDArray[Any],
        treatment: npt.NDArray[np.int_],
        y: npt.NDArray[np.float_ | np.int_],
        p: npt.NDArray[np.float_] | None = None,
        verbose: int = 1
    ) -> BaseLearner:
        pass

    @abstractmethod
    def predict(
        self, X: npt.NDArray[Any], p: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float64]:
        pass

```

## Reference

- [causalml](https://github.com/uber/causalml)
- [lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)
