from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from cate.base.inference.meta import (
    AbstractMetaLearner,
    Classifier,
    MetaLearnerException,
    Regressor,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class Slearner(AbstractMetaLearner):
    """A parent class for S-learner classes.
    An S-learner estimates treatment effects with one machine learning model.
    Details of S-learner are available at `Kunzel et al. (2018) <https://arxiv.org/abs/1706.03461>`_.
    """

    def __init__(self, learner: Classifier, control_name: int = 0) -> None:
        self.model = learner
        self.control_name = control_name

    def __repr__(self) -> str:
        return "{}(model={})".format(self.__class__.__name__, self.model.__repr__())

    def fit(
        self,
        X: npt.NDArray[Any],
        w: npt.NDArray[np.int_],
        y: npt.NDArray[np.float_ | np.int_],
        p: npt.NDArray[np.float_] | None = None,
        eval_set: list[
            tuple[
                npt.NDArray[Any],
                npt.NDArray[np.int_],
                npt.NDArray[np.float_ | np.int_],
                npt.NDArray[np.float_] | None,
            ]
        ]
        | None = None,
        params: dict[str, Any] | None = None,
    ) -> Slearner:
        self.t_groups = np.unique(w[w != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models = {group: deepcopy(self.model) for group in self.t_groups}

        for group in self.t_groups:
            mask = (w == group) | (w == self.control_name)
            w_filt = w[mask]
            X_filt = X[mask]
            y_filt = y[mask]

            w = (w_filt == group).astype(int)
            X_new = np.hstack((w.reshape((-1, 1)), X_filt))
            self.models[group].fit(X_new, y_filt)
        return self

    def predict(
        self,
        X: npt.NDArray[Any],
        p: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float64]:
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model = self.models[group]

            # set the treatment column to zero (the control group)
            X_new = np.hstack((np.zeros((X.shape[0], 1)), X))
            yhat_cs[group] = model.predict_proba(X_new)

            # set the treatment column to one (the treatment group)
            X_new[:, 0] = 1
            yhat_ts[group] = model.predict_proba(X_new)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = yhat_ts[group] - yhat_cs[group]

        return te
