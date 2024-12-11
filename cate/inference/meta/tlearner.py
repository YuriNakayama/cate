from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from cate.inference.meta.base import AbstractMetaLearner, Classifier, MetaLearnerException

if TYPE_CHECKING:
    import numpy.typing as npt


class Tlearner(AbstractMetaLearner):
    """A parent class for T-learner regressor classes.

    A T-learner estimates treatment effects with two machine learning models.

    Details of T-learner are available at `Kunzel et al. (2018) <https://arxiv.org/abs/1706.03461>`_.
    """

    def __init__(
        self,
        learner: Classifier | None = None,
        control_learner: Classifier | None = None,
        treatment_learner: Classifier | None = None,
        control_name: int = 0,
    ) -> None:
        if learner is None:
            if control_learner is not None and treatment_learner is not None:
                self.model_c = deepcopy(control_learner)
                self.model_t = deepcopy(treatment_learner)
            else:
                raise MetaLearnerException(
                    "Either learner or control_learner and treatment_learner must be provided."
                )
        else:
            self.model_c = deepcopy(learner)
            self.model_t = deepcopy(learner)

        self.control_name = control_name

    def __repr__(self) -> str:
        return "{}(model_c={}, model_t={})".format(
            self.__class__.__name__, self.model_c.__repr__(), self.model_t.__repr__()
        )

    def fit(
        self,
        X: npt.NDArray[Any],
        treatment: npt.NDArray[np.int_],
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
        verbose: int = 1,
    ) -> Tlearner:
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_c = {group: deepcopy(self.model_c) for group in self.t_groups}
        self.models_t = {group: deepcopy(self.model_t) for group in self.t_groups}

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            w = (treatment_filt == group).astype(int)

            self.models_c[group].fit(X_filt[w == 0], y_filt[w == 0])
            self.models_t[group].fit(X_filt[w == 1], y_filt[w == 1])
        return self

    def predict(
        self,
        X: npt.NDArray[Any],
        p: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float64]:
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model_c = self.models_c[group]
            model_t = self.models_t[group]
            yhat_cs[group] = model_c.predict_proba(X)
            yhat_ts[group] = model_t.predict_proba(X)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = yhat_ts[group] - yhat_cs[group]

        return te
