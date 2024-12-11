from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from causalml.inference.meta.utils import (
    check_treatment_vector,
    convert_pd_to_np,
)
from sklearn.dummy import DummyClassifier

from cate.base.inference.meta import (
    AbstractMetaLearner,
    Classifier,
    MetaLearnerException,
    Regressor,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class Xlearner(AbstractMetaLearner):
    """A parent class for X-learner regressor classes.

    An X-learner estimates treatment effects with four machine learning models.

    Details of X-learner are available at `Kunzel et al. (2018) <https://arxiv.org/abs/1706.03461>`_.
    """

    def __init__(
        self,
        learner: Classifier | None = None,
        control_outcome_learner: Classifier | None = None,
        treatment_outcome_learner: Classifier | None = None,
        control_effect_learner: Classifier | None = None,
        treatment_effect_learner: Classifier | None = None,
        propensity_model: Classifier | None = None,
        control_name: int = 0,
    ) -> None:
        if (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (control_effect_learner is not None)
            and (treatment_effect_learner is not None)
        ):
            raise MetaLearnerException(
                "Either learner or control_outcome_learner, treatment_outcome_learner, control_effect_learner, and treatment_effect_learner must be provided."
            )

        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if control_effect_learner is None:
            self.model_tau_c = deepcopy(learner)
        else:
            self.model_tau_c = control_effect_learner

        if treatment_effect_learner is None:
            self.model_tau_t = deepcopy(learner)
        else:
            self.model_tau_t = treatment_effect_learner

        self.control_name = control_name

        if propensity_model is None:
            self.propensity_model = DummyClassifier(strategy="prior")
        else:
            self.propensity_model = propensity_model

    def __repr__(self) -> str:
        return (
            "{}(control_outcome_learner={},\n"
            "\ttreatment_outcome_learner={},\n"
            "\tcontrol_effect_learner={},\n"
            "\ttreatment_effect_learner={})".format(
                self.__class__.__name__,
                self.model_mu_c.__repr__(),
                self.model_mu_t.__repr__(),
                self.model_tau_c.__repr__(),
                self.model_tau_t.__repr__(),
            )
        )

    def fit(
        self,
        X: npt.NDArray[Any],
        w: npt.NDArray[np.int_],
        y: npt.NDArray[np.float_ | np.int_],
        eval_set: list[
            tuple[
                npt.NDArray[Any],
                npt.NDArray[np.int_],
                npt.NDArray[np.float_ | np.int_],
            ]
        ]
        | None = None,
        params: dict[str, Any] | None = None,
    ) -> Xlearner:
        self.t_groups = np.unique(w[w != self.control_name])
        self.t_groups.sort()

        
        self.propensity_model.fit(X, w)
        
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_mu_c = {group: deepcopy(self.model_mu_c) for group in self.t_groups}
        self.models_mu_t = {group: deepcopy(self.model_mu_t) for group in self.t_groups}
        self.models_tau_c = {
            group: deepcopy(self.model_tau_c) for group in self.t_groups
        }
        self.models_tau_t = {
            group: deepcopy(self.model_tau_t) for group in self.t_groups
        }
        self.vars_c = {}
        self.vars_t = {}

        for group in self.t_groups:
            mask = (w == group) | (w == self.control_name)
            w_filt = w[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            w = (w_filt == group).astype(int)

            # Train outcome models
            self.models_mu_c[group].fit(X_filt[w == 0], y_filt[w == 0])
            self.models_mu_t[group].fit(X_filt[w == 1], y_filt[w == 1])

            # Calculate variances and treatment effects
            var_c = (
                y_filt[w == 0] - self.models_mu_c[group].predict(X_filt[w == 0])
            ).var()
            self.vars_c[group] = var_c
            var_t = (
                y_filt[w == 1] - self.models_mu_t[group].predict(X_filt[w == 1])
            ).var()
            self.vars_t[group] = var_t

            # Train treatment models
            d_c = self.models_mu_t[group].predict(X_filt[w == 0]) - y_filt[w == 0]
            d_t = y_filt[w == 1] - self.models_mu_c[group].predict(X_filt[w == 1])
            self.models_tau_c[group].fit(X_filt[w == 0], d_c)
            self.models_tau_t[group].fit(X_filt[w == 1], d_t)
        return self

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            logger.info("Generating propensity score")
            p = dict()
            for group in self.t_groups:
                p_model = self.propensity_model[group]
                p[group] = p_model.predict(X)
        else:
            p = self._format_p(p, self.t_groups)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            model_tau_c = self.models_tau_c[group]
            model_tau_t = self.models_tau_t[group]
            dhat_cs[group] = model_tau_c.predict(X)
            dhat_ts[group] = model_tau_t.predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_ts[group]).reshape(
                -1, 1
            )
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict(X_filt[w == 0])
                yhat[w == 1] = self.models_mu_t[group].predict(X_filt[w == 1])

                logger.info("Error metrics for group {}".format(group))
                regression_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts
