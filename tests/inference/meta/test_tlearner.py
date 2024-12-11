from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cate.inference.meta.tlearner import Tlearner
from cate.inference.meta.base import MetaLearnerException


class MockModel:
    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> MockModel:
        return self

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int_]:
        return np.array([[0.0, 1.0]] * X.shape[0])

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        return np.array([0] * X.shape[0])

    def __repr__(self) -> str:
        return "mock model"


def test_init_with_single_learner() -> None:
    mock_model = MockModel()
    tlearner = Tlearner(learner=mock_model)
    assert tlearner.model_c.__repr__() == "mock model"
    assert tlearner.model_t.__repr__() == "mock model"
    assert tlearner.control_name == 0


def test_init_with_control_and_treatment_learners() -> None:
    mock_model_c = MockModel()
    mock_model_t = MockModel()
    tlearner = Tlearner(control_learner=mock_model_c, treatment_learner=mock_model_t)
    assert tlearner.model_c.__repr__() == "mock model"
    assert tlearner.model_t.__repr__() == "mock model"
    assert tlearner.control_name == 0


def test_init_with_no_learners_raises_exception() -> None:
    with pytest.raises(MetaLearnerException) as excinfo:
        Tlearner()
    assert (
        "Either learner or control_learner and treatment_learner must be provided."
        in str(excinfo.value)
    )

def test_repr_with_single_learner() -> None:
    mock_model = MockModel()
    tlearner = Tlearner(learner=mock_model)
    expected_repr = "Tlearner(model_c=mock model, model_t=mock model)"
    assert repr(tlearner) == expected_repr


def test_repr_with_control_and_treatment_learners() -> None:
    mock_model_c = MockModel()
    mock_model_t = MockModel()
    tlearner = Tlearner(control_learner=mock_model_c, treatment_learner=mock_model_t)
    expected_repr = "Tlearner(model_c=mock model, model_t=mock model)"
    assert repr(tlearner) == expected_repr
    
def test_fit() -> None:
    mock_model = MockModel()
    tlearner = Tlearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    treatment = np.array([0, 1, 0, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    tlearner.fit(X, treatment, y)

    assert len(tlearner.t_groups) == 1
    assert tlearner.t_groups[0] == 1
    assert len(tlearner.models_c) == 1
    assert len(tlearner.models_t) == 1
    assert isinstance(tlearner.models_c[1], MockModel)
    assert isinstance(tlearner.models_t[1], MockModel)


def test_fit_with_multiple_treatment_groups() -> None:
    mock_model = MockModel()
    tlearner = Tlearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    treatment = np.array([0, 1, 0, 2, 1, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    tlearner.fit(X, treatment, y)

    assert len(tlearner.t_groups) == 2
    assert tlearner.t_groups[0] == 1
    assert tlearner.t_groups[1] == 2
    assert len(tlearner.models_c) == 2
    assert len(tlearner.models_t) == 2
    assert isinstance(tlearner.models_c[1], MockModel)
    assert isinstance(tlearner.models_c[2], MockModel)
    assert isinstance(tlearner.models_t[1], MockModel)
    assert isinstance(tlearner.models_t[2], MockModel)


def test_fit_with_no_control_group() -> None:
    mock_model = MockModel()
    tlearner = Tlearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    treatment = np.array([1, 1, 2, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    tlearner.fit(X, treatment, y)

    assert len(tlearner.t_groups) == 2
    assert tlearner.t_groups[0] == 1
    assert tlearner.t_groups[1] == 2
    assert len(tlearner.models_c) == 2
    assert len(tlearner.models_t) == 2
    assert isinstance(tlearner.models_c[1], MockModel)
    assert isinstance(tlearner.models_c[2], MockModel)
    assert isinstance(tlearner.models_t[1], MockModel)
    assert isinstance(tlearner.models_t[2], MockModel)

