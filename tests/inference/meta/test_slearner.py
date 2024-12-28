from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from lightgbm import LGBMClassifier

from cate.base.inference.meta import MetaLearnerException
from cate.dataset import synthetic_data
from cate.inference.meta import Slearner


class MockModel:
    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        eval_set: list[
            tuple[
                npt.NDArray[Any],
                npt.NDArray[np.int_],
                npt.NDArray[np.float_ | np.int_],
            ]
        ]
        | None = None,
    ) -> MockModel:
        return self

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int_]:
        return np.array([[0.0, 1.0]] * X.shape[0])

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        return np.array([[0.5, 0.5]] * X.shape[0])

    def __repr__(self) -> str:
        return "mock model"


def test_init_with_single_learner() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)
    assert slearner.model.__repr__() == "mock model"
    assert slearner.control_name == 0


def test_repr_with_single_learner() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)
    expected_repr = "Slearner(model=mock model)"
    assert repr(slearner) == expected_repr


def test_fit() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    w = np.array([0, 1, 0, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    slearner.fit(X, w, y)

    assert len(slearner.t_groups) == 1
    assert slearner.t_groups[0] == 1
    assert len(slearner.models) == 1
    assert isinstance(slearner.models[1], MockModel)


def test_fit_with_multiple_treatment_groups() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    w = np.array([0, 1, 0, 2, 1, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    slearner.fit(X, w, y)

    assert len(slearner.t_groups) == 2
    assert slearner.t_groups[0] == 1
    assert slearner.t_groups[1] == 2
    assert len(slearner.models) == 2
    assert isinstance(slearner.models[1], MockModel)
    assert isinstance(slearner.models[2], MockModel)


def test_fit_with_no_control_group() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    w = np.array([1, 1, 2, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    slearner.fit(X, w, y)

    assert len(slearner.t_groups) == 2
    assert slearner.t_groups[0] == 1
    assert slearner.t_groups[1] == 2
    assert len(slearner.models) == 2
    assert isinstance(slearner.models[1], MockModel)
    assert isinstance(slearner.models[2], MockModel)


def test_predict() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    w = np.array([0, 1, 0, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    slearner.fit(X, w, y)
    te = slearner.predict(X)

    assert te.shape == (4, 1)
    assert np.all(te == 0)


def test_predict_with_multiple_treatment_groups() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    w = np.array([0, 1, 0, 2, 1, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    slearner.fit(X, w, y)
    te = slearner.predict(X)

    assert te.shape == (6, 2)
    assert np.all(te == 0)


def test_predict_with_no_control_group() -> None:
    mock_model = MockModel()
    slearner = Slearner(learner=mock_model)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    w = np.array([1, 1, 2, 2])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    slearner.fit(X, w, y)
    te = slearner.predict(X)

    assert te.shape == (4, 2)
    assert np.all(te == 0)


def test_fit_with_synthetic_data_and_lgbm() -> None:
    base_model = LGBMClassifier(verbosity=-1)
    slearner = Slearner(learner=base_model)

    X, w, y = synthetic_data()
    slearner.fit(X, w, y)

    assert len(slearner.t_groups) == 1
    assert len(slearner.models) == 1
    for group in slearner.t_groups:
        assert isinstance(slearner.models[group], LGBMClassifier)


def test_predict_with_synthetic_data_and_lgbm_classifier() -> None:
    base_model = LGBMClassifier(verbosity=-1)
    slearner = Slearner(learner=base_model)

    X, w, y = synthetic_data()
    slearner.fit(X, w, y)
    te = slearner.predict(X)

    assert te.shape[0] == X.shape[0]
    assert te.shape[1] == len(slearner.t_groups)
