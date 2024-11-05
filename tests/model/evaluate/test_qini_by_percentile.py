import numpy as np
import pytest

from cate.model.evaluate import QiniByPercentile


def test_qini_by_percentile_calculate() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([1, 0, 1, 0, 1])
    k = 0.6

    qini_metric = QiniByPercentile(k)
    result = qini_metric._calculate(pred, y, w)

    assert isinstance(result, float)
    assert result == pytest.approx(2.0, rel=1e-2)


def test_qini_by_percentile_calculate_no_treatment() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([0, 0, 0, 0, 0])
    k = 0.6

    qini_metric = QiniByPercentile(k)
    result = qini_metric._calculate(pred, y, w)

    assert isinstance(result, float)
    assert result == 0.0


def test_qini_by_percentile_calculate_no_control() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([1, 1, 1, 1, 1])
    k = 0.6

    qini_metric = QiniByPercentile(k)
    result = qini_metric._calculate(pred, y, w)

    assert isinstance(result, float)
    assert result == 0.0
