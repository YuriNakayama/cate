import numpy as np
import pandas as pd
import pytest

from cate.metrics import Auuc


@pytest.fixture
def sample_data() -> tuple[pd.Series, pd.Series, pd.Series]:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([1, 0, 1, 0, 0])
    return score, group, conversion


def test_auuc_call(
    sample_data: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    score, group, conversion = sample_data
    auuc = Auuc(bin_num=5)
    result = auuc(score, group, conversion)
    assert isinstance(result, float)


def test_auuc_call_with_empty_data() -> None:
    score = pd.Series([])
    group = pd.Series([])
    conversion = pd.Series([])
    auuc = Auuc(bin_num=5)
    result = auuc(score, group, conversion)
    assert result == 0.0

    @pytest.fixture
    def sample_data() -> tuple[pd.Series, pd.Series, pd.Series]:
        score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
        group = pd.Series([1, 0, 1, 0, 1])
        conversion = pd.Series([1, 0, 1, 0, 0])
        return score, group, conversion


def test_auuc_call_with_all_zero_conversion() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([0, 0, 0, 0, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result == 0.0


def test_auuc_call_with_all_one_conversion() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6])
    group = pd.Series([1, 0, 1, 0])
    conversion = pd.Series([1, 1, 1, 1])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result == 0.0


def test_auuc_call_with_mixed_conversion() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6])
    group = pd.Series([1, 0, 1, 0])
    conversion = pd.Series([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result == 1.0


def test_auuc_call_with_random_data() -> None:
    score = pd.Series(np.random.rand(100))
    group = pd.Series(np.random.randint(0, 2, 100_00))
    conversion = pd.Series(np.random.randint(0, 2, 100_00))
    auuc = Auuc(bin_num=10)
    result = auuc(score, group, conversion)
    assert isinstance(result, float)
    pytest.approx(result, 0.0, 0.1)


def test_auuc_call_with_no_treatment_group() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6])
    group = pd.Series([0, 0, 0, 0])
    conversion = pd.Series([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result == 0.0


def test_auuc_call_with_no_control_group() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6])
    group = pd.Series([1, 1, 1, 1])
    conversion = pd.Series([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result == 0.0
