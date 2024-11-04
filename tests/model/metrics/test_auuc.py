import numpy as np
import numpy.typing as npt
import pytest

from cate.model.evaluate import Auuc


@pytest.fixture
def sample_data() -> (
    tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.int_ | np.float_],
        npt.NDArray[np.int_ | np.float_],
    ]
):
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([1, 0, 1, 0, 0])
    return score, group, conversion


def test_auuc_call(
    sample_data: tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.int_ | np.float_],
        npt.NDArray[np.int_ | np.float_],
    ],
) -> None:
    score, group, conversion = sample_data
    auuc = Auuc(bin_num=5)
    result = auuc(score, group, conversion)
    assert isinstance(result.data, float)
    assert result.name == "auuc"


def test_auuc_call_with_empty_data() -> None:
    score = np.array([])
    group = np.array([])
    conversion = np.array([])
    auuc = Auuc(bin_num=5)
    result = auuc(score, group, conversion)
    assert result.data == 0.0


def test_auuc_call_with_all_zero_conversion() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([0, 0, 0, 0, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result.data == 0.0


def test_auuc_call_with_all_one_conversion() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    group = np.array([1, 0, 1, 0])
    conversion = np.array([1, 1, 1, 1])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result.data == 0.0


def test_auuc_call_with_mixed_conversion() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    group = np.array([1, 0, 1, 0])
    conversion = np.array([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result.data == 1.0


def test_auuc_call_with_random_data() -> None:
    score = np.array(np.random.rand(10_000))
    group = np.array(np.random.randint(0, 2, 10_000))
    conversion = np.array(np.random.randint(0, 2, 10_000))
    auuc = Auuc(bin_num=10)
    result = auuc(score, group, conversion)
    assert isinstance(result.data, float)
    pytest.approx(result.data, 0.0, 0.1)


def test_auuc_call_with_no_treatment_group() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    group = np.array([0, 0, 0, 0])
    conversion = np.array([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result.data == 0.0


def test_auuc_call_with_no_control_group() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    group = np.array([1, 1, 1, 1])
    conversion = np.array([1, 0, 1, 0])
    auuc = Auuc(bin_num=2)
    result = auuc(score, group, conversion)
    assert result.data == 0.0
