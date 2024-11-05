import numpy as np

from cate.model.evaluate import UpliftByPercentile


def test_uplift_by_percentile() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([1, 0, 1, 0, 0])
    uplift = UpliftByPercentile(k=0.4)

    result = uplift(score, group, conversion)

    assert result.data == 1.0, f"Expected uplift to be 1.0, but got {result.data}"
    assert (
        result.name == "uplift_at_40"
    ), f"Expected name to be uplift_at_40, but got {result.name}"


def test_uplift_by_percentile_no_conversion() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([0, 0, 0, 0, 0])
    uplift = UpliftByPercentile(k=0.4)

    result = uplift(score, conversion, group)

    assert result.data == 0.0, f"Expected uplift to be 0.0, but got {result.data}"
    assert (
        result.name == "uplift_at_40"
    ), f"Expected name to be uplift_at_40, but got {result.name}"


def test_uplift_by_percentile_all_conversion() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([1, 1, 1, 1, 1])
    uplift = UpliftByPercentile(k=0.4)

    result = uplift(score, conversion, group)

    assert result.data == 0.0, f"Expected uplift to be 0.0, but got {result.data}"
    assert (
        result.name == "uplift_at_40"
    ), f"Expected name to be uplift_at_40, but got {result.name}"


def test_uplift_by_percentile_different_k() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([1, 0, 1, 0, 1])
    conversion = np.array([1, 0, 0, 0, 0])
    uplift = UpliftByPercentile(k=0.6)

    result = uplift(score, conversion, group)

    assert result.data == 0.5, f"Expected uplift to be 0.5, but got {result.data}"
    assert (
        result.name == "uplift_at_60"
    ), f"Expected name to be uplift_at_60, but got {result.name}"


def test_uplift_by_percentile_no_treatment() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    group = np.array([0, 0, 0, 0, 0])
    conversion = np.array([1, 0, 1, 0, 0])
    uplift = UpliftByPercentile(k=0.4)

    result = uplift(score, conversion, group)

    assert result.data == 0.0, f"Expected uplift to be 0.0, but got {result.data}"
    assert (
        result.name == "uplift_at_40"
    ), f"Expected name to be uplift_at_40, but got {result.name}"
