import pandas as pd
import pytest

from cate.metrics import UpliftByPercentile


def test_uplift_by_percentile() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([1, 0, 1, 0, 0])
    uplift = UpliftByPercentile(k=40)

    result = uplift(score, group, conversion)

    assert result == 1.0, f"Expected uplift to be 1.0, but got {result}"


def test_uplift_by_percentile_no_conversion() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([0, 0, 0, 0, 0])
    uplift = UpliftByPercentile(k=40)

    result = uplift(score, group, conversion)

    assert result == 0.0, f"Expected uplift to be 0.0, but got {result}"


def test_uplift_by_percentile_all_conversion() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([1, 1, 1, 1, 1])
    uplift = UpliftByPercentile(k=40)

    result = uplift(score, group, conversion)

    assert result == 0.0, f"Expected uplift to be 0.0, but got {result}"


def test_uplift_by_percentile_different_k() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    group = pd.Series([1, 0, 1, 0, 1])
    conversion = pd.Series([1, 0, 0, 0, 0])
    uplift = UpliftByPercentile(k=60)

    result = uplift(score, group, conversion)

    assert result == 0.5, f"Expected uplift to be 0.5, but got {result}"
