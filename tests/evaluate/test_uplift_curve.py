import numpy as np
import pandas as pd

from cate.metrics import UpliftCurve


def test_calculate_uplift_curve() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([1, 0, 1, 0, 1])
    uplift_curve = UpliftCurve(bin_num=5)

    result = uplift_curve._calculate(pred, y, w)

    assert isinstance(result, pd.DataFrame)
    assert "baseline_x" in result.columns
    assert "baseline_y" in result.columns
    assert "uplift_x" in result.columns
    assert "uplift_y" in result.columns
    assert len(result) == 5


def test_calculate_uplift_curve_tg_all_cv() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([1, 0, 1, 0, 1])
    uplift_curve = UpliftCurve(bin_num=5)

    actual = uplift_curve._calculate(pred, y, w)
    expect = pd.DataFrame(
        {
            "baseline_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "baseline_y": [0.0, 0.2, 0.4, 0.6, 0.8],
            "uplift_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "uplift_y": [0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    pd.testing.assert_frame_equal(actual, expect)


def test_calculate_uplift_curve_cg_all_cv() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([0, 1, 0, 1, 0])
    w = np.array([1, 0, 1, 0, 1])
    uplift_curve = UpliftCurve(bin_num=5)

    actual = uplift_curve._calculate(pred, y, w)
    expect = pd.DataFrame(
        {
            "baseline_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "baseline_y": [0.0, -0.2, -0.4, -0.6, -0.8],
            "uplift_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "uplift_y": [0.0, -1.0, -1.0, -1.0, -1.0],
        }
    )

    pd.testing.assert_frame_equal(actual, expect)


def test_calculate_uplift_curve_all_cv() -> None:
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 1, 1])
    w = np.array([1, 0, 1, 0, 1])
    uplift_curve = UpliftCurve(bin_num=5)

    actual = uplift_curve._calculate(pred, y, w)
    expect = pd.DataFrame(
        {
            "baseline_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "baseline_y": [0.0, 0.1, 0.2, 0.3, 0.4],
            "uplift_x": [0.0, 0.2, 0.4, 0.6, 0.8],
            "uplift_y": [0.0, 1.0, 1.0, 0.5, 0.5],
        }
    )

    pd.testing.assert_frame_equal(actual, expect)
