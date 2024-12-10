import numpy as np

from cate.dataset import synthetic_data


def test_synthetic_data_shapes() -> None:
    n, p = 1000, 5
    X, w, y = synthetic_data(n=n, p=p)
    assert X.shape == (n, p), f"Expected X shape {(n, p)}, but got {X.shape}"
    assert w.shape == (n,), f"Expected w shape {(n,)}, but got {w.shape}"
    assert y.shape == (n,), f"Expected y shape {(n,)}, but got {y.shape}"


def test_synthetic_data_random_state() -> None:
    n, p, random_state = 1000, 5, 42
    X1, w1, y1 = synthetic_data(n=n, p=p, random_state=random_state)
    X2, w2, y2 = synthetic_data(n=n, p=p, random_state=random_state)
    assert np.array_equal(X1, X2), "X arrays are not equal for the same random state"
    assert np.array_equal(w1, w2), "w arrays are not equal for the same random state"
    assert np.array_equal(y1, y2), "y arrays are not equal for the same random state"


# def test_synthetic_data_different_random_state() -> None:
#     n, p = 1000, 5
#     X1, w1, y1 = synthetic_data(n=n, p=p, random_state=42)
#     X2, w2, y2 = synthetic_data(n=n, p=p, random_state=43)
#     assert not np.array_equal(X1, X2), "X arrays are equal for different random states"
#     assert not np.array_equal(w1, w2), "w arrays are equal for different random states"
#     assert not np.array_equal(y1, y2), "y arrays are equal for different random states"
