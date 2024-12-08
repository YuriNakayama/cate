import pandas as pd

from cate.dataset import to_rank


def test_to_rank_ascending() -> None:
    primary_key = pd.Series([1, 2, 3, 4, 5], name="id")
    score = pd.Series([10, 20, 30, 40, 50], name="score")
    expected_ranks = pd.Series([1, 2, 3, 4, 5], name="rank", index=primary_key)

    ranks = to_rank(primary_key, score, ascending=True, k=5)

    pd.testing.assert_series_equal(ranks, expected_ranks)


def test_to_rank_descending() -> None:
    primary_key = pd.Series([1, 2, 3, 4, 5], name="id")
    score = pd.Series([10, 20, 30, 40, 50], name="score")
    # TODO: Fix the expected ranks
    expected_ranks = pd.Series(
        [5, 4, 3, 2, 1], name="rank", index=primary_key
    ).sort_index(ascending=False)

    ranks = to_rank(primary_key, score, ascending=False, k=5)

    pd.testing.assert_series_equal(ranks, expected_ranks)


def test_to_rank_with_k() -> None:
    primary_key = pd.Series([1, 2, 3, 4, 5], name="id")
    score = pd.Series([10, 20, 30, 40, 50], name="score")
    expected_ranks = pd.Series([20, 40, 60, 80, 100], name="rank", index=primary_key)

    ranks = to_rank(primary_key, score, ascending=True, k=100)

    pd.testing.assert_series_equal(ranks, expected_ranks)


def test_to_rank_with_ties() -> None:
    primary_key = pd.Series([1, 2, 3, 4, 5], name="id")
    score = pd.Series([10, 20, 20, 40, 50], name="score")
    expected_ranks = pd.Series([1, 2, 3, 4, 5], name="rank", index=primary_key)

    ranks = to_rank(primary_key, score, ascending=True, k=5)

    pd.testing.assert_series_equal(ranks, expected_ranks)
