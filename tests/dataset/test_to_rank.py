import polars as pl
from polars.testing import assert_series_equal

from cate.dataset import to_rank


def test_to_rank_ascending() -> None:
    primary_key = pl.Series("id", [1, 2, 3, 4, 5])
    score = pl.Series("score", [10, 20, 30, 40, 50])
    expected_ranks = pl.Series("rank", [1, 2, 3, 4, 5])

    ranks = to_rank(primary_key, score, descending=False, k=5)

    assert_series_equal(ranks, expected_ranks)


def test_to_rank_descending() -> None:
    primary_key = pl.Series("id", [1, 2, 3, 4, 5])
    score = pl.Series("score", [10, 20, 30, 40, 50])
    # TODO: Fix the expected ranks
    expected_ranks = pl.Series("rank", [5, 4, 3, 2, 1])

    ranks = to_rank(primary_key, score, descending=True, k=5)

    assert_series_equal(ranks, expected_ranks)


def test_to_rank_with_k() -> None:
    primary_key = pl.Series("id", [1, 2, 3, 4, 5])
    score = pl.Series("score", [10, 20, 30, 40, 50])
    expected_ranks = pl.Series("rank", [20, 40, 60, 80, 100])

    ranks = to_rank(primary_key, score, descending=False, k=100)

    assert_series_equal(ranks, expected_ranks)


def test_to_rank_with_ties() -> None:
    primary_key = pl.Series("id", [1, 2, 3, 4, 5])
    score = pl.Series("score", [10, 20, 20, 40, 50])
    expected_ranks = pl.Series("rank", [1, 2, 3, 4, 5])

    ranks = to_rank(primary_key, score, descending=False, k=5)

    assert_series_equal(ranks, expected_ranks)
