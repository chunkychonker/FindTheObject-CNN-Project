import numpy as np
import pytest

from findobject.eval.real_report import SizeBucket, discordant, rates_by, size_bucket


@pytest.mark.parametrize("L, bucket", [(1, SizeBucket.SMALL), (23, SizeBucket.SMALL), (24, SizeBucket.MEDIUM),
                                       (47, SizeBucket.MEDIUM), (48, SizeBucket.LARGE), (129, SizeBucket.LARGE)])
def test_size_bucket_edges(L, bucket):
    assert size_bucket(L) is bucket


def test_size_bucket_rejects_nonpositive():
    with pytest.raises(ValueError):
        size_bucket(0)


def test_rates_by_counts_each_group():
    ok = np.array([True, False, True, True, False])
    r = rates_by(ok, ["a", "b", "a", "c", "b"])
    assert list(r) == ["a", "b", "c"]
    assert (r["a"].k, r["a"].n) == (2, 2) and (r["b"].k, r["b"].n) == (0, 2) and (r["c"].k, r["c"].n) == (1, 1)
    assert list(rates_by(ok, ["a", "b", "a", "c", "b"], min_n=2)) == ["a", "b"]


def test_rates_by_rejects_mismatch_and_non_bool():
    with pytest.raises(ValueError):
        rates_by(np.array([True, False]), ["a"])
    with pytest.raises(ValueError):
        rates_by(np.array([1, 0]), ["a", "b"])


def test_discordant_counts_only_disagreements():
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, False])
    assert discordant(a, b) == (2, 1)
    with pytest.raises(ValueError):
        discordant(a, b[:3])
