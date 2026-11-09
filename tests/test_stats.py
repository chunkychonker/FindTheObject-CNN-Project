import pytest

from findobject.eval.stats import mcnemar_exact, wilson


def test_wilson_known_values():  # A12
    r = wilson(0, 10)
    assert r.lo == 0.0 and r.hi == pytest.approx(0.2775, abs=5e-4)
    r = wilson(10, 10)
    assert r.hi == pytest.approx(1.0) and r.lo == pytest.approx(0.7225, abs=5e-4)
    r = wilson(85, 100)  # textbook check: 0.85, n=100 -> [0.767, 0.907]
    assert r.value == 0.85 and r.lo == pytest.approx(0.7672, abs=5e-4) and r.hi == pytest.approx(0.9069, abs=5e-4)


def test_wilson_interval_contains_rate_and_narrows_with_n():
    for k, n in [(1, 3), (7, 9), (500, 1000)]:
        r = wilson(k, n)
        assert r.lo <= r.value <= r.hi
    assert (wilson(800, 1000).hi - wilson(800, 1000).lo) < (wilson(80, 100).hi - wilson(80, 100).lo)


@pytest.mark.parametrize("k, n", [(0, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_bad_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


def test_mcnemar_known_values():  # A12
    assert mcnemar_exact(41, 34) == pytest.approx(0.488, abs=2e-3)
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)  # all ten one way: 2 * (1/2)^10
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0  # capped


def test_mcnemar_is_symmetric_and_rejects_negatives():
    assert mcnemar_exact(3, 17) == mcnemar_exact(17, 3)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)
