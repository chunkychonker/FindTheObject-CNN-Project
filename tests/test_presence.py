"""A5-A7 (docs/v3a-spec.md §2.10): presence, τ, AUROC, the clustered bootstrap."""

import numpy as np
import pytest
import torch

from findobject.eval.presence import presence_scores, threshold_at_false_alarm
from findobject.eval.stats import auroc, cluster_bootstrap


# --- presence ---

def test_presence_is_the_largest_cell_probability():
    logits = torch.randn(5, 1, 32, 32) * 4
    expected = torch.sigmoid(logits).flatten(1).max(dim=1).values
    got = presence_scores(logits)
    assert got.dtype == torch.float32 and got.shape == (5,) and torch.equal(got, expected)
    assert torch.equal(presence_scores(logits.squeeze(1)), expected)


@pytest.mark.parametrize("shape", [(5, 2, 8, 8), (5, 64), (0, 1, 8, 8)])
def test_presence_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        presence_scores(torch.zeros(shape))


# --- A6: τ ---

def _admitted(neg, tau) -> int:
    return int((np.asarray(neg, np.float32) >= tau).sum())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rate", [0.0, 0.01, 0.05, 0.29])
def test_tau_admits_at_most_k_and_is_the_smallest_such_float32(seed, rate):
    rs = np.random.default_rng(seed)
    neg = rs.random(rs.integers(100, 6000)).astype(np.float32) ** 3  # skewed, like real presence scores
    neg[:20] = neg[20]  # some ties
    k = int(round(rate * 100)) * len(neg) // 100  # floor(rate * n) without float error
    tau = threshold_at_false_alarm(neg, rate)
    assert tau.dtype == np.float32
    assert _admitted(neg, tau) <= k
    assert _admitted(neg, np.nextafter(tau, np.float32(-np.inf))) > k  # one float32 lower admits too many


def test_tau_with_distinct_scores_hits_k_exactly():
    neg = np.arange(100, dtype=np.float32)
    assert _admitted(neg, threshold_at_false_alarm(neg, 0.29)) == 29  # 0.29 * 100 = 28.999... in floats
    assert _admitted(np.arange(5000, dtype=np.float32), threshold_at_false_alarm(np.arange(5000, dtype=np.float32), 0.01)) == 50


def test_ties_at_the_cut_fall_below_tau():
    neg = np.array([0.9, 0.5, 0.5, 0.5, 0.1], np.float32)
    assert _admitted(neg, threshold_at_false_alarm(neg, 0.4)) == 1  # k = 2, but the tie at s_3 admits only 1


def test_all_tied_scores_admit_none():
    neg = np.full(1000, 0.3, np.float32)
    assert _admitted(neg, threshold_at_false_alarm(neg, 0.05)) == 0


@pytest.mark.parametrize("neg,rate", [([], 0.01), ([0.1, np.nan], 0.01), ([0.1, 0.2], 1.0), ([0.1, 0.2], -0.1),
                                      ([[0.1, 0.2]], 0.01)])
def test_tau_rejects_bad_input(neg, rate):
    with pytest.raises(ValueError):
        threshold_at_false_alarm(np.asarray(neg, np.float32), rate)


# --- A5: AUROC ---

def test_auroc_extremes_and_ties():
    assert auroc([0.6, 0.9], [0.1, 0.5]) == 1.0
    assert auroc([0.1, 0.5], [0.6, 0.9]) == 0.0
    assert auroc([0.4] * 3, [0.4] * 7) == 0.5


def test_auroc_hand_computed_3_vs_3_with_one_tie():
    # 0.9 beats all 3 negatives (3); 0.5 beats 0.2 and 0.1 and ties 0.5 (2.5); 0.3 beats 0.2 and 0.1 (2).
    assert auroc([0.9, 0.5, 0.3], [0.5, 0.2, 0.1]) == pytest.approx(7.5 / 9)


def test_auroc_matches_brute_force_on_random_scores():
    rs = np.random.default_rng(0)
    pos, neg = rs.integers(0, 20, 300) / 20, rs.integers(0, 15, 200) / 20  # many ties
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (300 * 200)
    assert auroc(pos, neg) == pytest.approx(brute, abs=1e-12)


@pytest.mark.parametrize("pos,neg", [([], [0.1]), ([0.1], []), ([np.inf], [0.1]), ([0.1], [np.nan])])
def test_auroc_rejects_bad_input(pos, neg):
    with pytest.raises(ValueError):
        auroc(np.asarray(pos, float), np.asarray(neg, float))


# --- A7: clustered bootstrap ---

def test_bootstrap_same_seed_same_interval_and_constant_stat_gives_a_point():
    clusters = np.repeat(np.arange(30), 4)
    data = np.random.default_rng(0).random(len(clusters))
    stat = lambda idx: float(data[idx].mean())
    assert cluster_bootstrap(stat, clusters, 200, np.random.default_rng(7)) == \
        cluster_bootstrap(stat, clusters, 200, np.random.default_rng(7))
    assert cluster_bootstrap(lambda idx: 0.25, clusters, 50, np.random.default_rng(0)) == (0.25, 0.25)


def test_bootstrap_draws_whole_clusters_as_many_as_there_are():
    clusters = np.repeat(np.array([7, 3, 11, 5, 2]), 3)[np.random.default_rng(0).permutation(15)]  # shuffled ids
    seen = []

    def stat(idx):
        seen.append(idx)
        return 0.0

    cluster_bootstrap(stat, clusters, 100, np.random.default_rng(1))
    for idx in seen:
        assert len(idx) == 15  # 5 clusters drawn, 3 rows each
        rows_per_cluster = {c: np.bincount(idx, minlength=15)[clusters == c] for c in np.unique(clusters)}
        assert all(len(set(r)) == 1 for r in rows_per_cluster.values())  # a cluster comes whole or not at all


def test_bootstrap_with_singleton_clusters_is_the_ordinary_bootstrap():
    data = np.random.default_rng(0).normal(size=400)
    lo, hi = cluster_bootstrap(lambda idx: float(data[idx].mean()), np.arange(400), 2000, np.random.default_rng(0))
    assert lo < data.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 * data.std() / 20, rel=0.15)  # textbook width of a mean's interval


@pytest.mark.parametrize("clusters,n", [(np.array([]), 10), (np.arange(5), 0), (np.zeros((2, 2)), 10)])
def test_bootstrap_rejects_bad_input(clusters, n):
    with pytest.raises(ValueError):
        cluster_bootstrap(lambda idx: 0.0, clusters, n, np.random.default_rng(0))
