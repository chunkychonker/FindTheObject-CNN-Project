from pathlib import Path

import pytest

from findobject.data.gtsrb import SignSample, split_by_track

TRACKS_PER_CLASS = {0: 7, 1: 20, 2: 45}
FRAMES = 30


def _samples() -> list[SignSample]:
    return [
        SignSample(Path(f"Images/{c:05d}/{t:05d}_{f:05d}.ppm"), (0, 0, 9, 9), c)
        for c, n_tracks in TRACKS_PER_CLASS.items()
        for t in range(n_tracks)
        for f in range(FRAMES)
    ]


def _tracks(samples):
    return {(s.class_id, s.path.name.split("_")[0]) for s in samples}


def test_whole_tracks_split_disjointly_and_nothing_is_lost():
    samples = _samples()
    kept, held = split_by_track(samples, 0.1, seed=0)
    assert _tracks(kept).isdisjoint(_tracks(held))
    assert sorted(map(str, (s.path for s in kept + held))) == sorted(map(str, (s.path for s in samples)))
    assert len(held) % FRAMES == 0  # every held-out track has all its frames


def test_per_class_holdout_count_with_minimum_one():
    _, held = split_by_track(_samples(), 0.1, seed=0)
    per_class = {c: len({t for cc, t in _tracks(held) if cc == c}) for c in TRACKS_PER_CLASS}
    assert per_class == {0: 1, 1: 2, 2: 4}  # round(0.7) -> 1, round(2.0), round(4.5) -> 4 (banker's)


def test_deterministic_for_a_seed():
    a = split_by_track(_samples(), 0.1, seed=3)
    b = split_by_track(_samples(), 0.1, seed=3)
    assert a == b


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_rejects_bad_fraction(fraction):
    with pytest.raises(ValueError):
        split_by_track(_samples(), fraction, seed=0)


def test_rejects_class_with_single_track():
    one = [SignSample(Path(f"00000_{f:05d}.ppm"), (0, 0, 9, 9), 5) for f in range(FRAMES)]
    with pytest.raises(ValueError):
        split_by_track(one, 0.1, seed=0)
