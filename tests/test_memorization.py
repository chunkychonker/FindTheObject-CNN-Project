from pathlib import Path

import pytest
import torch

from findobject.data.gtsrb import SignSample
from findobject.eval.memorization import subset_class_acc, tracks_per_class, weak_classes


def _signs(spec: dict[int, int], frames: int = 3) -> list[SignSample]:
    """class_id -> number of tracks, each with `frames` frames named <track>_<frame>.ppm as in GTSRB."""
    return [SignSample(Path(f"{k:02d}{t:03d}_{f:05d}.ppm"), (0, 0, 9, 9), k)
            for k, n in spec.items() for t in range(n) for f in range(frames)]


def test_tracks_counts_physical_signs_not_frames():
    assert tracks_per_class(_signs({0: 2, 5: 7})) == {0: 2, 5: 7}


def test_weak_classes_use_inclusive_limit():
    assert weak_classes(_signs({0: 6, 1: 12, 2: 13, 3: 40}), max_tracks=12) == [0, 1]


def test_subset_class_acc_ignores_other_classes():
    y_true = torch.tensor([0, 0, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 0, 0])
    assert subset_class_acc(y_true, y_pred, [0]) == pytest.approx(0.5)
    assert subset_class_acc(y_true, y_pred, [0, 1]) == pytest.approx(2 / 3)


def test_subset_class_acc_empty_subset_raises():
    with pytest.raises(ValueError):
        subset_class_acc(torch.tensor([0, 1]), torch.tensor([0, 1]), [7])
