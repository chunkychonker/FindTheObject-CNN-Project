"""Summarizes how much better a model does on signs it trained on than on signs it never saw (pure, no I/O)."""

from collections import defaultdict

import torch

from findobject.data.gtsrb import SignSample, track_id

WEAK_MAX_TRACKS = 12  # classes with at most this many training tracks count as "weak"


def tracks_per_class(signs: list[SignSample]) -> dict[int, int]:
    """class_id -> number of distinct physical-sign tracks among `signs`. Classes with no signs are absent."""
    tracks: dict[int, set[str]] = defaultdict(set)
    for s in signs:
        tracks[s.class_id].add(track_id(s))
    return {k: len(v) for k, v in sorted(tracks.items())}


def weak_classes(signs: list[SignSample], max_tracks: int = WEAK_MAX_TRACKS) -> list[int]:
    """Sorted class ids with 1..max_tracks training tracks: the classes most at risk of sign memorization."""
    return [k for k, n in tracks_per_class(signs).items() if n <= max_tracks]


def subset_class_acc(y_true: torch.Tensor, y_pred: torch.Tensor, classes: list[int]) -> float:
    """Top-1 class accuracy over the scenes whose true class is in `classes`.

    Raises ValueError if no scene has a class in `classes` (an empty subset has no accuracy).
    """
    keep = torch.isin(y_true, torch.tensor(classes, dtype=y_true.dtype))
    if not keep.any():
        raise ValueError(f"no scenes with a true class in {classes}")
    return (y_true[keep] == y_pred[keep]).float().mean().item()
