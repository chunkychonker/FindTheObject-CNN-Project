"""Match one photo's detections to its signs, and say what kind of mistake each unmatched detection is (docs/v3b-spec.md §2.3). Pure."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from findobject.eval.photo_truth import TruthSign, box_center
from findobject.inference.detection import Detection, detection_order


class FalsePositive(Enum):
    DUPLICATE = "duplicate"  # points at a sign of its own class that an earlier detection already matched
    WRONG_CLASS = "wrong_class"  # points at a sign, but at none of its own class
    BACKGROUND = "background"  # points at no labeled sign (includes unlabeled signs)


@dataclass(frozen=True)
class PhotoMatch:
    found: tuple[int | None, ...]  # per kept detection: the gt_index it found (class-aware), or None
    located: tuple[int | None, ...]  # the same, ignoring class (a separate pass)
    fp_kind: tuple[FalsePositive | None, ...]  # per kept detection; None exactly where found is not None


def points_at(d: Detection, sign: TruthSign) -> bool:
    """The detection's point lies in the sign's half-open box: x1 <= x < x2 and y1 <= y < y2."""
    x1, y1, x2, y2 = sign.box
    return x1 <= d.x < x2 and y1 <= d.y < y2


def _nearest(d: Detection, signs: list[TruthSign]) -> TruthSign:
    """The sign whose box center is nearest the point (Euclidean); ties go to the lower gt_index."""
    return min(signs, key=lambda s: (math.dist((d.x, d.y), box_center(s.box)), s.gt_index))


def _greedy(kept: Sequence[Detection], truth: Sequence[TruthSign], class_aware: bool) -> list[int | None]:
    matched: set[int] = set()
    out: list[int | None] = []
    for d in kept:
        options = [s for s in truth if s.gt_index not in matched and points_at(d, s)
                   and (not class_aware or s.class_id == d.class_id)]
        if options:
            sign = _nearest(d, options)
            matched.add(sign.gt_index)
            out.append(sign.gt_index)
        else:
            out.append(None)
    return out


def match_photo(kept: Sequence[Detection], truth: Sequence[TruthSign]) -> PhotoMatch:
    """Greedy matching in detection order, each sign matched at most once, once class-aware (found) and once
    ignoring class (located). A detection that finds nothing is a false positive. Its kind is decided from
    the detections before it only, in this order: background if it points at no sign; else duplicate if it
    points at a sign of its own class (all of them already matched); else wrong_class.

    Every outcome depends only on earlier detections, so matching the detections with score >= t alone gives
    the same outcomes for them as this full pass (the prefix property).
    Raises ValueError if kept is not in detection_order.
    """
    keys = [detection_order(d) for d in kept]
    if any(a > b for a, b in zip(keys, keys[1:])):
        raise ValueError("kept detections must be in detection_order")
    found = _greedy(kept, truth, class_aware=True)
    located = _greedy(kept, truth, class_aware=False)
    kinds: list[FalsePositive | None] = []
    for d, f in zip(kept, found):
        if f is not None:
            kinds.append(None)
            continue
        pointed = [s for s in truth if points_at(d, s)]
        if not pointed:
            kinds.append(FalsePositive.BACKGROUND)
        elif any(s.class_id == d.class_id for s in pointed):
            kinds.append(FalsePositive.DUPLICATE)
        else:
            kinds.append(FalsePositive.WRONG_CLASS)
    return PhotoMatch(found=tuple(found), located=tuple(located), fp_kind=tuple(kinds))


def best_raw_score(raw: Sequence[Detection], truth: Sequence[TruthSign]) -> dict[int, float]:
    """gt_index -> the highest score among raw (unsuppressed) detections pointing at that sign; a sign no raw
    detection points at is absent. Gives "located before suppression" at any threshold. Never raises."""
    best: dict[int, float] = {}
    for d in raw:
        for s in truth:
            if points_at(d, s) and d.score > best.get(s.gt_index, -1.0):
                best[s.gt_index] = d.score
    return best
