"""The signs a whole photo is graded against: labels with exact repeats removed, each flagged stacked or not (docs/v3b-spec.md §2.3). Pure."""

import math
from dataclasses import dataclass

from findobject.data.gtsdb import Box, RealPhoto

STACK_FACTOR = 1.5  # stacked: another sign's center is strictly closer than this many of the sign's own long sides


@dataclass(frozen=True)
class TruthSign:
    gt_index: int  # row index of this sign among the photo's gt.txt rows (real_dev's sign_index is the same)
    box: Box  # x1, y1, x2, y2; x2/y2 exclusive
    class_id: int
    long_side: int  # max(x2 - x1, y2 - y1), native px
    stacked: bool


def box_center(box: Box) -> tuple[float, float]:
    """Center of a half-open box, in the same continuous pixel coordinates as a detection's point."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def photo_truth(photo: RealPhoto) -> tuple[TruthSign, ...]:
    """The photo's signs with exact repeats (same box and class) removed, keeping the first, in gt.txt order.

    Each keeps its original gt.txt row index, so dropping a repeat never shifts another sign's index. A sign
    is stacked if another kept sign's box center is strictly closer than STACK_FACTOR x its long side
    (Euclidean). Never raises for a RealPhoto that read_gtsdb produced.
    """
    seen: set[tuple[Box, int]] = set()
    kept: list[tuple[int, Box, int]] = []
    for i, sign in enumerate(photo.signs):
        key = (sign.box, sign.class_id)
        if key not in seen:
            seen.add(key)
            kept.append((i, sign.box, sign.class_id))
    centers = [box_center(box) for _, box, _ in kept]
    out = []
    for j, (i, box, class_id) in enumerate(kept):
        long_side = max(box[2] - box[0], box[3] - box[1])
        stacked = any(math.dist(centers[j], centers[k]) < STACK_FACTOR * long_side
                      for k in range(len(kept)) if k != j)
        out.append(TruthSign(gt_index=i, box=box, class_id=class_id, long_side=long_side, stacked=stacked))
    return tuple(out)
