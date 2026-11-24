"""Greedy suppression across pyramid scales: one detection per sign (docs/v3b-spec.md §2.2 step 3). Pure."""

from collections.abc import Sequence

import numpy as np

from findobject.inference.detection import Detection, detection_order
from findobject.models.baseline import STRIDE

SUPPRESS_CELLS = 3  # radius in cells of the kept detection's scale; from sign geometry, not tuned


def suppress(dets: Sequence[Detection]) -> list[Detection]:
    """Keep the best detection, drop every later one strictly closer than SUPPRESS_CELLS * STRIDE / s_kept
    photo px to it in both x and y (Chebyshev distance), and repeat on what is left. Class is ignored.

    Returns the kept detections in detection_order; [] for []. The result doesn't depend on input order.
    Never raises.
    """
    ordered = sorted(dets, key=detection_order)
    if not ordered:
        return []
    xy = np.array([(d.x, d.y) for d in ordered])
    alive = np.ones(len(ordered), dtype=bool)
    kept = []
    for i, d in enumerate(ordered):
        if not alive[i]:
            continue
        kept.append(d)
        radius = SUPPRESS_CELLS * STRIDE / d.scale
        later = slice(i + 1, None)
        close = np.abs(xy[later] - xy[i]).max(axis=1) < radius
        alive[later] &= ~close
    return kept
