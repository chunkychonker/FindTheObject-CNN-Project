"""One detection in a whole photo, and the single order every v3b step sorts detections by (docs/v3b-spec.md §2.2). Pure."""

import math
from dataclasses import dataclass

from findobject.data.gtsrb import NUM_CLASSES

PYRAMID_SCALES = (1.0, 0.5, 0.25)  # each divides 1360x800 into a size that is a multiple of STRIDE


@dataclass(frozen=True)
class Detection:
    """A heatmap peak, mapped back to the native photo, with the class read there."""

    x: float  # native photo px
    y: float
    score: float  # sigmoid of the peak cell's logit
    class_id: int
    class_prob: float  # softmax probability of class_id
    scale: float  # the pyramid scale the peak came from

    def __post_init__(self) -> None:
        """Raises ValueError for a non-finite point, a score or class_prob outside [0, 1], a class outside
        [0, NUM_CLASSES), or a scale outside (0, 1]."""
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError(f"point must be finite, got ({self.x}, {self.y})")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if not 0.0 <= self.class_prob <= 1.0:
            raise ValueError(f"class_prob must be in [0, 1], got {self.class_prob}")
        if not 0 <= self.class_id < NUM_CLASSES:
            raise ValueError(f"class_id must be in [0, {NUM_CLASSES}), got {self.class_id}")
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")


def detection_order(d: Detection) -> tuple[float, float, float, float]:
    """The sort key used everywhere: score descending, then the finer scale (larger s), then smaller y,
    then smaller x. Two detections from different cells never share all four (cell centers at scales
    1, 1/2 and 1/4 never coincide), so sorting by it doesn't depend on input order."""
    return (-d.score, -d.scale, d.y, d.x)
