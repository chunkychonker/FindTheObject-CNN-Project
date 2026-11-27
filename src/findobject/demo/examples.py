"""Choose the demo's example inputs from dev photos and GTSRB training signs only, never test (docs/demo-spec.md A6). Pure."""

from collections.abc import Sequence

from findobject.data.gtsdb import DEV_PHOTOS, RealPhoto, RealSplit, photos_in
from findobject.data.gtsrb import NUM_CLASSES, SignSample

# Dev photo index -> what it shows. Picked from seed 0's stored dev report at τ_W (outputs/eval/<run>/
# detect_dev.json); descriptions name what's in the photo, so they stay true at any threshold.
EXAMPLE_PHOTOS: dict[int, str] = {
    206: "five signs at three sizes",
    149: "two big signs (91 and 122 px) and a small one",
    5: "one 115 px sign: too big for scale 1",
    162: "three small signs (19-20 px)",
    552: "four tiny signs (17-19 px): the weak spot",
    54: "four signs stacked close together",
    146: "a big sign and two small ones",
    108: "no signs",
    235: "no signs, but sign-like clutter",
}


def example_photos(photos: Sequence[RealPhoto]) -> dict[int, RealPhoto]:
    """EXAMPLE_PHOTOS looked up among the dev photos, in EXAMPLE_PHOTOS order.
    Raises ValueError if an example isn't a dev photo (index >= DEV_PHOTOS) or isn't in `photos`."""
    dev = {p.index: p for p in photos_in(photos, RealSplit.DEV)}
    bad = [i for i in EXAMPLE_PHOTOS if i not in dev]
    if bad:
        raise ValueError(f"example photos {bad} are not dev photos (dev is index < {DEV_PHOTOS})")
    return {i: dev[i] for i in EXAMPLE_PHOTOS}


def sign_per_class(train_samples: Sequence[SignSample]) -> dict[int, SignSample]:
    """For each class, the GTSRB training sign with the largest region of interest (the sharpest crop); ties go
    to the first in input order. Pass read_train_index's output: the test signs come from a different file.
    Raises ValueError if a class has no sample."""
    best: dict[int, SignSample] = {}
    area = lambda s: (s.roi[2] - s.roi[0] + 1) * (s.roi[3] - s.roi[1] + 1)  # noqa: E731  (roi is inclusive)
    for s in train_samples:
        if s.class_id not in best or area(s) > area(best[s.class_id]):
            best[s.class_id] = s
    missing = [c for c in range(NUM_CLASSES) if c not in best]
    if missing:
        raise ValueError(f"no training sign for classes {missing}")
    return dict(sorted(best.items()))
