"""The demo's examples come only from dev photos and GTSRB training signs (docs/demo-spec.md A6)."""

from pathlib import Path

import pytest

from findobject.data.gtsdb import DEV_PHOTOS, PHOTO_SIZE, RealPhoto
from findobject.data.gtsrb import NUM_CLASSES, SignSample
from findobject.demo.examples import EXAMPLE_PHOTOS, example_photos, sign_per_class


def _photo(i):
    return RealPhoto(index=i, path=Path(f"{i:05d}.ppm"), signs=())


def test_a6_every_example_is_a_dev_photo():
    assert all(0 <= i < DEV_PHOTOS for i in EXAMPLE_PHOTOS)
    got = example_photos([_photo(i) for i in range(900)])
    assert list(got) == list(EXAMPLE_PHOTOS)


def test_a6_missing_or_test_photos_are_refused():
    with pytest.raises(ValueError):
        example_photos([_photo(i) for i in range(DEV_PHOTOS, 900)])  # only test photos given


def test_sign_per_class_takes_the_largest_roi():
    samples = [SignSample(Path(f"{c}-{k}.ppm"), (0, 0, k, k), c) for c in range(NUM_CLASSES) for k in (5, 30, 10)]
    best = sign_per_class(samples)
    assert list(best) == list(range(NUM_CLASSES))
    assert all(s.roi == (0, 0, 30, 30) for s in best.values())


def test_sign_per_class_needs_every_class():
    with pytest.raises(ValueError):
        sign_per_class([SignSample(Path("a.ppm"), (0, 0, 9, 9), 0)])
