"""Builds each split's sign and background lists from the data config."""

from pathlib import Path
from typing import NamedTuple

from findobject.data.coco import list_backgrounds
from findobject.data.gtsrb import SignSample, read_test_index, read_train_index, split_by_track

SPLITS = ("train", "val", "test")


class SceneSource(NamedTuple):
    signs: list[SignSample]
    backgrounds: list[Path]


def split_sources(data_cfg: dict) -> dict[str, SceneSource]:
    """{"train", "val", "test"} -> the signs and backgrounds that split may draw from.

    Signs: train and val split the GTSRB training set by whole track (no near-duplicate frames across
    them); test uses only the GTSRB test set. Backgrounds: COCO images by index, index % every == 0 ->
    test, == 1 -> val, the rest -> train (every = data_cfg["test_background_every"]). The three
    splits share no sign track and no background.

    Raises FileNotFoundError if the raw GTSRB/COCO files are missing, KeyError for a missing config
    key, ValueError as gtsrb.split_by_track does.
    """
    raw = Path(data_cfg["raw_dir"])
    bgs = list_backgrounds(raw / "coco/val2017", raw / "coco/annotations/instances_val2017.json")
    every = data_cfg["test_background_every"]
    train_signs, val_signs = split_by_track(
        read_train_index(raw / "gtsrb"), data_cfg["holdout_fraction"], seed=data_cfg["seed"]
    )
    return {
        "train": SceneSource(train_signs, [b for i, b in enumerate(bgs) if i % every not in (0, 1)]),
        "val": SceneSource(val_signs, bgs[1::every]),
        "test": SceneSource(read_test_index(raw / "gtsrb"), bgs[::every]),
    }
