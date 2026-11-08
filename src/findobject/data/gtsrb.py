"""Read the official GTSRB release: per-image sign ROI + class id, and load ROI crops."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 43


@dataclass(frozen=True)
class SignSample:
    path: Path
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2 as in the CSV (x2/y2 inclusive)
    class_id: int


def read_train_index(gtsrb_root: Path) -> list[SignSample]:
    """All 39,209 training signs, from the per-class GT-*.csv files.

    Raises FileNotFoundError if no CSVs are found, ValueError on a malformed row.
    """
    csvs = sorted((gtsrb_root / "GTSRB/Final_Training/Images").glob("*/GT-*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no GTSRB training CSVs under {gtsrb_root}")
    return [s for f in csvs for s in _read_csv(f, f.parent)]


def read_test_index(gtsrb_root: Path) -> list[SignSample]:
    """All 12,630 test signs. Labels come from GT-final_test.csv (the CSV inside
    Final_Test/Images has no ClassId). Raises FileNotFoundError / ValueError as above."""
    return _read_csv(gtsrb_root / "GT-final_test.csv", gtsrb_root / "GTSRB/Final_Test/Images")


def split_by_track(
    samples: list[SignSample], holdout_fraction: float, seed: int
) -> tuple[list[SignSample], list[SignSample]]:
    """Split training signs into (kept, held_out) by whole physical-sign track.

    GTSRB training files are named <track>_<frame>.ppm: ~30 near-identical video frames of
    one physical sign. Splitting by image would put sibling frames on both sides, so a track
    always lands entirely on one side. Per class, round(holdout_fraction * n_tracks) tracks
    (at least 1) are held out, chosen by a RNG seeded with `seed`; order within each side
    follows the input order. Raises ValueError unless 0 < holdout_fraction < 1, or if a class
    has a single track (it cannot appear on both sides).
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    tracks_by_class: dict[int, set[str]] = defaultdict(set)
    for s in samples:
        tracks_by_class[s.class_id].add(track_id(s))
    rng = np.random.default_rng(seed)
    held: set[tuple[int, str]] = set()
    for class_id in sorted(tracks_by_class):
        tracks = sorted(tracks_by_class[class_id])
        if len(tracks) < 2:
            raise ValueError(f"class {class_id} has {len(tracks)} track(s); need >= 2 to split")
        k = max(1, round(holdout_fraction * len(tracks)))
        held.update((class_id, tracks[j]) for j in rng.choice(len(tracks), size=k, replace=False))
    kept = [s for s in samples if (s.class_id, track_id(s)) not in held]
    held_out = [s for s in samples if (s.class_id, track_id(s)) in held]
    return kept, held_out


def track_id(sample: SignSample) -> str:
    return sample.path.name.split("_")[0]


def load_sign(sample: SignSample) -> Image.Image:
    """Open the image and crop to its ROI. Raises OSError if the file is missing/corrupt."""
    x1, y1, x2, y2 = sample.roi
    with Image.open(sample.path) as img:
        return img.convert("RGB").crop((x1, y1, x2 + 1, y2 + 1))


def _read_csv(csv_path: Path, image_dir: Path) -> list[SignSample]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    samples = []
    for row in rows:
        class_id = int(row["ClassId"])
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"{csv_path}: class id {class_id} out of range")
        roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
        samples.append(SignSample(image_dir / row["Filename"], roi, class_id))
    return samples
