"""Read the GTSDB release: the 900 road photos and the box + class of every labeled sign in them."""

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from PIL import Image

from findobject.data.gtsrb import NUM_CLASSES

N_PHOTOS = 900
DEV_PHOTOS = 600  # GTSDB's official split: photos 0-599 -> dev, 600-899 -> test
PHOTO_SIZE = (1360, 800)  # (width, height) of every GTSDB photo
GT_FILE = "gt.txt"
DOWNLOAD_HINT = "download FullIJCNN2013.zip into data/raw/gtsdb/ and unzip it there (see README)"
_FILE_NAME = re.compile(r"^(\d{5})\.ppm$")

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RealSign:
    box: Box  # x1, y1, x2, y2 in photo px; x2/y2 exclusive (gt.txt is inclusive)
    class_id: int


@dataclass(frozen=True)
class RealPhoto:
    path: Path
    index: int  # 0..N_PHOTOS-1, from the file name
    signs: tuple[RealSign, ...]  # in gt.txt row order; may be empty


class RealSplit(Enum):
    DEV = "dev"
    TEST = "test"


def read_gtsdb(root: Path) -> list[RealPhoto]:
    """All N_PHOTOS photos in index order, each with its labeled signs.

    root is the unzipped FullIJCNN2013 folder (holds 00000.ppm..00899.ppm and gt.txt). Only checks that
    the photo files exist; it does not open them (load_photo checks their size).
    Raises FileNotFoundError if gt.txt or a photo is missing; ValueError (with file and line) for a
    malformed row, a class id outside [0, NUM_CLASSES), or a box that is empty or outside the photo.
    """
    gt = root / GT_FILE
    if not gt.is_file():
        raise FileNotFoundError(f"{gt} not found: {DOWNLOAD_HINT}")
    signs: dict[int, list[RealSign]] = defaultdict(list)
    for lineno, line in enumerate(gt.read_text().splitlines(), start=1):
        if line.strip():
            index, sign = _parse_row(line, gt, lineno)
            signs[index].append(sign)
    photos = []
    for i in range(N_PHOTOS):
        path = root / f"{i:05d}.ppm"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: {DOWNLOAD_HINT}")
        photos.append(RealPhoto(path, i, tuple(signs.get(i, ()))))
    return photos


def photos_in(photos: Sequence[RealPhoto], split: RealSplit) -> list[RealPhoto]:
    """The photos of one official split: DEV = index < DEV_PHOTOS, TEST = the rest."""
    if split is RealSplit.DEV:
        return [p for p in photos if p.index < DEV_PHOTOS]
    return [p for p in photos if p.index >= DEV_PHOTOS]


def load_photo(photo: RealPhoto) -> Image.Image:
    """Open the photo as RGB. Raises OSError if unreadable, ValueError if it is not PHOTO_SIZE."""
    with Image.open(photo.path) as img:
        rgb = img.convert("RGB")
    if rgb.size != PHOTO_SIZE:
        raise ValueError(f"{photo.path}: size {rgb.size}, expected {PHOTO_SIZE}")
    return rgb


def load_sign_crop(photo: Image.Image, sign: RealSign) -> Image.Image:
    """The sign's box cut out of an opened photo (for hybrid scenes). Does not mutate photo."""
    return photo.crop(sign.box)


def _parse_row(line: str, gt: Path, lineno: int) -> tuple[int, RealSign]:
    def fail(msg: str) -> ValueError:
        return ValueError(f"{gt}:{lineno}: {msg}: {line!r}")

    parts = line.strip().split(";")
    if len(parts) != 6:
        raise fail("expected 6 ';'-separated fields")
    name = _FILE_NAME.match(parts[0])
    if not name or int(name.group(1)) >= N_PHOTOS:
        raise fail(f"file name must be 00000.ppm..{N_PHOTOS - 1:05d}.ppm")
    try:
        x1, y1, x2, y2, class_id = (int(p) for p in parts[1:])
    except ValueError:
        raise fail("box and class must be integers") from None
    if not 0 <= class_id < NUM_CLASSES:
        raise fail(f"class id must be in [0, {NUM_CLASSES})")
    w, h = PHOTO_SIZE
    if not (0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h):
        raise fail(f"box must satisfy 0 <= x1 <= x2 < {w} and 0 <= y1 <= y2 < {h} (inclusive corners)")
    return int(name.group(1)), RealSign((x1, y1, x2 + 1, y2 + 1), class_id)
