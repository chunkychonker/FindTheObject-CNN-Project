from pathlib import Path

import pytest
from PIL import Image

from findobject.data.gtsdb import (DEV_PHOTOS, N_PHOTOS, PHOTO_SIZE, RealSign, RealSplit, load_photo, photos_in,
                                   read_gtsdb)

REAL_ROOT = Path("data/raw/gtsdb/FullIJCNN2013")
needs_gtsdb = pytest.mark.skipif(not (REAL_ROOT / "gt.txt").exists(), reason="GTSDB not downloaded")


def _fake_release(root: Path, rows: list[str]) -> Path:
    """N_PHOTOS empty .ppm files (read_gtsdb only checks they exist) and a gt.txt with `rows`."""
    root.mkdir(exist_ok=True)
    for i in range(N_PHOTOS):
        (root / f"{i:05d}.ppm").touch()
    (root / "gt.txt").write_text("\n".join(rows) + "\n")
    return root


def test_rows_attach_to_their_photo_with_exclusive_corners(tmp_path):
    photos = read_gtsdb(_fake_release(tmp_path, ["00000.ppm;774;411;815;446;11", "00000.ppm;10;20;30;40;0",
                                                 "00599.ppm;0;0;1359;799;42"]))
    assert len(photos) == N_PHOTOS and [p.index for p in photos] == list(range(N_PHOTOS))
    assert photos[0].signs == (RealSign((774, 411, 816, 447), 11), RealSign((10, 20, 31, 41), 0))
    assert photos[599].signs == (RealSign((0, 0, 1360, 800), 42),)
    assert photos[1].signs == ()  # photos without a labeled sign are kept, with no signs


@pytest.mark.parametrize("row", [
    "00000.ppm;1;2;3;4",              # too few fields
    "0000.ppm;1;2;3;4;5",             # bad file name
    "00900.ppm;1;2;3;4;5",            # photo index out of range
    "00000.ppm;1;2;x;4;5",            # not an integer
    "00000.ppm;1;2;3;4;43",           # class out of range
    "00000.ppm;5;2;3;4;1",            # x2 < x1
    "00000.ppm;1;2;1360;4;1",         # x2 outside the photo (inclusive, so max is 1359)
    "00000.ppm;1;-1;3;4;1",           # negative
])
def test_bad_row_raises_with_line_number(tmp_path, row):
    with pytest.raises(ValueError, match=r"gt\.txt:2:"):
        read_gtsdb(_fake_release(tmp_path, ["00000.ppm;1;2;3;4;5", row]))


def test_missing_files_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        read_gtsdb(tmp_path)
    root = _fake_release(tmp_path, [])
    (root / "00123.ppm").unlink()
    with pytest.raises(FileNotFoundError, match="00123"):
        read_gtsdb(root)


def test_splits_are_disjoint_and_cover_everything(tmp_path):
    photos = read_gtsdb(_fake_release(tmp_path, []))
    dev, test = photos_in(photos, RealSplit.DEV), photos_in(photos, RealSplit.TEST)
    assert len(dev) == DEV_PHOTOS and len(test) == N_PHOTOS - DEV_PHOTOS
    assert {p.index for p in dev}.isdisjoint({p.index for p in test})
    assert {p.index for p in dev} | {p.index for p in test} == set(range(N_PHOTOS))


def test_load_photo_rejects_wrong_size(tmp_path):
    root = _fake_release(tmp_path, [])
    Image.new("RGB", (100, 80)).save(root / "00000.ppm")
    with pytest.raises(ValueError, match="size"):
        load_photo(read_gtsdb(root)[0])


@needs_gtsdb
def test_real_release_parses_completely():
    photos = read_gtsdb(REAL_ROOT)
    rows = [r for r in (REAL_ROOT / "gt.txt").read_text().splitlines() if r.strip()]
    assert sum(len(p.signs) for p in photos) == len(rows)
    assert load_photo(photos[0]).size == PHOTO_SIZE
