"""The v3b hand-audit sheet (eval/fp_sheet.py, scripts/fp_sheet.py): which false positives, and what is drawn."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from findobject.eval.fp_sheet import (COLUMNS, LABEL_PX, N_AUDIT, PAD_GRAY, TILE_PX, FpTile, choose_fps,
                                      crop_tile, render_fp_sheet)

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _photos(n_background=50):
    """Photo i holds one background FP scoring 0.5 + i/100, plus a found sign and a duplicate scoring higher."""
    return [{"index": i, "kept": [
        {"x": 10.0, "y": 10.0, "score": 0.99, "scale": 1.0, "fp_kind": None},
        {"x": 20.0, "y": 20.0, "score": 0.98, "scale": 1.0, "fp_kind": "duplicate"},
        {"x": 100.0 + i, "y": 50.0, "score": 0.5 + i / 100, "scale": 0.5, "fp_kind": "background"}]}
        for i in range(n_background)]


def test_choose_fps_takes_the_top_background_fps_above_the_threshold():
    tiles = choose_fps(_photos(), 0.55, np.random.default_rng(0))
    assert sorted(t.number for t in tiles) == list(range(1, N_AUDIT + 1))
    assert sorted(t.photo_index for t in tiles) == list(range(10, 50))  # the 40 highest of 45 above 0.55
    assert all(t.scale == 0.5 and t.x == 100.0 + t.photo_index for t in tiles)


def test_choose_fps_order_is_shuffled_and_seeded():
    a = choose_fps(_photos(), 0.0, np.random.default_rng(0))
    assert a == choose_fps(_photos(), 0.0, np.random.default_rng(0))
    assert [t.photo_index for t in sorted(a)] != sorted(t.photo_index for t in a)  # number order isn't rank order
    assert a != choose_fps(_photos(), 0.0, np.random.default_rng(1))


def test_choose_fps_ties_go_to_the_lower_photo_and_it_refuses_too_few():
    photos = [{"index": i, "kept": [{"x": 1.0, "y": 1.0, "score": 0.7, "scale": 1.0, "fp_kind": "background"}]}
              for i in range(45)]
    assert sorted(t.photo_index for t in choose_fps(photos, 0.0, np.random.default_rng(0))) == list(range(40))
    with pytest.raises(ValueError, match="need 40"):
        choose_fps(_photos(), 0.605, np.random.default_rng(0))  # only photos 11-49 score >= 0.605: 39


def test_crop_is_centered_scaled_and_padded_outside_the_photo():
    photo = Image.new("RGB", (1360, 800), (0, 0, 0))
    photo.paste((255, 255, 255), (200, 200, 210, 210))  # a white 10 px square
    tile = crop_tile(photo, FpTile(1, 0, 205.0, 205.0, 1.0, 0.9))
    a = np.asarray(tile)
    assert tile.size == (TILE_PX, TILE_PX)
    assert (a[TILE_PX // 2, TILE_PX // 2] == 255).all()  # the point itself is not covered by a mark
    corner = np.asarray(crop_tile(photo, FpTile(1, 0, 0.0, 0.0, 0.25, 0.9)))
    assert (corner[5, 5] == PAD_GRAY).all() and (corner[-5, -5] == 0).all()


def test_sheet_is_a_grid_in_number_order():
    photos = {0: Image.new("RGB", (1360, 800), (0, 0, 0)), 1: Image.new("RGB", (1360, 800), (255, 255, 255))}
    tiles = [FpTile(2, 1, 500.0, 400.0, 1.0, 0.9), FpTile(1, 0, 500.0, 400.0, 1.0, 0.8)]
    a = np.asarray(render_fp_sheet(photos, tiles))
    assert a.shape == (TILE_PX + LABEL_PX, COLUMNS * TILE_PX, 3)
    assert (a[LABEL_PX + 5, 5] == 0).all() and (a[LABEL_PX + 5, TILE_PX + 5] == 255).all()


def test_script_refuses_to_overwrite_labels(tmp_path):
    spec = importlib.util.spec_from_file_location("fp_sheet_cli", SCRIPTS / "fp_sheet.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path / "fp_audit.csv"
    cli.refuse_labeled(path)  # missing: fine
    path.write_text("number,label\n1,\n2,\n")
    cli.refuse_labeled(path)  # blank template: fine
    path.write_text("number,label\n1,other\n2,\n")
    with pytest.raises(ValueError, match="already holds labels"):
        cli.refuse_labeled(path)
