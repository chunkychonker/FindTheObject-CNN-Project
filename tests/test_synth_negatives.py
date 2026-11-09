"""A9 (docs/v3a-spec.md §2.10) and the EmptyScenes reader: synthetic empties and decoys."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from findobject.data.dataset import IGNORE_CLASS, EmptyScenes, has_sign
from findobject.data.gtsrb import SignSample
from findobject.data.sign_augment import SignAugment
from findobject.data.synth import SynthConfig, make_empty_scene
from findobject.data.synth_negatives import EMPTY_SEED, SeedOffset, build_decoys, build_empties, roi_size

CFG = SynthConfig()
AUG = dataclasses.replace(CFG, sign_augment=SignAugment(max_rotation_deg=10, max_perspective=0.1, occlusion_prob=0.5))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    rs = np.random.default_rng(0)
    bgs, signs = [], []
    for k in range(4):
        path = root / f"bg{k}.png"
        Image.fromarray(rs.integers(0, 256, (220 + 10 * k, 260, 3), dtype=np.uint8)).save(path)
        bgs.append(path)
    for k, roi in enumerate([(5, 5, 34, 44), (0, 0, 59, 59), (2, 3, 22, 33)]):  # inclusive corners
        signs.append(SignSample(path=root / "never_opened.png", roi=roi, class_id=k))
    return bgs, signs


def test_empties_are_deterministic_per_scene(fake):
    bgs, _ = fake
    a, b = build_empties(bgs, 6, SeedOffset.VAL_EMPTY, CFG), build_empties(bgs, 3, SeedOffset.VAL_EMPTY, CFG)
    assert np.array_equal(a["images"][:3], b["images"]) and np.array_equal(a["background_index"][:3], b["background_index"])
    c = build_empties(bgs, 6, SeedOffset.TEST_EMPTY, CFG)
    assert not np.array_equal(a["images"], c["images"])
    assert a["images"].shape == (6, 128, 128, 3) and a["images"].dtype == np.uint8
    assert [bgs[i].name for i in a["background_index"]] == a["background_name"].tolist()


def test_decoy_patch_is_a_sign_roi_sized_piece_of_another_background(fake):
    bgs, signs = fake
    d = build_decoys(signs, bgs, 40, SeedOffset.VAL_DECOY, CFG)
    assert (d["background_index"] != d["patch_background_index"]).all()
    for i in range(40):
        assert tuple(d["patch_size"][i]) == roi_size(signs[d["sign_index"][i]])
    assert {tuple(p) for p in d["patch_size"]} <= {(30, 40), (60, 60), (21, 31)}
    assert d["patch_background_name"].tolist() == [bgs[i].name for i in d["patch_background_index"]]


def test_plain_and_aug_decoys_share_their_draws_up_to_the_paste(fake):
    bgs, signs = fake
    plain, aug = (build_decoys(signs, bgs, 10, SeedOffset.VAL_DECOY, c) for c in (CFG, AUG))
    for k in ("background_index", "patch_background_index", "sign_index", "patch_size"):
        assert np.array_equal(plain[k], aug[k]), k
    assert not np.array_equal(plain["images"], aug["images"])  # the paste path differs


def test_decoy_changes_the_scene_only_where_the_patch_lands(fake):
    # No blur, noise or brightness change: a plain decoy equals its bare background crop except inside a
    # sign-sized rectangle (the pasted patch).
    bgs, signs = fake
    cfg = dataclasses.replace(CFG, blur_prob=0.0, noise_max_std=0.0, brightness=(1.0, 1.0))
    d = build_decoys(signs, bgs, 10, SeedOffset.VAL_DECOY, cfg)
    for i in range(10):
        rng = np.random.default_rng([EMPTY_SEED, SeedOffset.VAL_DECOY.value, i])
        rng.integers(len(bgs)), rng.integers(len(signs)), rng.integers(len(bgs) - 1)  # replay the pre-crop draws
        w, h = d["patch_size"][i]
        with Image.open(bgs[d["patch_background_index"][i]]) as src:
            rng.integers(0, src.width - w + 1), rng.integers(0, src.height - h + 1)
        with Image.open(bgs[d["background_index"][i]]) as bg:
            bare = make_empty_scene(bg, rng, cfg).image
        ys, xs = np.nonzero((d["images"][i] != bare).any(axis=2))
        assert len(ys) and xs.max() - xs.min() < cfg.sign_max and ys.max() - ys.min() < cfg.sign_max


def test_rejects_bad_inputs(fake):
    bgs, signs = fake
    with pytest.raises(ValueError):
        build_empties(bgs, 0, SeedOffset.VAL_EMPTY, CFG)
    with pytest.raises(ValueError):
        build_empties([], 5, SeedOffset.VAL_EMPTY, CFG)
    with pytest.raises(ValueError):
        build_decoys(signs, bgs[:1], 5, SeedOffset.VAL_DECOY, CFG)
    with pytest.raises(ValueError):
        build_decoys([SignSample(Path("x"), (0, 0, 999, 999), 0)], bgs, 5, SeedOffset.VAL_DECOY, CFG)  # patch too big


# --- EmptyScenes ---

def test_empty_scenes_serve_no_sign_samples(fake, tmp_path):
    bgs, _ = fake
    path = tmp_path / "e.npz"
    np.savez(path, **build_empties(bgs, 3, SeedOffset.VAL_EMPTY, CFG))
    ds = EmptyScenes(path)
    s = ds[2]
    assert len(ds) == 3 and s.image.shape == (3, 128, 128) and s.image.dtype == torch.float32
    assert 0 <= s.image.min() and s.image.max() <= 1
    assert s.heatmap.shape == (1, 32, 32) and s.heatmap.abs().sum() == 0
    assert s.class_id.item() == IGNORE_CLASS and not has_sign(s.class_id) and s.box.tolist() == [0, 0, 0, 0]
    with pytest.raises(IndexError):
        ds[3]


def test_empty_scenes_reject_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmptyScenes(tmp_path / "nope.npz")
    np.savez(tmp_path / "a.npz", images=np.zeros((2, 128, 128, 3), np.uint8))
    with pytest.raises(KeyError):
        EmptyScenes(tmp_path / "a.npz")
    np.savez(tmp_path / "b.npz", images=np.zeros((2, 128, 128, 3), np.float32), heatmap_size=np.int16(32))
    with pytest.raises(ValueError):
        EmptyScenes(tmp_path / "b.npz")
    np.savez(tmp_path / "c.npz", images=np.zeros((2, 128, 128, 3), np.uint8), heatmap_size=np.int16(30))
    with pytest.raises(ValueError):
        EmptyScenes(tmp_path / "c.npz")


# --- A9 on the built files (skipped until scripts/make_empty_dataset.py has run) ---

PROCESSED = Path("data/processed")
CONFIG = Path("configs/peak_fresh_aug.yaml")


def _built(name: str) -> dict:
    path = PROCESSED / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path} not built")
    with np.load(path) as d:
        return {k: d[k] for k in d}


@pytest.fixture(scope="module")
def sources():
    from findobject.data.splits import split_sources
    if not Path("data/raw/coco").exists():
        pytest.skip("raw data missing")
    return split_sources(yaml.safe_load(CONFIG.read_text())["data"])


@pytest.mark.parametrize("name,split", [("val_empty", "val"), ("test_empty", "test"),
                                        ("val_decoy_plain", "val"), ("val_decoy_aug", "val")])
def test_built_file_uses_only_its_splits_backgrounds(name, split, sources):
    a = _built(name)
    own = [p.name for p in sources[split].backgrounds]
    others = {p.name for s, src in sources.items() if s != split for p in src.backgrounds}
    used = set(a["background_name"].tolist()) | set(a.get("patch_background_name", np.array([])).tolist())
    assert used <= set(own) and not used & others
    assert [own[i] for i in a["background_index"]] == a["background_name"].tolist()


@pytest.mark.parametrize("name", ["val_decoy_plain", "val_decoy_aug"])
def test_built_decoy_patches_are_val_sign_roi_sizes(name, sources):
    a = _built(name)
    signs = sources["val"].signs
    assert all(tuple(size) == roi_size(signs[i]) for size, i in zip(a["patch_size"], a["sign_index"]))
