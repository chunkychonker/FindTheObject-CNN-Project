import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from findobject.data.data_config import train_synth_config
from findobject.data.dataset import IGNORE_CLASS, has_sign
from findobject.data.decoy_patch import PATCH_STREAM, draw_patch, eligible_patch_sources
from findobject.data.fresh import FreshScenes
from findobject.data.gtsrb import SignSample, load_sign
from findobject.data.sign_augment import SignAugment
from findobject.data.splits import SceneSource
from findobject.data.synth import SynthConfig, make_decoy_scene
from findobject.data.synth_negatives import roi_size

CFG = SynthConfig(sign_augment=SignAugment(max_rotation_deg=10, occlusion_prob=0.5))


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> SceneSource:
    root = tmp_path_factory.mktemp("raw")
    rs = np.random.default_rng(0)
    signs, bgs = [], []
    for k in range(4):
        path = root / f"0000{k}_00000.png"
        Image.fromarray(rs.integers(0, 256, (50, 46, 3), dtype=np.uint8)).save(path)
        signs.append(SignSample(path=path, roi=(3, 3, 42, 46), class_id=k * 10))
    for k in range(3):
        path = root / f"bg{k}.png"
        Image.fromarray(rs.integers(0, 256, (200, 260, 3), dtype=np.uint8)).save(path)
        bgs.append(path)
    return SceneSource(signs, bgs)


def _same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


# SHA-256 of FreshScenes scenes 0-9 of epochs 1 and 2 with v1.5b's exact training settings, recorded from
# the code v1.5b trained with (before any v3a change). It guards the augmented path that
# PRE_AUGMENT_DIGEST (tests/test_sign_augment.py) does not cover.
FRESH_AUG_DIGEST = "a81cf0b5b53d72fd39c75582425ac74d35a0059c11188ec5e249139fde30d503"
V15B_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "peak_fresh_aug.yaml"


def fresh_aug_digest(source: SceneSource) -> str:
    cfg = train_synth_config(yaml.safe_load(V15B_CONFIG.read_text())["data"])
    h = hashlib.sha256()
    for epoch in (1, 2):
        ds = FreshScenes(source, cfg, 10, seed=0, epoch=epoch)
        for i in range(10):
            for t in ds[i]:
                h.update(t.numpy().tobytes())
    return h.hexdigest()


def test_v15b_training_scenes_unchanged(source):
    assert fresh_aug_digest(source) == FRESH_AUG_DIGEST


def test_sample_format_matches_stored_scenes(source):
    s = FreshScenes(source, CFG, n=5, seed=0, epoch=1)[0]
    assert s.image.shape == (3, 128, 128) and s.image.dtype == torch.float32
    assert 0 <= s.image.min() and s.image.max() <= 1
    assert s.heatmap.shape == (1, 32, 32) and s.heatmap.dtype == torch.float32
    assert s.class_id.dtype == torch.int64 and s.class_id.item() in (0, 10, 20, 30)
    assert s.box.shape == (4,) and s.box.dtype == torch.int64


def test_scene_depends_only_on_seed_epoch_index(source):
    a, b = FreshScenes(source, CFG, 8, seed=3, epoch=2), FreshScenes(source, CFG, 8, seed=3, epoch=2)
    assert _same(a[5], b[5])
    assert _same(a[5], a[5])  # asking twice gives the same scene
    assert _same(b[5], FreshScenes(source, CFG, 100, seed=3, epoch=2)[5])  # n doesn't affect scene i


def test_new_epoch_or_seed_gives_new_scenes(source):
    base = FreshScenes(source, CFG, 4, seed=0, epoch=1)
    for other in (FreshScenes(source, CFG, 4, seed=0, epoch=2), FreshScenes(source, CFG, 4, seed=1, epoch=1)):
        assert not any(torch.equal(base[i].image, other[i].image) for i in range(4))


def test_identical_with_zero_or_several_workers(source):
    ds = FreshScenes(source, CFG, 12, seed=0, epoch=1)
    serial = list(DataLoader(ds, batch_size=4, num_workers=0))
    parallel = list(DataLoader(ds, batch_size=4, num_workers=2))
    assert len(serial) == len(parallel) == 3
    assert all(_same(a, b) for a, b in zip(serial, parallel))


@pytest.mark.parametrize("n,seed,epoch", [(0, 0, 1), (5, -1, 1), (5, 0, -1)])
def test_rejects_bad_arguments(source, n, seed, epoch):
    with pytest.raises(ValueError):
        FreshScenes(source, CFG, n, seed, epoch)


def test_rejects_empty_source(source):
    with pytest.raises(ValueError):
        FreshScenes(SceneSource([], source.backgrounds), CFG, 5, 0, 1)
    with pytest.raises(ValueError):
        FreshScenes(SceneSource(source.signs, []), CFG, 5, 0, 1)


def test_index_out_of_range(source):
    ds = FreshScenes(source, CFG, 3, 0, 1)
    with pytest.raises(IndexError):
        ds[3]


# --- v3a: the last n_empty scenes hold no sign ---

def test_empties_leave_sign_scenes_unchanged_and_fill_the_tail(source):
    plain, mixed = FreshScenes(source, CFG, 10, seed=0, epoch=1), FreshScenes(source, CFG, 10, seed=0, epoch=1, n_empty=4)
    assert len(mixed) == 10
    for i in range(6):
        assert _same(plain[i], mixed[i])
    for i in range(6, 10):
        s = mixed[i]
        assert s.class_id.item() == IGNORE_CLASS and not has_sign(s.class_id)
        assert s.heatmap.abs().sum().item() == 0 and s.box.tolist() == [0, 0, 0, 0]
        assert s.image.shape == plain[i].image.shape and s.heatmap.shape == plain[i].heatmap.shape


def test_empty_scene_has_the_background_its_sign_scene_was_pasted_onto(source):
    # No blur or noise, so outside the sign's box the sign scene is exactly its background crop.
    cfg = dataclasses.replace(CFG, blur_prob=0.0, noise_max_std=0.0)
    for i in range(5, 10):
        sign = FreshScenes(source, cfg, 10, seed=2, epoch=3)[i]
        empty = FreshScenes(source, cfg, 10, seed=2, epoch=3, n_empty=5)[i]
        x1, y1, x2, y2 = sign.box.tolist()
        outside = torch.ones(sign.image.shape[1:], dtype=torch.bool)
        outside[y1:y2, x1:x2] = False
        assert outside.any() and torch.equal(sign.image[:, outside], empty.image[:, outside])


@pytest.mark.parametrize("n_empty", [-1, 10, 11])
def test_rejects_n_empty_outside_0_to_n(source, n_empty):
    with pytest.raises(ValueError):
        FreshScenes(source, CFG, 10, 0, 1, n_empty=n_empty)


# --- v3a.1: decoys sit between the sign scenes and the empties ---

def _patches(source):
    with Image.open(source.backgrounds[0]) as bg:
        size = bg.size
    return eligible_patch_sources([size] * len(source.backgrounds), (40, 44))  # fixture ROIs are 40x44


def test_decoys_leave_sign_scenes_and_empties_unchanged(source):
    ps = _patches(source)
    plain = FreshScenes(source, CFG, 10, seed=0, epoch=1, n_empty=3)
    mixed = FreshScenes(source, CFG, 10, seed=0, epoch=1, n_empty=3, n_decoy=4, patch_sources=ps)
    for i in (*range(3), *range(7, 10)):  # sign scenes 0-2, empties 7-9
        assert _same(plain[i], mixed[i])
    assert not torch.equal(plain[3].image, mixed[3].image)  # scene 3 is now a decoy


def test_decoy_targets_are_an_empty_scenes(source):
    ds = FreshScenes(source, CFG, 6, seed=0, epoch=1, n_decoy=3, patch_sources=_patches(source))
    for i in range(3, 6):
        s = ds[i]
        assert s.class_id.item() == IGNORE_CLASS and not has_sign(s.class_id)
        assert s.heatmap.shape == (1, 32, 32) and s.heatmap.abs().sum().item() == 0
        assert s.box.tolist() == [0, 0, 0, 0]
    assert has_sign(ds[2].class_id)


@pytest.mark.parametrize("cfg", [CFG, SynthConfig()], ids=["augmented", "plain"])
def test_decoy_is_its_sign_scene_with_only_the_content_swapped(source, cfg):
    # No blur or noise: outside sign i's box, decoy i must be pixel-identical to sign scene i (same crop,
    # same paste geometry, so the same box); inside, the patch's pixels replace the sign's.
    cfg = dataclasses.replace(cfg, blur_prob=0.0, noise_max_std=0.0)
    for i in range(5, 10):
        sign = FreshScenes(source, cfg, 10, seed=2, epoch=3)[i]
        decoy = FreshScenes(source, cfg, 10, seed=2, epoch=3, n_decoy=5, patch_sources=_patches(source))[i]
        x1, y1, x2, y2 = sign.box.tolist()
        outside = torch.ones(sign.image.shape[1:], dtype=torch.bool)
        outside[y1:y2, x1:x2] = False
        assert torch.equal(sign.image[:, outside], decoy.image[:, outside])
        assert not torch.equal(sign.image[:, ~outside], decoy.image[:, ~outside])


def test_decoy_follows_the_documented_draws(source):
    # Contract: scene RNG draws sign index, background index, then make_scene's draws; the patch RNG
    # [seed, epoch, i, PATCH_STREAM] draws source and corner; the patch is cut at the sign's ROI size.
    ps, seed, epoch, i = _patches(source), 4, 2, 7
    got = FreshScenes(source, CFG, 8, seed, epoch, n_decoy=2, patch_sources=ps)[i]
    rng = np.random.default_rng([seed, epoch, i])
    sign = source.signs[rng.integers(len(source.signs))]
    bg_index = int(rng.integers(len(source.backgrounds)))
    src_index, (x, y) = draw_patch(ps, bg_index, roi_size(sign), np.random.default_rng([seed, epoch, i, PATCH_STREAM]))
    assert src_index != bg_index
    w, h = roi_size(sign)
    with Image.open(source.backgrounds[src_index]) as src:
        patch = src.convert("RGB").crop((x, y, x + w, y + h))
    assert patch.size == load_sign(sign).size
    with Image.open(source.backgrounds[bg_index]) as bg:
        want = make_decoy_scene(patch, bg, rng, CFG)
    assert torch.equal(got.image, torch.from_numpy(want.image).permute(2, 0, 1).float().div_(255))


def test_decoys_identical_with_zero_or_several_workers(source):
    ds = FreshScenes(source, CFG, 8, seed=0, epoch=1, n_empty=2, n_decoy=4, patch_sources=_patches(source))
    serial = list(DataLoader(ds, batch_size=4, num_workers=0))
    parallel = list(DataLoader(ds, batch_size=4, num_workers=2))
    assert all(_same(a, b) for a, b in zip(serial, parallel))


@pytest.mark.parametrize("n_empty,n_decoy", [(0, -1), (0, 10), (4, 6), (9, 1)])
def test_rejects_bad_decoy_counts(source, n_empty, n_decoy):
    with pytest.raises(ValueError):
        FreshScenes(source, CFG, 10, 0, 1, n_empty=n_empty, n_decoy=n_decoy, patch_sources=_patches(source))


def test_decoys_need_patch_sources_that_match_the_source(source):
    with pytest.raises(ValueError, match="patch_sources"):
        FreshScenes(source, CFG, 10, 0, 1, n_decoy=2)
    too_many = eligible_patch_sources([(260, 200)] * (len(source.backgrounds) + 1), (40, 44))
    with pytest.raises(ValueError, match="patch_sources"):
        FreshScenes(source, CFG, 10, 0, 1, n_decoy=2, patch_sources=too_many)
