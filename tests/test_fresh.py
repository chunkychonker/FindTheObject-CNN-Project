import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from findobject.data.data_config import train_synth_config
from findobject.data.fresh import FreshScenes
from findobject.data.gtsrb import SignSample
from findobject.data.sign_augment import SignAugment
from findobject.data.splits import SceneSource
from findobject.data.synth import SynthConfig

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
