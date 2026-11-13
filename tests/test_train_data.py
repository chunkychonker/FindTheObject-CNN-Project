from pathlib import Path

import numpy as np
import pytest

from findobject.data.data_config import synth_config, train_synth_config
from findobject.data.fresh import FreshScenes
from findobject.data.gtsrb import track_id
from findobject.data.sign_augment import SignAugment
from findobject.data.splits import split_sources
from PIL import Image

from findobject.data.gtsrb import SignSample
from findobject.data.splits import SceneSource
from findobject.training.train_data import TrainScenes, decoy_sources, epoch_datasets

SYNTH = {"scene_size": 128, "heatmap_size": 32, "sign_min": 16, "sign_max": 48, "brightness": [0.6, 1.4],
         "blur_prob": 0.5, "blur_max_radius": 1.0, "noise_max_std": 8.0, "heatmap_sigma": 1.0}
AUG = {"max_rotation_deg": 10, "max_perspective": 0.08, "contrast": [0.6, 1.4], "saturation": [0.6, 1.4],
       "occlusion_prob": 0.3, "occlusion_area": [0.10, 0.25]}
REAL = {"raw_dir": "data/raw", "processed_dir": "data/processed", "seed": 0, "n_train": 20000,
        "test_background_every": 5, "holdout_fraction": 0.1, "synth": SYNTH}
needs_raw = pytest.mark.skipif(not Path("data/raw/gtsrb").exists() or not Path("data/raw/coco").exists(),
                               reason="raw GTSRB/COCO data not downloaded")


def _tiny_train_npz(tmp_path) -> dict:
    n, s, h = 3, 16, 4
    np.savez(tmp_path / "train.npz", images=np.zeros((n, s, s, 3), np.uint8), class_ids=np.zeros(n, np.int64),
             boxes=np.tile(np.array([2, 2, 8, 8], np.int16), (n, 1)), heatmaps=np.zeros((n, h, h), np.float16))
    return {"processed_dir": str(tmp_path), "synth": SYNTH}


# --- data_config ---

def test_synth_config_never_augments_even_if_train_augment_present():
    assert synth_config({"synth": SYNTH, "train_augment": AUG}).sign_augment.is_identity


def test_train_synth_config_reads_augment_lists_as_tuples():
    cfg = train_synth_config({"synth": SYNTH, "train_augment": AUG})
    assert cfg.sign_augment == SignAugment(10, 0.08, (0.6, 1.4), (0.6, 1.4), 0.3, (0.10, 0.25))
    assert cfg.brightness == (0.6, 1.4) and cfg.scene_size == 128


def test_train_synth_config_without_augment_is_identity():
    assert train_synth_config({"synth": SYNTH}).sign_augment.is_identity


def test_train_synth_config_rejects_unknown_or_bad_keys():
    with pytest.raises(TypeError):
        train_synth_config({"synth": SYNTH, "train_augment": {"hue": 0.1}})
    with pytest.raises(ValueError):
        train_synth_config({"synth": SYNTH, "train_augment": {"max_rotation_deg": 90}})


# --- epoch_datasets ---

def test_fixed_is_default_and_reuses_one_dataset(tmp_path):
    for_epoch = epoch_datasets(_tiny_train_npz(tmp_path), seed=0)
    assert len(for_epoch(1)) == 3 and for_epoch(1) is for_epoch(7)


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        epoch_datasets({**_tiny_train_npz(tmp_path), "train_scenes": "random"}, seed=0)
    with pytest.raises(ValueError):
        TrainScenes("random")


def test_augment_with_fixed_scenes_rejected(tmp_path):
    with pytest.raises(ValueError, match="train_augment"):
        epoch_datasets({**_tiny_train_npz(tmp_path), "train_augment": AUG}, seed=0)


def test_empties_with_fixed_scenes_rejected(tmp_path):
    with pytest.raises(ValueError, match="n_train_empty"):
        epoch_datasets({**_tiny_train_npz(tmp_path), "n_train_empty": 1}, seed=0)


@pytest.mark.parametrize("n_empty", [-1, 20000, 25000, 1.5, True, "100"])
def test_bad_n_train_empty_rejected_before_reading_any_data(n_empty):
    cfg = {**REAL, "raw_dir": "/nonexistent", "train_scenes": "fresh", "n_train_empty": n_empty}
    with pytest.raises(ValueError, match="n_train_empty"):
        epoch_datasets(cfg, seed=0)


def test_decoys_with_fixed_scenes_rejected(tmp_path):
    with pytest.raises(ValueError, match="n_train_decoy"):
        epoch_datasets({**_tiny_train_npz(tmp_path), "n_train_decoy": 1}, seed=0)


@pytest.mark.parametrize("n_empty,n_decoy", [(0, -1), (0, 1.5), (0, True), (0, "100"), (0, 20000), (4000, 16000)])
def test_bad_n_train_decoy_rejected_before_reading_any_data(n_empty, n_decoy):
    cfg = {**REAL, "raw_dir": "/nonexistent", "train_scenes": "fresh", "n_train_empty": n_empty, "n_train_decoy": n_decoy}
    with pytest.raises(ValueError, match="n_train_decoy"):
        epoch_datasets(cfg, seed=0)


def test_decoy_sources_reads_sizes_and_drops_small_backgrounds(tmp_path):
    bgs = []
    for k, size in enumerate([(300, 300), (40, 300), (300, 300), (300, 40)]):  # (w, h); 1 too narrow, 3 too short
        bgs.append(tmp_path / f"bg{k}.png")
        Image.new("RGB", size).save(bgs[-1])
    signs = [SignSample(path=tmp_path / "s.png", roi=(0, 0, 49, 9), class_id=0),  # 50x10
             SignSample(path=tmp_path / "s.png", roi=(5, 5, 14, 54), class_id=1)]  # 10x50
    d = decoy_sources(SceneSource(signs, bgs))
    assert d.need == (50, 50) and d.index == (0, 2)
    with pytest.raises(ValueError):
        decoy_sources(SceneSource(signs, bgs[:2]))


@needs_raw
def test_fresh_with_decoys_keeps_the_epoch_length():
    for_epoch = epoch_datasets({**REAL, "train_scenes": "fresh", "train_augment": AUG, "n_train_decoy": 4000}, seed=0)
    e1 = for_epoch(1)
    assert len(e1) == 20000 and e1[15999].class_id.item() != -100 and e1[16000].class_id.item() == -100
    assert e1[19999].class_id.item() == -100


@needs_raw
def test_fresh_with_empties_keeps_the_epoch_length():
    for_epoch = epoch_datasets({**REAL, "train_scenes": "fresh", "train_augment": AUG, "n_train_empty": 4000}, seed=0)
    e1 = for_epoch(1)
    assert len(e1) == 20000 and e1[15999].class_id.item() != -100 and e1[16000].class_id.item() == -100


@needs_raw
def test_fresh_gives_n_train_new_scenes_per_epoch():
    for_epoch = epoch_datasets({**REAL, "train_scenes": "fresh", "train_augment": AUG}, seed=0)
    e1, e2 = for_epoch(1), for_epoch(2)
    assert isinstance(e1, FreshScenes) and len(e1) == len(e2) == 20000
    assert not np.array_equal(e1[0].image.numpy(), e2[0].image.numpy())


# --- split_sources: fresh training scenes may only draw from the train split ---

@needs_raw
def test_train_source_shares_no_track_or_background_with_val_or_test():
    src = split_sources(REAL)
    train_tracks = {(s.class_id, track_id(s)) for s in src["train"].signs}
    val_tracks = {(s.class_id, track_id(s)) for s in src["val"].signs}
    assert train_tracks and val_tracks and not train_tracks & val_tracks
    train_paths = {s.path for s in src["train"].signs}
    assert not train_paths & {s.path for s in src["test"].signs}
    train_bgs = set(src["train"].backgrounds)
    assert not train_bgs & set(src["val"].backgrounds) and not train_bgs & set(src["test"].backgrounds)
