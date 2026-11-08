import hashlib

import numpy as np
import pytest
from PIL import Image

from findobject.data.sign_augment import SignAugment, augment_sign
from findobject.data.synth import SynthConfig, make_scene

STRONG = SignAugment(max_rotation_deg=30, max_perspective=0.1, contrast=(0.6, 1.4), saturation=(0.6, 1.4))

# make_scene with the default (no augmentation) SynthConfig must give exactly what it gave before
# sign augmentation existed. Computed by running the pre-augmentation synth.py (commit f34db5f) on _digest's
# inputs. Stored val/test regenerate byte-identically only while this holds.
PRE_AUGMENT_DIGEST = "d956770536dd0ec636a1a9908e6bc8439d4ee04defc86fad890b4931039ec254"


def _sign(w: int = 38, h: int = 40) -> Image.Image:
    return Image.fromarray(np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8))


# --- SignAugment ---

def test_default_is_identity_and_configured_is_not():
    assert SignAugment().is_identity
    assert not STRONG.is_identity


@pytest.mark.parametrize("kwargs", [
    {"max_rotation_deg": -1}, {"max_rotation_deg": 46}, {"max_perspective": 0.3},
    {"contrast": (0.0, 1.0)}, {"saturation": (1.2, 0.8)}, {"occlusion_prob": 1.5},
    {"occlusion_area": (0.0, 0.2)}, {"occlusion_area": (0.1, 1.0)},
])
def test_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        SignAugment(**kwargs)


# --- augment_sign ---

def test_identity_returns_input_pixels_full_mask_and_draws_nothing():
    sign = _sign()
    rng = np.random.default_rng(0)
    rgb, mask = augment_sign(sign, rng, SignAugment())
    assert np.array_equal(np.asarray(rgb), np.asarray(sign))
    assert mask.mode == "L" and mask.size == sign.size and np.asarray(mask).min() == 255
    assert rng.random() == np.random.default_rng(0).random()  # rng untouched


def test_warp_mask_matches_rgb_and_leaves_corners_uncovered():
    for seed in range(10):
        rgb, mask = augment_sign(_sign(), np.random.default_rng(seed), STRONG)
        m = np.asarray(mask)
        assert mask.size == rgb.size and mask.mode == "L"
        assert m.max() == 255  # the sign's interior is fully covered
        assert min(m[0, 0], m[0, -1], m[-1, 0], m[-1, -1]) < 255  # a tilted sign leaves corners to the background


# --- make_scene with augmentation ---

def _clean_cfg(aug: SignAugment) -> SynthConfig:
    """No blur, noise or brightness change, so every non-black pixel on a black background is sign."""
    return SynthConfig(brightness=(1.0, 1.0), blur_prob=0.0, noise_max_std=0.0, sign_augment=aug)


def test_augmented_box_is_tight_around_every_sign_pixel():
    bg = Image.new("RGB", (300, 300), (0, 0, 0))
    sign = Image.new("RGB", (40, 40), (0, 255, 0))
    aug = SignAugment(max_rotation_deg=30, max_perspective=0.1)
    for seed in range(30):
        scene = make_scene(sign, bg, np.random.default_rng(seed), _clean_cfg(aug))
        ys, xs = np.nonzero(scene.image.max(axis=2))
        x1, y1, x2, y2 = scene.box
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (x1, y1, x2, y2), f"seed {seed}"


def test_augmented_heatmap_peaks_at_box_center_and_box_fits_scene():
    cfg = SynthConfig(sign_augment=SignAugment(max_rotation_deg=10, max_perspective=0.08, occlusion_prob=1.0))
    bg = Image.fromarray(np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8))
    stride = cfg.scene_size // cfg.heatmap_size
    for seed in range(30):
        scene = make_scene(_sign(), bg, np.random.default_rng(seed), cfg)
        x1, y1, x2, y2 = scene.box
        assert 0 <= x1 < x2 <= cfg.scene_size and 0 <= y1 < y2 <= cfg.scene_size
        r, c = np.unravel_index(scene.heatmap.argmax(), scene.heatmap.shape)
        assert abs((c + 0.5) * stride - (x1 + x2) / 2) <= stride / 2 + 1e-6
        assert abs((r + 0.5) * stride - (y1 + y2) / 2) <= stride / 2 + 1e-6


def test_occlusion_changes_pixels_inside_the_box_only():
    rs = np.random.default_rng(2)
    bg = Image.fromarray(rs.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    sign = Image.new("RGB", (40, 40), (0, 255, 0))
    base = SignAugment(max_rotation_deg=5)
    occluded = SignAugment(max_rotation_deg=5, occlusion_prob=1.0)
    for seed in range(10):
        a = make_scene(sign, bg, np.random.default_rng(seed), _clean_cfg(base))
        b = make_scene(sign, bg, np.random.default_rng(seed), _clean_cfg(occluded))
        assert a.box == b.box  # the occluder is drawn after placement and does not shrink the box
        x1, y1, x2, y2 = a.box
        diff = np.any(a.image != b.image, axis=2)
        assert diff[y1:y2, x1:x2].any()
        diff[y1:y2, x1:x2] = False
        assert not diff.any()


def test_no_augmentation_reproduces_pre_augmentation_scenes():
    rs = np.random.default_rng(123)
    bg = Image.fromarray(rs.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    sign = Image.fromarray(rs.integers(0, 256, (40, 38, 3), dtype=np.uint8))
    h = hashlib.sha256()
    for seed in range(10):
        sc = make_scene(sign, bg, np.random.default_rng(seed), SynthConfig())
        h.update(sc.image.tobytes())
        h.update(np.asarray(sc.box, np.int64).tobytes())
        h.update(sc.heatmap.tobytes())
    assert h.hexdigest() == PRE_AUGMENT_DIGEST
