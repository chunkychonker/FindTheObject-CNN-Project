import numpy as np
import pytest
from PIL import Image

from findobject.data.dataset import SignScenes
from findobject.data.gtsdb import PHOTO_SIZE, RealPhoto, RealSign
from findobject.data.real_scenes import build_real_scenes
from findobject.data.real_windows import Skip
from findobject.data.synth import SynthConfig

CFG = SynthConfig()


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("gtsdb")
    rs = np.random.default_rng(0)
    w, h = PHOTO_SIZE
    paths = []
    for i in range(4):
        p = root / f"{i:05d}.ppm"
        Image.fromarray(rs.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    bgs = []
    for k in range(2):
        b = root / f"bg{k}.png"
        Image.fromarray(rs.integers(0, 256, (200, 260, 3), dtype=np.uint8)).save(b)
        bgs.append(b)
    photos = [
        RealPhoto(paths[0], 0, (RealSign((100, 100, 140, 140), 3), RealSign((900, 500, 1000, 590), 14))),
        RealPhoto(paths[1], 1, ()),  # no signs: contributes nothing
        # 2: the middle sign has neighbors right next to it on both sides -> skipped; the outer two are not
        RealPhoto(paths[2], 2, (RealSign((600, 300, 630, 330), 1), RealSign((560, 300, 599, 330), 2),
                                RealSign((631, 300, 660, 330), 5))),
        RealPhoto(paths[3], 3, (RealSign((1300, 750, 1360, 800), 42),)),  # at the corner, downscaled
    ]
    return photos, bgs


def test_real_and_hybrid_pair_row_for_row(release):  # A4
    photos, bgs = release
    out = build_real_scenes(photos, bgs, seed=0, cfg=CFG)
    n = len(out.real["images"])
    assert n == len(out.hybrid["images"])
    for k in ("photo_index", "sign_index", "class_ids", "native_long_side"):
        assert np.array_equal(out.real[k], out.hybrid[k])
    assert out.n_signs == 6 and n + len(out.skipped) == out.n_signs
    assert [(s.photo_index, s.sign_index, s.reason) for s in out.skipped] == [(2, 0, Skip.OTHER_SIGN_IN_VIEW)]
    assert out.real["native_long_side"].tolist() == [40, 100, 39, 30, 60]  # longer side: 631-660 x 300-330 is 29 x 30


def test_same_inputs_same_bytes_and_seed_matters(release):  # A3
    photos, bgs = release
    a, b = build_real_scenes(photos, bgs, 0, CFG), build_real_scenes(photos, bgs, 0, CFG)
    for part in ("real", "hybrid"):
        for k in a.real:
            assert np.array_equal(getattr(a, part)[k], getattr(b, part)[k]), (part, k)
    c = build_real_scenes(photos, bgs, 1, CFG)
    assert not np.array_equal(a.real["images"], c.real["images"])


def test_a_sign_does_not_depend_on_the_photos_around_it(release):
    photos, bgs = release
    alone = build_real_scenes(photos[3:], bgs, 0, CFG)
    together = build_real_scenes(photos, bgs, 0, CFG)
    assert np.array_equal(alone.real["images"][0], together.real["images"][-1])
    assert np.array_equal(alone.hybrid["images"][0], together.hybrid["images"][-1])


def test_output_loads_as_sign_scenes(release, tmp_path):
    photos, bgs = release
    out = build_real_scenes(photos, bgs, 0, CFG)
    np.savez(tmp_path / "real.npz", **out.real)
    ds = SignScenes(tmp_path / "real.npz")
    assert len(ds) == len(out.real["images"]) and ds[0].image.shape == (3, 128, 128)


def test_no_windows_or_no_backgrounds_raise(release):
    photos, bgs = release
    with pytest.raises(ValueError, match="no sign got a window"):
        build_real_scenes([photos[1]], bgs, 0, CFG)
    with pytest.raises(ValueError, match="backgrounds"):
        build_real_scenes(photos, [], 0, CFG)
