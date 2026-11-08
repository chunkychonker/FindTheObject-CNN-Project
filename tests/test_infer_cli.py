"""End-to-end checks of scripts/infer.py and scripts/make_demo_figure.py on a tiny untrained checkpoint."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from findobject.data.class_names import CLASS_NAMES
from findobject.inference.predict import predict
from findobject.models.peak import PeakCNN
from findobject.training.checkpoint import load_model, save_checkpoint

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = _load_script("infer")
demo = _load_script("make_demo_figure")


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = PeakCNN()
    optimizer = AdamW(model.parameters())
    path = tmp_path / "run" / "best.pt"
    config = {"model": {"arch": "peak"}, "data": {"processed_dir": str(tmp_path / "data")}}
    save_checkpoint(path, model, optimizer, CosineAnnealingLR(optimizer, T_max=1), 1, config, {})
    return path


def _write_png(path, size=(128, 128), seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return arr


# --- infer.py ---

def test_infer_matches_the_model_on_the_same_scene_and_writes_overlays(tmp_path, checkpoint, capsys):
    arr = _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png", size=(300, 200), seed=1)  # non-square, gets cropped and resized
    out = tmp_path / "out"
    infer.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--out", str(out),
                str(tmp_path / "a.png"), str(tmp_path / "b.png")])

    # Same pixels straight into the model: the CLI must report the same class and peak.
    direct = predict(load_model(checkpoint, torch.device("cpu")),
                     torch.from_numpy(arr).permute(2, 0, 1).float().div(255)[None])[0]
    line_a = next(line for line in capsys.readouterr().out.splitlines() if "a.png" in line)
    assert f"class {direct.class_id} {CLASS_NAMES[direct.class_id]!r}" in line_a
    assert f"peak at ({direct.peak_xy[0]:.0f}, {direct.peak_xy[1]:.0f})" in line_a
    assert sorted(p.name for p in out.iterdir()) == ["a_overlay.png", "b_overlay.png"]


def test_infer_missing_image_fails_before_writing_anything(tmp_path, checkpoint):
    _write_png(tmp_path / "a.png")
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        infer.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--out", str(out),
                    str(tmp_path / "a.png"), str(tmp_path / "missing.png")])
    assert not out.exists()


def test_infer_bad_size_fails_before_writing_anything(tmp_path, checkpoint):
    _write_png(tmp_path / "a.png")
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        infer.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--out", str(out),
                    "--size", "130", str(tmp_path / "a.png")])
    assert not out.exists()


def test_infer_refuses_inputs_whose_overlays_would_collide(tmp_path):
    with pytest.raises(ValueError, match="stem"):
        infer.overlay_paths([Path("x/photo.png"), Path("y/photo.jpg")], tmp_path)


# --- make_demo_figure.py ---

def test_pick_examples_is_seeded_and_respects_correctness():
    correct = torch.tensor([True, False] * 10)
    right, wrong = demo.pick_examples(correct, 4, 3, seed=0)
    assert (right, wrong) == demo.pick_examples(correct, 4, 3, seed=0)
    assert len(right) == 4 and len(wrong) == 3
    assert all(correct[i] for i in right) and not any(correct[i] for i in wrong)
    assert right == sorted(right) and wrong == sorted(wrong)


def test_pick_examples_takes_all_when_too_few():
    right, wrong = demo.pick_examples(torch.tensor([True, True, True, False]), 2, 5, seed=0)
    assert len(right) == 2 and wrong == [3]


def test_demo_figure_end_to_end(tmp_path, checkpoint):
    n = 6
    rng = np.random.default_rng(0)
    (tmp_path / "data").mkdir()
    np.savez(tmp_path / "data" / "val.npz",
             images=rng.integers(0, 256, (n, 128, 128, 3), dtype=np.uint8),
             class_ids=rng.integers(0, 43, n),
             boxes=np.tile(np.array([[40, 40, 80, 80]]), (n, 1)),
             heatmaps=np.zeros((n, 32, 32), dtype=np.float16))
    out = tmp_path / "fig" / "demo.png"
    demo.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--n-right", "2", "--n-wrong", "2",
               "--out", str(out)])
    with Image.open(out) as img:
        assert img.format == "PNG"
    assert not out.with_name("demo.png.tmp").exists()
