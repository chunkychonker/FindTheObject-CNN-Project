"""Leak check for v2: blank each real-dev sign (box filled with the window mean color) and re-score v1.5b seed 0.

Usage: python scripts/mask_check_real.py. If class accuracy stays well above the majority-class rate with the sign
gone, something other than the sign gives the answer away. Writes the masked npz to outputs/eval/mask_check/.
"""
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from findobject.data.dataset import SignScenes
from findobject.eval.evaluate import evaluate
from findobject.training.checkpoint import load_model, load_checkpoint_info
from findobject.utils.device import pick_device
out = Path("outputs/eval/mask_check"); out.mkdir(parents=True, exist_ok=True); C = Path("outputs/checkpoints/peak_fresh_aug-20261108-024220/best.pt")
d = dict(np.load("data/processed/real_dev.npz"))
img = d["images"].copy()
for i, (x1, y1, x2, y2) in enumerate(d["boxes"]):
    img[i, y1:y2, x1:x2] = img[i].reshape(-1, 3).mean(0).astype(np.uint8)
d["images"] = img
np.savez(out / "real_dev_masked.npz", **d)
dev = pick_device("auto"); m = load_model(C, dev); w = load_checkpoint_info(C).config["train"]["heatmap_loss_weight"]
for name in ("data/processed/real_dev.npz", str(out / "real_dev_masked.npz")):
    r = evaluate(m, DataLoader(SignScenes(Path(name)), batch_size=256), dev, w)
    print(f"{Path(name).name:22s} class {r.class_acc:.3f} pointing {r.pointing_acc:.3f} joint {r.joint_acc:.3f}")
cls = d["class_ids"]; print("majority-class rate:", np.bincount(cls).max() / len(cls))
