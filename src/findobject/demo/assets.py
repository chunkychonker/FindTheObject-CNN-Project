"""Save and load what the demo app needs (weights, τ_W, run name), so the app doesn't depend on outputs/."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from findobject.models.factory import Arch, build_model
from findobject.models.peak import PeakCNN

MODEL_FILE = "model.pt"
META_FILE = "meta.json"
EXPORT_HINT = "run: python scripts/export_demo_model.py"


@dataclass(frozen=True)
class DemoMeta:
    run: str  # the training run the weights came from
    tau_w: float  # that run's whole-photo threshold, chosen on dev photos (docs/v3b-spec.md)
    checkpoint_sha256: str  # of the checkpoint the weights were copied from, as the dev report recorded it


def export_assets(model: nn.Module, meta: DemoMeta, out_dir: Path) -> None:
    """Write weights only (no optimizer state) to out_dir/MODEL_FILE and meta to out_dir/META_FILE, creating
    out_dir. Overwrites earlier exports; re-running with the same inputs writes the same weights.
    Raises TypeError if the model is not a PeakCNN; OSError on write failure."""
    if not isinstance(model, PeakCNN):
        raise TypeError(f"the demo needs a PeakCNN, got {type(model).__name__}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": Arch.PEAK.value, "model": model.state_dict()}, out_dir / MODEL_FILE)
    (out_dir / META_FILE).write_text(json.dumps(meta.__dict__, indent=1) + "\n")


def load_assets(asset_dir: Path, device: torch.device) -> tuple[PeakCNN, DemoMeta]:
    """The exported model, in eval mode on `device`, and its meta.
    Raises FileNotFoundError (naming the file and how to make it) if either file is missing; KeyError or
    RuntimeError if the files don't match what export_assets writes."""
    for name in (MODEL_FILE, META_FILE):
        if not (asset_dir / name).is_file():
            raise FileNotFoundError(f"{asset_dir / name} not found; {EXPORT_HINT}")
    saved = torch.load(asset_dir / MODEL_FILE, map_location=device, weights_only=True)
    model = build_model(Arch(saved["arch"]))
    model.load_state_dict(saved["model"])
    meta = DemoMeta(**json.loads((asset_dir / META_FILE).read_text()))
    return model.to(device).eval(), meta
