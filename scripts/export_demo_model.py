"""Copy a trained run's weights and its whole-photo threshold τ_W into demo/assets/ for the demo app.

Usage: python scripts/export_demo_model.py [--run outputs/checkpoints/peak_decoy-20261112-011953] [--out demo/assets]

τ_W comes from the run's stored dev report (outputs/eval/<run>/detect_dev.json), never from test. Safe to
re-run: it overwrites the previous export with the same content. Fails loudly if the checkpoint or the dev
report is missing, or if the report was made from a different checkpoint file than the one being exported.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from findobject.demo.assets import DemoMeta, export_assets
from findobject.training.checkpoint import load_model

DEFAULT_RUN = Path("outputs/checkpoints/peak_decoy-20261112-011953")  # seed 0, the run every demo figure uses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out", type=Path, default=Path("demo/assets"))
    args = ap.parse_args()

    ckpt = args.run / "best.pt"
    report_path = Path("outputs/eval") / args.run.name / "detect_dev.json"
    report = json.loads(report_path.read_text())  # FileNotFoundError names the missing file
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if report["checkpoint_sha256"] != sha:
        raise ValueError(f"{report_path} was made from a different {ckpt} (sha256 mismatch)")

    model = load_model(ckpt, torch.device("cpu"))
    meta = DemoMeta(run=args.run.name, tau_w=report["thresholds"]["tau_w"], checkpoint_sha256=sha)
    export_assets(model, meta, args.out)
    print(f"wrote {args.out}/: run {meta.run}, tau_w {meta.tau_w:.5f}")


if __name__ == "__main__":
    main()
