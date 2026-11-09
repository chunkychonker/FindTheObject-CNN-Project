"""CLI: build the v2 real-photo scenes: data/processed/{real,hybrid}_{dev,test}.npz + real_manifest.json.

Usage: python scripts/make_real_dataset.py --config configs/peak_fresh_aug.yaml

real_X: one 128x128 window per labeled GTSDB sign (docs/v2-spec.md §2.2). hybrid_X: the same signs' crops
pasted onto COCO backgrounds the way val/test scenes are made (val backgrounds for dev, test backgrounds
for test). Row i of real_X and hybrid_X is the same physical sign. Same array format as make_dataset.py,
plus photo_index, sign_index and native_long_side. Re-running overwrites with identical output.
Raises FileNotFoundError if GTSDB/GTSRB/COCO are missing; see build_real_scenes for the rest.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from findobject.data.data_config import synth_config
from findobject.data.gtsdb import GT_FILE, RealSplit, photos_in, read_gtsdb
from findobject.data.real_scenes import build_real_scenes
from findobject.data.splits import split_sources

GTSDB_DIR = Path("data/raw/gtsdb/FullIJCNN2013")
BACKGROUNDS_FOR = {RealSplit.DEV: "val", RealSplit.TEST: "test"}  # synthetic split whose COCO images to use
MANIFEST = "real_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/peak_fresh_aug.yaml"))
    parser.add_argument("--gtsdb-dir", type=Path, default=GTSDB_DIR)
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.config.read_text())["data"]
    cfg = synth_config(data_cfg)  # unaugmented, like the stored val/test scenes
    seed = data_cfg["seed"]
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = read_gtsdb(args.gtsdb_dir)
    sources = split_sources(data_cfg)

    manifest = {
        "gt_sha256": hashlib.sha256((args.gtsdb_dir / GT_FILE).read_bytes()).hexdigest(),
        "seed": seed,
        "config": str(args.config),
        "synth": _jsonable(dataclasses.asdict(cfg)),
        "splits": {},
    }
    for split in RealSplit:
        split_photos = photos_in(photos, split)
        bg_split = BACKGROUNDS_FOR[split]
        print(f"{split.value}: {len(split_photos)} photos, hybrid backgrounds from synthetic {bg_split}")
        out = build_real_scenes(split_photos, sources[bg_split].backgrounds, seed, cfg)
        for kind, arrays in (("real", out.real), ("hybrid", out.hybrid)):
            path = out_dir / f"{kind}_{split.value}.npz"
            _save_atomic(path, arrays)
            print(f"  wrote {path} ({len(arrays['images'])} scenes)")
        reasons: dict[str, int] = {}
        for s in out.skipped:
            reasons[s.reason.value] = reasons.get(s.reason.value, 0) + 1
        manifest["splits"][split.value] = {
            "photos": len(split_photos),
            "signs": out.n_signs,
            "windows": len(out.real["images"]),
            "skipped": reasons,
            "skipped_signs": [[s.photo_index, s.sign_index, s.reason.value] for s in out.skipped],
            "hybrid_backgrounds": bg_split,
        }
        print(f"  {out.n_signs} signs -> {len(out.real['images'])} windows, skipped {reasons}")

    tmp = out_dir / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out_dir / MANIFEST)
    print(f"wrote {out_dir / MANIFEST}")


def _save_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _jsonable(obj):
    """Tuples -> lists, recursively (dataclasses.asdict keeps tuples, which json writes fine, but be explicit)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
