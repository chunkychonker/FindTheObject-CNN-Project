"""CLI: build v3a's no-sign evaluation scenes (docs/v3a-spec.md §2.4) into data/processed/ + empty_manifest.json.

Usage: python scripts/make_empty_dataset.py --config configs/peak_fresh_aug.yaml

  val_empty, test_empty      5,000 make_empty_scene scenes on val / test COCO backgrounds
  val_decoy_plain            5,000 decoys through the val/test paste path (data.synth)
  val_decoy_aug              the same 5,000 draws through the training paste path (data.train_augment)
  real_empty_dev, _test      one UNIFORM and one BAND negative window per GTSDB dev / test photo

Needs v2's real_dev.npz / real_test.npz (their native_long_side gives the positive scales). Existing npz
files are never touched. Re-running overwrites these files with byte-identical output; the manifest
records each file's SHA-256 so a rebuild can be checked.
Raises FileNotFoundError if GTSDB/GTSRB/COCO or the real_*.npz files are missing; see the builders
(synth_negatives.py, real_negatives.py) for the rest.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from findobject.data.data_config import synth_config, train_synth_config
from findobject.data.gtsdb import GT_FILE, RealSplit, photos_in, read_gtsdb
from findobject.data.real_negatives import BAND_JITTER_PX, MARGIN_PX, NEG_STREAM, Stratum, build_real_negatives
from findobject.data.splits import split_sources
from findobject.data.synth_negatives import EMPTY_SEED, SeedOffset, build_decoys, build_empties

GTSDB_DIR = Path("data/raw/gtsdb/FullIJCNN2013")
N_SYNTH = 5000
MANIFEST = "empty_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/peak_fresh_aug.yaml"))
    parser.add_argument("--gtsdb-dir", type=Path, default=GTSDB_DIR)
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.config.read_text())["data"]
    plain, aug = synth_config(data_cfg), train_synth_config(data_cfg)
    seed = data_cfg["seed"]
    out_dir = Path(data_cfg["processed_dir"])
    sources = split_sources(data_cfg)
    val, test = sources["val"], sources["test"]
    manifest = {
        "config": str(args.config),
        "synth": _jsonable(dataclasses.asdict(plain)),
        "train_synth": _jsonable(dataclasses.asdict(aug)),
        "synthetic": {"empty_seed": EMPTY_SEED, "files": {}},
        "real": {"seed": seed, "neg_stream": NEG_STREAM, "margin_px": MARGIN_PX, "band_jitter_px": BAND_JITTER_PX,
                 "gt_sha256": hashlib.sha256((args.gtsdb_dir / GT_FILE).read_bytes()).hexdigest(), "files": {}},
    }

    synthetic = {
        "val_empty": (SeedOffset.VAL_EMPTY, lambda: build_empties(val.backgrounds, N_SYNTH, SeedOffset.VAL_EMPTY, plain)),
        "test_empty": (SeedOffset.TEST_EMPTY, lambda: build_empties(test.backgrounds, N_SYNTH, SeedOffset.TEST_EMPTY, plain)),
        "val_decoy_plain": (SeedOffset.VAL_DECOY,
                            lambda: build_decoys(val.signs, val.backgrounds, N_SYNTH, SeedOffset.VAL_DECOY, plain)),
        "val_decoy_aug": (SeedOffset.VAL_DECOY,
                          lambda: build_decoys(val.signs, val.backgrounds, N_SYNTH, SeedOffset.VAL_DECOY, aug)),
    }
    for name, (offset, build) in synthetic.items():
        print(f"{name}: {N_SYNTH} scenes, seed offset {offset.name}")
        path = out_dir / f"{name}.npz"
        _save_atomic(path, build())
        manifest["synthetic"]["files"][name] = {"n": N_SYNTH, "seed_offset": offset.value, "sha256": _sha256(path)}

    photos = read_gtsdb(args.gtsdb_dir)
    for split in RealSplit:
        split_photos = photos_in(photos, split)
        with np.load(out_dir / f"real_{split.value}.npz") as d:
            # Exactly v2's plan_window scale: min(1, sign_max / longer side), in float64.
            scales = np.minimum(1.0, plain.sign_max / d["native_long_side"].astype(np.float64))
        rows = np.asarray([(s.box[1] + s.box[3]) / 2 for p in split_photos for s in p.signs], np.float64)
        print(f"real_empty_{split.value}: {len(split_photos)} photos, {len(scales)} positive scales, {len(rows)} sign rows")
        out = build_real_negatives(split_photos, scales, rows, seed, plain)
        path = out_dir / f"real_empty_{split.value}.npz"
        _save_atomic(path, out.arrays)
        windows = {s.name.lower(): int((out.arrays["stratum"] == s.value).sum()) for s in Stratum}
        skipped = {s.name.lower(): n for s, n in out.skipped.items()}
        manifest["real"]["files"][f"real_empty_{split.value}"] = {
            "photos": len(split_photos), "windows": windows, "skipped": skipped, "sha256": _sha256(path)}
        print(f"  windows {windows}, skipped {skipped}")

    tmp = out_dir / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out_dir / MANIFEST)
    print(f"wrote {out_dir / MANIFEST}")


def _save_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
