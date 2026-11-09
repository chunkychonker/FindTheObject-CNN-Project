"""CLI: score a checkpoint's presence on one family of files and record the v3a report (docs/v3a-spec.md §2.6).

Usage:
    python scripts/evaluate_presence.py --checkpoint outputs/checkpoints/<run>/best.pt --split val   # first: chooses τ
    python scripts/evaluate_presence.py --checkpoint ... --split real_dev [--p6-sheet]
    python scripts/evaluate_presence.py --checkpoint ... --split test      # once per run; also real_test

  split      positives  negatives        also
  val        val        val_empty        val_decoy_plain, val_decoy_aug; chooses τ
  test       test       test_empty
  real_dev   real_dev   real_empty_dev   hybrid_dev (paired); --p6-sheet writes the P6 sheet and key
  real_test  real_test  real_empty_test  hybrid_test (paired)

Writes outputs/eval/<run>/presence_<split>.json: τ, the report, per-scene presence (and correct / hit /
y_pred for positives; photo_index, and stratum for negatives, on real splits), and the SHA-256 of the
checkpoint and of every data file. Only best.pt is accepted, since the file name holds no checkpoint name.

τ is chosen only on val (1% false alarms on val_empty) and stored with the checkpoint's SHA-256; every other
split reads it from presence_val.json and fails before scoring if that file is missing or belongs to a
different checkpoint. Test guard (eval/test_guard.py): exits with code 2 if the test family was already
scored for this run, by either script, unless --force. Raises FileNotFoundError for a missing checkpoint,
data or overlap file; ValueError for misaligned real/hybrid files or a window with no overlap record.
The JSON, sheet and key are each written atomically.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from findobject.data.dataset import EmptyScenes, SignScenes
from findobject.data.real_negatives import Stratum
from findobject.eval.p6_sheet import choose_tiles, render_sheet
from findobject.eval.presence import threshold_at_false_alarm
from findobject.eval.presence_report import (PhotoIntervals, SceneIntervals, Side, family_metrics, matched_found,
                                             seam_check, wilson_rate)
from findobject.eval.score import SceneScores, score_scenes
from findobject.eval.test_guard import scored_results
from findobject.training.checkpoint import load_checkpoint_info, load_model
from findobject.utils.device import pick_device
from findobject.utils.files import sha256_file, write_json_atomic

EVAL_ROOT = Path("outputs/eval")
OVERLAP = Path("results/v2-gtsdb-overlap/overlap.json")
BATCH_SIZE = 256
TAU_RATE = 0.01  # §2.5: τ admits at most 1% of val_empty
P6_SEED = 6
CHECKPOINT_STEM = "best"


class Family(NamedTuple):
    positives: str  # data file stems in the checkpoint's processed_dir
    negatives: str
    hybrid: str | None  # real splits: the same signs pasted onto the same photos, row for row
    decoys: tuple[str, ...]


FAMILIES = {
    "val": Family("val", "val_empty", None, ("val_decoy_plain", "val_decoy_aug")),
    "test": Family("test", "test_empty", None, ()),
    "real_dev": Family("real_dev", "real_empty_dev", "hybrid_dev", ()),
    "real_test": Family("real_test", "real_empty_test", "hybrid_test", ()),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=FAMILIES, required=True)
    parser.add_argument("--force", action="store_true", help="re-score a test family that was already scored")
    parser.add_argument("--p6-sheet", action="store_true", help="real_dev only: write the P6 contact sheet and key")
    parser.add_argument("--overlap", type=Path, default=OVERLAP)
    args = parser.parse_args(argv)
    if args.checkpoint.stem != CHECKPOINT_STEM:
        parser.error(f"only {CHECKPOINT_STEM}.pt is scored (presence_<split>.json names no checkpoint), got {args.checkpoint}")
    if args.p6_sheet and args.split != "real_dev":
        parser.error("--p6-sheet is only for --split real_dev")

    run_dir = EVAL_ROOT / args.checkpoint.parent.name
    out_path = run_dir / f"presence_{args.split}.json"
    refuse_test_rerun(run_dir, args.split, args.force)
    ckpt_sha = sha256_file(args.checkpoint)
    tau = None if args.split == "val" else read_tau(run_dir, ckpt_sha)  # fail before the slow part

    info = load_checkpoint_info(args.checkpoint)
    data_dir = Path(info.config["data"]["processed_dir"])
    family = FAMILIES[args.split]
    device = pick_device("auto")
    model = load_model(args.checkpoint, device)
    print(f"scoring {args.checkpoint} (epoch {info.epoch}) on the {args.split} family, {device}")

    def score(stem: str, dataset_cls: type[Dataset]) -> SceneScores:
        s = score_scenes(model, DataLoader(dataset_cls(data_dir / f"{stem}.npz"), batch_size=BATCH_SIZE), device)
        print(f"  {stem}: {len(s.presence)} scenes")
        return s

    pos = score(family.positives, SignScenes)
    neg = score(family.negatives, EmptyScenes)
    hybrid = score(family.hybrid, SignScenes) if family.hybrid else None
    decoys = {stem: score(stem, EmptyScenes).presence for stem in family.decoys}

    tau_record = None
    if tau is None:
        tau = threshold_at_false_alarm(neg.presence, TAU_RATE)
        tau_record = {"negatives": family.negatives, "rate": TAU_RATE, "n": len(neg.presence),
                      "k": math.floor(Fraction(TAU_RATE).limit_denominator(10**9) * len(neg.presence))}

    stems = [family.positives, family.negatives, *([family.hybrid] if family.hybrid else []), *family.decoys]
    data = {stem: {"file": str(data_dir / f"{stem}.npz"), "sha256": sha256_file(data_dir / f"{stem}.npz")} for stem in stems}
    per_scene = {"positives": _scene_record(pos), "negatives": {"presence": neg.presence.tolist()}}
    if hybrid is not None:
        meta = real_metadata(data_dir, family, args.overlap)
        per_scene["positives"]["photo_index"] = meta.pos_photo.tolist()
        per_scene["hybrid"] = _scene_record(hybrid)
        per_scene["negatives"] |= {"photo_index": meta.neg_photo.tolist(), "stratum": meta.stratum.tolist()}
        report = real_report(pos, neg, hybrid, meta, tau)
    else:
        report = synthetic_report(pos, neg, decoys, tau)
    per_scene |= {stem: {"presence": p.tolist()} for stem, p in decoys.items()}

    write_json_atomic(out_path, {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": info.epoch,
        "run_id": args.checkpoint.parent.name,
        "tau": float(tau),  # float32 -> float64 is exact; np.float32(this) gives τ back bit for bit
        "tau_chosen": tau_record,  # null: read from presence_val.json
        "data": data,
        "report": report,
        "per_scene": per_scene,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print_summary(args.split, tau, report)
    print(f"wrote {out_path}")

    if args.p6_sheet:
        write_p6(run_dir, data_dir / f"{family.negatives}.npz", neg.presence, per_scene["negatives"], ckpt_sha,
                 data[family.negatives]["sha256"])


def refuse_test_rerun(run_dir: Path, split: str, force: bool) -> None:
    """Exit with code 2 if this split's test family was already scored for the run (by either script) and
    force is not set. Non-test splits always pass."""
    existing = scored_results(run_dir, split)
    if existing and not force:
        print(f"refusing: test already scored for this run: {', '.join(map(str, existing))}\n"
              f"the test set is scored once per checkpoint; pass --force only if you mean to re-score it",
              file=sys.stderr)
        raise SystemExit(2)


def read_tau(run_dir: Path, checkpoint_sha256: str) -> np.float32:
    """τ from run_dir/presence_val.json. Raises FileNotFoundError if that file is missing, ValueError if it
    was written for a different checkpoint (τ must come from this checkpoint's own val run)."""
    path = run_dir / "presence_val.json"
    if not path.exists():
        raise FileNotFoundError(f"no val presence result at {path}: τ is chosen on val only, so score "
                                f"--split val with this checkpoint first")
    val = json.loads(path.read_text())
    if val["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"{path} was scored with a different checkpoint (SHA-256 {val['checkpoint_sha256'][:12]}… "
                         f"vs {checkpoint_sha256[:12]}…): re-score --split val first")
    return np.float32(val["tau"])


class RealMetadata(NamedTuple):
    pos_photo: np.ndarray  # photo of each positive (= hybrid) window
    neg_photo: np.ndarray  # photo of each negative window
    stratum: np.ndarray  # Stratum value of each negative window
    unseen: np.ndarray  # bool per positive: no exact-frame match in GTSRB train (v2's overlap check)


def real_metadata(data_dir: Path, family: Family, overlap: Path) -> RealMetadata:
    """Raises ValueError if the real and hybrid files' (photo, sign) rows differ, or a positive window has
    no overlap record."""
    keys = {}
    for stem in (family.positives, family.hybrid):
        with np.load(data_dir / f"{stem}.npz") as d:
            keys[stem] = d["photo_index"].astype(np.int64), d["sign_index"].astype(np.int64)
    (photo, sign), (h_photo, h_sign) = keys[family.positives], keys[family.hybrid]
    if not (np.array_equal(photo, h_photo) and np.array_equal(sign, h_sign)):
        raise ValueError(f"{family.positives} and {family.hybrid} rows are not the same signs in the same order")
    with np.load(data_dir / f"{family.negatives}.npz") as d:
        neg_photo, stratum = d["photo_index"].astype(np.int64), d["stratum"].astype(np.int64)
    flags = {(r["photo_index"], r["sign_index"]): r["exact_frame"] for r in json.loads(overlap.read_text())["signs"]}
    missing = [k for k in zip(photo.tolist(), sign.tolist()) if k not in flags]
    if missing:
        raise ValueError(f"{len(missing)} windows have no overlap record, e.g. {missing[0]}")
    unseen = ~np.array([flags[k] for k in zip(photo.tolist(), sign.tolist())])
    return RealMetadata(photo, neg_photo, stratum, unseen)


def synthetic_report(pos: SceneScores, neg: SceneScores, decoys: dict[str, np.ndarray], tau: np.float32) -> dict:
    """Wilson intervals for rates, the ordinary bootstrap for AUROC; decoy false alarms at τ."""
    intervals = SceneIntervals(len(pos.presence), len(neg.presence))
    return {
        "all": family_metrics(pos.presence, pos.y_true == pos.y_pred, pos.hit, neg.presence, tau, intervals),
        "decoy_false_alarm": {stem: wilson_rate(p >= tau) for stem, p in decoys.items()},
    }


def real_report(pos: SceneScores, neg: SceneScores, hybrid: SceneScores, meta: RealMetadata, tau: np.float32) -> dict:
    """Photo-clustered intervals throughout. Subsets restrict the positives; strata the negatives."""
    intervals = PhotoIntervals(meta.pos_photo, meta.neg_photo)
    correct = pos.y_true == pos.y_pred
    by_stratum = {}
    for s in Stratum:
        mask = meta.stratum == s.value
        by_stratum[s.name.lower()] = {"auroc": intervals.auroc(pos.presence, neg.presence, neg_within=mask),
                                      "false_alarm": intervals.rate(neg.presence >= tau, Side.NEG, mask)}
    return {
        "subsets": {name: family_metrics(pos.presence, correct, pos.hit, neg.presence, tau, intervals, within)
                    for name, within in (("all", None), ("unseen", meta.unseen))},
        "strata": by_stratum,
        "matched": matched_found(pos.presence, correct & pos.hit, neg.presence),
        "hybrid": family_metrics(hybrid.presence, hybrid.y_true == hybrid.y_pred, hybrid.hit, neg.presence, tau,
                                 intervals),
        "seam": seam_check(pos.presence, hybrid.presence, tau),
    }


def write_p6(run_dir: Path, neg_file: Path, presence: np.ndarray, neg_record: dict, ckpt_sha: str,
             data_sha: str) -> None:
    """The sheet holds no scores; the key holds everything else and is not opened before labels are committed."""
    tiles = choose_tiles(presence, np.random.default_rng(P6_SEED))
    with np.load(neg_file) as d:
        images = d["images"]
    sheet_path, key_path = run_dir / "p6_sheet.png", run_dir / "p6_key.json"
    tmp = sheet_path.with_name("p6_sheet.tmp.png")
    render_sheet(images, tiles).save(tmp)
    tmp.replace(sheet_path)
    write_json_atomic(key_path, {
        "sheet_sha256": sha256_file(sheet_path), "seed": P6_SEED, "checkpoint_sha256": ckpt_sha,
        "negatives_file": str(neg_file), "negatives_sha256": data_sha,
        "tiles": [{"number": t.number, "row": t.row, "top": t.top, "presence": float(presence[t.row]),
                   "photo_index": neg_record["photo_index"][t.row], "stratum": Stratum(neg_record["stratum"][t.row]).name}
                  for t in tiles],
    })
    print(f"wrote {sheet_path} and {key_path} (do not open the key before the labels are committed)")


def print_summary(split: str, tau: np.float32, report: dict) -> None:
    fam = report["all"] if "all" in report else report["subsets"]["all"]
    print(f"\n{split}: τ {float(tau):.6g} | AUROC {fam['auroc']['value']:.4f} "
          f"[{fam['auroc']['lo']:.4f}, {fam['auroc']['hi']:.4f}] | false alarm {_pct(fam['false_alarm'])} "
          f"| found {_pct(fam['found'])} | joint {_pct(fam['joint'])}")
    for stem, r in report.get("decoy_false_alarm", {}).items():
        print(f"  {stem} false alarm {_pct(r)}")
    if "seam" in report:
        s = report["seam"]
        print(f"  said yes: real {s['real_yes']['value']:.2%}, hybrid {s['hybrid_yes']['value']:.2%} "
              f"(gap {s['gap_points']:+.2f} pts, McNemar p {s['mcnemar_p']:.3g})")
        print(f"  unseen: AUROC {report['subsets']['unseen']['auroc']['value']:.4f}, "
              f"found {_pct(report['subsets']['unseen']['found'])}")


def _pct(r: dict) -> str:
    return f"{r['value']:.2%} ({r['k']}/{r['n']}) [{r['lo']:.2%}, {r['hi']:.2%}]"


def _scene_record(s: SceneScores) -> dict:
    return {"presence": s.presence.tolist(), "correct": (s.y_true == s.y_pred).astype(int).tolist(),
            "hit": s.hit.astype(int).tolist(), "y_pred": s.y_pred.tolist()}


if __name__ == "__main__":
    main()
