"""CLI: find GTSDB signs that are the same physical sign as a GTSRB training track (docs/v2-spec.md §2.6).

Usage: python scripts/check_gtsdb_overlap.py --config configs/peak_fresh_aug.yaml

Two tests per GTSDB sign:

1. Exact frame (primary, certain). GTSRB images are unresized crops of video frames, so if a GTSRB training
   image is the same frame as a GTSDB photo it appears pixel-for-pixel in that photo at the sign's
   location. Every same-class training image whose ROI is within SIZE_TOL px of the GTSDB box size is
   slid over +-SHIFT px around the box; mean absolute difference (0-255) below EXACT_MAD = same frame.
   On the real data the gap is clean: 297 of 1,213 signs at exactly 0, the next-lowest above 1.
2. Similarity (secondary, "possible"). Catches the same sign in a *different* frame, which test 1 can't.
   Same-class signs are designed to look alike, so raw similarity can't tell "same physical sign" from "same
sign type". The check therefore calibrates on a case with a known answer: GTSRB val tracks are, by
construction, different physical signs from the train tracks. For each val crop we take its best cosine
similarity to any same-class train crop; the 99th percentile of those is the threshold. A GTSDB sign whose
best match beats it is flagged. Crops include ~10% margin (GTSRB images have it built in; GTSDB boxes are
widened by 10% per side), since the surroundings are what make a true duplicate stand out.

Writes outputs/eval/gtsdb_overlap/overlap.json (summary + one record per GTSDB sign: exact_frame, mad,
best_sim, sim_flag, matched GTSRB image) and top_pairs.png (the 40 most similar pairs, for a person to
judge). A sign with neither flag can still be the same physical sign as a training track: this narrows
the overlap, it does not prove the rest clean. Raises FileNotFoundError if raw data is missing.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from findobject.data.gtsdb import RealSplit, load_photo, read_gtsdb  # noqa: E402
from findobject.data.gtsrb import NUM_CLASSES, SignSample  # noqa: E402
from findobject.data.splits import split_sources  # noqa: E402

GTSDB_DIR = Path("data/raw/gtsdb/FullIJCNN2013")
OUT_DIR = Path("outputs/eval/gtsdb_overlap")
SIDE = 24  # compare as SIDE x SIDE RGB
MARGIN = 0.10
NULL_PERCENTILE = 99.0
N_NULL = 3000  # val crops used for the null distribution
TOP_SHOWN = 40
EXACT_MAD = 0.5  # below this mean |difference| (0-255), the GTSRB image is the same frame
SIZE_TOL = 5  # px: candidate GTSRB ROIs within this of the GTSDB box's width and height
SHIFT = 4  # px: alignment search around the box position


def embed(img: Image.Image) -> np.ndarray:
    """SIDE x SIDE RGB, each channel standardized, flattened to unit length."""
    a = np.asarray(img.convert("RGB").resize((SIDE, SIDE), Image.BILINEAR), np.float32)
    a = (a - a.mean(axis=(0, 1))) / (a.std(axis=(0, 1)) + 1e-6)
    v = a.ravel()
    return v / (np.linalg.norm(v) + 1e-6)


def gtsrb_vectors(signs: list[SignSample]) -> np.ndarray:
    out = np.empty((len(signs), SIDE * SIDE * 3), np.float32)
    for i, s in enumerate(signs):
        with Image.open(s.path) as img:  # whole GTSRB image = ROI plus its built-in margin
            out[i] = embed(img)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(signs)}")
    return out


def best_match(q: np.ndarray, q_cls: np.ndarray, ref: np.ndarray, ref_cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the highest cosine similarity to a same-class reference and that reference's index."""
    sim = np.full(len(q), -1.0, np.float32)
    idx = np.full(len(q), -1, np.int64)
    for k in range(NUM_CLASSES):
        qi, ri = np.flatnonzero(q_cls == k), np.flatnonzero(ref_cls == k)
        if len(qi) and len(ri):
            s = q[qi] @ ref[ri].T
            sim[qi], idx[qi] = s.max(axis=1), ri[s.argmax(axis=1)]
    return sim, idx


def exact_frame_mad(photo: np.ndarray, box: tuple[int, int, int, int], candidates: list[SignSample]) -> tuple[float, int]:
    """Smallest mean |difference| of any candidate GTSRB image placed in `photo` (float32 HxWx3) so its ROI
    lands within SHIFT px of `box`, and that candidate's index (inf, -1 if none fits)."""
    best, best_i = float("inf"), -1
    h, w = photo.shape[:2]
    for i, s in enumerate(candidates):
        with Image.open(s.path) as g:
            G = np.asarray(g.convert("RGB"), np.float32)
        gh, gw = G.shape[:2]
        for dy in range(-SHIFT, SHIFT + 1):
            for dx in range(-SHIFT, SHIFT + 1):
                ox, oy = box[0] - s.roi[0] + dx, box[1] - s.roi[1] + dy
                if 0 <= ox and 0 <= oy and ox + gw <= w and oy + gh <= h:
                    mad = float(np.abs(photo[oy:oy + gh, ox:ox + gw] - G).mean())
                    if mad < best:
                        best, best_i = mad, i
    return best, best_i


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/peak_fresh_aug.yaml"))
    args = parser.parse_args()
    data_cfg = yaml.safe_load(args.config.read_text())["data"]
    src = split_sources(data_cfg)
    train, val = src["train"].signs, src["val"].signs
    rng = np.random.default_rng(0)
    val = [val[i] for i in sorted(rng.choice(len(val), min(N_NULL, len(val)), replace=False))]

    print(f"embedding {len(train)} GTSRB train crops")
    train_v, train_c = gtsrb_vectors(train), np.array([s.class_id for s in train])
    print(f"embedding {len(val)} GTSRB val crops (null: known different physical signs)")
    val_v, val_c = gtsrb_vectors(val), np.array([s.class_id for s in val])
    null_sim, _ = best_match(val_v, val_c, train_v, train_c)
    threshold = float(np.percentile(null_sim, NULL_PERCENTILE))

    by_class: dict[int, list[SignSample]] = {k: [] for k in range(NUM_CLASSES)}
    for t in train:
        by_class[t.class_id].append(t)
    rows, vecs, crops = [], [], []
    for photo in read_gtsdb(GTSDB_DIR):
        if not photo.signs:
            continue
        img = load_photo(photo)
        for j, sign in enumerate(photo.signs):
            x1, y1, x2, y2 = sign.box
            mx, my = round(MARGIN * (x2 - x1)), round(MARGIN * (y2 - y1))
            crop = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))
            split = RealSplit.DEV.value if photo.index < 600 else RealSplit.TEST.value
            bw, bh = x2 - x1, y2 - y1
            cands = [t for t in by_class[sign.class_id]
                     if abs(t.roi[2] + 1 - t.roi[0] - bw) <= SIZE_TOL and abs(t.roi[3] + 1 - t.roi[1] - bh) <= SIZE_TOL]
            mad, ci = exact_frame_mad(np.asarray(img, np.float32), sign.box, cands)
            rows.append({"photo_index": photo.index, "sign_index": j, "class_id": sign.class_id, "split": split,
                         "exact_frame": mad < EXACT_MAD, "mad": mad if np.isfinite(mad) else None,
                         "exact_gtsrb": str(cands[ci].path) if ci >= 0 and mad < EXACT_MAD else None,
                         "n_candidates": len(cands)})
            vecs.append(embed(crop))
            crops.append(crop)
    q_sim, q_idx = best_match(np.stack(vecs), np.array([r["class_id"] for r in rows]), train_v, train_c)
    for r, s, i in zip(rows, q_sim, q_idx):
        r["best_sim"], r["best_gtsrb"] = float(s), str(train[i].path) if i >= 0 else None
        r["sim_flag"] = bool(s > threshold)
    flagged = [r for r in rows if r["sim_flag"]]
    exact = [r for r in rows if r["exact_frame"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "threshold": threshold,
        "null": {"percentile": NULL_PERCENTILE, "n": len(null_sim), "median": float(np.median(null_sim)),
                 "max": float(null_sim.max())},
        "gtsdb": {"n": len(rows), "median_best_sim": float(np.median(q_sim)), "n_sim_flagged": len(flagged),
                  "n_exact_frame": len(exact), "n_either": sum(r["sim_flag"] or r["exact_frame"] for r in rows),
                  "n_exact_by_split": {sp: sum(r["split"] == sp for r in exact) for sp in ("dev", "test")}},
        "exact_mad": EXACT_MAD,
        "expected_sim_flag_rate_if_no_overlap": 1 - NULL_PERCENTILE / 100,
        "signs": rows,
    }
    (OUT_DIR / "overlap.json").write_text(json.dumps(summary, indent=2))

    order = np.argsort(-q_sim)[:TOP_SHOWN]
    fig, ax = plt.subplots(4, 20, figsize=(20, 4.6))
    for n, i in enumerate(order):
        a, b = ax[(n // 20) * 2, n % 20], ax[(n // 20) * 2 + 1, n % 20]
        a.imshow(crops[i])
        with Image.open(rows[i]["best_gtsrb"]) as g:
            b.imshow(g.convert("RGB"))
        a.set_title(f"{q_sim[i]:.2f}{' *' if q_sim[i] > threshold else ''}", fontsize=7)
        a.axis("off"), b.axis("off")
    fig.suptitle(f"top {TOP_SHOWN}: GTSDB (upper) vs best same-class GTSRB train (lower); * = above null p{NULL_PERCENTILE:.0f} "
                 f"({threshold:.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_pairs.png", dpi=110)
    print(f"null (val vs train, n={len(null_sim)}): median {np.median(null_sim):.3f}, p{NULL_PERCENTILE:.0f} {threshold:.3f}")
    print(f"GTSDB (n={len(rows)}): median best {np.median(q_sim):.3f}, flagged {len(flagged)} "
          f"({100 * len(flagged) / len(rows):.1f}%; {100 - NULL_PERCENTILE:.0f}% expected with no overlap)")
    print(f"exact-frame duplicates of a GTSRB training image: {len(exact)} "
          f"(dev {summary['gtsdb']['n_exact_by_split']['dev']}, test {summary['gtsdb']['n_exact_by_split']['test']}); "
          f"exact or similarity-flagged: {summary['gtsdb']['n_either']}")
    print(f"wrote {OUT_DIR}/overlap.json and top_pairs.png")


if __name__ == "__main__":
    main()
