# v2 spec: test on real road photos

Status: **approved 2026-11-09** (layers 1–3: intent, behavior, interfaces). Decisions in §5.

---

## 1. Intent

Every number so far (v1.5b: 97.6 ± 0.2% test joint accuracy over 3 seeds) comes from scenes we built
ourselves: a GTSRB sign crop pasted onto a COCO photo. v2 asks how much of that survives on **real road
photos**, where the sign sits in its real surroundings with real lighting, real blur, and other signs
nearby. The data is the German Traffic Sign Detection Benchmark (GTSDB): real 1360×800 photos with boxes
for the same 43 classes. The class head carries over unchanged.

v2 also asks *why* accuracy drops. It splits the synthetic→real drop into two parts using a middle tier:

| Tier | Sign pixels | Surroundings | What it isolates |
|---|---|---|---|
| synthetic (have it) | GTSRB crop | COCO photo | baseline: 97.6% |
| **hybrid** (new) | GTSDB crop | COCO photo, pasted the same way as training | **sign gap**: do real-photo signs look different? |
| **real** (new) | GTSDB photo | the real road around it | **context gap**: does the real scene confuse it? |

Hybrid and real use the *same* sign instances, so their difference is a paired comparison (McNemar).

**v2 is test-only.** It uses the v1.5b checkpoint as it is (seed 0 is the headline; seeds 1 and 2 give a
spread). Nothing is retrained or tuned on real data.

**Out of scope** (each is a possible later version):
- Full-image detection: finding every sign in a 1360×800 photo, photos with several signs, photos with
  none. The model always points somewhere and has no way to say "no sign", so v2 only scores windows
  that contain exactly one labeled sign.
- Fine-tuning on real photos (v2.1, with its own split).
- Any change to the model, training, augmentation, or the synthetic data.
- Non-German signs, video, speed.

---

## 2. Behavioral spec

### 2.1 Data

- **Source:** GTSDB `FullIJCNN2013` release, unpacked into `data/raw/gtsdb/` (gitignored, like
  GTSRB/COCO; not redistributed). It holds 900 photos `00000.ppm`–`00899.ppm` and `gt.txt` with one row
  per sign: `file;x1;y1;x2;y2;class_id`.
- **Splits:** GTSDB's official split. Photos 0–599 → **real-dev**, 600–899 → **real-test**. Dev can be
  looked at as often as we like (error analysis, figures). Test is scored once per checkpoint, enforced
  by the same guard as synthetic test.
- **Expected sizes** (to confirm on download, see §2.6): about 850 signs in dev and 360 in test. With
  n ≈ 800, a 95% interval on an accuracy near 85% is about ±2.5 points; with n ≈ 360 it's about ±3.7.
  Real-data numbers are much noisier than our 5,000-scene synthetic ones, and every reported rate
  carries its interval.

### 2.2 Real windows (one per labeled sign)

For each labeled sign *t* in a photo:

1. **Scale.** Let `L` be the sign's longer side in photo pixels. Scale the whole photo by
   `s = min(1, sign_max / L)` (`sign_max` = 48, the training maximum). Signs that already fit keep their
   real pixels (`s = 1`). Bigger signs shrink to 48 px. Signs are never enlarged. Signs under the
   training minimum (16 px) stay as they are and land in the "small" bucket.
2. **Place.** Choose a random 128×128 window in the scaled photo that contains the whole target box. The
   sign is not centered, matching training, where placement is random too.
3. **Exclude other signs.** The window must not overlap any other labeled sign's box, even partly.
   Try up to `PLACEMENT_TRIES` (= 50) random placements from the seeded RNG and keep the first valid one.
   If none is valid, skip the sign and record the reason `OTHER_SIGN_IN_VIEW`. Skips are counted and
   reported, never silently dropped.
4. **Targets.** Box = the target box in window pixels (x2/y2 exclusive). Heatmap = the same Gaussian as
   synthetic scenes, built by the same function. No blur, noise or brightness change is added: real
   photos bring their own.

Unlabeled signs (types outside the 43 classes, e.g. direction boards) can still appear in a window.
That is part of what "real" means, and the model may point at them. This is expected and is not
excluded.

### 2.3 Hybrid scenes

For every sign that got a real window (and only those, so the two files pair one to one): crop the
GTSDB box from the photo and pass it to the existing `make_scene` with the **val** synthesis config
(no training augmentation). Backgrounds: val-split COCO backgrounds for dev, test-split backgrounds for
test. Same seed handling as `make_dataset.py`.

### 2.4 Files

`data/processed/{real_dev,real_test,hybrid_dev,hybrid_test}.npz`, in the same format `SignScenes`
already reads (`images`, `class_ids`, `boxes`, `heatmaps`), plus metadata arrays that `SignScenes`
ignores:

- `photo_index` int16 [N]
- `sign_index` int16 [N] (row order within the photo)
- `native_long_side` int16 [N] (`L`, before scaling)

Scene *i* in `real_X` and scene *i* in `hybrid_X` are the same physical sign.

A sidecar `data/processed/real_manifest.json` records counts per split (signs, windows, skips by
reason), the seed, the `gt.txt` SHA-256, and the synthesis config used.

### 2.5 Scoring and reporting

- `scripts/evaluate.py --split {real_dev,real_test,hybrid_dev,hybrid_test}` works exactly as today.
  The "scored once" guard covers every split whose name ends in `test`.
- Every result JSON (all splits, old and new) also stores per-scene outcomes (`correct`, `hit`), so the
  breakdowns below need no second model run. In particular, test is never re-scored to get a breakdown.
- `scripts/report_real.py --run <run_id> --split {dev,test}` reads those JSONs and the npz metadata, and
  writes `outputs/eval/<run>/real_<split>_report.json`. It prints:
  - joint, class and pointing accuracy for real and hybrid, each with a Wilson 95% interval;
  - real vs hybrid exact McNemar p-value (paired on the same signs);
  - joint accuracy by native size: **small** `L < 24`, **medium** `24 ≤ L < 48`, **large** `L ≥ 48`;
  - per-class accuracy for classes with n ≥ 10;
  - skip counts.
- Results go into RESULTS.md as **v2** and into a README row "v2 real photos (GTSDB)".

### 2.6 Sanity check: did GTSRB's training set contain these physical signs?

GTSRB and GTSDB come from the same research group. **I don't know** whether they share recordings. If
a GTSDB sign is the same physical sign as a GTSRB training track, the model may have memorized it
(v1.5b measured a ~0.8-point gain from seeing a sign during training), which would inflate the real
number. GTSDB has no track IDs, so this can't be checked exactly.

`scripts/check_gtsdb_overlap.py` does a near-duplicate search: each GTSDB crop against every GTSRB
*training-split* crop of the same class, compared as 32×32 grayscale, mean-subtracted, by cosine
similarity. It writes a contact sheet of the 40 most similar pairs for a person to judge. We report how
many pairs look like the same physical sign, and if there are any, also report accuracy with those
signs removed. This flags likely overlap. It does not prove there is none.

### 2.7 Predictions (to commit to RESULTS.md before any real or hybrid scene is scored)

These are proposals to review. They get frozen in the commit that comes before the first scoring.

| # | Prediction | Measured on |
|---|---|---|
| P1 | Real-dev joint is between 75% and 92% (point guess 85%) | real_dev, seed 0 |
| P2 | Hybrid-dev joint is ≥ 90%: real sign pixels alone cost under 8 points | hybrid_dev, seed 0 |
| P3 | Real < hybrid on the same signs, McNemar p < 0.05: the real surroundings cost something on top | paired dev |
| P4 | Real pointing drops below 96% (synthetic: 99.8%), because the peak sometimes lands on other objects or unlabeled signs | real_dev |
| P5 | Small signs (`L < 24`) score ≥ 10 points lower in joint than large ones (`L ≥ 48`) | real_dev |
| P6 | Real-test joint falls inside real-dev's 95% interval | real_test, once |
| P7 | All three seeds' real-test joint lies within ±3 points of each other | real_test |

### 2.8 Acceptance criteria

Data:
- A1. `read_gtsdb` returns 900 photos. The number of signs equals the number of `gt.txt` rows, every
  class ID is in [0, 43), and every box has positive area and lies inside its 1360×800 photo.
  Otherwise it raises `ValueError` naming the line.
- A2. Dev and test are disjoint and together cover all 900 photos.
- A3. Running `make_real_dataset.py` twice gives byte-identical npz files (compared by SHA-256).
- A4. `real_X` and `hybrid_X` have equal N and identical `photo_index`, `sign_index` and `class_ids`.
  In the manifest, windows + skips = signs for each split.

Windows (property tests over many seeds and synthetic box layouts):
- A5. The target box lies fully inside the window, and the window lies fully inside the scaled photo.
- A6. No other box intersects the window. If no valid window exists, the result is `Skip`, never an
  error and never an overlapping window.
- A7. `s ≤ 1`. `s == 1` exactly when `L ≤ sign_max`. After scaling, the target's longer side is
  ≤ `sign_max` (allowing for rounding).
- A8. The heatmap's argmax cell contains the box center, the same check as the synthetic scenes.
- A9. The same RNG seed gives the same window.

Evaluation:
- A10. Re-scoring v1.5b seed 0 on synthetic **val** after the change gives the same joint (0.9846) and
  the same per-class numbers as `results/v1.5b-fresh-aug/best_val.json`. Adding `hit` changes no
  existing number.
- A11. `evaluate.py --split real_test` refuses a second run without `--force`.
- A12. `wilson(0, 10)` upper bound ≈ 0.278. `mcnemar_exact(41, 34)` ≈ 0.49 (the v1.5b A→B value).
- A13. `make_scene` with no augmentation still matches `PRE_AUGMENT_DIGEST` after the heatmap helper
  is extracted.

### 2.9 Failure modes

| Situation | Behavior |
|---|---|
| `data/raw/gtsdb/` or `gt.txt` missing | `FileNotFoundError` naming the expected path and where to download |
| Malformed `gt.txt` row, class out of range, box outside the photo | `ValueError` with file and line |
| A sign has no valid window | Recorded as a skip with its reason; build continues |
| A split ends up with zero windows | `ValueError` (something is wrong with the data or the rules) |
| A photo fails to decode | `OSError` from PIL, build stops: no partial npz (written atomically) |

---

## 3. Interfaces

No new dependencies: PIL reads `.ppm`, and the statistics are a few lines of math.

### `src/findobject/data/gtsdb.py`: reads the GTSDB annotation file

```python
DEV_PHOTOS = 600  # official split: photos 0-599 -> dev, 600-899 -> test

@dataclass(frozen=True)
class RealSign:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in photo px; x2/y2 exclusive (gt.txt is inclusive)
    class_id: int

@dataclass(frozen=True)
class RealPhoto:
    path: Path
    index: int                      # 0-899, from the file name
    signs: tuple[RealSign, ...]     # may be empty

class RealSplit(Enum):
    DEV = "dev"
    TEST = "test"

def read_gtsdb(root: Path) -> list[RealPhoto]: ...
def photos_in(photos: Sequence[RealPhoto], split: RealSplit) -> list[RealPhoto]: ...
def load_sign_crop(photo: Image.Image, sign: RealSign) -> Image.Image: ...  # for hybrid scenes
```

### `src/findobject/data/real_windows.py`: cuts one-sign windows out of real photos (pure)

```python
PLACEMENT_TRIES = 50

class Skip(Enum):
    OTHER_SIGN_IN_VIEW = "other_sign_in_view"

@dataclass(frozen=True)
class WindowPlan:
    scale: float                    # applied to the whole photo
    x0: int                         # window top-left, in scaled-photo px
    y0: int
    box: tuple[int, int, int, int]  # target sign in window px, x2/y2 exclusive

def plan_window(photo_size: tuple[int, int], boxes: Sequence[tuple[int, int, int, int]], target: int,
                rng: np.random.Generator, cfg: SynthConfig) -> WindowPlan | Skip: ...
def cut_window(photo: Image.Image, plan: WindowPlan, cfg: SynthConfig) -> Scene: ...
```

`cfg` supplies `scene_size`, `sign_max`, `heatmap_size` and `heatmap_sigma`, so real windows use exactly
the training geometry.

### `src/findobject/data/synth.py`: extract one helper (separate commit, no behavior change, guarded by A13)

```python
def box_heatmap(box: tuple[int, int, int, int], cfg: SynthConfig) -> np.ndarray: ...
# make_scene and cut_window both call it, so the heatmap formula lives in one place.
```

### `src/findobject/eval/evaluate.py`: additive field

```python
class EvalResult(NamedTuple):
    ...                             # existing fields unchanged
    hit: Tensor                     # bool [N], cpu: heatmap peak inside the true box
```

### `src/findobject/eval/stats.py`: interval and paired test (pure)

```python
@dataclass(frozen=True)
class Rate:
    k: int
    n: int
    lo: float                       # Wilson 95% interval
    hi: float
    @property
    def value(self) -> float: ...

def wilson(k: int, n: int) -> Rate: ...                          # ValueError if n == 0 or k not in [0, n]
def mcnemar_exact(only_a: int, only_b: int) -> float: ...       # two-sided p on the discordant counts
```

### `src/findobject/eval/real_report.py`: groups per-scene outcomes into the v2 report (pure)

```python
class SizeBucket(Enum):
    SMALL = "small"                 # L < 24
    MEDIUM = "medium"               # 24 <= L < 48
    LARGE = "large"                 # L >= 48

def size_bucket(native_long_side: int) -> SizeBucket: ...
def rates_by(ok: np.ndarray, keys: Sequence[Hashable], min_n: int = 1) -> dict[Hashable, Rate]: ...
```

### Scripts (I/O shell)

| Script | Does |
|---|---|
| `scripts/make_real_dataset.py --config configs/peak_fresh_aug.yaml --seed 0` | builds the 4 npz files and the manifest |
| `scripts/evaluate.py` | adds the 4 split names, stores per-scene `correct`/`hit`, guards `*test` |
| `scripts/report_real.py --run <id> --split dev\|test` | prints and writes the v2 report |
| `scripts/check_gtsdb_overlap.py` | near-duplicate contact sheet (§2.6) |

---

## 4. Order of work (after approval)

Each step is its own commit; tests are written against §2.8.

1. `box_heatmap` extraction (A13).
2. `gtsdb.py` + tests (A1, A2), plus README download instructions.
3. `real_windows.py` + tests (A5–A9).
4. `make_real_dataset.py` (A3, A4); build the files; run the overlap check (§2.6).
5. `stats.py`, `hit` in `EvalResult`, `evaluate.py` changes (A10–A12).
6. `real_report.py` + `report_real.py`.
7. **Commit predictions (§2.7) to RESULTS.md.** Then score dev (real + hybrid), write the analysis,
   score test once per seed, record v2.

Blast radius: everything new is additive. Existing synthetic data, checkpoints and results are only
read. The two edits to existing code (`box_heatmap`, `hit`) are covered by A13 and A10.

## 5. Decisions (review, 2026-11-09)

1. **Sizing rule: native.** Signs keep their real size when it fits (`s = min(1, sign_max / L)`). The
   hybrid tier covers the training-like uniform 16–48 case.
2. **Overlap check (§2.6): included.**
3. **Predictions P1–P7: unchanged.** Frozen as written in §2.7.
