# v3b spec: find every sign in a whole road photo

Status: **draft 1, for review.** Nothing below is implemented. Predictions (§2.6) are frozen in RESULTS.md
as the **first** step after approval, before any code.

## 1. Intent

Every version so far has answered a question about one **128×128 window** holding at most one sign. The
real task is different. A GTSDB photo is 1360×800 and may hold 0 to 6 signs of very different sizes, and
the model has to say where each one is and what it is. v3b asks: **with no retraining, how well does the
v3a.1 model (`peak_decoy`, seeds 0–2) find signs in whole photos, and what limits it?**

Whole photos differ from windows in four ways. Each one is measured below; none of them has been measured
before:

1. **Several signs per photo.** Dev photos 0–599 hold 851 labeled signs: 94 photos have none, 259 have
   one, 178 have two, and 69 have three to six. The model reads a class only at its single argmax, so v3b reads a class
   at **every** peak.
2. **Stacked signs.** 234 of the 851 dev signs are *stacked*: another labeled sign's center is within
   1.5× their long side, usually because two plates hang on one post. v2's window cutter skipped any window
   with another sign in view, so **206 of these 234 have never been scored.** All 206 dev signs missing
   from real_dev are stacked.
3. **Sign size.** Dev signs are 17–129 px on their long side (median 39). Training signs are 16–48 px.
   Windows shrank each sign to ≤ 48 px before scoring it. A whole photo has no single right scale, so v3b
   runs the model on an **image pyramid**: the photo at scales 1, ½ and ¼. Each of these is divisible by
   the stride (1360×800, 680×400, 340×200), and together they put every sign from 16 to 192 px into the
   16–48 px training range at some scale.
4. **Many more chances to fire.** At those three scales a photo has 89,250 heatmap cells. That is as many
   as 87 windows. τ was chosen so that 1% of *windows* fire, so a photo has many chances to set off a false
   alarm at that τ. v3b counts **false positives per photo (FPPI)** and chooses its threshold on dev photos.
   v3a deferred that choice to here (v3a spec §5.5).

**The decision this informs.** v3a §6 left a choice for v3b: train on scenes with several signs, or read a
class at every peak. v3b tries the cheap option, reading at every peak, and measures where it breaks:

| Result | Reading | Next step |
|---|---|---|
| Stacked signs are found as often as isolated ones (W2 fails) | Reading at every peak is enough | No multi-sign training needed |
| Stacked signs are clearly harder (W2 holds) | The one-sign training is a limit | Multi-sign training scenes (v3c) |
| Recall at the headline operating point is low (W5 fails) | Real signs score too low against real clutter. This is the seam gap (v3a.1 D1) costing recall directly | A paste that leaves no edge, or training on real signs |

**Out of scope:**
- Any retraining: multi-sign scenes, seamless pastes, real-sign training, a size or box head. v3b
  evaluates the three existing `peak_decoy` checkpoints only.
- IoU-based detection AP, and any comparison with published GTSDB results. The model outputs a point,
  not a box, and the GTSDB competition scored three sign categories by box overlap (§6).
- Learned or class-aware suppression. Suppression is one fixed geometric rule (§2.2).
- Speed work, video, and photos from outside GTSDB.
- Whole-photo hybrids (pasted signs in whole photos).
- Any change to the windowed splits, their files, τ's rule for windows, or their stored results.

## 2. Behavior

### 2.1 Running the model on a photo

- **Input:** one GTSDB photo, loaded with `load_photo` (RGB, 1360×800), as float [3, 800, 1360] in [0, 1],
  like every scene so far.
- **Pyramid:** for each scale s in `PYRAMID_SCALES = (1, 1/2, 1/4)`:
  - resize the photo to (1360·s, 800·s) with PIL bilinear, the same resize `cut_window` uses to shrink
    real windows;
  - run the trunk and heatmap head once on the whole resized photo.

  The network is fully convolutional (`check_images` needs only H and W divisible by 4), so this is the
  same computation as tiling the photo into windows, without the seams between tiles. A1 tests that
  directly.
- **No square requirement.** `predict()` refuses non-square input because `peak_centers` assumes a square
  scene. v3b does not call `predict()` or `PeakCNN.forward`. It calls `trunk`, `heatmap_head` and
  `class_head` itself, and maps cells to pixels with its own rule (below).
- **Cell → photo point:** cell (r, c) at scale s is the photo point
  (x, y) = ((c + 0.5)·4 / s, (r + 0.5)·4 / s). The mapping is exact, because every scaled size is an exact
  multiple of the native size.

### 2.2 From heatmaps to detections

1. **Peaks.** At each scale, a *peak* is a cell whose probability is ≥ `MIN_PEAK_SCORE` (0.05) and equal
   to the max of its 3×3 neighborhood. Cells on a flat top all count as peaks, and step 3 merges them. The
   0.05 floor only shortens the list; every threshold reported is above it.
2. **Class at each peak.** The class is read exactly as PeakCNN reads it: features averaged over the 3×3
   cells around the peak (`PEAK_WINDOW`), then `class_head`, then softmax. A new
   `window_features_at` computes this for many cells of one image at once. It must agree with
   `window_features` (A2).
3. **Suppression across all scales,** greedy by score (ties broken by scale, finest first, then row, then
   column):
   - keep the highest-scoring detection left;
   - drop every detection whose point is strictly closer than r = `SUPPRESS_CELLS`·4 / s_kept photo px
     to it in both x and y (a square, Chebyshev distance), where s_kept is the scale of the detection
     being kept and `SUPPRESS_CELLS` = 3;
   - repeat.

   Why 3 cells: at scale s the signs the model can see are at least 16/s px across, so two touching
   stacked signs have centers at least about 16/s px apart. A radius of 12/s keeps both, while still
   merging the several peaks one sign raises within a scale and across neighboring scales. Dev's closest
   stacked pair (photo 552) has centers 15.5 px apart in y. Their peaks land 3–4 cells apart at scale 1
   (12–16 px), so both survive: only peaks closer than 3 cells merge. Suppression uses
   no class.
4. **A detection** is (photo point x, y; score = the peak cell's probability; class_id; class_prob; scale).
   The detections ≥ a threshold t are the model's answer at t.

### 2.3 Ground truth and matching

- **Signs:** the labeled signs of each photo, with exact duplicate rows removed. Photo 340 lists the same
  box and class twice. That leaves 851 dev signs and 361 test signs.
- **A detection points at a sign** if its point lies in the sign's half-open box (x1 ≤ x < x2,
  y1 ≤ y < y2). This is the windows' "peak in box" rule.
- **Matching is greedy by score** (the same order as suppression), and each sign is matched at most once:
  - **found matching:** a detection matches the nearest-centered unmatched sign that it points at *and*
    whose class equals its class (ties go to the lower sign index);
  - **located matching:** the same, ignoring class.
- **Every unmatched detection is a false positive**, of one of three kinds (from found matching):
  - `duplicate`: it points at a sign that is already matched;
  - `wrong_class`: it points only at signs of another class;
  - `background`: it points at no sign. This includes unlabeled signs.
- **Prefix property:** matching in score order means a detection's outcome depends only on detections
  scored higher. So matching once over all detections gives the matching at every threshold t: the
  detections with score ≥ t keep their outcome. One pass yields the whole curve (A6).

### 2.4 Metrics

At a threshold t, over a set of photos:

- **found recall** = matched signs / signs (found matching). This is the windows' "found", for whole
  photos;
- **located recall** (class ignored), and **class accuracy given located**;
- **FPPI** = false positives / photos;
- **FROC curve:** found recall against FPPI as t sweeps all detection scores;
- **AP_point:** area under found precision vs found recall, with precision = matched / detections ≥ t.
  It is reported, never graded, and is not comparable to any published AP (§1, Out of scope);
- **τ_W** = the smallest threshold with dev FPPI ≤ 0.5, i.e. at most ⌊0.5 · 600⌋ = 300 dev false positives
  at or above it. It is computed like `threshold_at_false_alarm`, over the false positives' scores. τ_W is
  chosen per checkpoint on dev only and then carried, unchanged, to test.

**Intervals:** a photo-clustered bootstrap (`cluster_bootstrap`, 95% percentile, whole photos resampled
with all their signs and false positives), at a fixed threshold. Recall and FPPI are ratios of sums over
the resampled photos.

**Breakdowns** of found recall: size bucket by native long side (`size_bucket`: small < 24, medium,
large ≥ 48; dev has 85 / 482 / 284), stacked vs isolated (234 / 617), and in-real_dev-window vs not
(645 / 206).

### 2.5 Failure modes and risks

- **The seam gap costs recall at strict thresholds.** Real signs' median presence is 0.752, against 0.878
  for the same signs pasted (v3a.1). The stored window scores already show the cost. When the threshold is
  set so only 0.08% of real_dev negatives fire (1 of 1,200), real_dev found is 34% / 77% / 80% for seeds
  0 / 1 / 2. At 1% it is about 96% for all three. FPPI 0.5 spread over 87 window-equivalents is about
  0.6% per window, between those two points. FPPI 0.1 is about 0.1%, where the seeds disagree wildly.
  This is why the headline is FPPI 0.5 (§6).
- **Unlabeled signs count as false positives.** GTSDB labels only the 43 GTSRB classes. P6 (on v3a seed 0)
  found that 8 of the 20 strongest false alarms on real_dev negatives were unlabeled signs, such as
  parking signs, crossing squares and chevron boards. FPPI is therefore pessimistic. The
  hand audit (§2.6, reported) estimates by how much. It is not used for grading.
- **One sign, several scales.** A 40 px sign is in range at scales 1 and ½. Suppression should merge those
  two peaks. If the coarse peak wins, its larger radius (24 px at ½) can also delete a stacked neighbor
  found at scale 1. "Recall lost to suppression" (located recall before vs after step 3) measures this.
- **Big signs at scale 1.** A 100 px sign is twice the largest training sign at scale 1, and may raise
  several peaks across its face. Those that escape suppression are `duplicate` false positives, reported
  per scale.
- **Stacked signs share a class window.** At scale 1, the 3×3-cell class window spans 12 px. For two
  20 px plates stacked on one post, it can take in features of both, so the second plate gets the wrong
  class. That shows up as a `wrong_class` false positive plus a missed sign.
- **Photo borders.** The model pads with zeros, as it does on windows. GTSDB sign centers lie at y ≥ 56,
  so no sign is close to the top edge, where this would matter most.
- **Blast radius: new files only.** `window_features`, `predict()`, every existing script and every stored
  result are unchanged. The test guard gains a new family, and its existing families are untouched
  (existing tests pin them). If any v3b code is wrong, only `detect_*.json` files and RESULTS.md's v3b
  section are affected.

### 2.6 Predictions (frozen in RESULTS.md before any code)

Graded on **seed 0 (`peak_decoy-20261112-011953`), dev photos 0–599**, at that seed's synthetic
τ = 0.484, unless the row says otherwise. "Found" is found recall (§2.4).

| # | Prediction | Measured on |
|---|---|---|
| W1 | Whole photos agree with windows where both exist: on the 645 signs in real_dev, whole-photo found at τ is within 3 points of windowed found (94.42%), i.e. in [91.42, 97.42]%. Guess: 93.5% | dev, the 645 windowed signs |
| W2 | Stacked signs are harder: found on the 617 isolated signs minus found on the 234 stacked is ≥ 5 points, and the photo-clustered 95% interval of that difference excludes 0. Guess: 93% vs 80% | dev, all 851 signs |
| W3 | The pyramid is needed for large signs: on the 139 dev signs with native long side ≥ 64 px, found with the pyramid is ≥ 85%, and with scale 1 alone ≤ 60%. Guess: 90% vs 30% | dev, signs ≥ 64 px |
| W4 | The synthetic τ is not a whole-photo threshold: dev FPPI at τ is ≥ 0.3. A crude count gives 0.65 (0.75% real false alarms per window × 87 window-equivalents). Guess: 0.8 | dev, all 600 photos |
| W5 | A usable operating point exists: at τ_W (dev FPPI ≤ 0.5), found on all 851 dev signs is ≥ 80%, for both seed 0 and the 3-seed mean. Guess: 85% | dev, seeds 0–2 |
| W6 | τ_W transfers to new photos: on test photos 600–899, at each seed's dev τ_W, the 3-seed mean test FPPI is ≤ 0.75 and the 3-seed mean test found is within 5 points of the 3-seed mean dev found at τ_W. Guess: FPPI 0.5, found 86% | test photos, seeds 0–2, scored once |

**Reported but not graded:**
- the FROC curves of all three seeds on dev; found at FPPI 0.1 and 1.0; AP_point;
- located recall, class accuracy given located, and recall lost to suppression;
- found by size bucket, stacked vs isolated, windowed vs not, for each seed;
- false positives by kind (`duplicate`, `wrong_class`, `background`) and by scale;
- **hand audit:** the 40 highest-scoring seed-0 dev `background` false positives at τ_W, each labeled
  *unlabeled sign*, *sign-like object*, or *other*, the way P6 was. From that, FPPI with the audited
  unlabeled signs removed;
- W1's paired agreement (McNemar, `mcnemar_exact`);
- seconds per photo.

### 2.7 Acceptance criteria

- **A1** *Whole photo = windows.* On CPU, for a random [1, 3, 800, 1360] input, the trunk's heatmap
  logits on any crop whose corner is at a multiple of 4 px equal the whole-input logits at every cell at
  least 8 cells inside the crop, within 1e-4. (The receptive field is 64 px = 16 cells wide, so 8 cells
  is its half-width.)
- **A2** `window_features_at(features[i:i+1], cells)` equals `window_features` for one cell per image, on
  random features, at the four corners, at the edges and in the interior.
- **A3** `find_peaks`, on hand-made heatmaps:
  - one Gaussian gives one peak at its center;
  - two Gaussians 4 cells apart give two peaks;
  - a flat 2×2 top gives four peaks, which suppression merges into one;
  - a peak at 0.049 is dropped.
- **A4** `suppress`:
  - the result does not depend on input order;
  - two scale-1 detections 16 px apart both survive, and 8 px apart only the higher survives;
  - a scale-¼ detection suppresses a lower scale-1 detection 20 px away, but not one 60 px away;
  - ties break as §2.2 says.
- **A5** A single-peak heatmap at cell (r, c) and scale s gives the point ((c + 0.5)·4/s, (r + 0.5)·4/s).
- **A6** `match_photo` on hand-built cases:
  - the box edges: x1 is inside, x2 is not;
  - a point inside two boxes goes to the nearest center;
  - found and located matching differ when the class is wrong;
  - a second detection on a matched sign is a `duplicate`;
  - **prefix property:** for every threshold t, matching the detections ≥ t alone gives the same outcomes
    as the full matching restricted to them.
- **A7** Deduplication drops exactly photo 340's repeated row. Dev has 851 signs, test 361.
- **A8** `threshold_at_fppi`:
  - at most ⌊fppi · n_photos⌋ false-positive scores are ≥ τ;
  - ties fall below τ;
  - with fewer false positives than that, τ = `MIN_PEAK_SCORE`;
  - it raises for an fppi < 0, n_photos < 1, or a non-finite score.
- **A9** *Same pipeline as windows.* On 64 stored val sign scenes whose presence is ≥ `MIN_PEAK_SCORE`,
  `detect` at scales (1,), run on the 128×128 scene, gives a top detection whose score, point and class equal `predict()`'s peak_prob,
  peak_xy and class_id. Score within 1e-5; point and class exactly.
- **A10** *Test guard.*
  - `detect_photos.py --split test` refuses to run without that run's `detect_dev.json`. It reads τ_W
    from there and takes no threshold argument.
  - It refuses a second test scoring of the same run.
  - Existing guard families are unchanged.

## 3. Interfaces (no bodies)

### `src/findobject/models/peak.py` (add; nothing existing changes)
```python
def window_features_at(features: Tensor, cells: Tensor) -> Tensor:
    """[1, C, h, w] features of one image, int64 [N, 2] (row, col) cells -> [N, C]: for each cell, the mean
    over the PEAK_WINDOW x PEAK_WINDOW cells around it, clipped at borders exactly as window_features.
    Pools once, then indexes. Raises ValueError if features is not [1, C, h, w], or a cell is out of range."""
```

### `src/findobject/inference/peaks.py` (new, pure)
```python
MIN_PEAK_SCORE = 0.05

def find_peaks(probs: Tensor, min_score: float = MIN_PEAK_SCORE) -> Tensor:
    """[h, w] probabilities -> int64 [N, 2] (row, col) of every cell >= min_score that equals the max of its
    3x3 neighborhood (clipped at borders), in row-major order. N may be 0.
    Raises ValueError if probs is not 2-D or min_score is not in (0, 1)."""

def cell_to_photo(cells: Tensor, scale: float) -> Tensor:
    """int64 [N, 2] (row, col) at `scale` -> float [N, 2] (x, y) in native photo px: ((c+0.5)*STRIDE/scale, ...).
    Raises ValueError if scale <= 0."""
```

### `src/findobject/inference/suppress.py` (new, pure)
```python
SUPPRESS_CELLS = 3

@dataclass(frozen=True)
class Detection:
    x: float  # native photo px
    y: float
    score: float  # the peak cell's probability
    class_id: int
    class_prob: float
    scale: float  # pyramid scale it came from

def suppress(dets: Sequence[Detection]) -> list[Detection]:
    """Greedy suppression (docs/v3b-spec.md §2.2 step 3), result sorted by (score desc, scale asc, y, x).
    Never raises for valid Detections; [] for []."""
```

### `src/findobject/inference/detect.py` (new)
```python
PYRAMID_SCALES = (1.0, 0.5, 0.25)

@torch.no_grad()
def detect(model: PeakCNN, photo: Image.Image, scales: Sequence[float] = PYRAMID_SCALES) -> list[Detection]:
    """All detections in one photo, after suppression (docs/v3b-spec.md §2.1-2.2). Model expected in eval
    mode. Raises ValueError if a scaled size is not an integer divisible by STRIDE, or as check_images."""
```

### `src/findobject/eval/photo_match.py` (new, pure)
```python
class FalsePositive(Enum):
    DUPLICATE = "duplicate"
    WRONG_CLASS = "wrong_class"
    BACKGROUND = "background"

@dataclass(frozen=True)
class PhotoMatch:
    found: tuple[int | None, ...]  # per detection (in score order): matched sign index, or None
    located: tuple[int | None, ...]  # the same, ignoring class
    fp_kind: tuple[FalsePositive | None, ...]  # per detection; None if found-matched

def dedupe_signs(signs: Sequence[RealSign]) -> tuple[RealSign, ...]:
    """Drop exact repeats (same box and class), keeping first occurrences in order."""

def match_photo(dets: Sequence[Detection], signs: Sequence[RealSign]) -> PhotoMatch:
    """§2.3 matching over dets sorted as suppress returns them. Raises ValueError if dets are not in that order."""
```

### `src/findobject/eval/froc.py` (new, pure)
```python
def threshold_at_fppi(fp_scores: np.ndarray, n_photos: int, fppi: float) -> np.float32:
    """Smallest float32 τ with at most floor(fppi * n_photos) of fp_scores >= τ (ties fall below), or
    MIN_PEAK_SCORE if there are fewer. Raises ValueError as A8 says."""

def froc_curve(scores: np.ndarray, is_tp: np.ndarray, n_signs: int, n_photos: int) -> dict[str, np.ndarray]:
    """thresholds, found recall and FPPI at every distinct score, descending. Raises ValueError for unequal
    lengths, n_signs < 1 or n_photos < 1."""

def ap_point(scores: np.ndarray, is_tp: np.ndarray, n_signs: int) -> float:
    """Area under found precision vs recall (all-points interpolation). Raises as froc_curve."""
```

### `src/findobject/eval/detect_report.py` (new, pure)
```python
def detect_report(per_photo: Sequence[tuple[RealPhoto, list[Detection], PhotoMatch]], thresholds: dict[str, float],
                  windowed: set[tuple[int, int]]) -> dict:
    """§2.4 metrics, breakdowns and intervals at each named threshold (e.g. "tau_syn", "tau_w"), plus the FROC
    curve and AP_point. `windowed` holds the (photo, sign) pairs in real_dev. Raises ValueError for no photos."""
```

### `src/findobject/eval/test_guard.py` (add one family)
`"photos_test": ("detect_test.json",)`. This is a new family, separate from `real_test`. Whole-photo
detection is a new question, and nothing in v3b is tuned on the windowed real_test results. §6 asks
whether that is acceptable.

### Scripts
- `scripts/detect_photos.py --checkpoint <run>/best.pt --split dev|test [--scales 1 0.5 0.25] [--device ...]`
  - writes `outputs/eval/<run>/detect_dev.json`, `detect_dev_scale1.json` (with `--scales 1`) or
    `detect_test.json`;
  - each file holds every detection and its match, the report, τ_syn, τ_W (dev) or the τ_W it read
    (test), the checkpoint hash and the device.
- `scripts/fp_sheet.py --run <run>`: a contact sheet of the 40 highest-scoring dev `background` false
  positives at τ_W, for the hand audit. The labels go in `results/v3b_fp_audit.csv`.

## 4. Steps after approval

1. Freeze W1–W6 in RESULTS.md and commit, before any code.
2. The pure pieces, each with its tests: `find_peaks`, `cell_to_photo`, `suppress`, `window_features_at`,
   `dedupe_signs`, `match_photo`, `froc` (A2–A8).
3. `detect`, with A1 and A9.
4. `detect_report` and `detect_photos.py`. Score dev for seeds 0–2, plus seed 0 at scale 1 alone.
5. The FP sheet and hand audit. Write the dev analysis in RESULTS.md and commit it.
6. Score test once per seed at its τ_W, grade W6, and commit.
7. If wanted: a notebook 09 for whole photos (§6).

## 5. Cost

- **Code:** five small modules, one added function, one guard entry and two scripts. No new dependency.
- **Compute:** no training. One photo through all three scales takes about 0.09 s on the MPS GPU (0.070
  + 0.018 + 0.004 s, timed on random input). 600 dev photos are about a minute of model time per seed,
  plus loading. All of steps 4–6 are well under an hour.

## 6. Open questions for review

- **Scope.** This draft is evaluation-only: it reads a class at every peak and leaves multi-sign training
  to a v3c, if W2 says it is needed. The alternative is to fold multi-sign training into v3b now. That
  costs about 2.5 hours of training plus a new scene generator, and it gives up the before/after baseline
  this draft provides.
- **Headline operating point: FPPI 0.5.** That is one false positive every two photos, counting unlabeled
  signs. FPPI 0.1 is the common choice in pedestrian detection, but the stored window scores suggest that
  at about 0.1% per window, seed-to-seed spread (34–80% found) would swamp everything else. FPPI 0.1 and
  1.0 are reported either way.
- **Unlabeled signs are graded as false positives.** The alternative is to hand-label every `background`
  false positive and grade on the rest. Under 0.5 FPPI that is about 300 per seed. The audit of the top 40
  estimates the size of the effect without that cost.
- **`SUPPRESS_CELLS = 3` is fixed by the geometry argument in §2.2, not tuned.** Tuning it on dev before
  grading would be legitimate, but it adds a knob the predictions would depend on.
- **Octave pyramid (1, ½, ¼).** A finer pyramid (steps of 1/√2) would keep every sign nearer the 32 px
  training median, at about 1.5× the cells (1.94 against 1.31 scale-1 photos' worth) and therefore more false positives. Octaves are chosen
  because they divide the photo exactly and already cover 16–192 px.
- **Test photos 600–899 were already scored as windows** (real_test, v3a and v3a.1). Whole-photo
  detection on them is a new family that is scored once. Nothing here was chosen by looking at real_test.
  Is that clean enough, or should W6 be reported as provisional?
- **Point-in-box, not IoU.** Without a size output there is no box to overlap. A size head is the natural
  v3c companion to multi-sign training, if wanted.
- **Notebook 09.** A whole-photo notebook would add one more notebook, and `test_notebooks.py` would need
  its 00–08 check changed to 00–09. Should it be part of v3b, or wait until results exist?
