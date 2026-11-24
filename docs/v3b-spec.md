# v3b spec: find every sign in a whole road photo

Status: **draft 2.** The plan was approved on 2026-11-17 on draft 1 (0391cbf). Draft 2 applies a
fresh-session review of draft 1, and §7 lists what changed. The predictions (§2.6) are frozen in RESULTS.md
before any code.

## 1. Intent

Every version so far has answered a question about one **128×128 window** holding at most one sign. The
real task is different. A GTSDB photo is 1360×800 and may hold 0 to 6 signs of very different sizes, and
the model has to say where each one is and what it is. v3b asks: **with no retraining, how well does the
v3a.1 model (`peak_decoy`, seeds 0–2) find signs in whole photos, and what limits it?**

Whole photos differ from windows in four ways. Each one is measured below; none of them has been measured
before:

1. **Several signs per photo.** Dev photos 0–599 hold 851 labeled signs, after one duplicate label is
   removed (§2.3). 94 photos have none, 259 have one, 179 have two, and 68 have three to six. The model
   reads a class only at its single argmax, so v3b reads a class at **every** peak.
2. **Stacked signs.** 234 of the 851 dev signs are *stacked*: another labeled sign's center is within
   1.5× their long side, usually because two plates hang on one post. v2's window cutter skipped any window
   with another sign in view, so **206 of these 234 have never been scored**. All 206 dev signs missing
   from real_dev are stacked.
3. **Sign size.** Dev signs are 17–129 px on their long side (median 39). Training signs are 16–48 px.
   Windows shrank each sign to ≤ 48 px before scoring it. A whole photo has no single right scale, so v3b
   runs the model on an **image pyramid**: the photo at scales 1, ½ and ¼. Each of these is divisible by
   the stride (1360×800, 680×400, 340×200), and together they put every sign from 16 to 192 px into the
   16–48 px training range at some scale.
4. **Many more chances to fire.** At those three scales a photo has 89,250 heatmap cells, as many as 87
   windows. τ was chosen so that 1% of *windows* fire, so a photo has many chances to set off a false
   alarm at that τ. v3b counts **false positives per photo (FPPI)** and chooses its threshold on dev photos.
   v3a deferred that choice to here (v3a spec §5).

**The decision this informs.** v3a §6 left a choice for v3b: train on scenes with several signs, or read a
class at every peak. v3b tries the cheap option, reading at every peak, and measures where it breaks. The
stacked comparison is made within the 24–48 px bucket (§2.6, W2), because stacked signs are smaller
(median 35 vs 40 px), and size alone would otherwise move the comparison.

| Result (seed 0, 24–48 px bucket, at τ_syn) | Reading | Next step |
|---|---|---|
| W2 holds, and stacked signs are also ≥ 5 points behind on *located before suppression* | The model itself handles stacked signs worse: one-sign training is a limit | Multi-sign training scenes (v3c) |
| W2 holds, but before suppression the gap is < 5 points | Suppression deletes stacked neighbors | Change the suppression rule; no retraining |
| W2 fails, and stacked found is ≥ 85% | Reading a class at every peak is enough | No multi-sign training needed |
| W2 fails, and stacked found is < 85% | Both groups are weak, or the gap is noise: inconclusive | Read the breakdowns |
| W5 fails, and found at τ_W on the 617 isolated signs is also < 80% | Real signs score too low against real clutter. This is the seam gap (v3a.1 D1) costing recall directly | A paste that leaves no edge, or training on real signs |
| W5 fails, but the isolated signs reach ≥ 80% | The loss is in stacked or large signs | Read the breakdowns |

**Out of scope:**
- Any retraining: multi-sign scenes, seamless pastes, real-sign training, a size or box head. v3b
  evaluates the three existing `peak_decoy` checkpoints only.
- IoU-based detection AP, and any comparison with published GTSDB results. The model outputs a point,
  not a box, and the GTSDB competition scored three sign categories by box overlap.
- Learned or class-aware suppression. Suppression is one fixed geometric rule (§2.2).
- Speed work, video, and photos from outside GTSDB.
- Whole-photo hybrids (pasted signs in whole photos).
- A whole-photo notebook. It can follow once results exist.
- Any change to the windowed splits, their files, τ's rule for windows, or their stored results.

## 2. Behavior

### 2.1 Running the model on a photo

- **Input:** one GTSDB photo, loaded with `load_photo` (RGB, 1360×800), as float [3, 800, 1360] in [0, 1],
  like every scene so far. The model must be in **eval mode**. In train mode BatchNorm uses batch
  statistics, and the same crop's logits move by about 0.2. `detect` refuses a model in train mode.
- **Pyramid:** for each scale s in `PYRAMID_SCALES = (1, 1/2, 1/4)`:
  - resize the photo to (1360·s, 800·s) with PIL bilinear, the same resize `cut_window` uses to shrink
    real windows;
  - run the trunk and heatmap head once on the whole resized photo.

  The network is fully convolutional (`check_images` needs only H and W divisible by 4). Its receptive
  field is 64 px: cell c sees input pixels 4c − 30 to 4c + 33. So whole-photo logits equal a window's
  logits wherever the window's zero padding cannot reach (A1). This is tiling without the seams between
  tiles.
- **No square requirement.** `predict()` refuses non-square input because `peak_centers` assumes a square
  scene. v3b does not call `predict()` or `PeakCNN.forward`. It calls `trunk`, `heatmap_head` and
  `class_head` itself, and maps cells to pixels with its own rule (below).
- **Cell → photo point:** cell (r, c) at scale s is the photo point
  (x, y) = ((c + 0.5)·4 / s, (r + 0.5)·4 / s). The mapping is exact, because every scaled size is an exact
  multiple of the native size.

### 2.2 From heatmaps to detections

1. **Peaks.** At each scale, a *peak* is a cell whose **logit** is at least logit(`MIN_PEAK_SCORE` = 0.05)
   and equal to the max logit of its 3×3 neighborhood. Its score is sigmoid(logit). Peaks are found on
   logits, as `predict` and `presence_scores` find the argmax, because float32 sigmoid can tie logits that
   differ. Cells on a flat top all count as peaks, and step 3 merges them. The 0.05 floor only shortens
   the list; every threshold reported is above it.
2. **Class at each peak.** The class is read exactly as PeakCNN reads it: features averaged over the 3×3
   cells around the peak (`PEAK_WINDOW`), then `class_head`, then softmax. A new `window_features_at`
   computes this for many cells of one image at once. It must agree with `window_features` (A2).
3. **Suppression across all scales.** Detections are ordered by score, descending. Ties go to the finer
   scale first (larger s), then the smaller y, then the smaller x. Then repeat:
   - keep the first detection left;
   - drop every later detection whose point is **strictly closer** than r = `SUPPRESS_CELLS`·4 / s_kept
     photo px to it in both x and y (a square: Chebyshev distance), where s_kept is the kept detection's
     scale and `SUPPRESS_CELLS` = 3.

   **Why 3 cells:** at scale s, the signs the model can see are at least 16/s px across. Two touching
   stacked signs therefore have centers about 16/s px apart or more. A radius of 12/s keeps both, and still
   merges the several peaks one sign raises within a scale and across neighboring scales. Dev's closest
   stacked pair (photo 552) has centers 15.5 px apart in y. At scale 1 their peaks land 3–4 cells apart
   (12–16 px), so both survive: only peaks closer than 3 cells merge.

   Suppression ignores class. The list before step 3 is kept too (the *raw* detections), to measure what
   suppression removes.
4. **A detection** is: photo point (x, y); score; class_id; class_prob; scale. The detections with
   score ≥ t are the model's answer at threshold t.

### 2.3 Ground truth and matching

- **Signs:** the labeled signs of each photo. Exact duplicate rows (same box and class) are removed,
  keeping the first. Photo 340 lists one box and class twice (rows 1 and 2 of that photo; row 2 goes).
  That leaves 851 dev signs and 361 test signs. **Every sign keeps its gt.txt row index within the photo**
  (`gt_index`), and that is the index everywhere: in matches, in `windowed` lookups (real_dev's
  `sign_index` is the same index), and in reports.
- **Stacked:** a sign is stacked if another (deduplicated) sign of the same photo has its box center
  strictly closer than `STACK_FACTOR` = 1.5 × this sign's long side, by Euclidean distance.
- **A detection points at a sign** if its point lies in the sign's half-open box (x1 ≤ x < x2,
  y1 ≤ y < y2). This is the windows' "peak in box" rule. Dev has 90 pairs of overlapping boxes, so a
  point can lie in more than one box.
- **Matching** runs once, over the suppressed detections in the order of §2.2 step 3. Each sign is
  matched at most once.
  - **Found matching:** a detection matches the *unmatched* sign that it points at and whose class equals
    its own. If there are several, it takes the one whose box center is nearest by Euclidean distance;
    ties go to the lower `gt_index`.
  - **Located matching:** the same, ignoring class. It is run as a separate pass.
- **Every detection not found-matched is a false positive.** Its kind is decided *when it is processed*,
  using only the detections processed before it:
  - `background` if it points at no sign. This includes unlabeled signs;
  - else `duplicate` if it points at a sign of its own class, all of which are then already matched;
  - else `wrong_class`.
- **Prefix property:** every outcome, both matchings and the false-positive kind, depends only on
  detections earlier in the order. Running the same rule on just the detections with score ≥ t gives
  their same outcomes. So one pass yields every threshold's result (A6).
- **Located before suppression:** a sign counts as located before suppression at t if any *raw*
  detection with score ≥ t points at it. It is computed from each sign's best raw score.

### 2.4 Metrics

At a threshold t, over a set of photos:

- **found recall** = found-matched signs / signs. This is the windows' "found", for whole photos;
- **located recall** (class ignored), **located before suppression**, and **class accuracy given
  located**;
- **FPPI** = false positives / photos;
- **FROC curve:** found recall against FPPI as t sweeps every detection score;
- **AP_point:** area under found precision against found recall, with all-points interpolation, where
  precision = found-matched / detections ≥ t. It is reported, never graded, and is not comparable to any
  published AP;
- **τ_syn:** each seed's stored window τ (0.484, 0.562 and 0.529 for seeds 0, 1 and 2);
- **τ_W:** the smallest threshold with dev FPPI ≤ 0.5, i.e. at most ⌊0.5 · 600⌋ = 300 dev false positives
  at or above it. It is computed like `threshold_at_false_alarm`, over the false positives' scores alone,
  so no sign's score enters it. τ_W is chosen per checkpoint on dev and carried unchanged to test.

**Intervals:** a photo-clustered bootstrap (`cluster_bootstrap`, a 95% percentile interval,
`BOOTSTRAP_RESAMPLES` and `BOOTSTRAP_SEED` as in presence_report). It resamples whole photos, with all
their signs and false positives. Recall and FPPI are ratios of sums over the resampled photos. A
difference between two groups (isolated − stacked) is computed on the same resamples. Intervals are taken
at a fixed threshold. For τ_W this leaves out the variance from choosing τ_W, so W5's interval is a little
too narrow.

**Groups** for found, located and located-before-suppression:
- all signs;
- size bucket by native long side (`size_bucket`: small < 24, medium 24–47, large ≥ 48; dev has
  85 / 482 / 284);
- `LARGE_FOR_PYRAMID` = signs ≥ 64 px (139);
- stacked vs isolated (234 / 617), overall and within the medium bucket (146 / 336);
- in real_dev vs not (645 / 206).

### 2.5 Failure modes and risks

- **The seam gap costs recall at strict thresholds.** Real signs' median presence is 0.752, against 0.878
  for the same signs pasted (v3a.1). The stored window scores already show the cost, and the curve is
  steep:

  | Real_dev negatives firing | Found, seed 0 / 1 / 2 |
  |---|---|
  | 2 of 1,200 (0.17%) | 75.3 / 83.9 / 82.9% |
  | 5 (0.42%) | 86.8 / 95.8 / 96.0% |
  | 7 (0.58%) | 94.1 / 96.1 / 96.0% |
  | 12 (1%) | 96.4 / 96.7 / 96.7% |

  FPPI 0.5 spread over 87 window-equivalents is about 0.6% per window. That 87 is crude. W4's band allows
  an error of about 2× in it, which would put FPPI 0.5 anywhere from 0.3% to 1.1% per window. Seed 0 is
  on the steep part, which is why W5 is graded on the 3-seed mean. At FPPI 0.1 (about 0.1% per window),
  found falls to 34 / 77 / 80%, and the seeds disagree wildly. That is why the headline is FPPI 0.5
  (§6).
- **Unlabeled signs count as false positives.** GTSDB labels only the 43 GTSRB classes, so parking signs,
  crossing squares and chevron boards are unlabeled. For v3a seed 0, 8 of the 20 highest-presence real_dev
  negatives were signs: 6 outside the 43 classes and 2 of them. 3 of its 6 false alarms at τ were
  unlabeled signs. FPPI is therefore pessimistic. The hand audit (reported) estimates by how much; it is
  not used for grading.
- **One sign, several scales.** A 40 px sign is in range at scales 1 and ½, and suppression should merge
  the two peaks. If the coarse peak wins, its larger radius (24 px at ½) can also delete a stacked
  neighbor found at scale 1. "Located before suppression" against "located" measures this, and the
  decision table uses it.
- **Big signs at scale 1.** A 100 px sign is twice the largest training sign at scale 1, and may raise
  several peaks across its face. Those that escape suppression are `duplicate` false positives, reported
  per scale.
- **Neighbors share what each cell sees.** A cell sees 64 px at its scale, and the 3×3 class average sees
  72 px. At scale 1, any labeled sign within about 36 px therefore affects both the presence and the class
  read for its neighbor, not only through the class window. That is the mechanism W2 probes.
- **Photo borders.** Zero padding reaches up to half the receptive field: 32 px at scale 1, 128 native px
  at scale ¼. Signs sit near the vertical middle (center y from 56 to 693), but 76 dev sign centers are
  within 128 px of some edge, mostly the left or right one, and 13 are within 64 px. Large signs, which
  depend on scale ¼, are the ones most exposed.
- **Blast radius: new files only.** `window_features`, `predict()`, every existing script and every stored
  result are unchanged. The test guard gains a new family, and its existing families are untouched
  (existing tests pin them). If any v3b code is wrong, only `detect_*.json`, `results/v3b/` and
  RESULTS.md's v3b section are affected.

### 2.6 Predictions (frozen in RESULTS.md before any code)

Graded on **dev photos 0–599** at the run's own τ_syn, unless the row says otherwise. Seed 0 is
`peak_decoy-20261112-011953`. "Found" is found recall (§2.4). "Medium" is the 24–47 px bucket.

| # | Prediction | Measured on |
|---|---|---|
| W1 | Whole photos agree with windows where both exist: on the 645 signs in real_dev, seed 0's whole-photo found is within 3 points of windowed found (94.42%), i.e. in [91.42, 97.42]%. Guess: 93.5% | dev, the 645 windowed signs |
| W2 | Stacked signs are harder, with size held fixed: in the medium bucket, seed 0's found on the 336 isolated signs minus found on the 146 stacked is ≥ 5 points; the photo-clustered 95% interval of that difference excludes 0; and the difference is positive for all three seeds. Guess: 94% vs 83% | dev, medium bucket, seeds 0–2 |
| W3 | The pyramid is needed for large signs: on the 139 dev signs ≥ 64 px, seed 0's found with the pyramid is ≥ 85%, and at scale 1 alone ≤ 60%. Guess: 90% vs 30% | dev, signs ≥ 64 px |
| W4 | The window τ is not a whole-photo threshold, and the window-to-photo estimate holds roughly: seed 0's dev FPPI at τ_syn is in [0.4, 1.6]. The estimate is 0.65 (0.75% of real negatives × 87 window-equivalents), a lower bound that ignores `duplicate` and `wrong_class` false positives and the clutter near signs that real negatives were cut to avoid. Guess: 0.8 | dev, all 600 photos |
| W5 | A usable operating point exists: at each seed's τ_W (dev FPPI ≤ 0.5), the 3-seed mean found on all 851 dev signs is ≥ 80%. Guess: 85% | dev, seeds 0–2 |
| W6 | τ_W transfers to new photos: on test photos 600–899, at each seed's dev τ_W, the 3-seed mean test FPPI is ≤ 0.75, and the 3-seed mean test found is ≥ the 3-seed mean dev found at τ_W minus 5 points. Guess: FPPI 0.5, found 86% | test photos, seeds 0–2, scored once |

**Reported but not graded:**
- the FROC curves of all three seeds on dev; found at FPPI 0.1 and 1.0; AP_point;
- every group of §2.4 for each seed: found, located, located before suppression, class accuracy given
  located;
- W2's comparison over all sizes, and seed 0's found at τ_W;
- false positives by kind (`duplicate`, `wrong_class`, `background`) and by scale;
- **hand audit:** the 40 highest-scoring seed-0 dev `background` false positives at τ_W, each labeled from
  the image alone, as P6 was: *unlabeled sign*, *sign-like object*, or *other*. From that, an estimate of
  FPPI without unlabeled signs;
- W1's paired agreement, per sign, between windows and whole photos (`mcnemar_exact`);
- seconds per photo, measured by the script.

### 2.7 Acceptance criteria

- **A1** *Whole photo = windows, where padding can't reach.* On CPU, in eval mode, with a random
  [1, 3, 800, 1360] input, take any crop whose corner is at a multiple of 4 px. Its heatmap logits must
  equal the whole-input logits within 1e-6 at every cell c with 8 ≤ c ≤ w − 9 (and likewise for rows),
  where w is the crop's width in cells. Negative control: with a high-contrast input, some cell at margin
  ≤ 5 differs by more than 1e-6. For the class read, the margin is one cell more, because of the 3×3
  window.
- **A2** `window_features_at(features, cells)` equals `window_features` for one cell per image, on random
  features, at the four corners, at the edges and in the interior.
- **A3** `find_peaks`, on hand-made logits:
  - one Gaussian bump gives one peak at its center;
  - two bumps 4 cells apart give two peaks;
  - a flat 2×2 top gives four peaks, which suppression merges into one;
  - a peak at logit(0.049) is dropped;
  - two logits that tie after float32 sigmoid but differ as logits give the right single maximum.
- **A4** `suppress`:
  - the result does not depend on input order;
  - two scale-1 detections 16 px apart both survive, and 8 px apart only the higher survives;
  - a scale-¼ detection suppresses a lower scale-1 detection 20 px away, but not one 60 px away;
  - on equal scores, the scale-1 detection is kept over the scale-¼ one.
- **A5** A single-peak heatmap at cell (r, c) and scale s gives the point ((c + 0.5)·4/s, (r + 0.5)·4/s).
- **A6** `match_photo` on hand-built cases:
  - the box edges: x1 is inside, x2 is not;
  - a point inside two boxes goes to the nearer center;
  - found and located matching differ when the class is wrong;
  - a second same-class detection on a matched sign is a `duplicate`;
  - **the reviewer's case:** D1 (class 2) is above D2 (class 1), and both point only at sign A
    (class 1). D1 is `wrong_class`, and D2 is found;
  - **prefix property:** on random cases, for every threshold t, matching only the detections ≥ t gives
    the same matches and kinds as the full pass restricted to them.
- **A7** `photo_truth` on the real labels:
  - deduplication drops exactly photo 340's row 2;
  - `gt_index` is the gt.txt row index;
  - dev has 851 signs: 234 stacked / 617 isolated; 85 / 482 / 284 by size bucket; 139 ≥ 64 px; 146 / 336
    stacked / isolated in the medium bucket; 645 / 206 in real_dev or not;
  - test has 361 signs, 85 of them stacked;
  - a hand case pins the stacked rule: Euclidean distance, strict <, factor 1.5.
- **A8** `threshold_at_fppi`:
  - at most ⌊fppi · n_photos⌋ false-positive scores are ≥ τ;
  - ties fall below τ;
  - when there are at most ⌊fppi · n_photos⌋ false positives in all, τ = `MIN_PEAK_SCORE`;
  - it raises for fppi < 0, n_photos < 1, or a non-finite score.
- **A9** `froc_curve` and `ap_point` on a hand case: one photo, two signs, detections TP 0.9, FP 0.8,
  TP 0.7. Recall is 0.5 / 0.5 / 1.0, FPPI is 0 / 1 / 1, and AP_point = 0.5·1 + 0.5·⅔ = 0.8333.
- **A10** *Same pipeline as windows.* On 64 stored val sign scenes with presence ≥ `MIN_PEAK_SCORE`,
  `detect` at scales (1,) on the 128×128 scene gives a top detection whose score, point and class equal
  `predict()`'s peak_prob, peak_xy and class_id. The score must agree within 1e-6; the point and class
  must be equal.
- **A11** *Illegal states.*
  - `Detection` raises for a score or class_prob outside [0, 1], a class outside [0, 43), or a scale
    outside (0, 1].
  - `detect` raises RuntimeError for a model in train mode.
- **A12** *Test guard.* `detect_photos.py --split test`:
  - checks `scored_results(run, "photos_test")`, and refuses if `detect_test.json` exists;
  - refuses `--scales`;
  - refuses unless that run's `detect_dev.json` exists, has the same checkpoint SHA-256, and used
    `PYRAMID_SCALES`;
  - reads τ_W from that file and takes no threshold argument;
  - writes `detect_test.json` atomically (temp file, then rename), and prints no test number before the
    file is written.

  The existing guard families are unchanged.
- **A13** `detect_report` on a small hand-made set:
  - every group's counts and rates are right;
  - every interval contains its point estimate;
  - the difference interval is computed from the same resamples as its two groups;
  - the same seed gives the same intervals.

## 3. Interfaces (no bodies)

### `src/findobject/models/peak.py` (add; nothing existing changes)
```python
def window_features_at(features: Tensor, cells: Tensor) -> Tensor:
    """[1, C, h, w] features of one image, int64 [N, 2] (row, col) cells -> [N, C]: for each cell, the mean
    over the PEAK_WINDOW x PEAK_WINDOW cells around it, clipped at borders exactly as window_features.
    Pools once, then indexes. Raises ValueError if features is not [1, C, h, w], or a cell is out of range."""
```

### `src/findobject/inference/detection.py` (new, pure)
```python
PYRAMID_SCALES = (1.0, 0.5, 0.25)

@dataclass(frozen=True)
class Detection:
    x: float  # native photo px
    y: float
    score: float  # sigmoid of the peak cell's logit
    class_id: int
    class_prob: float
    scale: float  # pyramid scale it came from
    # __post_init__ raises ValueError for illegal values (A11)

def detection_order(d: Detection) -> tuple:
    """The one sort key used everywhere: (-score, -scale, y, x)."""
```

### `src/findobject/inference/peaks.py` (new, pure)
```python
MIN_PEAK_SCORE = 0.05

def find_peaks(logits: Tensor, min_score: float = MIN_PEAK_SCORE) -> Tensor:
    """[h, w] logits -> int64 [N, 2] (row, col) of every cell with logit >= logit(min_score) that equals the
    max logit of its 3x3 neighborhood (clipped at borders), in row-major order. N may be 0.
    Raises ValueError if logits is not 2-D or min_score is not in (0, 1)."""

def cell_to_photo(cells: Tensor, scale: float) -> Tensor:
    """int64 [N, 2] (row, col) at `scale` -> float [N, 2] (x, y) in native photo px. Raises ValueError if scale <= 0."""
```

### `src/findobject/inference/suppress.py` (new, pure)
```python
SUPPRESS_CELLS = 3

def suppress(dets: Sequence[Detection]) -> list[Detection]:
    """Greedy suppression (docs/v3b-spec.md §2.2 step 3), result in detection_order. [] for []. Never raises."""
```

### `src/findobject/inference/detect.py` (new)
```python
@dataclass(frozen=True)
class DetectResult:
    raw: tuple[Detection, ...]  # every peak at every scale, in detection_order
    kept: tuple[Detection, ...]  # after suppression, in detection_order

@torch.no_grad()
def detect(model: PeakCNN, photo: Image.Image, scales: Sequence[float] = PYRAMID_SCALES) -> DetectResult:
    """All detections in one photo (docs/v3b-spec.md §2.1-2.2). Raises RuntimeError if model.training;
    ValueError if a scaled size is not an integer divisible by STRIDE, or as check_images."""
```

### `src/findobject/eval/photo_truth.py` (new, pure)
```python
STACK_FACTOR = 1.5

@dataclass(frozen=True)
class TruthSign:
    gt_index: int  # row index of this sign in the photo's gt.txt rows
    box: Box
    class_id: int
    long_side: int
    stacked: bool

def photo_truth(photo: RealPhoto) -> tuple[TruthSign, ...]:
    """The photo's signs with exact duplicates removed (first kept), in gt.txt order, each flagged stacked
    or not (§2.3). Never raises for a RealPhoto that read_gtsdb produced."""
```

### `src/findobject/eval/photo_match.py` (new, pure)
```python
class FalsePositive(Enum):
    DUPLICATE = "duplicate"
    WRONG_CLASS = "wrong_class"
    BACKGROUND = "background"

@dataclass(frozen=True)
class PhotoMatch:
    found: tuple[int | None, ...]  # per kept detection: matched gt_index, or None
    located: tuple[int | None, ...]  # the same, ignoring class
    fp_kind: tuple[FalsePositive | None, ...]  # per kept detection; None if found-matched

def match_photo(kept: Sequence[Detection], truth: Sequence[TruthSign]) -> PhotoMatch:
    """§2.3 matching. Raises ValueError if kept is not in detection_order."""

def best_raw_score(raw: Sequence[Detection], truth: Sequence[TruthSign]) -> dict[int, float]:
    """gt_index -> the highest raw score pointing at that sign (absent if none), for located-before-suppression."""
```

### `src/findobject/eval/froc.py` (new, pure)
```python
def threshold_at_fppi(fp_scores: np.ndarray, n_photos: int, fppi: float) -> np.float32:
    """Smallest float32 τ with at most floor(fppi * n_photos) of fp_scores >= τ (ties fall below), or
    MIN_PEAK_SCORE if fp_scores has at most that many. Raises ValueError as A8 says."""

def froc_curve(scores: np.ndarray, is_tp: np.ndarray, n_signs: int, n_photos: int) -> dict[str, np.ndarray]:
    """thresholds, found recall and FPPI at every distinct score, descending. Raises ValueError for unequal
    lengths, n_signs < 1 or n_photos < 1."""

def ap_point(scores: np.ndarray, is_tp: np.ndarray, n_signs: int) -> float:
    """Area under found precision vs recall, all-points interpolation. Raises as froc_curve."""
```

### `src/findobject/eval/detect_report.py` (new, pure)
```python
LARGE_FOR_PYRAMID = 64  # px, native long side

@dataclass(frozen=True)
class PhotoRecord:
    index: int
    truth: tuple[TruthSign, ...]
    kept: tuple[Detection, ...]
    match: PhotoMatch
    best_raw: dict[int, float]

def detect_report(photos: Sequence[PhotoRecord], thresholds: dict[str, float],
                  windowed: dict[tuple[int, int], bool], rng_seed: int, n_resamples: int) -> dict:
    """§2.4 at each named threshold ("tau_syn", "tau_w", and the FPPI 0.1 and 1.0 points): every group, the
    isolated - stacked differences (overall and medium) with same-resample intervals, false positives by
    kind and scale, the FROC curve and AP_point. Also W1's McNemar test against `windowed`, which maps
    (photo, gt_index) to the windowed found outcome from presence_real_dev.json. Raises ValueError for no
    photos or a windowed key that names no sign."""
```

### `src/findobject/eval/detect_grade.py` (new, pure)
```python
def grade_dev(reports: Sequence[dict], scale1_seed0: dict) -> dict:
    """W1-W5 pass/fail with the numbers behind each, from the three seeds' dev reports (seed 0 first) and
    seed 0's scale-1 report. Raises ValueError if the reports are not three runs on the same photos."""

def grade_test(dev_reports: Sequence[dict], test_reports: Sequence[dict]) -> dict:
    """W6 from the three seeds' dev and test reports. Raises ValueError if the runs or τ_W disagree."""
```

### `src/findobject/eval/test_guard.py` (add one family)
`"photos_test": ("detect_test.json",)`. This is a new family, separate from `real_test`. Whole-photo
detection is a new question, and nothing in v3b is tuned on the windowed real_test results.

### Scripts
- `scripts/detect_photos.py --checkpoint <run>/best.pt --split dev|test [--scales 1 0.5 0.25] [--device ...]`
  - writes `outputs/eval/<run>/detect_dev.json`, `detect_dev_scale1.json` (dev with `--scales 1`) or
    `detect_test.json` (test, with no `--scales`; A12);
  - each file holds every kept detection with its match, every sign's best raw score, the report,
    τ_syn, τ_W (dev) or the τ_W it read (test), the scales, the checkpoint SHA-256, the device and
    seconds per photo.
- `scripts/grade_v3b.py`: reads the three runs' files, writes `results/v3b/grades_dev.json` and, once
  test exists, `grades_test.json`.
- `scripts/fp_sheet.py --run <run>`: a contact sheet of the 40 highest-scoring dev `background` false
  positives at τ_W, for the hand audit. The labels go in `results/v3b/fp_audit.csv`.

## 4. Steps after approval

1. Freeze W1–W6 in RESULTS.md and commit, before any code.
2. The pure pieces, each with its tests: `Detection`, `find_peaks`, `cell_to_photo`, `suppress`,
   `window_features_at`, `photo_truth`, `match_photo`, `froc`, `detect_report`, `detect_grade`
   (A2–A9, A11, A13).
3. `detect`, with A1, A10 and A11.
4. The test-guard family and `detect_photos.py` (A12). Score dev for seeds 0–2, plus seed 0 at scale 1
   alone.
5. `grade_v3b.py` (dev). The FP sheet and hand audit. Write the dev analysis in RESULTS.md and commit it.
6. Score test once per seed at its τ_W, grade W6, and commit.

## 5. Cost

- **Code:** nine small modules, one added function, one guard entry and three scripts. No new
  dependency.
- **Compute:** no training. One photo through all three scales takes about 0.09 s on the MPS GPU (0.070
  + 0.018 + 0.004 s, timed on random input). 600 dev photos are about a minute of model time per seed,
  plus loading. All of steps 4–6 are well under an hour.

## 6. Open questions for review

- **Scope.** This draft is evaluation-only: it reads a class at every peak and leaves multi-sign training
  to a v3c, if the decision table calls for it. Folding training in now would cost about 2.5 hours plus a
  new scene generator, and would lose the no-retraining baseline.
- **Headline operating point: FPPI 0.5.** That is one false positive every two photos, counting unlabeled
  signs. FPPI 0.1 is common in pedestrian detection, but at about 0.1% per window the stored scores show
  seed spread of 34–80% found. FPPI 0.1 and 1.0 are reported either way.
- **Unlabeled signs are graded as false positives.** Hand-labeling every `background` false positive would
  mean about 300 per seed. The audit of the top 40 estimates the effect.
- **`SUPPRESS_CELLS = 3` comes from the geometry in §2.2 and is not tuned.** Tuning it would add a knob
  the predictions depend on.
- **Octave pyramid.** A 1/√2 pyramid keeps signs nearer the 32 px median at about 1.5× the cells (1.94
  against 1.31 scale-1 photos' worth), so it would bring more false positives. Octaves divide the photo
  exactly and already cover 16–192 px.
- **Test photos 600–899 were already scored as windows** (real_test, v3a and v3a.1). Whole-photo detection
  on them is a new family, scored once, and nothing here was chosen by looking at real_test.
- **Point-in-box, not IoU.** With no size output there is no box to overlap. A size head is the natural
  companion to multi-sign training in a v3c.

## 7. Changes from draft 1 (after fresh-session review)

- **W2** is graded in the medium bucket (146 stacked vs 336 isolated), because stacked signs are smaller
  (median 35 vs 40 px; 26.9% vs 35.8% are ≥ 48 px). It also has to hold in direction on all three seeds.
  The decision table now separates a model limit from suppression (located before suppression) and has an
  absolute bar (85%) for "reading at every peak is enough". A W5 failure is attributed to the seam gap
  only if the isolated signs also fall below 80%.
- **W4** is now a band, [0.4, 1.6], rather than ≥ 0.3, which was nearly guaranteed. **W5** is graded on
  the 3-seed mean only, because seed 0 sits on a steep part of the window curve (75–96% found across 2–12
  negatives firing). **W6** is one-sided, because test has come out easier than dev before.
- **False-positive kinds** are decided when each detection is processed, with a fixed precedence
  (`background` → `duplicate` → `wrong_class`). That makes the prefix property true for the kinds too.
  "Nearest center" is Euclidean. The stacked rule is pinned (Euclidean, strict <, 1.5).
- **The suppression tie order** is finest scale first everywhere, through one `detection_order` key. Draft
  1's docstring had it reversed.
- **A1** names eval mode, the exact cell range (8 ≤ c ≤ w − 9), a 1e-6 tolerance, and a negative control.
  Measured on an untrained model with random input, a 256×256 crop agrees exactly (difference 0) from 8
  cells in, and differs by 1.1e-5 at 7 cells and 8e-5 at 6. The reviewer saw a similar pattern on a
  trained one. So draft 1's 1e-4 could barely fail.
- **New interfaces** for what the grading needs:
  - `DetectResult.raw` and `best_raw_score`, for located before suppression;
  - `TruthSign.gt_index`, so dedup can't shift indices;
  - windowed outcomes (not just membership), for McNemar;
  - bootstrap seed and count;
  - `LARGE_FOR_PYRAMID`;
  - `detect_grade` and `grade_v3b.py`, for 3-seed means and the W3 two-file comparison.
- **The test guard:** the script names its key (`photos_test`; the existing `test` key would have
  refused, since `presence_test.json` exists), refuses `--scales` on test, checks the dev file's SHA-256
  and scales, and writes atomically.
- **New criteria:** A7 (truth counts), A9 (FROC and AP), A11 (illegal states, train mode) and A13 (report).
  A8's edge case now reads "at most".
- **Smaller fixes:**
  - the photo counts are after dedup (94/259/179/68);
  - peaks are found on logits;
  - the border risk covers scale ¼ (76 dev sign centers within 128 px of an edge);
  - the neighbor risk covers the 64 px receptive field, not only the 12 px class window;
  - the P6 citation is corrected (6 + 2 of the top 20; 3 of the 6 false alarms at τ);
  - the notebook is out of scope.
