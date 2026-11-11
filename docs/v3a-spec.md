# v3a spec: teach the model to say "no sign here"

Status: **draft 2, for review.** Nothing below is implemented. Draft 1 (ba4c3c5) was reviewed by a fresh
session. §7 lists what changed and why. Predictions (§2.8) are frozen in RESULTS.md as the **first**
step after approval, before any code is written and before any model's presence score exists.

## 1. Intent

Every scene the model has ever seen holds exactly one sign, so it always finds one, even in an empty
image. That blocks the next step: running it over a whole road photo (v3b), where most of the image is
empty. v3a teaches the model to say "nothing here" and measures:

- how well it can tell sign from no-sign, on synthetic and on real photos;
- what that costs on the existing one-sign task;
- whether the untrained v1.5b heatmap could already tell them apart;
- whether "sign present" was partly learned as "paste seam present".

**Presence is read from the heatmap's peak height**, not from a new output. Training replaces some sign
scenes with empty ones whose target heatmap is all zero. The heatmap head is fully convolutional, so the
same "how tall is this peak" score works unchanged when v3b runs the model over a 1360×800 photo.

**Out of scope:**
- Whole-photo detection, detection AP, and non-maximum suppression (v3b).
- Scenes with two or more signs (a v3b question; see §6).
- Any change to the network's layers.
- Regenerating the stored val/test sign scenes. They stay byte-identical, so every earlier number stays
  comparable.
- Scoring v1.5b on any test split again. Its test numbers already exist, and the test guard (§2.6)
  refuses a second scoring.

## 2. Behavior

### 2.1 Presence score

`presence(x) = max over the 32×32 cells of sigmoid(heatmap_logit)`, stored as float32 in (0, 1). A scene
counts as "said yes" when `presence ≥ τ`.

On a sign scene the target peak is 0.7788–1.0. It depends on where the sign center falls relative to the
cell grid, since the Gaussian has σ = 1 cell. On an empty scene every target cell is 0.

### 2.2 Training scenes

- New config key `data.n_train_empty` (int, default 0), valid only with `train_scenes: fresh` and
  `0 ≤ n_train_empty < n_train`. It counts **within** `n_train`. Each epoch is still `n_train` scenes:
  - scenes `i < n_train − n_train_empty` are sign scenes;
  - the last `n_train_empty` scenes are empty.
- Each scene's RNG is still `[seed, epoch, i]`, so sign scene i is exactly the scene v1.5b saw at
  index i.
- **Why replace rather than add.** The epoch length doesn't change, which keeps all of these identical
  to v1.5b's run with the same seed:
  - the step count (313 per epoch) and the cosine schedule;
  - the model's initialisation;
  - the sampler's permutation each epoch, and with it the batch order.

  v3a seed 0 and v1.5b seed 0 therefore differ only in what fills slots 16,000–19,999, plus GPU
  nondeterminism, which any retrain has. The cost is 20% fewer sign scenes per epoch: 320k sign scenes
  over the run instead of 400k. That bias works *against* v3a on P3, so a small measured cost is a
  conservative one.
- **An empty scene** runs the background half of make_scene with no paste:
  1. draw the sign index and discard it, so the following draws line up with sign scene i;
  2. draw the background index;
  3. a random square crop of that train-split COCO background, then a resize to 128;
  4. blur with probability blur_prob;
  5. Gaussian noise.

  Steps 3–5 are one helper shared with make_scene, so empties can't be told apart by their noise or blur
  statistics. Blur and noise use the same distributions as in a sign scene, but they come from different
  positions in the stream, since a sign scene's paste draws come first.
- An empty scene's `Sample` has an all-zero heatmap, box `(0, 0, 0, 0)`, and `class_id = IGNORE_CLASS`.
  IGNORE_CLASS is −100, PyTorch's `cross_entropy` ignore index, defined once. "Has a sign" is *derived*
  from `class_id != IGNORE_CLASS` and never stored separately, so the two can't disagree.
- **`Sample.class_id` contract** becomes: int64 scalar in `[0, 43)`, or `IGNORE_CLASS` for a FreshScenes
  empty. Its consumers:
  - `training/losses.py`: masked (§2.3).
  - `training/epoch.py`: metrics over sign rows only.
  - `eval/heatmap_metrics.peak_in_box`: only ever called on sign rows.
  - `eval/evaluate.py`: only ever receives `SignScenes`. `SignScenes._validate` still rejects anything
    outside `[0, 43)`, so a stored sign file can never hold an empty.
- v3a config (`configs/peak_empty.yaml`, plus `_s1` and `_s2` for the other seeds): v1.5b's config with
  `n_train_empty: 4000`, so each epoch is 16,000 sign scenes and 4,000 empty. Nothing else changes,
  including the epochs, `n_train: 20000` and the seeds.

### 2.3 Loss

- Heatmap BCE over every scene, unchanged: summed over cells, divided by batch size b. On an empty scene
  every cell is pushed toward 0.
- Class cross-entropy becomes `cross_entropy(logits, class_id, ignore_index=IGNORE_CLASS,
  reduction="sum") / b`. That is, the sum over sign rows divided by the **whole** batch size. This gives:
  - Each sign row's weight stays 1/b, as in v1.5b. Averaging over sign rows instead would raise it by
    about 1.25× and confound the comparison.
  - An all-empty batch gets class loss exactly 0 with finite gradients. A plain mean over zero rows
    would be NaN.
  - On a batch with no empties the value equals today's mean, up to float rounding, because the
    reduction order differs (A4).

### 2.4 Evaluation data (new files; existing npz files untouched)

| File | What | n |
|---|---|---|
| `val_empty.npz` / `test_empty.npz` | synthetic empties on val / test COCO backgrounds | 5,000 each |
| `val_decoy_plain.npz` | diagnostic: a sign-shaped patch of *another* val background pasted with the val/test paste path (identity, rectangular seam) | 5,000 |
| `val_decoy_aug.npz` | the same, through the **training** paste path with v1.5b's `train_augment` (warped mask, contrast/saturation, occlusion) | 5,000 |
| `real_empty_dev.npz` / `real_empty_test.npz` | real negative windows from GTSDB dev (600 photos) / test (300 photos) | ≤ 1,200 / ≤ 600 |

**Synthetic empties and decoys:**
- Seeded per scene: `[EMPTY_SEED, split_offset, i]`.
- Each file stores the background index of every scene, so A9 can check that val files use only val
  backgrounds and test files only test ones.
- A decoy is made by **calling make_scene with the patch in place of the sign**, so its paste path is
  literally the same code as a sign's. Then the heatmap is set to zero and the box to `(0, 0, 0, 0)`.
  - The patch is a random crop of a second val background.
  - Its size is the ROI size of a val GTSRB sign drawn by the same RNG (the sign's pixels are unused).
    make_scene then resizes the patch to a 16–48 px long side exactly as it would a sign, so the patch
    gets a sign's resampling blur.
- Decoys are never trained on.

**Real negative windows:** two per photo, on every photo in the split, with or without signs, each from
its own stratum:
- **uniform**: the window position is uniform over the scaled photo.
- **band**: the window's center row is the scaled center row of a sign drawn uniformly from the split's
  labeled signs (from any photo), plus a uniform jitter of ±16 px, clamped to the photo. The column is
  uniform. This puts the negative where signs actually sit (roadside, mid-height) rather than in sky or
  road surface.

For each window:
- Its scale is drawn uniformly from the split's positive-window scales, rebuilt exactly as v2 computed
  them: `min(1, 48 / L)` from each `real_<split>.npz` row's `native_long_side`. Two-thirds of dev
  positives have scale 1.0.
- It is 128×128, cut with v2's `cut_window`. It must not intersect any labeled box after the box is
  widened by **4 photo-px** per side and then scaled.
- RNG `[seed, NEG_STREAM, photo_index, k]`, with k = 0 for uniform and 1 for band. Up to 50 tries;
  otherwise the window is skipped, and the skip count per stratum goes in `empty_manifest.json`.
- Each window stores `photo_index` and `stratum`.

Under GTSDB's definition, a sign outside the 43 classes (parking, no stopping, street names) is
background, and so is any unlabeled sign. A "yes" on one counts as a false alarm. P6 measures how much
of the real false-alarm rate is this.

Positives are the existing `val.npz` / `test.npz` (synthetic) and v2's `real_dev.npz` / `real_test.npz`.
v2's `hybrid_dev.npz` / `hybrid_test.npz` hold the same signs as the real files, row for row, pasted
with a seam. They are the paired seam check (§2.7).

### 2.5 The threshold τ

`τ = threshold_at_false_alarm(val_empty presence, 0.01)`, chosen once per checkpoint and saved in that
checkpoint's val presence result. It is defined as:
- with n negatives and `k = floor(0.01 · n)` (50 for n = 5,000), sort the scores in descending order
  `s₁ ≥ … ≥ sₙ`;
- `τ = nextafter(s_{k+1}, +∞)` in float32.

So at most k negatives score ≥ τ, and τ is the smallest float32 that guarantees it. Ties at `s_{k+1}` all
fall below τ, so ties can push the realised rate below 1%, never above. Every other split reads τ from
the val result. It is never chosen on, or passed in from, any other split.

### 2.6 Scoring

**`scripts/evaluate_presence.py --checkpoint C --split {val,test,real_dev,real_test}`** scores one family
of files in one pass and writes `outputs/eval/<run>/presence_<split>.json`. For each scene it records
presence, and for positives also `correct`, `hit` and `y_pred`. It is the only script that scores v3a
checkpoints on any test split.

| `--split` | Positives | Negatives | Also |
|---|---|---|---|
| `val` | val | val_empty | val_decoy_plain, val_decoy_aug; chooses τ |
| `test` | test | test_empty | |
| `real_dev` | real_dev | real_empty_dev | hybrid_dev (paired); P6 contact sheet |
| `real_test` | real_test | real_empty_test | hybrid_test (paired) |

Reported, for every family:
- **AUROC** of presence, positives vs negatives. This is the chance that a random positive outscores a
  random negative, with ties counted half. Real splits also report it per stratum.
- **False-alarm rate**: the share of negatives with presence ≥ τ.
- **Found rate**: the share of positives with presence ≥ τ *and* peak in box *and* class right.
- **Plain joint accuracy**, ignoring τ, for comparison with every earlier version.

Per family:
- `val`: decoy false-alarm rate for each decoy file.
- `real_*`:
  - said-yes on real vs hybrid for the same signs (exact McNemar, using `real_report.discordant`);
  - the median paired presence difference, hybrid minus real;
  - results for v2's all / unseen subsets (positives only);
  - found rate at matched false-alarm rates of 1% and 5% of that split's negatives. This is a
    descriptive ROC point, never a threshold used for grading.

**Intervals:**
- Synthetic splits: Wilson 95%.
- Real splits: a **photo-clustered bootstrap**, 2,000 resamples with a fixed seed, percentile 95%. A
  resample draws photos with replacement and brings along every positive and negative window of each
  photo drawn. Windows from one photo are not independent, so per-window Wilson intervals would be too
  narrow.

**`scripts/compare_presence.py --a RUN_A --b RUN_B --split {val,real_dev}`** compares two checkpoints on
the same windows:
- It reads both presence JSONs and refuses to compare unless their data SHA-256s match.
- It reports ΔAUROC = AUROC_a − AUROC_b with a paired photo-clustered bootstrap interval: both models are
  scored on each resample. On `val` it uses an ordinary paired bootstrap.
- It reports found at matched false-alarm rates (1%, 5%) for both models.

It replaces draft 1's McNemar at each model's own τ. That comparison was decided mostly by where each τ
landed, not by how well each model separates.

**Test guard:** one rule shared by both scripts (`eval/test_guard.py`). For a run, a test split counts
as scored if *any* of its result files exist:
- `test`: `best_test.json`, `presence_test.json`;
- `real_test`: `best_real_test.json`, `best_hybrid_test.json`, `presence_real_test.json`.

Either script then refuses without `--force`. evaluate.py keeps its current behavior for every run that
has no presence file.

**Who gets scored:**
- v1.5b seeds 0–2 on `val` and `real_dev`.
- v3a seeds 0–2 on all four families. Test families are scored once per seed, last.

### 2.7 Seam check

A pasted sign leaves a seam; real signs don't. A model trained on signs against empties could learn
"seam = sign". Two checks:

1. **Decoys** (seam, no sign): a decoy false-alarm rate far above the plain-empty rate means the model
   says yes to seams.
2. **Real vs hybrid** (same signs, with and without a seam): a seam-reliant model says yes *less* often
   on a real sign window than on its hybrid twin. This is the check that real photos can give. Real
   negatives can't reveal the shortcut, because a seam-reliant model would score them *low*. Draft 1
   had this backwards.

A seam shortcut found here argues for training on decoys in a follow-up (v3a.1, not here).

### 2.8 Predictions (frozen in RESULTS.md before any code)

Every prediction is graded on **v3a seed 0** against **v1.5b seed 0**, the run-B headline checkpoint,
unless the row says otherwise. Seeds 1 and 2 are reported alongside. At the time of writing, no presence
score from any checkpoint has been computed in this project. `infer.py` prints `peak_prob` for single
images, but it has never been run over a set of negatives.

| # | Prediction | Measured on |
|---|---|---|
| P1 | v1.5b, never trained on empties, already separates real sign windows from real negative windows: AUROC between 0.80 and 0.95 (guess 0.88) | real_dev, both strata pooled |
| P2 | v3a separates synthetic signs from empties almost perfectly: val AUROC ≥ 0.995, and test false alarms at τ ≤ 2% | val; test |
| P3 | Saying "no" costs little (one-sided): v3a seed 0's test joint ≥ 97.28% (v1.5b's 97.78% − 0.5), and the 3-seed mean ≥ 97.09% (v1.5b's 97.59% − 0.5). Guess: seed 0 at 97.4% | test, seeds 0–2 |
| P4 | Training on empties improves real separation: v3a real_dev AUROC ≥ 0.95, and ΔAUROC (v3a − v1.5b) > 0 with the clustered 95% interval excluding 0 | real_dev, compare_presence |
| P5 | The synthetic τ transfers imperfectly: real_dev false alarms at τ between 2% and 15% (guess 6%), and real_dev found ≥ 92% | real_dev |
| P6 | At least 10 of the 20 highest-presence real_dev negatives are labeled A or B (an unlabeled real sign) under the §2.9 protocol | real_dev contact sheet |
| P7 | No seam shortcut on synthetic: false alarms at τ ≤ 3% on both val_decoy_plain and val_decoy_aug | val decoys |
| P8 | No seam shortcut on real: v3a's said-yes rate on real_dev signs is at most 3 points below its rate on their hybrid twins (guess 1 point). v1.5b's gap at its own τ is reported, not graded | real_dev vs hybrid_dev |

Real-test is reported but carries no prediction, as v2 showed it runs easier than dev.

### 2.9 P6 labeling protocol (Claude labels, blind to scores)

1. `evaluate_presence --split real_dev` on v3a seed 0 writes `p6_sheet.png`: 40 numbered tiles, with no
   scores and no other marking.
   - Tiles: the 20 highest-presence real_empty_dev windows, plus 20 drawn uniformly (seeded) from the
     rest.
   - Order: shuffled by a seeded permutation.
   - The mapping and scores go to `p6_key.json`, which is not opened until step 3.
2. Claude labels each tile from the sheet image alone into `results/v3a/p6_labels.json`, then commits
   it. The categories are:
   - **A**: an unlabeled sign outside the 43 classes, with at least half its face inside the window;
   - **B**: an unlabeled sign of one of the 43 classes (for example, below GTSDB's size floor), same
     half-face rule;
   - **C**: a sign-like non-sign: a round, triangular or octagonal object in red/blue/white, a traffic
     light, vehicle lights, a logo;
   - **D**: nothing sign-like.

   When unsure between two categories, pick the later letter. That is the conservative choice for P6.
3. Only after that commit does a script join labels to the key. It reports the A+B count among the top
   20 (graded) and among the 20 controls (descriptive).

The commit order makes the blindness checkable in git. Claude wrote the sheet builder but never sees the
scores or which tiles are top-scoring. Anyone can relabel the sheet to check agreement.

### 2.10 Acceptance criteria

- A1. With `n_empty = 0`, `FreshScenes` is byte-identical to the pre-v3a code. Two digests must still
  match:
  - `PRE_AUGMENT_DIGEST`;
  - a new `FRESH_AUG_DIGEST`. It covers indices 0–9 of epochs 1 and 2, built from the test fixtures with
    v1.5b's exact `train_augment` settings, and is recorded from the current code before any refactor.
- A2. With `n_empty = k`:
  - scenes `0..n−k−1` equal the k = 0 scenes;
  - scenes `n−k..n−1` have an all-zero heatmap, `class_id == IGNORE_CLASS` and box `(0, 0, 0, 0)`;
  - the length is n;
  - `k < 0` or `k ≥ n` raises ValueError.
- A3. Empty scene i draws the sign index, background index and crop exactly as sign scene i does. With
  blur_prob 0 and noise_max_std 0, its image equals the resized canvas sign scene i was pasted onto.
- A4. `detection_loss`:
  - On a batch with no empties it matches today's value within rtol 1e-6.
  - With empties, the class loss equals the sum of per-row cross-entropies over sign rows, divided by b.
  - On an all-empty batch the class loss is 0, the total is finite, and backward gives finite
    gradients.
- A5. `auroc` gives:
  - 1.0 for perfect separation;
  - 0.0 when reversed;
  - 0.5 when every score ties;
  - the hand-computed value on a 3-vs-3 case with one tie;
  - ValueError on empty or non-finite input.
- A6. `threshold_at_false_alarm`:
  - at most `floor(rate·n)` negatives are ≥ τ;
  - the next float32 below τ admits more, except when ties at `s_{k+1}` cause it;
  - all-tied scores give a τ that admits none;
  - ValueError if the rate is outside [0, 1) or the scores are empty.
- A7. `cluster_bootstrap` returns the same interval for the same seed. A constant statistic gives
  (c, c). Every resample draws as many clusters as there are clusters.
- A8. Every real negative window:
  - lies inside its scaled photo;
  - touches no box widened by 4 photo-px;
  - has a scale from the split's positive-scale set.

  Band windows are centred within ±16 px of the drawn row unless clamped. Building twice gives
  byte-identical npz files.
- A9. Every val_empty and val_decoy background index (and every decoy patch source) is a val background,
  and every test_empty background is a test one. Decoy patch sizes are val sign ROI sizes.
- A10. For v1.5b seed 0, evaluate_presence's per-scene `correct` and `hit` on real_dev and hybrid_dev
  equal the `per_scene` arrays already stored in its v2 result JSONs, scene for scene.
- A11. `evaluate_presence` on a non-val split fails with a clear error if that checkpoint has no val
  presence result to read τ from.
- A12. The test guard:
  - with either `best_test.json` or `presence_test.json` present for a run, both scripts refuse `test`
    without `--force`;
  - likewise for the three `real_test` files;
  - the existing evaluate.py guard tests still pass.
- A13. Training-epoch class and pointing accuracy are computed over sign rows only. An all-empty batch
  adds no rows.
- A14. `n_train_empty > 0` with `train_scenes: fixed`, or `n_train_empty ≥ n_train`, raises ValueError
  in `epoch_datasets`, before training starts.
- A15. `compare_presence` refuses two presence results whose data SHA-256s differ.

## 3. Interfaces (no bodies)

### `src/findobject/data/dataset.py`
```python
IGNORE_CLASS = -100  # class_id of a scene with no sign; equals torch cross_entropy's default ignore_index

def has_sign(class_id: Tensor) -> Tensor: ...     # bool, same shape: class_id != IGNORE_CLASS

class EmptyScenes(Dataset[Sample]):                # stored empties/decoys: images (+ background_index) in the npz
    def __init__(self, npz_path: Path) -> None: ...  # FileNotFoundError / KeyError / ValueError like SignScenes
```

### `src/findobject/data/synth.py`
```python
def make_empty_scene(bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene: ...
#   box (0, 0, 0, 0), heatmap all zero; crop/resize, blur and noise via the helper shared with make_scene
```
The helper extraction is its own commit, after `FRESH_AUG_DIGEST` is recorded, guarded by both digests.
Decoys need no new synth function: `make_empty_dataset.py` calls `make_scene` with the patch and then
zeroes the heatmap and box.

### `src/findobject/data/fresh.py`
```python
class FreshScenes(Dataset[Sample]):
    def __init__(self, source, cfg, n: int, seed: int, epoch: int, n_empty: int = 0) -> None: ...
    # indices >= n - n_empty are empty; ValueError unless 0 <= n_empty < n
```

### `src/findobject/data/real_negatives.py` (new, pure apart from what it is passed)
```python
class Stratum(Enum): UNIFORM = 0; BAND = 1        # value = k in the RNG key

def plan_negative(photo_size, boxes, scale: float, stratum: Stratum, band_row: float | None,
                  rng, cfg) -> WindowPlan | Skip: ...
def build_real_negatives(photos, positive_scales: np.ndarray, sign_rows: np.ndarray,
                         seed: int, cfg) -> RealNegatives: ...
```
These reuse `real_windows.WindowPlan`, `cut_window` and `Skip`.

### `src/findobject/training/losses.py`
`detection_loss` keeps its signature. The class term becomes the masked sum divided by b (§2.3), and the
`LossParts.class_loss` comment is updated to match.

### `src/findobject/eval/presence.py` (new, pure)
```python
def presence_scores(heatmap_logits: Tensor) -> Tensor: ...                 # [B,1,h,w] -> [B] float32, max sigmoid
def threshold_at_false_alarm(neg: np.ndarray, rate: float) -> np.float32: ...  # §2.5
```

### `src/findobject/eval/stats.py` (additions)
```python
def auroc(pos: np.ndarray, neg: np.ndarray) -> float: ...  # Mann-Whitney, ties half; ValueError if empty/non-finite
def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray,
                      n_resamples: int, rng: np.random.Generator) -> tuple[float, float]: ...
#   stat receives the row indices of one resample; returns the percentile 95% interval
```

### `src/findobject/eval/test_guard.py` (new, pure)
```python
def scored_results(run_dir: Path, split: str) -> list[Path]: ...  # every file that marks this test family as scored
```
`scripts/evaluate.py`'s `refuse_test_rerun` and `evaluate_presence.py` both use it.

### Scripts
| Script | Does |
|---|---|
| `scripts/make_empty_dataset.py` | writes the six npz files in §2.4 and `empty_manifest.json` |
| `scripts/evaluate_presence.py` | §2.6: scores, τ, report JSON, P6 sheet and key; refuses repeat test runs |
| `scripts/compare_presence.py` | §2.6: paired ΔAUROC and matched-false-alarm found rates for two runs |
| `scripts/p6_join.py` | §2.9 step 3: joins committed labels to the key |
| `scripts/train.py` | unchanged; `epoch_datasets` passes `n_train_empty` through |

## 4. Order of work (after approval)

0. **Freeze P1–P8 in RESULTS.md**, its own commit, before any code is written.
1. Record `FRESH_AUG_DIGEST` from the current code (A1), its own commit.
2. Extract the shared background helper from `make_scene` (A1).
3. `IGNORE_CLASS`, `has_sign`, the masked loss, epoch metrics over sign rows (A4, A13).
4. `make_empty_scene`, `FreshScenes(n_empty)`, config plumbing (A2, A3, A14).
5. `real_negatives.py`, `make_empty_dataset.py`, then build the files (A8, A9).
6. `presence.py`, `auroc`, `cluster_bootstrap`, `test_guard.py`, `evaluate_presence.py`,
   `compare_presence.py` (A5–A7, A11, A12, A15).
7. Score v1.5b seeds 0–2 on val and real_dev (A10). These are the P1 measurements.
8. Train v3a seeds 0, 1, 2, about 45 min each, the same step count as v1.5b. Score val and real_dev, and
   run compare_presence. Do the P6 labeling (§2.9). Write the analysis. Then score test and real_test
   once per seed.

**Blast radius:**
- The edits to existing code are:
  - the helper extraction (guarded by A1);
  - the masked class loss (A4);
  - epoch metrics (A13);
  - `refuse_test_rerun` gaining extra files to check (A12, with the existing tests unchanged).
- Existing stored data, checkpoints and results are only read.
- If v3a breaks, nothing earlier is affected. A v3a bug can't change a v1.5b number, because v1.5b is
  only re-scored on val and dev, and A10 checks that re-scoring against its stored outputs.

## 5. Decisions

**Decided in review round 1:**
1. **Replace, not add:** 16,000 sign + 4,000 empty scenes per epoch (§2.2).
2. **P6 is labeled by Claude**, blind to scores, under §2.9.
3. **P1 is a committed guess** (0.80–0.95, guess 0.88), frozen before any presence score exists.

**Recommended; carried over from draft 1, awaiting sign-off:**

4. **Presence from the heatmap peak.** It needs no new layers and carries straight to whole photos. The
   alternatives:
   - a global "is there a sign" head, which doesn't carry over to whole photos;
   - a 44th "background" class read at the peak, which mixes "where" with "what".
5. **τ = 1% false alarms on synthetic val empties.** It is fixed before any real photo is seen, so P5
   measures how a synthetic threshold transfers. Tuning τ on real dev is left to v3b.
6. **Checkpoint selection unchanged: best synthetic val joint.** The selection rule stays identical to
   v1.5b's. The risk is that it ignores presence; if val AUROC swings across epochs, report it.
7. **Three seeds.** P3 is about a cost the size of seed noise (±0.2 points), so one seed can't answer it.
8. **Decoys for diagnosis only, not training.** Measure the seam shortcut first, then fix it if it
   exists.

## 6. Not covered, and known risks

- **Multiple signs.** A fully convolutional heatmap should fire on each sign, but the class is read only
  at the single argmax, and no training scene has two signs. v3b has to decide between multi-sign
  training and reading a class at every peak.
- **The class readout on an empty scene is meaningless.** The model still outputs a class at its peak.
  Callers must check presence first.
- **Two causes behind a P3 cost.** A cost on P3 could come from the empties themselves or from seeing
  20% fewer sign scenes. This design can't separate them. Doing so would take a 16,000-sign,
  no-empties run, which is not planned.
- **Unlabeled signs in real negatives** count as false alarms by GTSDB's definition. P6 measures how
  much of the real rate they explain. Its labels come from a single labeler (Claude), so they are
  checkable but not cross-validated.
- **COCO is filtered only for stop signs.** Other signs in COCO backgrounds sit unlabeled in training
  empties and get trained toward "no sign". This was already true for v1.5b's backgrounds, but empties
  add more of it.
- **Synthetic intervals assume independent scenes**, though COCO backgrounds repeat across scenes.
  Synthetic n is large enough that this barely matters for the graded thresholds.
- **The loss is not bit-identical** to v1.5b's on all-sign batches, only within float rounding (A4).
  v1.5b is never retrained, so nothing depends on bit-identity there.

## 7. Changes from draft 1 (after the fresh-session review)

| Review point | Change |
|---|---|
| P1 was measured (step 6) before predictions were frozen (step 7) | Predictions are frozen at step 0, before code; P1 is a committed guess |
| Real negatives can't reveal a seam shortcut | Added the paired real-vs-hybrid check (§2.7, P8) |
| No digest guards v1.5b's augmented FreshScenes path | `FRESH_AUG_DIGEST`, recorded before any refactor (A1) |
| "On top" changed steps, the LR schedule and the shuffle | Replace: the epoch length is unchanged (§2.2) |
| Mean over sign rows: weight about 1.25× and NaN on all-empty batches | Sum over sign rows / b (§2.3, A4) |
| McNemar at each model's own τ is decided by τ | Paired ΔAUROC with a clustered bootstrap, plus found at matched false-alarm rates (§2.6) |
| Predictions were ambiguous | Seed named for every prediction; P3 one-sided with explicit numbers; P7 absolute; P6 categories and blind protocol (§2.9) |
| Decoys skipped the training paste path | Two decoy sets, both built by calling make_scene itself (§2.4) |
| Real negatives missed where signs sit | Band stratum; scales rebuilt as `min(1, 48/L)` (§2.4) |
| The test guard was per script | One shared guard over all result files (§2.6, A12) |
| No check that the new scorer matches the old | A10 |
| Correlated windows were only noted | Photo-clustered bootstrap on real splits (§2.6) |
| Nits | τ tie rule (§2.5); A3 draws and discards the sign index; widening in photo px; A9 background splits; the `Sample.class_id` contract and its consumers (§2.2). Pairing val_empty with val scenes was dropped: the stored val scenes come from one sequential RNG, so it would need a replay of the whole stream |
