# Results log

One section per version, newest first. Rules for every entry:

- The checkpoint is chosen on **val** (highest val joint accuracy). Test plays no part in any choice.
- Each checkpoint is scored on **test once** (`scripts/evaluate.py` refuses a second run without `--force`).
- Every number is tied to the exact data via the SHA-256 of the `.npz` it came from. Regenerating the data
  means a new test set, so its numbers are not comparable to earlier entries.

Raw result files (all metrics, per-class accuracy, 43×43 confusion matrix) live in `results/<version>/`.

---

## v1.5 — classify at the heatmap peak — 2026-11-06

### Headline

Same data and test set as v1 (identical SHA-256), so the two rows are directly comparable.

| Split | Model | Joint acc | Class acc | Pointing acc | Center error (median / mean) |
|---|---|---|---|---|---|
| **Test** | **v1.5** | **95.88%** | 95.94% | 99.56% | 1.58 px / 1.88 px |
| Test | v1 | 94.22% | 94.22% | 99.74% | 1.58 px / 1.80 px |
| Val | v1.5 | 97.30% | 97.32% | 99.90% | 1.58 px / 1.71 px |
| Val | v1 | 95.92% | 95.96% | 99.90% | 1.58 px / 1.73 px |

Test errors fell from 289 to 206 (29% fewer); val errors from 204 to 135 (34% fewer).

### What changed

One thing: where the class head gets its input. v1 averaged the trunk's features over all 1,024 heatmap
cells; v1.5 averages only the 3×3 cells around the heatmap's own peak, in training and in eval alike.
Everything else is identical, down to the initial weights (same seed, same parameter names and shapes).

### Setup (differences from v1 only)

| | |
|---|---|
| Model | `PeakCNN` (`models/peak.py`), 588,556 parameters, same as v1. Class head input = mean of the 3×3 trunk-feature window at the heatmap argmax (clipped at borders) |
| Training | identical to v1: 20 epochs, same optimizer, schedule, seed and data; ~34 min on MPS |
| Checkpoint | `best.pt` from run `peak-20261105-195425`, epoch 14 of 20 |
| Config | `configs/peak.yaml` (differs from `configs/baseline.yaml` only in `model.arch` and `run_name`) |
| Code | commit `c167e58` (the run used that code, uncommitted at the time) |

### Predictions vs. outcome

Written down before training, with a decision rule: adopt if val joint beats v1 by at least 0.5 points
(about twice the ±0.28-point sampling noise of a 5,000-scene val set).

| Prediction | v1 | v1.5 | Held? |
|---|---|---|---|
| Class accuracy rises faster early (val, after epoch 1) | 10.2% | 76.6% | yes |
| Fewer val errors on small signs (16–23 px, longer side) | 107 | 69 | yes |
| Val joint ≥ 96.42% | 95.92% | 97.30% | yes → adopted |

### What the numbers say

**The gain is real and broad.** Scene by scene on val, v1.5 fixed 112 of v1's errors and introduced 43 new
ones. If the models were equally good, a split that lopsided would occur with probability 2.8×10⁻⁸ (exact
McNemar test). That covers sampling noise in the val set only: there is one training run of each model,
so seed-to-seed variation is not yet measured (planned for v2). The val plateau is stable (epochs 12–20
all 97.2–97.3%), so the result does not hinge on one lucky epoch.

**It did not help small signs specifically.** Errors fell by a similar fraction across sizes (val):

| Sign size (longer side) | Scenes | v1 errors | v1.5 errors | Change |
|---|---|---|---|---|
| 16–23 px | 1,156 | 107 | 69 | −36% |
| 24–31 px | 1,242 | 36 | 21 | −42% |
| 32–39 px | 1,242 | 19 | 19 | 0% |
| 40–48 px | 1,360 | 42 | 26 | −38% |
| dark (< 40/255) | 1,105 | 85 | 72 | −15% |

The motivating idea was that whole-image averaging dilutes small signs most. The data support "reading
at the peak helps" but not that explanation: small signs are still 51% of val errors (v1: 52%).

**Dark signs are now the largest error group**: 53% of val errors, and they improved least.

**Pointing got slightly worse on test**: 22 misses vs. v1's 13 (mean center error 1.80 → 1.88 px; median
unchanged). Val pointing is unchanged at 99.90%. The counts are small and the cause is unknown.

**Weakest classes on test** (accuracy, number of test scenes, v1 for comparison):

| Class | v1.5 | v1 | n |
|---|---|---|---|
| 21 double curve | 57.1% | 61.9% | 42 |
| 22 bumpy road | 78.2% | 76.4% | 55 |
| 27 pedestrians | 80.8% | 38.5% | 26 |
| 30 ice/snow | 89.7% | 83.8% | 68 |

4 of 43 classes are below 90% (v1: 14); 5 are at 100% (v1: 3). Small classes still swing widely
between val and test: bumpy road is 100.0% on val but 78.2% on test.

**Most frequent confusions on test** (true → predicted, of 203 class errors):

| True | Predicted | v1.5 | v1 |
|---|---|---|---|
| 3 speed 60 | 5 speed 80 | 11 | 15 |
| 8 speed 120 | 3 speed 60 | 10 | 8 |
| 11 right-of-way | 30 ice/snow | 7 | 2 |
| 22 bumpy road | 25 road work | 6 | 0 |
| 21 double curve | 29 bicycles crossing | 6 | 4 |
| 14 stop | 15 no vehicles | 6 | 0 |

The pedestrians → right-of-way confusion that led v1 (15 scenes) is gone from the list; some new ones
appear, e.g. stop → no vehicles (both round-ish with a red border).

### Known limits of v1.5

- **One training run per model.** Seed-to-seed variance is unmeasured.
- **Memorization persists.** Training class accuracy hits 100% by epoch 9 while val stays at 97.3%. The
  training scenes are fixed at generation time; fresh augmentation each epoch is the next planned change.
- **Dark signs** are now the main error source (see above).
- **Val quirk, class 34 (turn left ahead):** 51.2% on val for both v1 and v1.5, but 100% (44/44) on test for
  both. Probably a hard val track (val is split by whole track), not a model weakness; not investigated.
- Everything listed under v1's known limits still applies: synthetic scenes only, one sign per scene.

### Raw results

- `results/v1.5-peak/best_test.json`
- `results/v1.5-peak/best_val.json`

---

## v1 baseline — 2026-11-05

### Headline

| Split | Joint acc | Class acc | Pointing acc | Center error (median / mean) |
|---|---|---|---|---|
| **Test** (5,000 scenes) | **94.22%** | 94.22% | 99.74% | 1.58 px / 1.80 px |
| Val (5,000 scenes) | 95.92% | 95.96% | 99.90% | 1.58 px / 1.73 px |

- **Joint acc**: the sign was both found (heatmap peak inside its box) and named correctly. This is the main number.
- **Pointing acc**: the brightest heatmap cell lies inside the true sign box.
- **Center error**: pixels from the peak cell's center to the true sign center, in a 128×128 scene.

### Setup

| | |
|---|---|
| Data | 20,000 train / 5,000 val / 5,000 test synthetic 128×128 scenes: one GTSRB sign crop (16–48 px) pasted on a COCO val2017 photo, random brightness / blur / noise, fixed at generation time |
| Splits | train/val: GTSRB training set split by whole track (10% of tracks per class to val); test: GTSRB official test set. The three splits use disjoint COCO backgrounds |
| Model | `BaselineCNN`, 588,556 parameters: 8 conv layers (last two dilated), stride 4; 1×1 conv heatmap head (32×32), global-average-pool + linear class head (43) |
| Loss | cross-entropy + 0.1 × heatmap BCE (summed over cells) |
| Training | 20 epochs, AdamW lr 0.001, weight decay 0.0001, cosine decay, batch 64, Apple MPS, ~32 min |
| Checkpoint | `best.pt` from run `baseline-20261103-135333`, epoch 19 of 20 |
| Config | `configs/baseline.yaml` |
| Code | commit `c1b06b2` (the run itself used that code minus a logging-only `.detach()` fix) |
| Test data | `test.npz` SHA-256 `5b12263894a8848abf833d9d8f7c7b12a05ab3e6d4ca915be5ba5f741b30589a` |
| Val data | `val.npz` SHA-256 `86879aff4e9c4615abaad1f4dfc55dc689b262238470e88d5f8c7f647c22a747` |

### What the numbers say

**Finding the sign is close to solved; naming it is where the errors are.**

- Pointing holds up on unseen data: 13 misses in 5,000 test scenes, and every one of those 13 was also
  misclassified (so joint acc = class acc here).
- The median center error of 1.58 px equals the limit of the 32×32 grid: picking the correct cell every time
  gives the same median (each cell is 4 px wide). Doing better needs sub-cell prediction, not a better model.
- Classification drops 1.7 points from val to test (95.96% → 94.22%), which is about expected: test signs come
  from GTSRB's separately recorded test set.

**Weakest classes on test** (accuracy, number of test scenes):

| Class | Acc | n | | Class | Acc | n |
|---|---|---|---|---|---|---|
| 27 pedestrians | 38.5% | 26 | | 32 end of all limits | 76.5% | 17 |
| 21 double curve | 61.9% | 42 | | 19 dangerous curve left | 78.3% | 23 |
| 0 speed limit 20 | 72.7% | 22 | | 42 end of no passing (trucks) | 79.4% | 34 |
| 22 bumpy road | 76.4% | 55 | | 23 slippery road | 80.0% | 40 |

14 of 43 classes are below 90%; 3 are at 100%. All of the weak classes are rare, and each is represented by only
a handful of test scenes, so a single class's figure can swing by tens of points. For example, pedestrians scored
80.6% on val but 38.5% on test, and bumpy road 97.1% on val but 76.4% on test.

**Most frequent confusions on test** (true → predicted, count of 289 errors):

| True | Predicted | Count | Why they look alike |
|---|---|---|---|
| 27 pedestrians | 11 right-of-way | 15 | red-rimmed triangle, dark central figure |
| 3 speed 60 | 5 speed 80 | 15 | one digit differs |
| 8 speed 120 | 7 speed 100 | 11 | one digit differs |
| 22 bumpy road | 26 traffic signals | 10 | red-rimmed triangle, small dark symbol |
| 5 speed 80 | 2 speed 50 | 9 | one digit differs |
| 8 speed 120 | 3 speed 60 | 8 | one digit differs |

### Known limits of v1

- **Synthetic scenes only.** Signs are pasted on everyday photos (food, rooms, animals), not roads. The model has
  never seen a real road scene, a sign with no box-shaped patch around it, a scene with no sign, or a scene with two.
- **Hard cases drive the errors.** On val, signs of 16–23 px are 23% of scenes but 53% of errors; near-black signs
  (mean brightness < 40/255) are 22% of scenes and 42% of errors. These are what real road photos look like, so they
  were kept, not filtered out.
- **Paste-shortcut check: passed, with one leak.** A rectangle cut from another photo, pasted exactly like a sign,
  makes the heatmap fire in 2.1% of scenes, versus 98.9% for a real sign. Uniform *dark* patches, however, fire 59%
  of the time, because 21% of training signs are nearly black. Expect false alarms on dark windows and shadows in
  real photos. (`scripts/shortcut_check.py`)
- **One label per image.** The class head averages over the whole image, so two signs in one scene blend
  together (classifier confidence drops from 0.99 to 0.86 in the shortcut check).

### Raw results

- `results/v1-baseline/best_test.json`
- `results/v1-baseline/best_val.json`
