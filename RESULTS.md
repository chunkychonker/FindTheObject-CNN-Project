# Results log

One section per version, newest first. Rules for every entry:

- The checkpoint is chosen on **val** (highest val joint accuracy). Test plays no part in any choice.
- Each checkpoint is scored on **test once** (`scripts/evaluate.py` refuses a second run without `--force`).
- Every number is tied to the exact data via the SHA-256 of the `.npz` it came from. Regenerating the data
  means a new test set, so its numbers are not comparable to earlier entries.

Raw result files (all metrics, per-class accuracy, 43×43 confusion matrix) live in `results/<version>/`.

---

## v1.6: seed variance of v1.5b — 2026-11-08

### Why

Every result so far is one training run per model, so "v1.5b scores 97.78% on test" could partly be a lucky
or unlucky seed. v1.6 retrains v1.5b twice more, changing nothing but `train.seed` (1 and 2), which sets the
initial weights, the batch order and the stream of fresh training scenes. `data.seed` stays 0, so the
splits and the val/test files are unchanged.

### Plan

- Configs `configs/peak_fresh_aug_s1.yaml` and `_s2.yaml`, identical to `configs/peak_fresh_aug.yaml` except
  `train.seed` and `run_name`.
- Each new checkpoint (best val epoch, as always) is scored on val, probed, and scored on test once.
- Report mean and spread over the three seeds. **No selection:** the headline model stays the seed-0 run.
  Picking the best seed after the fact would turn the seed into a tuned hyperparameter.

### Predictions (written before training)

| Prediction | Measure |
|---|---|
| Seeds 1 and 2 land within ±0.3 pts of seed 0's val joint (98.46%) | val joint |
| All three seeds reach ≥ 97.80% val joint, so v1.5b's gain over v1.5 does not depend on the seed | val joint |
| Seeds 1 and 2 land within ±0.5 pts of seed 0's test joint (97.78%) | test joint |
| The sign gap stays at 0.7–1.3 pts in every seed | probe |

If a seed falls below 97.80% on val, the v1.5b headline gets a caveat that its gain depends on the seed.

### Results

Seed 0 is run B from v1.5b (`peak_fresh_aug-20261108-024220`); seeds 1 and 2 are
`peak_fresh_aug_s1-20261108-062248` and `peak_fresh_aug_s2-20261108-071534`. Every number below comes from
the JSONs in `results/v1.5b-fresh-aug/`, `results/v1.6-seed1/` and `results/v1.6-seed2/`. Each test split was
scored once.

| Seed | Best epoch | Val joint | Test joint (errors / 5,000) | Probe: fresh training signs | Sign gap | Weak-class val acc |
|---|---|---|---|---|---|---|
| 0 (headline) | 19 | 98.46% | 97.78% (111) | 99.48% | 1.02 pts | 96.99% |
| 1 | 16 | 98.32% | 97.50% (125) | 99.02% | 0.70 pts | 96.62% |
| 2 | 18 | 98.42% | 97.50% (125) | 99.02% | 0.60 pts | 97.18% |
| **Mean ± sd** | | **98.40 ± 0.07%** | **97.59 ± 0.16%** | 99.17 ± 0.27% | 0.77 ± 0.22 pts | 96.93 ± 0.29% |

(sd = sample standard deviation over 3 runs, so it is itself a rough estimate.)

| Prediction | Result | Held? |
|---|---|---|
| Seeds 1, 2 within ±0.3 pts of 98.46% val | −0.14, −0.04 pts | yes |
| All seeds ≥ 97.80% val | lowest is 98.32% | yes → no caveat needed |
| Seeds 1, 2 within ±0.5 pts of 97.78% test | −0.28, −0.28 pts | yes |
| Sign gap 0.7–1.3 pts in every seed | 1.02, 0.70, 0.60 | **no**: seed 2 is below the range |

### What it means

- **v1.5b's gain over v1.5 is not a lucky seed.** The worst seed (98.32% val, 97.50% test) still beats
  v1.5 (97.30% val, 95.88% test) by a full point on val and 1.6 points on test.
- **The headline is the luckiest of the three seeds on test.** The 3-seed mean is 97.6%, about 0.2 pts below
  the 97.8% headline. The headline stays seed 0 (that was decided before training), but the README now
  shows the mean next to it. The 14-scene difference (111 vs 125 errors) is about the size of the sampling
  noise (±0.24 pts), so seed 0 is not clearly a better model, only a higher draw.
- **Augmentation's edge over plain fresh scenes (v1.5b A vs B) disappears into seed noise.** A scored 98.32%
  val; B's three seeds span 98.32–98.46%, and seed 1 ties A exactly. That fits the McNemar result in v1.5b
  (p = 0.49): choosing B over A was effectively a coin flip. B is kept because the decision rule picked it,
  and it is no worse.
- **The sign gap is smaller than one run suggested.** The prediction failed because seed 0 fit its training
  signs unusually well (99.48% vs 99.02% on fresh training-sign scenes) while val accuracy barely moved.
  Across seeds the gap is 0.77 ± 0.22 pts: some memorization of the physical signs remains, but about
  0.6–1.0 pts, not the ~1 pt the single run showed.
- Hardest classes stay the same across seeds: 21 (double curve) is the worst test class in seeds 1 and 2
  (71% and 64%, n = 42), with 42 and 32 next in both. All three are weak (≤ 12 track) classes, so more
  distinct physical signs (or real-photo data in v2), not more training, is the likely fix.

### Not covered

- 3 seeds give a rough spread, not a confidence interval; a sd of 0.07 from 3 runs could easily be 0.15.
- Only v1.5b was re-seeded. v1 and v1.5 are still single runs, so their numbers carry the same
  ±0.2-ish seed uncertainty (the gaps between versions are much larger than that).

---

## v1.5b: fresh, augmented training scenes — 2026-11-08

### Headline

Same val and test data as v1 and v1.5 (identical SHA-256), so the rows are directly comparable.

| Split | Model | Joint acc | Class acc | Pointing acc | Center error (median / mean) |
|---|---|---|---|---|---|
| **Test** | **v1.5b (run B)** | **97.78%** | 97.78% | 99.78% | 1.58 px / 1.82 px |
| Test | v1.5 | 95.88% | 95.94% | 99.56% | 1.58 px / 1.88 px |
| Val | v1.5b (run B) | 98.46% | 98.48% | 99.90% | 1.58 px / 1.79 px |
| Val | run A (fresh, no augmentation) | 98.32% | 98.34% | 99.92% | 1.58 px / 1.72 px |
| Val | v1.5 | 97.30% | 97.32% | 99.90% | 1.58 px / 1.71 px |

Test errors fell from 206 to 111 (46% fewer); val errors from 135 to 77 (43% fewer). Run A was not scored
on test (only the adopted checkpoint is).

**The gain comes from fresh scenes; augmentation's extra effect is unproven.** Most of the improvement is
already in run A. B's lead over A is small and could be noise (see below). B was adopted because the rule
written before training said "the better of A and B by val joint", and B is better by that measure.

### Why

v1.5 reaches 100% training class accuracy by epoch 9 while val stays at 97.3%. A probe
(`scripts/probe_memorization.py`) splits that 2.7-point gap by scoring v1.5 on 5,000 *new* scenes built
from *training* signs and backgrounds:

| v1.5 on | Joint acc | Gap to the row above |
|---|---|---|
| its own training scenes | 100% | |
| new scenes, training signs | 98.40% | 1.60 pts: scene memorization |
| val (new scenes, unseen signs) | 97.30% | 1.10 pts: sign memorization |

Weak classes (≤ 12 training tracks; 15 of 43) score 95.30% class accuracy on val.

### What changes

Training data only; model, optimizer, schedule, seed, val and test are untouched (same SHA-256s).

- **Run A** (`configs/peak_fresh.yaml`): 20,000 newly generated scenes every epoch instead of the fixed
  `train.npz`. Same generator, same signs and backgrounds, so this targets scene memorization only.
- **Run B** (`configs/peak_fresh_aug.yaml`): A plus per-sign augmentation of the training signs: rotation
  ±10°, perspective warp (corners move up to 8% of the sign's size), contrast and saturation ×0.6–1.4, and a
  30% chance a background patch covers 10–25% of the sign. No flips (they swap left/right meanings) and no
  hue shifts (color carries meaning). This targets sign memorization too.

### Predictions (written before training)

| Prediction | Measure |
|---|---|
| A: val joint > 97.30% (v1.5) | val joint |
| A: sign gap stays about where it was (~1.1 pts), because A shows the same signs the same way | probe: fresh − val joint |
| B: val joint > A's | val joint |
| B: sign gap < 1.0 pt | probe |
| B: weak-class val accuracy > 95.30% | probe |
| A and B: training class accuracy stays below 100% in every epoch | train_epoch/class_acc |

Decision rule: adopt the better of A and B (by val joint) only if its val joint is ≥ 97.80% (v1.5 + 0.5
points, about twice the sampling noise of the 5,000-scene val set). Only an adopted checkpoint is scored on
test, once. Otherwise this entry records a negative result and v1.5 stays the headline.

### Setup (differences from v1.5 only)

| | |
|---|---|
| Training data | 20,000 scenes generated on the fly each epoch (`data.train_scenes: fresh`), scene *i* of epoch *e* seeded by (0, *e*, *i*); 4 loader worker processes. B adds `data.train_augment` (above) |
| Training time | ~40 min per run on MPS (v1.5: ~34); epochs 108–138 s vs ~100 s |
| Checkpoints | B: `best.pt` from `peak_fresh_aug-20261108-024220`, epoch 19 of 20. A: `best.pt` from `peak_fresh-20261108-032133`, epoch 19 of 20 |
| Configs | `configs/peak_fresh_aug.yaml` (B), `configs/peak_fresh.yaml` (A) |
| Code | commit `6403e61` |

### Predictions vs. outcome

| Prediction | v1.5 | A | B | Held? |
|---|---|---|---|---|
| A: val joint > 97.30% | 97.30% | 98.32% | | yes |
| A: sign gap stays ~1.1 pts | 1.10 | 1.28 | | yes (roughly; within probe noise) |
| B: val joint > A's | | 98.32% | 98.46% | yes, but not significant (p = 0.49) |
| B: sign gap < 1.0 pt | 1.10 | | 1.02 | **no** (barely) |
| B: weak-class val acc > 95.30% | 95.30% | 97.93% | 96.99% | yes, but A is higher still |
| Training class acc < 100% every epoch | 100% by epoch 9 | max 99.5% | max 93.6% | yes |
| Best val joint ≥ 97.80% | | | 98.46% | yes → **B adopted** |

### What the numbers say

**Fresh scenes removed scene memorization.** Scene by scene on val (exact McNemar tests):

| Comparison | Errors fixed | New errors | p |
|---|---|---|---|
| v1.5 → A | 76 | 25 | 3.7×10⁻⁷ |
| v1.5 → B | 92 | 34 | 2.4×10⁻⁷ |
| A → B | 41 | 34 | 0.49 |

Run A's accuracy on its own training scenes (99.5% in the last epoch) now matches its accuracy on new
scenes with training signs (99.60%), so the 1.60-point scene-memorization gap v1.5 had is gone.
These tests cover val sampling noise only; there is still one training run per model.

**Sign memorization did not shrink measurably.** The probe's sign gap (new scenes with training signs vs.
val) is 1.10 pts for v1.5, 1.28 for A, 1.02 for B. On 5,000 scenes per side, the gap carries roughly ±0.3
points of sampling noise, so all three are about the same. Augmentation at these strengths does not make the
model treat a training sign as unfamiliar enough to stop memorizing it. Closing that gap probably needs
more distinct physical signs (GTSRB has 6–12 tracks for the weakest classes) or stronger augmentation.

**Weak classes improved, but from fresh scenes, not augmentation.** Class accuracy on val for the 15
classes with ≤ 12 training tracks: v1.5 95.30%, A 97.93%, B 96.99%. That A-vs-B difference is not tested.

**Test improved more than val** (+1.90 pts vs. +1.16). The cause is unknown. Test signs come from
GTSRB's separately recorded test set, so more varied training scenes may transfer better to it, but this
was not tested. Test was scored once, for B only.

**Weakest classes on test** (accuracy, v1.5 for comparison, number of test scenes):

| Class | v1.5b | v1.5 | n |
|---|---|---|---|
| 21 double curve | 76.2% | 57.1% | 42 |
| 42 end no passing (trucks) | 79.4% | 91.2% | 34 |
| 22 bumpy road | 87.3% | 78.2% | 55 |
| 30 ice/snow | 89.7% | 89.7% | 68 |

4 of 43 classes are below 90% (v1.5: 4); 15 are at 100% (v1.5: 5). Pedestrians, v1's worst class at 38.5%,
is now 92.3%. End no passing (trucks) got worse, mostly confused with end no passing (5 scenes; v1.5: 0).

**Most frequent confusions on test** (true → predicted, of 111 class errors):

| True | Predicted | v1.5b | v1.5 |
|---|---|---|---|
| 5 speed 80 | 2 speed 50 | 9 | 5 |
| 22 bumpy road | 25 road work | 6 | 6 |
| 3 speed 60 | 5 speed 80 | 6 | 11 |
| 42 end no passing (trucks) | 41 end no passing | 5 | 0 |

**Val quirk, class 34 (turn left ahead):** 51.2% for v1.5, 88.4% for B (79.1% for A). The hard val track
is not unlearnable after all; it benefits from more varied training scenes.

### Known limits of v1.5b

- **One training run per model.** Seed-to-seed variance is unmeasured, so the A-vs-B ranking could flip
  with another seed.
- **Sign memorization remains** (~1 pt gap; see above).
- **Training is a bit slower** (~40 vs ~34 min) and now depends on the raw GTSRB/COCO files at training
  time, not only on `train.npz`.
- **The augmentation can stack into very dark signs**: contrast 0.6 on an already-dark sign at brightness
  0.6. Not filtered; dark signs were already the largest error group in v1.5.
- Everything listed under v1.5's and v1's known limits still applies.

### Raw results

- `results/v1.5b-fresh-aug/best_test.json`, `best_val.json`, `best_probe.json` (run B)
- `results/v1.5b-fresh/best_val.json`, `best_probe.json` (run A)
- `results/v1.5-peak/best_probe.json` (the v1.5 probe baseline above)

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
