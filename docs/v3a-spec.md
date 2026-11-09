# v3a spec: teach the model to say "no sign here"

Status: **draft for review**. Nothing below is implemented. Predictions (§2.7) get frozen in RESULTS.md
before the first v3a model is trained.

## 1. Intent

Every scene the model has ever seen holds exactly one sign, so it always finds one, even in an empty
image. That blocks the next step: running it over a whole road photo (v3b), where most of the
image is empty. v3a teaches the model to say "nothing here" and measures three things:

- how well it can tell sign from no-sign, on synthetic and on real photos;
- what that costs on the existing one-sign task;
- whether the untrained v1.5b heatmap could already tell them apart.

**Presence is read from the heatmap's peak height**, not from a new output. Training adds empty scenes
whose target heatmap is all zero. The heatmap head is fully convolutional, so the same "how tall is
this peak" score works unchanged when v3b runs the model over a 1360×800 photo.

**Out of scope:**
- Whole-photo detection, detection AP, and non-maximum suppression (v3b).
- Scenes with two or more signs (a v3b question; see §6).
- Any change to the network's layers.
- Regenerating the stored val/test sign scenes. They stay byte-identical, so every earlier number stays
  comparable.

## 2. Behavior

### 2.1 Presence score

`presence(x) = max over the 32×32 cells of sigmoid(heatmap_logit)`, a number in (0, 1). A scene counts as
"sign present" when `presence ≥ τ`. τ is chosen once per checkpoint on synthetic **val** (§2.5) and then
fixed for every other split.

On a sign scene the target peak is 0.78–1.0. It depends on where the sign center falls relative to the
cell grid, since the Gaussian has σ = 1 cell. On an empty scene every target cell is 0.

### 2.2 Training scenes

- New config key `data.n_train_empty` (int ≥ 0, default 0), valid only with `train_scenes: fresh`.
- Each epoch is `n_train` sign scenes **plus** `n_train_empty` empty scenes. Scene i < n_train is exactly
  the scene v1.5b saw at index i, because each scene's RNG is still `[seed, epoch, i]`. Scenes
  i ≥ n_train are empty. Adding empties therefore never changes a sign scene, and the sign exposure per
  epoch matches v1.5b. Any difference from v1.5b comes from the empties, not from fewer signs.
- An empty scene goes through exactly the background pipeline of a sign scene, with no paste:
  1. a random square crop of a train-split COCO background;
  2. a resize to 128;
  3. blur with probability blur_prob;
  4. Gaussian noise.

  This shares one code path with make_scene, so empties can't be told apart by their noise or blur
  statistics.
- An empty scene's `Sample` has an all-zero heatmap, box `(0, 0, 0, 0)`, and
  `class_id = IGNORE_CLASS` (−100, PyTorch's `cross_entropy` ignore index, defined once). "Has a sign"
  is *derived* from `class_id != IGNORE_CLASS` and never stored separately, so the two can't disagree.
- Proposed v3a config (`configs/peak_empty.yaml`): v1.5b's config plus `n_train_empty: 5000`, so 20% of
  each epoch is empty. Everything else, including the 20 epochs and the cosine schedule, is unchanged.

### 2.3 Loss

- Heatmap BCE over every scene, as today; on an empty scene, every cell is pushed toward 0.
- Class cross-entropy over sign scenes only (`ignore_index`). It is averaged over the sign scenes in the
  batch. A batch with no sign scene gets class loss 0 (not NaN), which still carries a gradient path.
- On a batch with no empties the loss equals today's loss exactly. v1.5b's training is a special case.

### 2.4 Evaluation data (new files; existing npz files untouched)

| File | What | n |
|---|---|---|
| `val_empty.npz` / `test_empty.npz` | synthetic empties from val / test COCO backgrounds, seeded per scene | 5,000 each |
| `val_decoy.npz` | diagnostic, never trained on: an empty scene with a sign-shaped patch of *another* COCO background pasted through make_scene's paste code (same seam and size range, no sign) | 5,000 |
| `real_empty_dev.npz` / `real_empty_test.npz` | real negative windows from GTSDB dev / test photos | ~2 per photo |

**Real negative windows:**
- Up to 2 per photo, on every photo in the split, with or without signs.
- Each is 128×128, placed at random. It must intersect no labeled box, after widening each box by
  4 px per side.
- The window's scale is drawn from that split's own positive-window scales (v2's `s`), so negatives and
  positives get the same resampling blur.
- Same RNG scheme as v2: `[seed, NEG_STREAM, photo_index, k]`. Up to 50 tries; otherwise the window is
  skipped and logged.
- Under GTSDB's definition, a sign outside the 43 classes (parking, no stopping, street names) is
  background. A detection on one counts as a false alarm. This makes real negatives harder than
  synthetic ones, on purpose.

Positives are the existing `val.npz` / `test.npz` (synthetic) and v2's `real_dev.npz` / `real_test.npz`.

### 2.5 Scoring

`scripts/evaluate_presence.py --checkpoint C --split {val,test,real_dev,real_test}` scores the split's
positives together with its negatives. It writes per-scene presence scores and, for positives, the
existing per-scene `correct`/`hit`. The report contains:

- **AUROC** of presence, positives vs negatives. This is the chance that a random sign scene outscores
  a random empty one, and it needs no threshold.
- **τ**: on `val`, set to the smallest threshold with at most 1% false alarms on `val_empty`. It is saved
  in the val result. Every other split reads τ from that file; it is never chosen or passed on the
  command line.
- **False-alarm rate** = share of negatives with presence ≥ τ (Wilson interval).
- **Found rate** = share of positives with presence ≥ τ, peak in box, *and* class right: v1.5b's joint
  accuracy plus "and it said yes". The plain joint accuracy, ignoring τ, is also reported, for
  comparison with every earlier version.
- On `val`, the decoy false-alarm rate as well (§2.6).
- For real splits, results for the all / unseen subsets from v2 (positives only; negatives have no
  overlap question).
- A contact sheet of the 20 highest-scoring negatives on dev splits.

The existing test guard applies: any split ending in `test` is scored once per checkpoint.

**Comparisons with v1.5b** use v1.5b's own τ, which comes from its own val run, on the same windows:
- exact McNemar on false alarms over the paired negatives;
- exact McNemar on found over the paired positives.

v1.5b gets scored by the same script. It has never seen an empty scene, which is the point of the
baseline.

### 2.6 Seam diagnostic

A pasted sign leaves a sharp rectangular or masked seam. Real signs don't. A model trained on signs vs
empties could learn "seam = sign". The decoy set has the seam without the sign. If the decoy false-alarm
rate is far above the plain-empty rate, the model is partly detecting pastes. That would argue for
training on decoys in a follow-up (v3a.1, not here). Real negatives are the final check: they contain
neither seams nor pasted signs.

### 2.7 Predictions (to be frozen in RESULTS.md before training)

| # | Prediction | Measured on |
|---|---|---|
| P1 | v1.5b, never trained on empties, already separates real sign windows from real empty ones: AUROC between 0.80 and 0.95 | real_dev, v1.5b |
| P2 | v3a separates them on synthetic val almost perfectly: AUROC ≥ 0.995; synthetic test false alarms at τ ≤ 2% | val, test |
| P3 | Saying "no" costs little: v3a seed 0's synthetic-test joint is within 0.5 points of v1.5b's 97.8%, and the 3-seed mean is within 0.5 of v1.5b's 97.6% | test, 3 seeds |
| P4 | v3a raises v1.5b's real-dev AUROC to ≥ 0.95 and makes fewer real-dev false alarms than v1.5b (McNemar p < 0.05) | real_dev |
| P5 | The synthetic τ transfers imperfectly: real-dev false alarms between 2% and 15% (guess 6%), and real-dev found ≥ 92% | real_dev |
| P6 | At least 10 of the 20 highest-scoring real-dev negatives contain a real sign outside the 43 classes | real_dev contact sheet |
| P7 | No strong seam shortcut: decoy false alarms ≤ 3× plain-empty false alarms on val | val_decoy |

Test is scored once per seed at the end, as in v2. P1–P7 are graded on dev and synthetic val/test.
Real-test is reported but carries no prediction, as v2 showed it runs easier than dev.

### 2.8 Acceptance criteria

- A1. With `n_train_empty` absent or 0, `FreshScenes` returns byte-identical scenes to today for every
  index, and `PRE_AUGMENT_DIGEST` still matches.
- A2. With `n_train_empty = k`, scenes 0..n_train−1 equal the k = 0 scenes. Scenes n_train..n_train+k−1
  have an all-zero heatmap, `class_id == IGNORE_CLASS` and box `(0, 0, 0, 0)`. Length is n_train + k.
- A3. With blur_prob = 0 and noise_max_std = 0, an empty scene equals the resized random square crop the
  same RNG gives make_scene's canvas.
- A4. `detection_loss` on a batch with no empties equals today's value exactly. With empties, the class
  loss equals the cross-entropy over the sign rows alone. On an all-empty batch the class loss is 0 and
  the total is finite.
- A5. `auroc` gives 1.0 for perfect separation, 0.0 when reversed and 0.5 for all ties, and matches a
  hand-computed 3-vs-3 case.
- A6. Every real negative window lies inside the scaled photo and touches no widened labeled box.
  Building twice gives byte-identical npz files.
- A7. Re-scoring v1.5b on `real_dev` with `evaluate.py` still gives joint 0.9752: sign-scene scoring is
  unchanged.
- A8. `evaluate_presence.py` on a non-val split fails with a clear error if the checkpoint has no val
  result to read τ from.
- A9. A second run on `test` or `real_test` is refused without `--force`.
- A10. Training-epoch class and pointing accuracy are computed over sign scenes only.
- A11. `n_train_empty > 0` with `train_scenes: fixed` raises ValueError before training starts.

## 3. Interfaces (no bodies)

### `src/findobject/data/dataset.py`
```python
IGNORE_CLASS = -100  # class_id of a scene with no sign; equals torch cross_entropy's default ignore_index

def has_sign(class_id: Tensor) -> Tensor: ...     # bool, same shape: class_id != IGNORE_CLASS

class EmptyScenes(Dataset[Sample]):                # stored empties: images only in the npz
    def __init__(self, npz_path: Path) -> None: ...  # FileNotFoundError / KeyError / ValueError like SignScenes
```

### `src/findobject/data/synth.py`
```python
def make_empty_scene(bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene: ...
#   box = (0, 0, 0, 0), heatmap all zero; shares crop/blur/noise with make_scene (one helper)
def make_decoy_scene(patch_src: Image.Image, bg: Image.Image, rng, cfg: SynthConfig) -> Scene: ...
#   diagnostic only: pastes a sign-sized crop of patch_src via make_scene's paste path; heatmap all zero
```
The helper extraction comes first, as its own commit, guarded by `PRE_AUGMENT_DIGEST`, like v2's
`box_heatmap`.

### `src/findobject/data/fresh.py`
```python
class FreshScenes(Dataset[Sample]):
    def __init__(self, source, cfg, n: int, seed: int, epoch: int, n_empty: int = 0) -> None: ...
```

### `src/findobject/data/real_negatives.py` (new, pure apart from what it is passed)
```python
def plan_negative(photo_size, boxes, scale: float, rng, cfg) -> WindowPlan | Skip: ...
def build_real_negatives(photos, positive_scales: np.ndarray, seed: int, cfg) -> RealNegatives: ...
```
It reuses `real_windows.WindowPlan`, `cut_window` and `Skip`.

### `src/findobject/training/losses.py`
`detection_loss` keeps its signature; the class term becomes masked (§2.3).

### `src/findobject/eval/presence.py` (new, pure)
```python
def presence_scores(heatmap_logits: Tensor) -> Tensor: ...             # [B,1,h,w] -> [B], max sigmoid
def threshold_at_false_alarm(neg: np.ndarray, rate: float) -> float: ...  # smallest τ with share(neg >= τ) <= rate
```

### `src/findobject/eval/stats.py` (addition)
```python
def auroc(pos: np.ndarray, neg: np.ndarray) -> float: ...  # Mann-Whitney with ties counted half; ValueError if either is empty
```

### Scripts
| Script | Does |
|---|---|
| `scripts/make_empty_dataset.py` | writes the five npz files in §2.4 and `empty_manifest.json` |
| `scripts/evaluate_presence.py` | §2.5: scores, τ, report JSON, contact sheet; refuses repeat test runs |
| `scripts/train.py` | unchanged, except that it passes `n_train_empty` through `epoch_datasets` |

## 4. Order of work (after approval)

1. Extract the shared background helper from `make_scene` (A1, A13 of v2).
2. `IGNORE_CLASS`, `has_sign`, the masked loss, epoch metrics over sign scenes (A4, A10).
3. `make_empty_scene`, `FreshScenes(n_empty)`, config plumbing (A1–A3, A11).
4. `make_decoy_scene`, `real_negatives.py`, `make_empty_dataset.py` (A6).
5. `presence.py`, `auroc`, `evaluate_presence.py` (A5, A7–A9).
6. Score v1.5b (all 3 seeds) on val and real_dev. This only measures the baseline; no v3a model exists
   yet. Look at the v1.5b contact sheet.
7. **Commit predictions to RESULTS.md**, then train v3a seeds 0, 1, 2 (about 3 h total). Score val and
   real_dev, write the analysis, then score test and real_test once per seed.

**Blast radius:**
- The two edits to existing training code (the helper extraction and the masked class loss) are covered
  by A1 and A4.
- Existing stored data, checkpoints and results are only read.
- `evaluate.py` is untouched; A7 checks that it still gives the same numbers.

## 5. Decisions for review

1. **Presence from the heatmap peak** (recommended). It needs no new layers and carries straight to
   whole photos. Alternatives:
   - a global "is there a sign" head, which doesn't carry over to whole photos;
   - a 44th "background" class read at the peak, which mixes "where" with "what".
2. **5,000 extra empties per epoch (20%), added on top** of the 20,000 sign scenes rather than replacing
   some (recommended). The comparison with v1.5b stays clean, at about 25% more training time.
3. **τ = 1% false alarms on synthetic val empties** (recommended). It is fixed before any real photo is
   seen, so P5 measures how a synthetic threshold transfers. Tuning τ on real dev is left to v3b.
4. **Checkpoint selection unchanged: best synthetic val joint** (recommended). It keeps the selection rule
   identical to v1.5b's. The risk is that it ignores presence. If val AUROC swings across epochs, report
   it.
5. **Three seeds** (recommended). P3 is about a cost the size of seed noise (±0.2 points), so one seed
   can't answer it.
6. **Decoys for diagnosis only, not training** (recommended). Measure the seam shortcut first, then fix it
   if it exists.

## 6. Not covered, and known risks

- **Multiple signs.** A fully convolutional heatmap should fire on each sign, but the class is read only
  at the single argmax, and no training scene has two signs. v3b has to decide between multi-sign
  training and reading a class at every peak.
- **The class readout on an empty scene is meaningless.** The model still outputs a class at its peak.
  Callers must check presence first.
- **Unlabeled signs in real negatives.** These count as false alarms by GTSDB's definition. The model has
  never been taught that a parking sign is "nothing", so the real false-alarm rate will partly measure
  that, not general confusion. P6 and the contact sheet make the split visible.
- **Correlated negatives.** Two windows from one photo are not independent, so the Wilson intervals on
  real false-alarm rates are somewhat too narrow.
- **COCO is filtered only for stop signs.** Other signs in COCO backgrounds sit unlabeled in training
  empties and in the backgrounds of sign scenes, and get trained toward "no sign". This is already true
  for v1.5b's backgrounds, so it is not a new effect, but empties add more of it.
