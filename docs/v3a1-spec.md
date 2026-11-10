# v3a.1 spec: train on decoys, to remove the paste-edge cue

Status: **draft 1, for review.** Nothing below is implemented. Predictions (§2.5) are proposals until
approved; they are then frozen in RESULTS.md as the **first** step, before any code.

## 1. Intent

v3a found that the model treats the edge of a pasted patch as evidence of a sign:

- a background patch pasted where a sign would go (a decoy) sets off a false alarm 9–19% of the time;
- a real sign gets "yes" 3–9 points less often than the same pixels pasted onto COCO, and its median
  presence is 0.11–0.12 lower.

Training on plain empties didn't touch this, because an empty scene has no paste. v3a.1 tests one
hypothesis: **if training shows pasted patches that are not signs, the model stops treating the paste
edge as part of a sign.** The test is measured on real photos, where there is no paste.

The change is a single swap: v3a's 4,000 empty training scenes per epoch become 4,000 decoys. v3a.1
seed s is compared with v3a seed s, so the only difference is what fills slots 16,000–19,999.

**Out of scope:**
- Whole-photo detection (v3b).
- Mixing empties and decoys, or changing the decoy share. That would be a second variable (see §6).
- Any change to the network, the loss, the evaluation files or τ's rule.
- Labeling a new P6 sheet.
- Scoring v3a or v1.5b on any test split again.

## 2. Behavior

### 2.1 Training scenes

- New config key `data.n_train_decoy` (int, default 0), valid only with `train_scenes: fresh`.
  `n_train_empty + n_train_decoy < n_train`, both ints ≥ 0.
- The layout of each epoch's `n_train` scenes is **sign | decoy | empty**:
  - `i < n_train − n_decoy − n_empty`: sign scenes, byte-identical to v1.5b and v3a at the same
    (seed, epoch, i);
  - the next `n_decoy`: decoys;
  - the last `n_empty`: empties, as in v3a.
- **A training decoy is built the way `build_decoys` builds `val_decoy_aug`**, from training data only.
  Its draws, from the scene's RNG `[seed, epoch, i]`:
  1. the sign index. That sign's ROI size (width, height) becomes the patch size; its pixels are never
     loaded;
  2. the scene's background index;
  3. the patch's background index, never the scene's own;
  4. the patch's top-left corner, uniform;
  5. then `make_scene`'s draws, with the patch in place of a sign. It goes through the **training**
     paste: resize to a 16–48 px long side, brightness, rotation, perspective, the soft mask, and the
     30% occlusion.
- A decoy's `Sample` is the same as an empty's: all-zero heatmap, box `(0, 0, 0, 0)`, and
  `class_id = IGNORE_CLASS`. The loss and metrics code is unchanged; it already masks these rows.
- **Why decoys use the training paste and not the plain one:** training signs use the training paste,
  so a sign and a decoy differ only in what is inside the edge. The plain rectangular paste, used by
  `val_decoy_plain` and every hybrid twin, is never shown in training. That makes it a held-out test
  of whether the lesson carries over to an edge it never saw.
- Config: `configs/peak_decoy.yaml`, plus `_s1` and `_s2`. Each is the matching v3a config with
  `n_train_empty: 0` and `n_train_decoy: 4000`. Nothing else changes.

### 2.2 Evaluation

The same as v3a, with no new evaluation code:
- `evaluate_presence.py --split val`, then `--split real_dev`, for each seed;
- `compare_presence.py`, v3a.1 against v3a and against v1.5b, on val and real_dev;
- after the dev analysis is committed, test and real_test once per seed.

τ keeps its rule: the 1% point of `val_empty`, per checkpoint. `val_empty` is still a fair negative set,
since no training empty uses a val background.

### 2.3 What counts as the seam effect

The **median paired presence difference, hybrid minus real**, over the 645 real_dev signs, is the
primary measure. Values so far: v1.5b 0.111 / 0.125 / 0.127, v3a 0.119 / 0.113 / 0.123.

It doesn't depend on τ. v3a's said-yes gap does: v3a seed 2's low τ (0.343) alone shrank its gap to
3.3 points, because nearly everything clears a low threshold. The said-yes gap is still reported.

### 2.4 Failure modes and risks

- **The model learns "COCO texture inside an edge = not a sign" and keeps the edge as sign evidence.**
  Then decoy false alarms fall, but hybrids still outscore reals. D1 is built to catch exactly this,
  and a D2 pass with a D1 fail would mean it happened.
- **Decoys make "yes" harder for small or dim signs,** which also have little content inside the edge.
  This would show as lower real_dev found, and D3 watches for it.
- **Slower data loading:** a decoy opens two backgrounds. If the 4 workers can't keep up, training takes
  longer, but the results are unchanged. v3a took about 50 minutes per seed.
- **Blast radius:** training data only. No existing file, result or checkpoint changes. A bug in the
  decoy path can't touch sign scenes, because A2 pins them byte for byte.

### 2.5 Predictions (proposed; frozen in RESULTS.md before any code)

Graded on **v3a.1 seed 0 against v3a seed 0** unless the row says otherwise.

| # | Prediction | Measured on |
|---|---|---|
| D1 | The seam effect shrinks: median presence (hybrid − real) ≤ 0.06 for seed 0 (v3a: 0.119), and all three seeds below v3a's lowest (0.113). Guess: 0.05 | real_dev vs hybrid_dev |
| D2 | The lesson carries over to the unseen plain paste: val_decoy_plain false alarms at τ ≤ 3% (v3a: 9.0%). val_decoy_aug, which is in-distribution, is reported but not graded | val decoys |
| D3 | More real signs found at τ: real_dev found ≥ 93% (v3a: 90.1%) | real_dev |
| D4 | Real separation is unchanged: ΔAUROC (v3a.1 − v3a) interval includes 0. This is a null prediction: the effect should move pasted scores, not real ones | real_dev, compare_presence |
| D5 | No cost on the one-sign task: seed 0 test joint ≥ 97.28% and 3-seed mean ≥ 97.09% (the P3 bars) | test, seeds 0–2 |

### 2.6 Acceptance criteria

- **A1** With `n_decoy = 0`, FreshScenes is byte-identical to today. `FRESH_AUG_DIGEST` and the existing
  v3a empty-scene tests pass unchanged.
- **A2** With `n_decoy > 0`, every scene below the first decoy is byte-identical to the same scene with
  `n_decoy = 0`. With both `n_decoy` and `n_empty` set, the empties are byte-identical to v3a's empties
  at the same index.
- **A3** A decoy sample has an all-zero heatmap, box `(0, 0, 0, 0)` and `class_id == IGNORE_CLASS`. Its
  image differs from the empty scene at the same (seed, epoch, i), so a paste happened.
- **A4** A decoy's patch comes from a train-split background other than the scene's own. Its cut size
  equals the drawn sign's ROI size.
- **A5** `build_decoys` is refactored onto the shared helpers (§3). The regenerated `val_decoy_plain.npz`
  and `val_decoy_aug.npz` match their current SHA-256 hashes.
- **A6** `epoch_datasets` raises ValueError for `n_train_decoy` with `fixed` scenes, for a non-int or
  negative value, and for `n_train_empty + n_train_decoy ≥ n_train`.
- **A7** A decoy scene is identical across `num_workers` 0 and 2. This is the existing determinism test,
  extended to decoys.

## 3. Interfaces (no bodies)

### `src/findobject/data/synth.py`
```python
def make_decoy_scene(patch: Image.Image, bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene:
    """make_scene with `patch` in place of a sign, then the sign's targets dropped: box (0, 0, 0, 0),
    heatmap all zero. Same draws as make_scene. Raises as make_scene does."""
```

### `src/findobject/data/synth_negatives.py`
```python
def cut_patch(src: Image.Image, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """A (width, height) RGB crop of src at a uniform top-left corner: two draws, x then y.
    Raises ValueError if src is smaller than size."""

def other_index(n: int, exclude: int, rng: np.random.Generator) -> int:
    """A uniform index in [0, n) other than `exclude`: one draw. Raises ValueError if n < 2."""
```
`build_decoys` calls these in its current draw order, so A5 holds.

### `src/findobject/data/fresh.py`
```python
class FreshScenes(Dataset[Sample]):
    def __init__(self, source: SceneSource, cfg: SynthConfig, n: int, seed: int, epoch: int,
                 n_empty: int = 0, n_decoy: int = 0) -> None:
        """Raises ValueError as today, and if n_decoy < 0 or n_empty + n_decoy >= n."""
```
This is an expand step: the new argument is keyword-only and defaults to 0, so existing callers are
unchanged.

### `src/findobject/training/train_data.py`
`epoch_datasets` reads `data.n_train_decoy` (default 0) and validates it with `n_train_empty` (A6).

### Configs
`configs/peak_decoy.yaml`, `peak_decoy_s1.yaml`, `peak_decoy_s2.yaml`.

## 4. Steps after approval

1. Freeze D1–D5 in RESULTS.md and commit, before any code.
2. Refactor `build_decoys` onto `cut_patch` and `other_index`; check A5.
3. `make_decoy_scene`, then FreshScenes `n_decoy`, then `epoch_datasets`, each with its tests (A1–A4,
   A6, A7).
4. Configs; train seeds 0, 1 and 2, about 50 minutes each.
5. Score val and real_dev, and run compare_presence. Write the dev analysis and commit it.
6. Score test and real_test once per seed. Grade D5 and commit.

## 5. Cost

- Code: about 3 small functions and 1 constructor argument. No new dependency.
- Compute: 3 trainings, about 2.5 hours total on the MPS GPU, plus scoring, which takes minutes.

## 6. Open questions for review

- **4,000 decoys, or 2,000 decoys plus 2,000 empties?** I recommend all decoys. It changes one thing
  relative to v3a. A decoy is still about 95% plain background, so it keeps teaching "background =
  empty". A mix would need its own comparison.
- **Is D1's bar right?** Halving the median gap (0.119 → ≤ 0.06) is my guess at "the cue is mostly
  gone". If decoys only shave it to about 0.09, D1 fails, and the honest reading is that the cue is
  partly something else. One candidate is the GTSRB camera look of the sign crops, which no decoy
  removes.
