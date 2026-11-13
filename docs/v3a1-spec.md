# v3a.1 spec: train on decoys, to remove the paste-edge cue

Status: **draft 2, for review.** Nothing below is implemented. Draft 1 (e8ba011) was reviewed by a fresh
session, and §7 lists what changed. Predictions (§2.5) are frozen in RESULTS.md as the **first** step
after approval, before any code.

## 1. Intent

v3a found that the model scores pasted things higher than real ones:

- A background patch pasted where a sign would go (a *decoy*) sets off a false alarm 9–19% of the
  time.
- A real sign gets "yes" 3–9 points less often than the same pixels pasted onto COCO. Its median
  presence is 0.11–0.12 lower.

Training on plain empties didn't change this, since an empty scene has no paste. v3a.1 asks a narrow
question: **if training shows pasted patches that are not signs, does the paste edge stop being
evidence of a sign?**

**What decoys can and cannot teach.** After the change, 80% of pasted things in training are signs,
and nothing that isn't pasted is ever a sign.

- Decoys teach that an edge is not *enough*: an edge with the wrong content is not a sign. D2 tests
  this.
- They do not teach that an edge is not *needed*. Every training sign still has one, so the model can
  learn "edge AND sign content". That rule rejects decoys and still marks real signs down.

D1 measures whether the edge is still needed, because a real sign is a sign with no edge. So:

| D2 | D1 | Reading |
|---|---|---|
| passes | fails | The edge is no longer enough, but it is still needed. That is the expected failure of decoys, not proof that the gap has another cause. |
| passes | passes | Decoys removed both. |

The follow-up that tests "needed" directly is a paste with no edge (a seamless blend) or training on
real GTSDB signs. Both are out of scope here.

**The change is one swap:** v3a's 4,000 empty training scenes per epoch become 4,000 decoys. v3a.1 seed
s is compared with v3a seed s, so the two runs differ only in what fills slots 16,000–19,999, plus GPU
nondeterminism.

**Out of scope:**
- Whole-photo detection (v3b).
- Mixing empties and decoys, or changing the decoy share.
- Seamless pastes, or training on real signs.
- Any change to the network, the loss, the evaluation files or τ's rule.
- A new P6 sheet.
- Scoring v3a or v1.5b on any test split again.

## 2. Behavior

### 2.1 Training scenes

- **New config key `data.n_train_decoy`** (int, default 0), valid only with `train_scenes: fresh`. Both
  `n_train_empty` and `n_train_decoy` are ints ≥ 0, and `n_train_empty + n_train_decoy < n_train`.
- **Layout** of each epoch's `n_train` scenes: **sign | decoy | empty**.
  - `i < n_train − n_decoy − n_empty`: sign scenes, byte-identical to v1.5b and v3a at the same
    (seed, epoch, i).
  - The next `n_decoy`: decoys.
  - The last `n_empty`: empties, as in v3a.
- **A decoy is sign scene i with the sign's pixels replaced by a background patch.** The scene's own
  RNG `[seed, epoch, i]` draws exactly what sign scene i draws:
  1. the sign index;
  2. the background index;
  3. then `make_scene`'s draws, with the patch in place of the sign crop, through the **training**
     paste: resize to a 16–48 px long side, brightness, rotation, perspective, the soft mask, the 30%
     occlusion, then blur and noise.

  The patch has the drawn sign's ROI size, so the sign's own crop would have had the same width and
  height. The sign's pixels are never loaded.
- **The patch comes from a separate RNG**, `[seed, epoch, i, PATCH_STREAM]`. It draws:
  1. the patch's source background, uniform over the eligible sources (below), excluding the scene's
     own background;
  2. the patch's top-left corner, x then y, uniform.

  Because the patch has its own stream, decoy i has the same crop, paste geometry, box, blur and noise
  as sign scene i. (This holds if `augment_sign`'s draws depend only on the crop's size. That is
  checked by A3.) Decoy i and sign i are a matched pair: they differ only inside the pasted shape.
- **Eligible patch sources.** These are the train backgrounds whose width and height are both at least
  the largest train-sign ROI (192 × 183). That excludes 8 of the 2,958 backgrounds as patch sources; they
  still appear as scene backgrounds.
  - Without this rule, a patch can be bigger than its source image. That happens with probability
    9.0e-7 per decoy, about a 19% chance of a crash somewhere in the 240,000 decoys of 3 runs. There is
    no resume, so a late crash costs most of a 50-minute run.
  - The eligible list is computed once, at the edge (`train_data.py`), from image headers. It is passed
    in, so FreshScenes does no extra I/O.
- **A decoy's `Sample`** is the same as an empty's: all-zero heatmap, box `(0, 0, 0, 0)`, and
  `class_id = IGNORE_CLASS`. The loss and metrics code is unchanged; it already masks these rows.
- **How this differs from `build_decoys`** (val_decoy_*). That builder draws the background before the
  sign index, and draws the patch from the scene's own stream. Training decoys use the same paste
  recipe in a different draw order. The val files are not rebuilt.
- **Why the training paste.** Training signs use the training paste, so a training sign and a training
  decoy differ only in what is inside the edge. The plain rectangular paste, used by `val_decoy_plain`
  and every hybrid twin, never appears in training, so `val_decoy_plain` holds out the *edge style*. It
  does not hold out the *patch content*: val and training decoys are both COCO crops at GTSRB ROI size.
  A D2 pass can therefore mean "COCO texture inside an edge = no". It does not show that the model now
  reads sign content.
- **Configs:** `configs/peak_decoy.yaml`, plus `_s1` and `_s2`. Each is the matching v3a config with
  `n_train_empty: 0` and `n_train_decoy: 4000`. Nothing else changes.

### 2.2 Evaluation

The same as v3a:
- `evaluate_presence.py --split val`, then `--split real_dev`, for each seed;
- `compare_presence.py`, v3a.1 against v3a and against v1.5b, on val and real_dev;
- after the dev analysis is committed, test and real_test once per seed.

τ keeps its rule, the 1% point of `val_empty` per checkpoint. No training scene uses a val background,
so `val_empty` is still a fair reference. D2 compares every model at its own such point.

One addition: the **seam report** (§2.3), computed from the per-scene arrays that `presence_real_dev.json`
already stores and the size metadata in `real_dev.npz` and `hybrid_dev.npz`.

### 2.3 What counts as the seam effect

**The primary measure** is the median paired presence difference, hybrid minus real, over the 645
real_dev signs. Its 95% interval comes from the photo-clustered bootstrap.
- It does not depend on τ.
- Saturation near 1 is not a problem: median hybrid presence is 0.87–0.88, and only 2–3% of hybrids
  score above 0.95.
- Values so far: v1.5b 0.111 / 0.125 / 0.127, v3a 0.119 / 0.113 / 0.123.

**It is not a pure edge measure.** A hybrid twin also differs from its real sign in two other ways:
- **Size.** Hybrids get a uniform 16–48 px long side (median 32). Real signs keep the size they have in
  their 128 px window (median 40). The two sizes are uncorrelated (r = −0.05). "Size" here and in the
  groups below is the box's long side in the 128 px scene, read from each npz's `boxes`. It is not
  `native_long_side`, which differs for 32% of real signs and drives the size buckets of D3.
- **Finishing.** Hybrids get the brightness change, blur and noise; real windows get none.

The v3a seed 0 gap depends strongly on relative size, and v1.5b shows the same pattern:

| Pairs | n | v3a s0 median gap | v1.5b s0 |
|---|---|---|---|
| hybrid larger than real (by > 4 px) | 148 | 0.057 | 0.040 |
| similar size (±4 px) | 139 | 0.088 | 0.088 |
| hybrid smaller than real (by > 4 px) | 358 | 0.158 | 0.155 |

**The seam report** therefore gives, for each run:
1. the median gap with its interval, for all pairs and for each of the three relative-size groups;
2. the share of pairs where the hybrid scores above the real sign (v3a 0.88, v1.5b 0.88). No
   order-preserving rescaling of the scores can change it;
3. median real presence and median hybrid presence separately (v3a s0: 0.755 and 0.874). If the gap
   shrinks because hybrids fall, the model has lost confidence in general; that is not the cue going
   away. If it shrinks because real signs rise, it is;
4. the said-yes gap at τ, as in v3a.

### 2.4 Failure modes and risks

- **"Edge AND content."** Decoy false alarms fall, but the edge is still needed, so D1 fails (§1).
- **Decoys make "yes" harder for small or dim signs**, which have little content inside the edge.
  This would show as real_dev found falling in the small bucket (63 signs), reported with D3.
- **A gap that shrinks for the wrong reason.** Hybrids could fall instead of real signs rising, or
  only one size group could move. The seam report separates these (§2.3).
- **Slower data loading.** A decoy opens two images. If the 4 workers can't keep up, training takes
  longer, but the results are unchanged.
- **Blast radius: training data only.** No existing data file, result or checkpoint changes. A2 pins
  sign scenes and empties byte for byte, so a bug in the decoy path can't touch them.

### 2.5 Predictions (frozen in RESULTS.md before any code)

Graded on **v3a.1 seed 0 against v3a seed 0** unless the row says otherwise.

| # | Prediction | Measured on |
|---|---|---|
| D1 | The seam effect shrinks, because real signs rise. Seed 0's median presence gap (hybrid − real) is ≤ 0.06 (v3a: 0.119), all three seeds are below v3a's lowest (0.113), and seed 0's median real presence is above v3a's (0.755). Guess: gap 0.05 | real_dev vs hybrid_dev |
| D2 | Decoy rejection carries over to the unseen plain edge: val_decoy_plain false alarms at τ ≤ 3% (v3a: 9.0%). val_decoy_aug is in-distribution, so it is reported but not graded | val decoys |
| D3 | More real signs found at τ: the 3-seed mean real_dev found is ≥ 95% (v3a: 92.8%; its seeds span 90.1–95.5%, so a seed-0 bar would mostly measure where τ lands) | real_dev, seeds 0–2 |
| D4 | Real separation is unchanged: the paired 95% interval of ΔAUROC (v3a.1 − v3a) lies inside ±0.003 | real_dev, compare_presence |
| D5 | No cost on the one-sign task: seed 0 test joint ≥ 97.28% and 3-seed mean ≥ 97.09% (the P3 bars) | test, seeds 0–2 |

**Reported but not graded:**
- the full seam report (§2.3);
- found at the threshold that gives 1% false alarms on real_dev negatives (v3a s0: 94.7%);
- found per native-size bucket (small, medium, large);
- val_decoy_aug false alarms;
- real_test.

### 2.6 Acceptance criteria

- **A1** With `n_decoy = 0`, FreshScenes is byte-identical to today: `FRESH_AUG_DIGEST` and the v3a
  empty-scene tests pass unchanged.
- **A2** With `n_decoy > 0`:
  - every sign scene is byte-identical to the same scene with `n_decoy = 0`;
  - with `n_empty` also set, every empty is byte-identical to the same empty without decoys.
- **A3** A decoy is sign scene i with only its content swapped. With blur off and noise std 0:
  - its pixels equal sign scene i's everywhere outside sign i's box;
  - they differ inside the box;
  - and with these settings no draw depends on pixel content.
- **A4** A decoy's `Sample` has an all-zero heatmap, box `(0, 0, 0, 0)` and `class_id == IGNORE_CLASS`.
- **A5** A decoy's patch comes from an eligible source other than the scene's own background. Its cut
  size equals the drawn sign's ROI size.
- **A6** The eligibility rule makes a decoy failure impossible. A test builds a source list with a
  background smaller than the largest ROI and shows that it is never chosen as a patch source.
  FreshScenes raises ValueError in its constructor when `n_decoy > 0` and there are fewer than 2 eligible
  sources.
- **A7** `epoch_datasets` raises ValueError for `n_train_decoy` with `fixed` scenes, for a non-int or
  negative value, and for `n_train_empty + n_train_decoy ≥ n_train`.
- **A8** A decoy scene is identical with `num_workers` 0 and 2.
- **A9** The seam report's median and relative-size groups reproduce the §2.3 table from v3a seed 0's
  and v1.5b seed 0's stored results. Its interval contains the point estimate.

`build_decoys` and the val decoy files are not touched, so there is no regeneration and nothing to
pin.

## 3. Interfaces (no bodies)

### `src/findobject/data/synth.py`
```python
def make_decoy_scene(patch: Image.Image, bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene:
    """make_scene with `patch` in place of a sign crop, then the sign's targets dropped: box (0, 0, 0, 0),
    heatmap all zero. Draws exactly what make_scene draws. Raises as make_scene does."""
```

### `src/findobject/data/decoy_patch.py` (new, pure)
```python
PATCH_STREAM = 1  # fourth word of a training decoy's patch RNG key; fixed once

def eligible_patch_sources(sizes: Sequence[tuple[int, int]], need: tuple[int, int]) -> np.ndarray:
    """Indices of the sources whose width >= need[0] and height >= need[1], ascending.
    Raises ValueError if fewer than 2 qualify."""

def draw_patch(sources: np.ndarray, scene_bg: int, size: tuple[int, int],
               src_size_of: Callable[[int], tuple[int, int]], rng: np.random.Generator) -> tuple[int, tuple[int, int]]:
    """(source background index, top-left corner) for a size patch: a uniform eligible source other than
    scene_bg, then a uniform corner, x then y. Never fails for a valid eligible list."""
```

### `src/findobject/data/fresh.py`
```python
class FreshScenes(Dataset[Sample]):
    def __init__(self, source: SceneSource, cfg: SynthConfig, n: int, seed: int, epoch: int,
                 n_empty: int = 0, n_decoy: int = 0,
                 patch_sources: DecoySources | None = None) -> None:
        """Raises ValueError as today; also if n_decoy < 0, if n_empty + n_decoy >= n, or if n_decoy > 0
        without patch_sources (or with fewer than 2)."""
```
`DecoySources` is a small frozen dataclass: the eligible indices plus each one's (width, height), read
once from headers. Both new arguments default to "none", so existing callers are unchanged (an expand
step).

### `src/findobject/training/train_data.py`
`epoch_datasets` reads `data.n_train_decoy` (default 0). When it is nonzero, it reads the train
backgrounds' header sizes and builds `DecoySources` from the largest train ROI, before training starts
(A7).

### `src/findobject/eval/seam_report.py` (new, pure) and `scripts/seam_report.py`
```python
def seam_report(real: np.ndarray, hybrid: np.ndarray, real_long: np.ndarray, hybrid_long: np.ndarray,
                photo: np.ndarray, tau: np.float32) -> dict:
    """§2.3 items 1-4. Pairs are row-aligned. Intervals use PhotoIntervals over `photo`.
    Raises ValueError for unequal lengths or an empty group."""
```
The script reads one run's `presence_real_dev.json` and the two npz files' `boxes`. It writes
`outputs/eval/<run>/seam_real_dev.json`.

### Configs
`configs/peak_decoy.yaml`, `peak_decoy_s1.yaml`, `peak_decoy_s2.yaml`.

## 4. Steps after approval

1. Freeze D1–D5 in RESULTS.md and commit, before any code.
2. `seam_report`, with A9 checked on the stored v3a and v1.5b results. This uses existing numbers
   only; no new model is scored.
3. `decoy_patch.py`, `make_decoy_scene`, FreshScenes, `epoch_datasets`, each with its tests (A1–A8).
4. Configs, then train seeds 0, 1 and 2, about 50 minutes each.
5. Score val and real_dev, and run compare_presence and the seam report. Write the dev analysis and
   commit it.
6. Score test and real_test once per seed, grade D5, and commit.

## 5. Cost

- **Code:** two small pure modules, one scene function, one constructor change, one script. No new
  dependency.
- **Compute:** about 2.5 hours of training on the MPS GPU, plus minutes of scoring.

## 6. Open questions for review

- **D1 is graded on all pairs, but the size confound is large.** Size-matched pairs have a 0.088 gap,
  against 0.158 where the hybrid is smaller. If decoys remove the edge cue completely, the size-driven
  part of the gap may stay, and the all-pairs median might not reach 0.06. The alternative is to grade
  D1 on the 139 size-matched pairs: ≤ 0.044, half of 0.088. The cost is a wider interval. Draft 2 keeps
  the approved all-pairs bar and reports the size groups.
- **Why the gap grows with real sign size is unknown.** Possible causes are the shrinking of signs over
  48 px, the real surroundings, or the edge itself. The seam report shows the pattern; it does not
  explain it.

## 7. Changes from draft 1 (after fresh-session review)

| # | Review finding | Change |
|---|---|---|
| 1 | Decoys teach that an edge is not enough, but not that it is not needed. D1 tests "needed", so draft 1's reading of a D1 failure didn't follow | §1 now says what decoys can and cannot teach, gives the D1/D2 readings, and names the follow-up |
| 2 | The primary measure mixes in sign size and finishing (verified: 0.057 / 0.088 / 0.158 by relative size) | §2.3 seam report: size groups, the share of pairs where hybrid scores higher, real and hybrid medians, a photo-clustered interval. D1 now also requires real presence to rise |
| 3 | A seed-0 D3 bar sits inside v3a's own seed spread | D3 graded on the 3-seed mean; matched-false-alarm found and per-size found reported |
| 4 | About a 19% chance of a crash from patches bigger than their source (verified: 8 backgrounds, 9.0e-7 per decoy) | Eligible patch sources computed at the edge; constructor-time errors (A6) |
| 5 | A3 always passed, and the draw order wasn't build_decoys' | Separate patch RNG stream, so decoy i is sign i with its content swapped. A3 checks that. The draw order is stated |
| 6 | Checking A5 would regenerate the val decoy files in place | build_decoys is no longer refactored; its files are untouched |
| 7 | D4 ("interval includes 0") passes almost automatically | Equivalence bound: interval inside ±0.003 |
| 8 | D2 holds out the edge style but not the patch content | Stated in §2.1 |
| 9 | D1 had no interval | Photo-clustered bootstrap in the seam report |
