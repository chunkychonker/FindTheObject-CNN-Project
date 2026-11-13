# Results log

One section per version, newest first. Rules for every entry:

- The checkpoint is chosen on **val** (highest val joint accuracy). Test plays no part in any choice.
- Each checkpoint is scored on **test once** (`scripts/evaluate.py` refuses a second run without `--force`).
- Every number is tied to the exact data via the SHA-256 of the `.npz` it came from. Regenerating the data
  means a new test set, so its numbers are not comparable to earlier entries.

Raw result files (all metrics, per-class accuracy, 43×43 confusion matrix) live in `results/<version>/`.

Commit hashes changed on 2026-11-10, when attribution trailers were removed from every commit message.
File contents and commit dates are unchanged, and every hash cited here was updated to its new value.

---

## v3a.1: decoys as training negatives — pre-registered 2026-11-12

### Why

v3a scores pasted things higher than real ones. A background patch pasted where a sign would go sets
off a false alarm 9–19% of the time. A real sign's median presence is 0.11–0.12 below its pasted twin.
Every pasted thing in training was a sign, so the paste edge may have become evidence of a sign.
v3a.1 trains on pasted non-signs (decoys) to test that. Full spec:
[docs/v3a1-spec.md](docs/v3a1-spec.md) (draft 2, approved).

Decoys can teach that an edge is not *enough* for "sign" (D2). They cannot teach that an edge is not
*needed*, because every training sign still has one. D1 tests "needed". So a D2 pass with a D1 fail
points to a seamless paste or real-sign training next, not to a different cause.

### Setup (fixed before any code was written)

- **v3a.1 training:** v3a's config, with the 4,000 empty slots of each epoch replaced by 4,000 decoys.
  - Decoy i is sign scene i with the sign's pixels replaced by a same-size patch from another train
    background, through the training paste.
  - Its target is the same as an empty's: all-zero heatmap, no class.
  - The patch comes from its own RNG stream, and only from backgrounds big enough to cut it.
  - Seeds 0, 1 and 2.
- **Baseline:** v3a, not retrained: `peak_empty-20261109-140021`, `peak_empty_s1-20261109-144931` and
  `peak_empty_s2-20261109-153419`. v1.5b is reported alongside.
- **τ, splits and intervals:** unchanged from v3a.
- **Seam effect:** the median presence difference, hybrid minus real, over the 645 real_dev signs. Its
  interval comes from the photo-clustered bootstrap. Hybrids are smaller than their real twins (median
  32 vs 40 px), and the gap grows as the hybrid gets smaller (v3a s0: 0.057 / 0.088 / 0.158). So the
  seam report also gives:
  - the gap per relative-size group;
  - the share of pairs where the hybrid scores higher;
  - median real and median hybrid presence separately.
- **Grading:** every prediction is graded on v3a.1 seed 0 against v3a seed 0 unless its row says
  otherwise. Test is scored once per seed, after the dev analysis is committed.

### Predictions (frozen before any code; copied from spec §2.5)

| # | Prediction | Measured on |
|---|---|---|
| D1 | The seam effect shrinks, because real signs rise. Seed 0's median presence gap (hybrid − real) is ≤ 0.06 (v3a: 0.119), all three seeds are below v3a's lowest (0.113), and seed 0's median real presence is above v3a's (0.755). Guess: gap 0.05 | real_dev vs hybrid_dev |
| D2 | Decoy rejection carries over to the unseen plain edge: val_decoy_plain false alarms at τ ≤ 3% (v3a: 9.0%). val_decoy_aug is reported, not graded | val decoys |
| D3 | More real signs found at τ: the 3-seed mean real_dev found is ≥ 95% (v3a: 92.8%) | real_dev, seeds 0–2 |
| D4 | Real separation is unchanged: the paired 95% interval of ΔAUROC (v3a.1 − v3a) lies inside ±0.003 | real_dev |
| D5 | No cost on the one-sign task: seed 0 test joint ≥ 97.28% and 3-seed mean ≥ 97.09% (the P3 bars) | test, seeds 0–2 |

Reported but not graded:
- the full seam report;
- found at a 1% real_dev false-alarm point (v3a s0: 94.7%);
- found per native-size bucket;
- val_decoy_aug false alarms;
- real_test.

### Dev results (written before test or real_test was scored)

Runs: `peak_decoy-20261112-011953`, `peak_decoy_s1-20261112-021910` and `peak_decoy_s2-20261112-030852`.
The best checkpoints by val joint came from epochs 18, 19 and 18. All three trained without error, about
50 minutes each.

Seed 0 against v3a seed 0. Brackets are 95% intervals: Wilson or two-sample bootstrap on synthetic
splits, photo-clustered bootstrap on real ones.

| | v3a seed 0 | v3a.1 seed 0 |
|---|---|---|
| τ (1% of val empties) | 0.566 | 0.484 |
| val AUROC, signs vs empties | 0.9991 | 0.9993 [0.9987, 0.9996] |
| val found at τ | 98.46% | 98.04% [97.62, 98.39] |
| val joint (τ ignored) | 98.62% | 98.22% [97.81, 98.55] |
| **val decoy FA, plain paste (D2)** | 8.98% | **1.68% [1.36, 2.08]** |
| val decoy FA, training paste | 11.54% | 2.22% [1.85, 2.67] |
| real_dev AUROC | 0.9967 | 0.9973 [0.9951, 0.9989] |
| real_dev FA at τ | 6/1200 = 0.50% | 9/1200 = 0.75% [0.33, 1.25] |
| real_dev found at τ | 90.08% | 94.42% [92.68, 96.13] |
| real_dev found at the threshold giving 1% FA on real negatives | 94.73% | 96.43% |
| real_dev joint (τ ignored) | 98.14% | 97.98% |
| said yes: real sign vs its hybrid twin (of 645) | 588 vs 643, gap 8.5 pts | 619 vs 643, gap 3.7 pts, McNemar p = 8e-7 |

**Seam report** (`scripts/seam_report.py`): median presence, hybrid minus real, over the same 645 pairs.

| | v3a s0 | v3a.1 s0 | v3a.1 s1 | v3a.1 s2 |
|---|---|---|---|---|
| **all pairs (D1)** | 0.119 [0.109, 0.127] | **0.123 [0.112, 0.132]** | 0.103 [0.090, 0.117] | 0.110 [0.099, 0.120] |
| hybrid larger (148) | 0.057 | 0.051 | 0.033 | 0.036 |
| similar size (139) | 0.088 | 0.093 [0.076, 0.111] | 0.075 | 0.087 |
| hybrid smaller (358) | 0.158 | 0.154 | 0.144 | 0.150 |
| share of pairs where the hybrid scores higher | 0.881 | 0.891 | 0.848 | 0.857 |
| median real presence | 0.755 | 0.752 | 0.778 | 0.757 |
| median hybrid presence | 0.874 | 0.878 | 0.880 | 0.867 |

v3a seeds 1 and 2 had gaps of 0.113 and 0.123, with median real presence 0.762 and 0.747.

**Paired AUROC difference**, v3a.1 minus v3a on the same scenes (`scripts/compare_presence.py`):

| Seed | real_dev ΔAUROC (D4) | val ΔAUROC | real_dev ΔAUROC vs v1.5b |
|---|---|---|---|
| 0 | +0.0006 [−0.0002, +0.0015] | +0.0001 [−0.0002, +0.0005] | +0.0009 [−0.0009, +0.0029] |
| 1 | +0.0015 [−0.0003, +0.0033] | +0.0005 [−0.0001, +0.0011] | +0.0009 [−0.0002, +0.0021] |
| 2 | +0.0011 [+0.0001, +0.0022] | −0.0004 [−0.0008, −0.0001] | +0.0024 [+0.0008, +0.0042] |

**All three seeds**:

| Seed | τ | val decoy FA plain / training paste | real_dev FA | real_dev found | found at 1% real FA | found small / medium / large |
|---|---|---|---|---|---|---|
| v3a.1 s0 | 0.484 | 1.68% / 2.22% | 0.75% | 94.42% | 96.43% | 85.7 / 96.1 / 94.2% |
| v3a.1 s1 | 0.562 | 1.44% / 1.70% | 0.25% | 92.87% | 96.74% | 88.9 / 96.1 / 88.9% |
| v3a.1 s2 | 0.529 | 1.76% / 2.22% | 0.33% | 94.11% | 96.74% | 85.7 / 96.6 / 92.5% |
| v3a s0 | 0.566 | 8.98% / 11.54% | 0.50% | 90.08% | 94.73% | 87.3 / 94.9 / 83.2% |
| v3a s1 | 0.518 | 12.20% / 16.02% | 0.58% | 92.87% | 95.35% | 87.3 / 95.5 / 90.3% |
| v3a s2 | 0.343 | 18.86% / 24.66% | 0.67% | 95.50% | 96.12% | 88.9 / 96.6 / 95.6% |

The native-size buckets hold 63 small, 356 medium and 226 large signs.

Three-seed means, v3a → v3a.1:
- τ: 0.476 → 0.525
- real_dev AUROC: 0.9970 → 0.9981
- real_dev FA: 0.58% → 0.44%
- real_dev found: 92.82% → 93.80%
- plain-decoy FA: 13.3% → 1.6%
- median presence gap: 0.118 → 0.112
- said-yes gap: 5.6 → 4.5 points
- val joint: 98.41% → 98.30%

### Predictions vs outcome (dev)

| # | Prediction | Outcome | |
|---|---|---|---|
| D1 | Seed 0 gap ≤ 0.06, all seeds < 0.113, seed 0 median real presence above 0.755 | Seed 0 gap 0.123 (interval 0.112–0.132); seeds 0.123 / 0.103 / 0.110; median real presence 0.752 | ✗ |
| D2 | val_decoy_plain FA ≤ 3% | 1.68% [1.36, 2.08]; seeds 1 and 2: 1.44%, 1.76% | ✓ |
| D3 | 3-seed mean real_dev found ≥ 95% | 93.80% (v3a 92.82%) | ✗ |
| D4 | ΔAUROC interval inside ±0.003 | [−0.0002, +0.0015] | ✓ |
| D5 | test joint | scored after this commit | — |

### What it means (dev)

**Decoys did what they can do.** Plain-decoy false alarms fell about fivefold, from 9.0% to 1.7%. This
happened in all three seeds (1.4–1.8%, against v3a's 9–19%), and for the plain paste, which no
training scene uses. After v3a.1, an edge with COCO texture inside it is no longer read as a sign.

**The real-vs-hybrid gap did not move.**
- The median gap is 0.123 for seed 0 against v3a's 0.119, and the seed range (0.103–0.123) overlaps
  v3a's (0.113–0.123).
- Real signs barely rose: median real presence is 0.75–0.78 across v3a.1 seeds, against 0.75–0.76 for v3a.
- The gap is unchanged within each size group too, including size-matched pairs (0.093 vs 0.088).

So the size confound doesn't hide a fix: there is no fix to hide.

**Reading the D1/D2 pattern**, as the spec set out before the data (§1):
- A D2 pass with a D1 fail means the paste edge is no longer *enough* for "sign".
- It does not say what the remaining gap is made of. Two explanations fit these numbers equally well:
  1. The model needs the edge *and* sign content, as §1 warned. Decoys cannot remove that, because
     every training sign still has an edge.
  2. The gap was never mainly about the edge. A hybrid also differs from its real twin in finishing
     (brightness change, blur, noise) and in surroundings (COCO vs a road scene).
- v3a.1 cannot tell these apart. A seamless paste, or hybrids made without the finishing step, would
  test them.

**The said-yes gap at τ shrank (8.5 → 3.7 points for seed 0), but that is τ moving, not the cue.**
- v3a.1 seed 0's τ is lower (0.484 vs 0.566), so more real signs clear it.
- The median presence gap, which does not depend on τ, did not change.
- That is why the spec made the τ-free measure primary.

**No cost elsewhere on dev.**
- Real separation is a little better in every seed. ΔAUROC is +0.0006 to +0.0015, and the seed 2
  interval excludes 0.
- Found at a matched 1% real false-alarm rate rose in every seed: 96.4–96.7% against v3a's 94.7–96.1%.
- Real false alarms at τ stayed under 1%.
- val joint fell slightly (98.41% → 98.30% on the 3-seed mean); D5 grades the test equivalent.

**D3 missed its bar but moved the right way.** The 3-seed mean found is 93.80% against v3a's 92.82%.
The seeds are also closer together: τ ranges 0.48–0.56, against v3a's 0.34–0.57. A reason may be that
val decoys no longer decide where empties and signs separate, but I have not checked that.

### Not covered (dev)

- Which of the two explanations above holds. Both need a new kind of hybrid, which is outside v3a.1.
- Decoy content is COCO texture in both training and val. D2 shows the edge style carries over, not
  that the model now reads sign content (spec §2.1).
- One training recipe and one decoy share (all 4,000 slots); no mix of empties and decoys.

### Test (each split scored once per seed, after the dev analysis above was committed)

| v3a.1 seed | τ | test AUROC | test FA at τ | test found | **test joint** | real_test AUROC | real_test FA | real_test found | real said yes vs hybrid |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.484 | 0.9982 [0.9974, 0.9989] | 0.70% (35/5000) [0.50, 0.97] | 96.46% | **97.30%** [96.81, 97.71] | 0.9976 [0.9942, 0.9999] | 3/600 | 96.19% [93.91, 98.25] | 97.6% vs 99.7% (p = 0.03) |
| 1 | 0.562 | 0.9983 | 0.66% (33/5000) | 96.48% | **97.68%** | 0.9972 | 2/600 | 95.16% | 95.2% vs 100.0% (p = 1e-4) |
| 2 | 0.529 | 0.9982 | 0.66% (33/5000) | 96.50% | **97.34%** | 0.9981 | 3/600 | 94.81% | 96.2% vs 99.7% (p = 0.006) |

3-seed mean test joint: **97.44%** (v3a: 97.67%).

| # | Prediction | Outcome | |
|---|---|---|---|
| D5 | seed 0 test joint ≥ 97.28%, 3-seed mean ≥ 97.09% | 97.30% and 97.44% | ✓ |

**Final tally: 3 of 5** (D2, D4, D5). The three that held say that decoys teach the model to reject a
pasted edge, and that this costs nothing measurable. The two that failed (D1, D3) are the ones that
expected real signs to benefit.

- **D5 passed narrowly.** Seed 0 cleared its bar by 0.02 points, and its interval (96.81–97.71) spans
  the bar. The 3-seed mean cleared it by 0.35.
- **Synthetic joint is a little lower than v3a's**: 97.44% against 97.67% on the 3-seed mean, with two
  seeds lower and one higher. On dev, val joint also fell (98.30% against 98.41%). These are separate
  runs, not paired, so I cannot say whether the difference is real. If it is, it is a small cost.
- **Test false alarms at τ fell**: 0.67% against v3a's 1.09% on the 3-seed mean. The synthetic τ still
  carries over to test.
- **Real test repeats dev.**
  - Real found is the same in both models: 95.39% on the 3-seed mean.
  - Real FA at τ is 2–3 per 600, against v3a's 2–7.
  - The median presence gap, hybrid minus real, is 0.101–0.110, against v3a's 0.114–0.117. That is on
    289 pairs with no interval and was not predicted, so I don't read it as a fix.
  - Real signs still get "yes" less often than their hybrid twins, by 2–5 points.

---

## v3a: saying "no sign here" — pre-registered 2026-11-09

### Why

Every training scene so far holds exactly one sign, so the model always finds one. v3b will run it over
whole road photos, which are mostly empty. v3a trains on empty scenes too and reads "is there a sign" from
the heatmap's peak height. Full spec: [docs/v3a-spec.md](docs/v3a-spec.md) (draft 2, approved).

### Setup (fixed before any code was written and before any presence score existed)

- **Presence** = the max over the 32×32 heatmap cells of sigmoid(logit). "Said yes" means presence ≥ τ.
  τ is set per checkpoint so that at most 1% of the 5,000 synthetic val empties score ≥ τ (spec §2.5).
  It is then fixed for every other split.
- **v3a training:** v1.5b's config, except that the last 4,000 of each epoch's 20,000 fresh scenes are
  empty (all-zero heatmap, no class). This keeps v1.5b's step count, learning-rate schedule,
  initialisation and batch order. The class loss is the sum over sign scenes divided by batch size.
  Seeds 0, 1 and 2.
- **Baseline:** v1.5b, not retrained. Seed 0 is `peak_fresh_aug-20261108-024220`; seeds 1 and 2 are
  `peak_fresh_aug_s1-20261108-062248` and `peak_fresh_aug_s2-20261108-071534`.
- **Negatives:**
  - 5,000 synthetic val and 5,000 test empties;
  - 5,000 val decoys each with the plain and the training paste (a background patch pasted where a
    sign would go);
  - real GTSDB windows that touch no labeled sign, one uniform and one at sign height per photo, from
    600 dev and 300 test photos.
- **Intervals:** Wilson on synthetic splits. Real splits use a photo-clustered bootstrap (2,000
  resamples).
- **Grading:** every prediction is graded on v3a seed 0 against v1.5b seed 0 unless its row says
  otherwise. Real-test carries no prediction, and each test split is scored once per seed.

### Predictions (frozen before any code; copied from spec §2.8)

| # | Prediction | Measured on |
|---|---|---|
| P1 | v1.5b, never trained on empties, already separates real sign windows from real negative windows: AUROC between 0.80 and 0.95 (guess 0.88) | real_dev, both strata pooled |
| P2 | v3a separates synthetic signs from empties almost perfectly: val AUROC ≥ 0.995, and test false alarms at τ ≤ 2% | val; test |
| P3 | Saying "no" costs little (one-sided): v3a seed 0's test joint ≥ 97.28% (v1.5b's 97.78% − 0.5), and the 3-seed mean ≥ 97.09% (v1.5b's 97.59% − 0.5). Guess: seed 0 at 97.4% | test, seeds 0–2 |
| P4 | Training on empties improves real separation: v3a real_dev AUROC ≥ 0.95, and ΔAUROC (v3a − v1.5b) > 0 with the clustered 95% interval excluding 0 | real_dev |
| P5 | The synthetic τ transfers imperfectly: real_dev false alarms at τ between 2% and 15% (guess 6%), and real_dev found ≥ 92% | real_dev |
| P6 | At least 10 of the 20 highest-presence real_dev negatives are labeled A or B (an unlabeled real sign), labeled blind under spec §2.9 | real_dev contact sheet |
| P7 | No seam shortcut on synthetic: false alarms at τ ≤ 3% on both decoy sets | val decoys |
| P8 | No seam shortcut on real: v3a's said-yes rate on real_dev signs is at most 3 points below its rate on their hybrid twins (guess 1 point) | real_dev vs hybrid_dev |

"Found" means said yes, *and* peak inside the sign's box, *and* class right.

### Dev results (written before test or real_test was scored)

**Read this first:** "presence" asks only whether the model thinks a 128×128 window holds a sign. Real
negatives are windows that touch no *labeled* GTSDB sign, and some of them hold signs GTSDB does not
label (see P6).

Seed 0 for each model. Brackets are 95% intervals. Synthetic splits use Wilson for rates and a two-sample
bootstrap for AUROC. Real splits use the photo-clustered bootstrap. FA means false alarms (negatives with
presence ≥ τ).

| | v1.5b seed 0 | v3a seed 0 |
|---|---|---|
| τ (1% of val empties) | 0.549 | 0.566 |
| val AUROC, signs vs empties | 0.9988 [0.9981, 0.9994] | 0.9991 [0.9985, 0.9996] |
| val found at τ | 98.40% [98.01, 98.71] | 98.46% [98.08, 98.77] |
| val joint (τ ignored) | 98.46% | 98.62% |
| val decoy FA, plain paste | 12.92% [12.02, 13.88] | 8.98% [8.22, 9.80] |
| val decoy FA, training paste | 15.78% [14.80, 16.82] | 11.54% [10.68, 12.46] |
| real_dev AUROC | 0.9964 [0.9941, 0.9984] | 0.9967 [0.9943, 0.9986] |
| real_dev AUROC, unseen signs | 0.9960 [0.9935, 0.9982] | 0.9962 [0.9935, 0.9982] |
| real_dev AUROC, uniform / sign-height negatives | 0.9962 / 0.9966 | 0.9981 / 0.9952 |
| real_dev FA at τ | 12/1200 = 1.00% [0.50, 1.58] | 6/1200 = 0.50% [0.17, 0.92] |
| real_dev found at τ | 93.02% [90.94, 94.99] | 90.08% [87.56, 92.34] |
| real_dev found at τ, unseen signs | 91.58% [88.93, 94.01] | 88.71% [85.65, 91.55] |
| real_dev joint (τ ignored) | 97.52% | 98.14% |
| real_dev found at the threshold giving 1% / 5% FA on real negatives | 93.2% / 97.4% | 94.7% / 97.7% |
| said yes: real sign vs its hybrid twin (of 645) | 610 vs 644: gap 5.3 pts, McNemar p = 1e-9 | 588 vs 643: gap 8.5 pts, p = 8e-16 |
| median presence, hybrid minus real | +0.111 | +0.119 |

AUROC difference, v3a minus v1.5b on the same scenes, with a paired interval
(`scripts/compare_presence.py`):

| Seed | real_dev ΔAUROC | val ΔAUROC |
|---|---|---|
| 0 | +0.0003 [−0.0015, +0.0022] | +0.0003 [−0.0001, +0.0008] |
| 1 | −0.0005 [−0.0026, +0.0014] | −0.0004 [−0.0011, +0.0002] |
| 2 | +0.0013 [+0.0001, +0.0026] | +0.0003 [−0.0002, +0.0008] |

Three-seed means, v1.5b → v3a:
- τ: 0.548 → 0.476
- real_dev AUROC: 0.9967 → 0.9970
- real_dev FA: 0.69% → 0.58%
- real_dev found: 92.1% → 92.8%
- plain-decoy FA: 13.0% → 13.3%
- seam gap: 6.5 → 5.6 points

v3a's own seeds spread widely, because their τ differs:

| v3a seed | τ | real_dev found | plain-decoy FA |
|---|---|---|---|
| 0 | 0.566 | 90.1% | 9.0% |
| 1 | 0.518 | 92.9% | 12.2% |
| 2 | 0.343 | 95.5% | 18.9% |

A lower τ finds more real signs and fires on more decoys, which is one trade-off curve.

**P6 contact sheet** (`results/v3a/p6_sheet.png`). I labeled it from the image alone and committed the
labels (17e24b9) before `scripts/p6_join.py` read the key. Results are in `results/v3a/p6_result.json`.

| Group | A: sign outside the 43 classes | B: sign of the 43 classes | C: sign-like | D: nothing sign-like |
|---|---|---|---|---|
| top 20 by presence | 6 | 2 | 6 | 6 |
| 20 random controls | 1 | 0 | 4 | 15 |

The one borderline call in the top 20 was tile 19, two notice plates behind a fence. I labeled it C;
counting it as A gives 9/20, which still fails P6. v3a seed 0 has 6 real_dev false alarms at τ:
- 3 are A: a no-parking disc, a pedestrian-crossing square and a P sign;
- 2 are C: a tail light and a green traffic light;
- 1 is D.

### Predictions vs outcome (dev)

| # | Prediction | Outcome | |
|---|---|---|---|
| P1 | v1.5b real_dev AUROC 0.80–0.95 (guess 0.88) | 0.9964 | ✗ far too pessimistic, again |
| P2 | val AUROC ≥ 0.995; test FA ≤ 2% | val 0.9991; test below | ✓ val half |
| P3 | test joint costs ≤ 0.5 pts | graded below | |
| P4 | v3a real_dev AUROC ≥ 0.95, and ΔAUROC > 0 with interval excluding 0 | 0.9967, but Δ = +0.0003 [−0.0015, +0.0022] | ✗ no improvement shown |
| P5 | real_dev FA at τ 2–15% (guess 6%), found ≥ 92% | FA 0.50%, found 90.08% | ✗ both halves; τ is *stricter* on real |
| P6 | ≥ 10 of the top 20 real negatives are A or B | 8 of 20 | ✗ |
| P7 | decoy FA ≤ 3% on both decoy sets | 8.98% and 11.54% | ✗ |
| P8 | real said-yes at most 3 pts below hybrid | 8.5 pts below | ✗ |

### What it means (dev)

- **Training on empties changed almost nothing.** v3a and v1.5b differ by less than the seed-to-seed
  spread on every number. Two of three ΔAUROC intervals cover zero, and seed 1's point estimate is
  negative.

  My explanation, which I have not tested: v1.5b had already learned "no sign here". Its heatmap loss is
  per cell. Each training scene has about 1,015 background cells out of 1,024, and all of them are pushed
  toward zero. An empty scene is just 1,024 more cells of the same kind, so the 4,000 empties add about
  25% more of what the model already sees. P1 is the evidence: v1.5b, never shown an empty scene, already
  had real AUROC 0.996.
- **Real background is easy. A pasted patch is not.**
  - On real_dev negatives, both models fire less often (0.5–1.0%) than on the synthetic empties τ was set
    on (1%).
  - A background patch pasted where a sign would go fires 9–19% of the time.
  - Real signs get "yes" 5–9 points less often than the same sign pixels pasted onto COCO, and their
    median presence is 0.11–0.12 lower.

  So part of what raises the peak is the paste edge itself. That cue is present on every synthetic sign
  and on no real one. v1.5b has it to the same degree, so empties neither created it nor removed it.

  It affects the peak's height, not the naming: real_dev joint is 97.5–98.1% with τ ignored. That is why
  P5 failed in both directions:
  - real negatives fall under τ more easily than expected;
  - real signs also fall under it more easily, so found is 90.1%.
- **Half of the real false alarms are real signs.** GTSDB labels only the 43 GTSRB classes. Parking
  signs, crossing squares and chevron boards are unlabeled, so the measured 0.50% overstates the true
  false-alarm rate. Still, the top 20 is not mostly signs (8/20): lights, logos and plain road scenes fill
  the other 12.
- **For v3b (an idea, not a result):** the direct fix for the seam cue is to train with decoys as
  negatives, so that a pasted rectangle with no sign counts as "no sign". Another option is a paste that
  leaves no edge. Either one, not more empties.

### Not covered (dev)

- Whole-photo search. The model still only sees 128×128 windows.
- One labeler for P6 (me), with no second pass. Only seed 0 has a sheet.
- The "unseen" subset removes exact-frame overlaps only, as in v2.

### Test (each split scored once per seed, after the dev analysis above was committed)

| v3a seed | τ | test AUROC | test FA at τ | test found | **test joint** | real_test AUROC | real_test FA | real_test found | real said yes vs hybrid |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.566 | 0.9977 [0.9967, 0.9985] | 0.90% (45/5000) [0.67, 1.20] | 96.54% | **97.58%** [97.12, 97.97] | 0.9974 [0.9950, 0.9994] | 3/600 | 92.39% [89.24, 95.32] | 93.1% vs 100.0% (p = 2e-6) |
| 1 | 0.518 | 0.9983 | 0.90% (45/5000) | 96.78% | **97.56%** | 0.9981 | 2/600 | 96.54% | 96.9% vs 99.7% (p = 0.02) |
| 2 | 0.343 | 0.9988 | 1.46% (73/5000) | 97.46% | **97.88%** | 0.9985 | 7/600 | 97.23% | 99.0% vs 100.0% (p = 0.25) |

3-seed mean test joint: **97.67%** (v1.5b: 97.59%).

| # | Prediction | Outcome | |
|---|---|---|---|
| P2 | val AUROC ≥ 0.995, and test FA at τ ≤ 2% | val 0.9991, test 0.90% | ✓ |
| P3 | seed 0 test joint ≥ 97.28%, 3-seed mean ≥ 97.09% (guess 97.4%) | 97.58% and 97.67% | ✓ |

**Final tally: 2 of 8** (P2, P3). The two that held are the ones that said "v3a does no harm", and v3a
also did little good.

- On synthetic test, v3a matches v1.5b on joint: seed 0 is 0.20 points lower, and the mean is 0.08 higher.
- The synthetic τ carries over to test empties: 0.90% against the 1% it was set for on val.
- Real test repeats dev. Real FA at τ is low (0.3–1.2%), and real signs get "yes" less often than their
  hybrid twins, by 1–7 points depending on the seed.

---

## v2: real road photos (GTSDB) — 2026-11-09

### Why

Every number so far is on synthetic scenes: a GTSRB sign crop pasted onto a COCO photo. v2 asks how much
of v1.5b's 97.8% survives on real road photos, and separates two causes of any drop. Full spec:
[docs/v2-spec.md](docs/v2-spec.md).

### Setup (fixed before any real or hybrid scene was scored)

- **Model:** v1.5b seed 0 (`peak_fresh_aug-20261108-024220/best.pt`), not retrained. Seeds 1 and 2 only
  for P7.
- **Data:** GTSDB (900 photos, 1360×800, 1,213 labeled signs). Photos 0–599 = **dev**, 600–899 = **test**
  (the official split), and test is scored once per seed.
- **Three tiers, paired on the same signs:**
  - *real*: a 128×128 window cut from the photo around one labeled sign. The sign keeps its native size,
    shrunk only when its longer side exceeds 48 px. The window is placed at random but contains no other
    labeled sign.
  - *hybrid*: the same sign's pixels pasted onto a COCO background exactly like val/test scenes.
  - *synthetic*: the existing test set (97.8%).
- **Windows:** dev 852 signs → 645 windows, test 361 → 289. The rest (207 dev, 72 test) are skipped because
  no window fits around them without another labeled sign; this leaves out tightly clustered signs.
- **Size buckets** (native longer side L) in dev: small L < 24: 63, medium: 356, large L ≥ 48: 226.
- **Data hashes (SHA-256):** real_dev `4dcd6310…`, hybrid_dev `8aad6c35…`, real_test `da057051…`,
  hybrid_test `1bc58def…`. The full hashes are in each result JSON.

### Overlap with GTSRB training (found before scoring)

GTSRB and GTSDB come from the same group, and they share recordings. `scripts/check_gtsdb_overlap.py` found
that **297 of the 1,213 GTSDB signs (24%) appear pixel-for-pixel in a GTSRB training image**: the same
video frame, cropped. The match is exact (mean difference 0.0; the next-closest candidate is above 1.0 on
a 0–255 scale). A calibrated similarity test flags 32 more possible matches. The model has trained on
these exact signs.

So every result is reported on four subsets:

| Subset | Meaning | dev windows | test windows |
|---|---|---|---|
| all | every window (what P1–P7 are graded on) | 645 | 289 |
| unseen | no exact-frame match | 487 | 216 |
| unflagged | neither exact nor similarity-flagged | 470 | 209 |
| seen | exact-frame match | 158 | 73 |

"Unseen" is the honest real-world number. It is still not proof: the same physical sign in another frame
of the same video would pass the exact test. The gap between seen and unseen is a second measure of
memorization, this time on real data. Per-sign flags and the contact sheet are in
`results/v2-gtsdb-overlap/`.

### Predictions (frozen before scoring; unchanged from the spec)

| # | Prediction | Measured on |
|---|---|---|
| P1 | Real-dev joint is between 75% and 92% (point guess 85%) | real_dev, seed 0 |
| P2 | Hybrid-dev joint is ≥ 90%: real sign pixels alone cost under 8 points | hybrid_dev, seed 0 |
| P3 | Real < hybrid on the same signs, McNemar p < 0.05: the real surroundings cost something on top | paired dev |
| P4 | Real pointing drops below 96% (synthetic: 99.8%), because the peak sometimes lands on other objects or unlabeled signs | real_dev |
| P5 | Small signs (`L < 24`) score ≥ 10 points lower in joint than large ones (`L ≥ 48`) | real_dev |
| P6 | Real-test joint falls inside real-dev's 95% interval | real_test, once |
| P7 | All three seeds' real-test joint lies within ±3 points of each other | real_test |

These were written before the overlap was known, and are graded on *all* windows as written. The overlap
probably makes P1 and P2 easier to meet. Dev small has only 63 windows, so P5's interval will be wide.

### Results

**Read this first:** the model is shown a 128×128 window that was chosen to contain one labeled sign. This
measures *finding and naming a sign within a window that has one*, on real pixels. It is not a detector
running over whole road photos.

Seed 0 (the headline model). Brackets are Wilson 95% intervals.

| Tier | Split | n | Joint | Class | Pointing |
|---|---|---|---|---|---|
| synthetic | val / test | 5,000 each | 98.5% / 97.8% | 98.5% / 97.8% | 99.9% / 99.8% |
| hybrid | dev | 645 | 98.3% [97.0, 99.0] | 98.3% | 99.7% |
| **real** | **dev** | 645 | **97.5% [96.0, 98.5]** | 97.5% | 98.6% |
| hybrid | test | 289 | 99.7% [98.1, 99.9] | 99.7% | 100.0% |
| **real** | **test** | 289 | **98.6% [96.5, 99.5]** | 98.6% | 99.0% |

By overlap subset, real joint:

| Subset | dev (seed 0) | test seed 0 | test seed 1 | test seed 2 |
|---|---|---|---|---|
| all | 97.5% (629/645) | 98.6% (285/289) | 98.6% (285/289) | 97.9% (283/289) |
| **unseen** | **97.1% [95.2, 98.3]** (n=487) | 98.6% (213/216) | 98.1% (212/216) | 97.2% (210/216) |
| unflagged | 97.4% (n=470) | 98.6% (n=209) | — | — |
| seen | 98.7% (n=158) | 98.6% (n=73) | — | — |

Real vs hybrid, same signs (pairs where only one tier was right, then exact McNemar p): dev 9 vs 14,
p = 0.40. Test: seed 0 1 vs 4 (p = 0.38), seed 1 3 vs 3 (p = 1), seed 2 0 vs 6 (p = 0.031).

Real dev by native size: small 98.4% [91.5, 99.7] (n=63), medium 97.2% (n=356), large 97.8% (n=226).

Leak check (`scripts/mask_check_real.py`): blanking the sign in every real-dev window drops class accuracy
from 97.5% to 5.4%, below the 8.5% majority-class rate. The model is reading the sign, not its
surroundings.

### Predictions vs outcome

| # | Prediction | Outcome | |
|---|---|---|---|
| P1 | Real-dev joint 75–92% (guess 85%) | 97.5% | ✗ far too pessimistic |
| P2 | Hybrid-dev joint ≥ 90% | 98.3% | ✓ |
| P3 | Real < hybrid, McNemar p < 0.05 | right direction, 9 vs 14, p = 0.40 | ✗ not shown |
| P4 | Real pointing < 96% | 98.6% | ✗ |
| P5 | Small ≥ 10 pts below large | small 98.4%, large 97.8% | ✗ no gap |
| P6 | Real-test inside real-dev's 95% CI | 98.62% vs upper bound 98.47% | ✗ just above it |
| P7 | Three seeds' real-test within ±3 pts | 97.9–98.6% | ✓ |

2 of 7. Every miss was in the same direction: I expected real photos to be much harder than they were.

### What it means

- **Most of v1.5b's accuracy carries over to real photos.** It loses about 1 point from synthetic val (98.5%) to
  real dev (97.5%), and about 1.4 on unseen signs (97.1%). Real test is higher still (98.6%; mean of three
  seeds 98.4%). The synthetic recipe, real sign crops pasted onto real photos, was a better stand-in for
  real windows than I assumed.
- **Why my predictions were off.** Three reasons, all visible in the spec before scoring, which I didn't
  weigh:
  1. GTSRB crops are already real photos from the same kind of camera, roads and year. The hybrid tier
     (98.3%) shows the sign pixels cost almost nothing.
  2. With native sizing, every sign ends up inside the 16–48 px training range. The smallest GTSDB sign is
     17 px, and anything bigger than 48 is shrunk to 48. So P5's size effect had nothing to work with.
  3. Windows contain no other labeled sign, and 24% of signs (the clustered ones) were skipped. That makes
     the task easier than real driving.
- **The real surroundings do cost something, but it's small.** Real scored at or below hybrid in all four
  paired comparisons, and pointing drops from 99.7% to 98.6% on dev. Of the 16 real-dev misses, 9 put the
  peak somewhere other than the sign, usually on an *unlabeled* sign in view: parking, no-stopping, a
  chevron. The rest are very dark, overexposed or faded signs (`results/v2-real/real_dev_errors.png`). With
  only 9–23 disagreeing pairs per comparison, the gap is significant in just one of four (seed 2 test).
- **Memorization on real data: probably about 1–2 points, but not established.** On dev, seen signs score
  98.7% real and 100% hybrid, against 97.1% and 97.7% for unseen. That fits the 0.8-point sign gap v1.5b's
  synthetic probe measured, but the intervals overlap. The headline real number should be the unseen one:
  **97.1% [95.2, 98.3] on dev**.
- **Test came out easier than dev.** Hybrid test is 98.6–100%. That is why P6 failed: test landed above
  dev's interval, not below it.

### Not covered

- Whole-photo detection. The model never searches a 1360×800 photo, and it is never shown a window without
  a sign (it always names one; see the README).
- Clustered signs: the 24% skipped for having another labeled sign too close to fit a clean window.
- "Unseen" is not proven unseen. A different frame of a sign that GTSRB recorded passes the exact test.
- One country, one camera, one group's recordings. GTSDB's class mix is skewed: classes 12, 13, 38 and the
  speed limits dominate.
- Per-class numbers: only 22 of 43 classes have n ≥ 10 windows in dev. They are in the report JSONs.

Files: `results/v2-real/seed{0,1,2}/` (eval JSONs with per-scene outcomes and the reports),
`results/v2-gtsdb-overlap/`.

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
| Code | commit `0277d4d` |

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
| Code | commit `2c71d0d` (the run used that code, uncommitted at the time) |

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
| Code | commit `722c65a` (the run itself used that code minus a logging-only `.detach()` fix) |
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
