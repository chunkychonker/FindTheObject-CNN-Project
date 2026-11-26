# Wrap-up ideas

Notes for finishing the project, written 2026-11-25 after v3b was merged (PR #9). Nothing here is decided
or scheduled. Each idea notes what already exists that it could build on.

## The plan right now

1. **Understand it.** Work through notebooks 00 → 09 (09 is v3b, whole photos). Use a one-page map of the
   pipeline and the key ideas alongside them: heatmap peaks, joint accuracy, presence and τ, FPPI, the
   pyramid, pre-registration.
2. **Build a demo** (below).
3. **Close it out** (below).

## Demo ideas

### Watch it train, or watch it guess, live (the favourite so far)

An animation of the model learning, or of the guessing process, as it happens.

- **Training, live.** Show loss falling, the heatmap on a few fixed scenes sharpening, and the weights
  changing, epoch by epoch.
  - What exists: `scripts/train.py` already logs the loss, weight and gradient histograms, and heatmap
    panels for 8 fixed val scenes to TensorBoard every epoch (`src/findobject/viz/tensorboard.py`).
    Notebook 03 trains a small model in about 2 minutes, which is short enough to watch.
  - Missing: a full run keeps only `best.pt` and `last.pt`, so "watch the weights change" needs a
    snapshot every N steps, saved either by a hook in the training loop or by a separate small run.
  - Candidate visuals: the first conv layer's filters as little images (they usually turn into edge and
    colour detectors), the heatmap on fixed scenes, and the class probabilities for a few signs.
- **Guessing, live.** Step through one forward pass on a whole photo, one stage at a time:
  - the photo at each pyramid scale (1, ½, ¼);
  - the heatmap at each scale;
  - the peaks;
  - suppression removing repeats;
  - the class read at each kept peak;
  - the final answer.
  - What exists: `findobject.inference.detect` does all of this. Notebook 09 shows each step as a static
    figure, so an animation would reuse its code.
- Format to decide: a notebook with live-updating plots (cheapest), a local web page or app (new
  dependency, e.g. Gradio or Streamlit), or a recorded video/GIF to share.

### Other demo ideas

- **Whole-photo demo script.** Run `detect()` on any image and draw a point and class name for every sign.
  `scripts/infer.py` and notebook 08 crop the middle square and answer only "is there one sign here?",
  so this is the missing showcase. Small, because `detect()` already exists.
- **Gallery page.** A static page of GTSDB photos with detections drawn on: hits, misses, and the
  interesting false positives (unlabeled parking signs, traffic lights). Shareable without installing
  anything.
- **Video.** At about 0.12 s per photo on MPS (about 8 frames per second), the model could run over
  German dashcam footage.
- **Known limit to show on purpose:** the 43 classes are German signs, so photos of US signs (Seattle)
  will mostly fail. Show this as a limit rather than discover it during a demo.

## How to frame it (applications)

- Traffic-sign recognition is a real driver-assistance feature. This model points at signs but doesn't
  draw boxes, and it was tested on 900 photos from one country. Frame it as "a careful study of how far a
  simple model goes", not as a product.
- The distinctive part is the method: predictions written down before any code, each test set scored
  once, and failed predictions reported openly.

## Possible next steps

- **YOLO comparison.** Fine-tune an off-the-shelf detector on GTSDB and score both with the same
  matching, to show what 98.5% found is worth. It also teaches the standard toolchain. Note: our model
  outputs points, not boxes, so the usual IoU-based benchmark numbers aren't comparable without this.
- **Open leads, if more versions are ever wanted** (each is low value per hour; see RESULTS.md, v3b):
  - small signs (< 24 px) are the weakest group, at 89% found;
  - most misses get no detection at all, not a wrong class;
  - seed 2 is weaker on stacked signs, unexplained;
  - whole photos find more signs than single windows did, unexplained.
- **Close-out:**
  - a project summary at the top of README.md;
  - a `v1.0` release tag;
  - tidy up old branches, including `backup/*`.
