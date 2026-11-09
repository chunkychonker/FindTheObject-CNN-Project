# FindTheObject-CNN

A small CNN that finds a traffic sign in an image and names it: one output is a 32×32 heatmap of where the
sign is, the other is which of the 43 GTSRB sign classes it is. See [DETAILS.md](DETAILS.md) for the plan.

## Results

Scored once on a held-out test set: 5,000 synthetic scenes, or for v2, real road-photo windows. Full
write-up, setup and caveats in [RESULTS.md](RESULTS.md).

| Version | Test joint acc | Class acc | Pointing acc | Median center error |
|---|---|---|---|---|
| v2 real photos (GTSDB): v1.5b on 289 real 128×128 windows | **98.6%** | 98.6% | 99.0% | 1.80 px |
| v1.5b fresh, augmented training scenes | **97.8%** | 97.8% | 99.8% | 1.58 px |
| v1.5 classify at the heatmap peak | 95.9% | 95.9% | 99.6% | 1.58 px |
| v1 baseline | 94.2% | 94.2% | 99.7% | 1.58 px |

*Joint* = the heatmap's peak lands inside the sign **and** the class is right.

v1.5b retrained with 3 seeds scores **97.6% ± 0.2** test joint (range 97.5–97.8%). The table shows seed 0,
the headline chosen before training (see v1.6 in RESULTS.md).

v2 runs the same model, unchanged, on windows cut from real German road photos (GTSDB) around one
labeled sign. It is not whole-photo detection. 24% of GTSDB signs are exact frames from GTSRB's training
set. On the signs that aren't, real-dev joint is **97.1% [95.2, 98.3]**.

## What the model sees

Random validation scenes from v1.5: eight it gets right (top) and four it gets wrong (bottom row, true class in
red). The orange glow is the predicted heatmap; the cyan ring marks its peak; the green box is the true sign.
In all four misses the model found the sign but named a look-alike.

![v1.5 predictions on validation scenes](docs/demo_val.png)

Regenerate with `python scripts/make_demo_figure.py --checkpoint outputs/checkpoints/<run>/best.pt`.

## Data

Raw downloads go in `data/raw/` (gitignored):

| Dataset | Used for | Put it at |
|---|---|---|
| GTSRB (training + `GTSRB_Final_Test_GT.zip`) | sign crops for synthetic scenes | `data/raw/gtsrb/` |
| COCO val2017 + annotations | backgrounds | `data/raw/coco/` |
| GTSDB `FullIJCNN2013.zip` (1.66 GB) | real road photos, v2 only | `data/raw/gtsdb/`, unzipped to `data/raw/gtsdb/FullIJCNN2013/` |

GTSDB is published at
<https://erda.ku.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html>. The zip used
here has SHA-256 `45faf2894b9fc455d14ecaa7942e976992b55ca54ee7184a2fbaf2cfef9e4c7b`. If you use it in a
publication, cite Houben et al., IJCNN 2013.

## Try it on your own images

```
python scripts/infer.py --checkpoint outputs/checkpoints/<run>/best.pt photo1.jpg photo2.png
```

Prints the predicted sign, its confidence and where the heatmap peaks, and saves an overlay per image to
`outputs/infer/`. Images are center-cropped and resized to 128×128 (`--size` to change). The model was
trained only on synthetic scenes with one sign each, so it always names a sign, even when there is none.
