# FindTheObject-CNN

A small CNN that finds a traffic sign in an image and names it: one output is a 32×32 heatmap of where the
sign is, the other is which of the 43 GTSRB sign classes it is. See [DETAILS.md](DETAILS.md) for the plan.

## Results

Scored once on a held-out test set of 5,000 synthetic scenes. Full write-up, setup and caveats in
[RESULTS.md](RESULTS.md).

| Version | Test joint acc | Class acc | Pointing acc | Median center error |
|---|---|---|---|---|
| v1 baseline | **94.2%** | 94.2% | 99.7% | 1.58 px |

*Joint* = the heatmap's peak lands inside the sign **and** the class is right.
