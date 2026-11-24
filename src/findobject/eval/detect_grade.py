"""Grade v3b's frozen predictions W1-W6 from detect_report outputs (docs/v3b-spec.md §2.6; RESULTS.md v3b). Pure."""

from collections.abc import Sequence
from statistics import mean

from findobject.eval.detect_report import LARGE_FOR_PYRAMID, TAU_SYN, TAU_W

N_SEEDS = 3
# The bars, exactly as frozen in RESULTS.md (2026-11-17).
W1_WITHIN = 0.03  # whole-photo found within 3 points of windowed found, on the windowed signs
W2_MIN_GAP = 0.05  # medium isolated - stacked found >= 5 points
W3_PYRAMID_MIN, W3_SCALE1_MAX = 0.85, 0.60  # found on signs >= LARGE_FOR_PYRAMID px
W4_BAND = (0.4, 1.6)  # dev FPPI at τ_syn
W5_MIN_FOUND = 0.80  # 3-seed mean found at τ_W
W6_MAX_FPPI, W6_MAX_DROP = 0.75, 0.05  # 3-seed mean test FPPI at τ_W; test found >= dev found - 5 points


def _check_runs(reports: Sequence[dict], what: str) -> None:
    if len(reports) != N_SEEDS:
        raise ValueError(f"need {N_SEEDS} {what} reports (seed 0 first), got {len(reports)}")
    if any(r["photo_indices"] != reports[0]["photo_indices"] for r in reports):
        raise ValueError(f"{what} reports cover different photos")
    for r in reports:
        missing = {TAU_SYN, TAU_W} - set(r["at"])
        if missing:
            raise ValueError(f"{what} report lacks thresholds {sorted(missing)}")


def _found(report: dict, threshold: str, group: str = "all") -> float:
    return report["at"][threshold]["groups"][group]["found"]["value"]


def grade_dev(reports: Sequence[dict], scale1_seed0: dict) -> dict:
    """W1-W5 pass/fail with the numbers behind each, from the three seeds' dev reports (seed 0 first) and seed
    0's scale-1-only dev report. Raises ValueError if there are not three reports, they or the scale-1 report
    cover different photos, a threshold is missing, seed 0's report has no windowed comparison, or W2's
    interval is undefined. Never rounds: a bar is met or not on the exact value."""
    _check_runs(reports, "dev")
    if scale1_seed0["photo_indices"] != reports[0]["photo_indices"]:
        raise ValueError("the scale-1 report covers different photos")
    s0 = reports[0]["at"][TAU_SYN]

    if "windowed_vs_whole" not in s0:
        raise ValueError("seed 0's dev report has no windowed comparison")
    w = s0["windowed_vs_whole"]
    whole, window = w["whole_found"] / w["n"], w["window_found"] / w["n"]
    w1 = {"pass": abs(whole - window) <= W1_WITHIN, "whole_found": whole, "window_found": window, "n": w["n"],
          "mcnemar_p": w["mcnemar_p"]}

    gaps = [r["at"][TAU_SYN]["differences"]["medium_isolated_minus_stacked"] for r in reports]
    if gaps[0]["ci"] is None:
        raise ValueError("W2's interval is undefined")
    w2 = {"pass": gaps[0]["value"] >= W2_MIN_GAP and gaps[0]["ci"][0] > 0 and all(g["value"] > 0 for g in gaps),
          "gap_seed0": gaps[0]["value"], "ci_seed0": gaps[0]["ci"], "gap_by_seed": [g["value"] for g in gaps],
          "isolated_seed0": _found(reports[0], TAU_SYN, "medium_isolated"),
          "stacked_seed0": _found(reports[0], TAU_SYN, "medium_stacked")}

    big = f"ge{LARGE_FOR_PYRAMID}"
    pyramid, scale1 = _found(reports[0], TAU_SYN, big), _found(scale1_seed0, TAU_SYN, big)
    w3 = {"pass": pyramid >= W3_PYRAMID_MIN and scale1 <= W3_SCALE1_MAX, "pyramid": pyramid, "scale1": scale1,
          "n": s0["groups"][big]["found"]["n"]}

    fppi = s0["fppi"]["value"]
    w4 = {"pass": W4_BAND[0] <= fppi <= W4_BAND[1], "fppi": fppi, "ci": s0["fppi"]["ci"]}

    at_w = [_found(r, TAU_W) for r in reports]
    w5 = {"pass": mean(at_w) >= W5_MIN_FOUND, "mean_found": mean(at_w), "found_by_seed": at_w,
          "tau_w_by_seed": [r["thresholds"][TAU_W] for r in reports]}
    return {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5}


def grade_test(dev_reports: Sequence[dict], test_reports: Sequence[dict]) -> dict:
    """W6 from the three seeds' dev and test reports (seed 0 first). Raises ValueError if either side is not
    three reports on common photos, or a seed's test τ_W is not its dev τ_W."""
    _check_runs(dev_reports, "dev")
    _check_runs(test_reports, "test")
    for d, t in zip(dev_reports, test_reports):
        if d["thresholds"][TAU_W] != t["thresholds"][TAU_W]:
            raise ValueError(f"test τ_W {t['thresholds'][TAU_W]} is not dev τ_W {d['thresholds'][TAU_W]}")
    test_fppi = [t["at"][TAU_W]["fppi"]["value"] for t in test_reports]
    dev_found, test_found = [_found(d, TAU_W) for d in dev_reports], [_found(t, TAU_W) for t in test_reports]
    return {"W6": {"pass": mean(test_fppi) <= W6_MAX_FPPI and mean(test_found) >= mean(dev_found) - W6_MAX_DROP,
                   "mean_test_fppi": mean(test_fppi), "mean_test_found": mean(test_found),
                   "mean_dev_found": mean(dev_found), "test_fppi_by_seed": test_fppi,
                   "test_found_by_seed": test_found}}
