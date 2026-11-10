"""Which result files mark a run's test split as already scored (docs/v3a-spec.md §2.6). Pure apart from
checking that files exist.

Test splits are scored once: re-running them while tweaking turns them into a second val set. A family
counts as scored if *any* of its files exist, whichever script wrote it, so evaluate.py and
evaluate_presence.py cannot each score the same test split once.
"""

from pathlib import Path

# Split -> every file name (in outputs/eval/<run>/) that marks its family as scored.
_TEST_FAMILY = ("best_test.json", "presence_test.json")
_REAL_TEST_FAMILY = ("best_real_test.json", "best_hybrid_test.json", "presence_real_test.json")
_FAMILY = {"test": _TEST_FAMILY, "real_test": _REAL_TEST_FAMILY, "hybrid_test": _REAL_TEST_FAMILY}


def scored_results(run_dir: Path, split: str) -> list[Path]:
    """The existing files in run_dir that mark `split`'s test family as scored; [] for a split that is not
    a test split (val, real_dev, hybrid_dev, ...), which may be re-scored freely. Never raises for a
    missing run_dir (nothing is scored yet)."""
    return [run_dir / name for name in _FAMILY.get(split, ()) if (run_dir / name).exists()]
