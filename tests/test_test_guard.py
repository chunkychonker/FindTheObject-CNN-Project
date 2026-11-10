"""A12 (docs/v3a-spec.md §2.10): one test-split guard shared by evaluate.py and evaluate_presence.py."""

import importlib.util
from pathlib import Path

import pytest

from findobject.eval.test_guard import scored_results

TEST_FAMILY = ["best_test.json", "presence_test.json"]
REAL_TEST_FAMILY = ["best_real_test.json", "best_hybrid_test.json", "presence_real_test.json"]


@pytest.mark.parametrize("name", TEST_FAMILY)
def test_any_test_file_marks_test_scored(tmp_path, name):
    (tmp_path / name).write_text("{}")
    assert scored_results(tmp_path, "test") == [tmp_path / name]
    assert scored_results(tmp_path, "real_test") == []  # the families are separate


@pytest.mark.parametrize("name", REAL_TEST_FAMILY)
@pytest.mark.parametrize("split", ["real_test", "hybrid_test"])
def test_any_real_test_file_marks_both_real_test_splits_scored(tmp_path, name, split):
    (tmp_path / name).write_text("{}")
    assert scored_results(tmp_path, split) == [tmp_path / name]
    assert scored_results(tmp_path, "test") == []


@pytest.mark.parametrize("split", ["val", "real_dev", "hybrid_dev"])
def test_non_test_splits_are_never_scored(tmp_path, split):
    for name in TEST_FAMILY + REAL_TEST_FAMILY + [f"presence_{split}.json", f"best_{split}.json"]:
        (tmp_path / name).write_text("{}")
    assert scored_results(tmp_path, split) == []


def test_missing_run_dir_has_nothing_scored(tmp_path):
    assert scored_results(tmp_path / "no_such_run", "test") == []


def _evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_cli", Path(__file__).parents[1] / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("presence,split", [("presence_test.json", "test"), ("presence_real_test.json", "real_test"),
                                            ("presence_real_test.json", "hybrid_test"),
                                            ("best_real_test.json", "hybrid_test")])
def test_evaluate_refuses_a_test_split_another_file_already_scored(tmp_path, presence, split):
    (tmp_path / presence).write_text("{}")
    with pytest.raises(SystemExit) as exc:
        _evaluate_script().refuse_test_rerun(tmp_path / f"best_{split}.json", split, force=False)
    assert exc.value.code == 2


def test_evaluate_force_overrides_and_val_is_never_guarded(tmp_path):
    (tmp_path / "presence_test.json").write_text("{}")
    cli = _evaluate_script()
    cli.refuse_test_rerun(tmp_path / "best_test.json", "test", force=True)  # must not raise
    (tmp_path / "presence_val.json").write_text("{}")
    cli.refuse_test_rerun(tmp_path / "best_val.json", "val", force=False)
