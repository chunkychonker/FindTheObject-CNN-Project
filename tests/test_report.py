import importlib.util
from pathlib import Path

import pytest
import torch

from findobject.eval.report import class_report

# 3 classes, 6 scenes. Class 0: 2 of 3 right (one called 2). Class 1: 1 of 1. Class 2: 1 of 2 (one called 0).
Y_TRUE = torch.tensor([0, 0, 0, 1, 2, 2])
Y_PRED = torch.tensor([0, 0, 2, 1, 2, 0])


def test_counts_accuracy_and_confusion_match_hand_counts():
    r = class_report(Y_TRUE, Y_PRED, num_classes=3)
    assert r.counts == [3, 1, 2]
    assert r.accuracy == pytest.approx([2 / 3, 1.0, 0.5])
    assert r.confusion == [[2, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]]


def test_report_is_consistent_with_overall_accuracy():
    r = class_report(Y_TRUE, Y_PRED, num_classes=3)
    overall = (Y_TRUE == Y_PRED).float().mean().item()
    assert sum(n * a for n, a in zip(r.counts, r.accuracy)) / len(Y_TRUE) == pytest.approx(overall)
    assert [sum(row) for row in r.confusion] == r.counts
    assert sum(map(sum, r.confusion)) == len(Y_TRUE)


def test_class_absent_from_split_has_no_accuracy():
    r = class_report(torch.tensor([0, 0]), torch.tensor([0, 3]), num_classes=4)
    assert r.counts == [2, 0, 0, 0]
    assert r.accuracy == [0.5, None, None, None]
    assert r.confusion[0][3] == 1  # predicted classes still count even if no scene is truly that class


@pytest.mark.parametrize("y_true,y_pred", [
    (torch.tensor([0, 1]), torch.tensor([0])),  # length mismatch
    (torch.tensor([0, 3]), torch.tensor([0, 1])),  # true id out of range
    (torch.tensor([0, 1]), torch.tensor([0, -1])),  # predicted id out of range
    (torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)),  # not 1-D
])
def test_bad_inputs_raise(y_true, y_pred):
    with pytest.raises(ValueError):
        class_report(y_true, y_pred, num_classes=3)


# --- scripts/evaluate.py test guard (scripts/ is not a package, so load it by path) ---

def _evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_cli", Path(__file__).parents[1] / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_test_result_is_refused_without_force(tmp_path):
    cli = _evaluate_script()
    existing = tmp_path / "best_test.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        cli.refuse_test_rerun(existing, "test", force=False)
    assert exc.value.code == 2
    assert existing.read_text() == "{}"


@pytest.mark.parametrize("split,force,exists", [("test", True, True), ("val", False, True), ("test", False, False)])
def test_guard_allows_force_val_and_first_run(tmp_path, split, force, exists):
    path = tmp_path / f"best_{split}.json"
    if exists:
        path.write_text("{}")
    _evaluate_script().refuse_test_rerun(path, split, force)  # must not raise
