"""A11, A12 (evaluate_presence side), A15 (docs/v3a-spec.md §2.10), and the P6 join's commit check."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def presence_cli(tmp_path, monkeypatch):
    cli = _script("evaluate_presence")
    monkeypatch.setattr(cli, "EVAL_ROOT", tmp_path / "eval")
    ckpt = tmp_path / "checkpoints" / "runX" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not a real checkpoint")  # every check below must fire before it is loaded
    return cli, ckpt, tmp_path / "eval" / "runX"


# --- A11: τ comes only from this checkpoint's val result ---

@pytest.mark.parametrize("split", ["test", "real_dev", "real_test"])
def test_non_val_split_without_val_result_fails_clearly(presence_cli, split):
    cli, ckpt, _ = presence_cli
    with pytest.raises(FileNotFoundError, match="--split val"):
        cli.main(["--checkpoint", str(ckpt), "--split", split])


def test_val_result_from_another_checkpoint_is_refused(presence_cli):
    cli, ckpt, run_dir = presence_cli
    run_dir.mkdir(parents=True)
    (run_dir / "presence_val.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64, "tau": 0.5}))
    with pytest.raises(ValueError, match="different checkpoint"):
        cli.main(["--checkpoint", str(ckpt), "--split", "real_dev"])


def test_tau_round_trips_bit_for_bit(presence_cli):
    import numpy as np

    cli, ckpt, run_dir = presence_cli
    tau = np.nextafter(np.float32(0.3), np.float32(1))
    run_dir.mkdir(parents=True)
    sha = cli.sha256_file(ckpt)
    (run_dir / "presence_val.json").write_text(json.dumps({"checkpoint_sha256": sha, "tau": float(tau)}))
    got = cli.read_tau(run_dir, sha)
    assert got.dtype == np.float32 and got == tau


# --- A12, evaluate_presence side ---

@pytest.mark.parametrize("existing,split", [("best_test.json", "test"), ("presence_test.json", "test"),
                                            ("best_real_test.json", "real_test"),
                                            ("best_hybrid_test.json", "real_test"),
                                            ("presence_real_test.json", "real_test")])
def test_presence_refuses_an_already_scored_test_family(presence_cli, existing, split):
    cli, ckpt, run_dir = presence_cli
    run_dir.mkdir(parents=True)
    (run_dir / existing).write_text("{}")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--checkpoint", str(ckpt), "--split", split])
    assert exc.value.code == 2


def test_presence_force_passes_the_guard(presence_cli):
    cli, ckpt, run_dir = presence_cli
    run_dir.mkdir(parents=True)
    (run_dir / "best_test.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="--split val"):  # got past the guard, stopped at τ
        cli.main(["--checkpoint", str(ckpt), "--split", "test", "--force"])


@pytest.mark.parametrize("argv", [["--split", "val", "--p6-sheet"], ["--split", "val", "--checkpoint-name", "last"]])
def test_presence_rejects_bad_arguments(presence_cli, argv):
    cli, ckpt, _ = presence_cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["--checkpoint", str(ckpt), *argv])
    assert exc.value.code == 2  # argparse usage error


def test_presence_only_scores_best(presence_cli):
    cli, ckpt, _ = presence_cli
    last = ckpt.with_name("last.pt")
    last.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--checkpoint", str(last), "--split", "val"])
    assert exc.value.code == 2


# --- A15: compare_presence refuses results scored on different data ---

def _result(**shas):
    return {"split": "real_dev", "data": {stem: {"file": f"{stem}.npz", "sha256": sha} for stem, sha in shas.items()}}


def test_compare_accepts_same_data():
    _script("compare_presence").check_same_data(_result(real_dev="a", real_empty_dev="b"),
                                                _result(real_dev="a", real_empty_dev="b"))


@pytest.mark.parametrize("b", [_result(real_dev="a", real_empty_dev="CHANGED"), _result(real_dev="a")])
def test_compare_refuses_different_data(b):
    with pytest.raises(ValueError, match="real_empty_dev"):
        _script("compare_presence").check_same_data(_result(real_dev="a", real_empty_dev="b"), b)


def test_compare_refuses_different_splits():
    a, b = _result(val="a"), _result(val="a")
    b["split"] = "val"
    with pytest.raises(ValueError, match="different splits"):
        _script("compare_presence").check_same_data(a, b)


def test_compare_of_a_model_with_itself_is_zero():
    rng = __import__("numpy").random.default_rng(0)
    pos, neg = rng.random(30).tolist(), rng.random(40).tolist()
    r = {"per_scene": {"positives": {"presence": pos, "correct": [1] * 30, "hit": [1] * 30},
                       "negatives": {"presence": neg}}}
    out = _script("compare_presence").compare(r, r, "val")
    assert out["delta_auroc"] == {"value": 0.0, "lo": 0.0, "hi": 0.0}
    assert out["matched_a"] == out["matched_b"]


# --- P6 join: labels must be committed before the key is read ---

def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_join_refuses_uncommitted_labels(repo):
    labels = repo / "labels.json"
    labels.write_text("{}")
    with pytest.raises(SystemExit, match="must be committed"):
        _script("p6_join").require_committed(Path("labels.json"))


def test_join_refuses_labels_edited_after_commit(repo):
    labels = repo / "labels.json"
    labels.write_text("{}")
    _git(repo, "add", "labels.json")
    _git(repo, "commit", "-qm", "labels")
    _script("p6_join").require_committed(Path("labels.json"))  # committed and clean: passes
    labels.write_text('{"edited": true}')
    with pytest.raises(SystemExit, match="must be committed"):
        _script("p6_join").require_committed(Path("labels.json"))
