"""A12 (docs/v3b-spec.md §2.7) and the dev-side reads of scripts/detect_photos.py.

Every refusal must fire before the checkpoint is loaded, so the checkpoint here is a few junk bytes. The
end-to-end cases stub out the slow part (loading the model and running detect) with hand-made photos.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from findobject.data.gtsdb import RealPhoto, RealSign
from findobject.eval.detect_report import TAU_SYN, TAU_W, PhotoRecord
from findobject.eval.photo_match import match_photo
from findobject.eval.photo_truth import photo_truth
from findobject.inference.detection import PYRAMID_SCALES, Detection
from findobject.utils.files import sha256_file

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    cli = _script("detect_photos")
    monkeypatch.setattr(cli, "EVAL_ROOT", tmp_path / "eval")
    ckpt = tmp_path / "checkpoints" / "runX" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not a real checkpoint")
    run_dir = tmp_path / "eval" / "runX"
    run_dir.mkdir(parents=True)
    return cli, ckpt, run_dir


def _write_dev(run_dir, sha, scales=PYRAMID_SCALES, thresholds=None):
    thresholds = thresholds or {TAU_SYN: 0.48, TAU_W: 0.7, "fppi_0.1": 0.9, "fppi_1.0": 0.6}
    (run_dir / "detect_dev.json").write_text(json.dumps(
        {"checkpoint_sha256": sha, "scales": list(scales), "thresholds": thresholds}))


# --- argument checks ---

def test_only_best_pt_is_accepted(cli_env, tmp_path):
    cli, _, _ = cli_env
    other = tmp_path / "checkpoints" / "runX" / "last.pt"
    with pytest.raises(SystemExit) as e:
        cli.main(["--checkpoint", str(other), "--split", "dev"])
    assert e.value.code == 2


def test_test_refuses_scales(cli_env, capsys):
    cli, ckpt, run_dir = cli_env
    _write_dev(run_dir, sha256_file(ckpt))
    with pytest.raises(SystemExit) as e:
        cli.main(["--checkpoint", str(ckpt), "--split", "test", "--scales", "1", "0.5", "0.25"])
    assert e.value.code == 2 and "--scales is dev only" in capsys.readouterr().err


@pytest.mark.parametrize("scales", [["0.5"], ["1", "0.5"], ["0.25", "0.5", "1"]])
def test_dev_takes_only_the_pyramid_or_scale_1(cli_env, scales, capsys):
    cli, ckpt, _ = cli_env
    with pytest.raises(SystemExit) as e:
        cli.main(["--checkpoint", str(ckpt), "--split", "dev", "--scales", *scales])
    assert e.value.code == 2 and "--scales must be" in capsys.readouterr().err


def test_test_takes_no_threshold_argument(cli_env):
    cli, ckpt, run_dir = cli_env
    _write_dev(run_dir, sha256_file(ckpt))
    with pytest.raises(SystemExit) as e:
        cli.main(["--checkpoint", str(ckpt), "--split", "test", "--tau", "0.5"])
    assert e.value.code == 2


# --- A12: the test guard ---

def test_test_refused_when_already_scored(cli_env, capsys):
    cli, ckpt, run_dir = cli_env
    _write_dev(run_dir, sha256_file(ckpt))
    (run_dir / "detect_test.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])
    assert e.value.code == 2 and "already scored" in capsys.readouterr().err


def test_windowed_test_results_do_not_block_whole_photo_test(cli_env):
    # real_test is a different question: its files must not count as photos_test scored. The run then
    # fails for a different reason (no detect_dev.json), which shows the guard let it through.
    cli, ckpt, run_dir = cli_env
    for name in ("presence_real_test.json", "real_test.json", "presence_test.json"):
        (run_dir / name).write_text("{}")
    with pytest.raises(FileNotFoundError, match="--split dev"):
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])


def test_test_needs_detect_dev(cli_env):
    cli, ckpt, _ = cli_env
    with pytest.raises(FileNotFoundError, match="--split dev"):
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])


def test_test_refuses_dev_from_another_checkpoint(cli_env):
    cli, ckpt, run_dir = cli_env
    _write_dev(run_dir, "0" * 64)
    with pytest.raises(ValueError, match="different checkpoint"):
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])


def test_test_refuses_dev_made_at_other_scales(cli_env):
    cli, ckpt, run_dir = cli_env
    _write_dev(run_dir, sha256_file(ckpt), scales=(1.0,))
    with pytest.raises(ValueError, match="used scales"):
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])


# --- dev: what it reads ---

def test_dev_needs_this_checkpoints_val_tau(cli_env):
    cli, ckpt, run_dir = cli_env
    with pytest.raises(FileNotFoundError, match="--split val"):
        cli.main(["--checkpoint", str(ckpt), "--split", "dev"])
    (run_dir / "presence_val.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64, "tau": 0.5}))
    with pytest.raises(ValueError, match="different checkpoint"):
        cli.main(["--checkpoint", str(ckpt), "--split", "dev"])


def _real_dev(tmp_path, run_dir, sha, *, tau=0.5, photo=(3, 3, 7), sign=(0, 1, 0),
              presence=(0.9, 0.9, 0.4), correct=(True, False, True), hit=(True, True, True), json_photo=None):
    npz = tmp_path / "real_dev.npz"
    np.savez(npz, photo_index=np.array(photo, np.int16), sign_index=np.array(sign, np.int16))
    (run_dir / "presence_real_dev.json").write_text(json.dumps({
        "checkpoint_sha256": sha, "tau": tau,
        "data": {"real_dev": {"file": str(npz), "sha256": sha256_file(npz)}},
        "per_scene": {"positives": {"presence": list(presence), "correct": list(correct), "hit": list(hit),
                                    "photo_index": list(json_photo or photo)}}}))
    return npz


def test_read_windowed_keys_by_photo_and_gt_row(cli_env, tmp_path):
    cli, ckpt, run_dir = cli_env
    sha = sha256_file(ckpt)
    _real_dev(tmp_path, run_dir, sha)
    # found = correct & hit & presence >= τ: only row 0 has all three
    assert cli.read_windowed(run_dir, sha, np.float32(0.5)) == {(3, 0): True, (3, 1): False, (7, 0): False}


def test_read_windowed_refuses_misaligned_rows_another_tau_and_a_changed_file(cli_env, tmp_path):
    cli, ckpt, run_dir = cli_env
    sha = sha256_file(ckpt)
    _real_dev(tmp_path, run_dir, sha, json_photo=[3, 7, 7])
    with pytest.raises(ValueError, match="line up"):
        cli.read_windowed(run_dir, sha, np.float32(0.5))
    _real_dev(tmp_path, run_dir, sha, tau=0.6)
    with pytest.raises(ValueError, match="used τ"):
        cli.read_windowed(run_dir, sha, np.float32(0.5))
    npz = _real_dev(tmp_path, run_dir, sha)
    np.savez(npz, photo_index=np.array([3, 3, 8], np.int16), sign_index=np.array([0, 1, 0], np.int16))
    with pytest.raises(ValueError, match="changed"):
        cli.read_windowed(run_dir, sha, np.float32(0.5))


# --- thresholds and the end-to-end write ---

def _photo(index, boxes):
    return RealPhoto(path=Path(f"{index:05d}.ppm"), index=index, signs=tuple(RealSign(b, 1) for b in boxes))


def _record(photo, dets):
    truth = photo_truth(photo)
    kept = tuple(sorted(dets, key=lambda d: (-d.score, -d.scale, d.y, d.x)))
    return PhotoRecord(index=photo.index, truth=truth, kept=kept, match=match_photo(kept, truth),
                       best_raw={s.gt_index: max((d.score for d in kept), default=0.0) for s in truth})


def _det(x, y, score, class_id=1):
    return Detection(x=x, y=y, score=score, class_id=class_id, class_prob=0.9, scale=1.0)


def test_tau_w_counts_false_positives_only(cli_env):
    cli, _, _ = cli_env
    p0, p1 = _photo(0, [(10, 10, 40, 40)]), _photo(1, [(10, 10, 40, 40)])
    records = [_record(p0, [_det(25, 25, 0.95), _det(300, 300, 0.9)]),  # a found sign and an FP
               _record(p1, [_det(25, 25, 0.2), _det(300, 300, 0.6), _det(500, 500, 0.3)])]
    t = cli.dev_thresholds(records, np.float32(0.48))
    # FPPI 0.5 over 2 photos allows 1 FP at or above τ_W: the FPs are 0.9, 0.6, 0.3, so τ_W sits just
    # above 0.6. The found sign's 0.95 and 0.2 don't enter.
    assert np.float32(0.6) < t[TAU_W] <= np.nextafter(np.float32(0.6), np.float32(1))
    assert t[TAU_SYN] == float(np.float32(0.48))
    assert set(t) == {TAU_SYN, TAU_W, "fppi_0.1", "fppi_1.0"}
    assert np.float32(0.9) < t["fppi_0.1"] and np.float32(0.3) < t["fppi_1.0"] <= np.float32(0.6)


@pytest.fixture
def stubbed(cli_env, monkeypatch):
    """The slow part replaced: two dev photos and two test photos, hand-made detections, no model."""
    cli, ckpt, run_dir = cli_env
    photos = [_photo(i, [(10, 10, 40, 40)]) for i in (0, 1, 600, 601)]
    monkeypatch.setattr(cli, "read_gtsdb", lambda root: photos)
    monkeypatch.setattr(cli, "load_checkpoint_info", lambda path: SimpleNamespace(epoch=7))
    monkeypatch.setattr(cli, "load_model", lambda path, device: None)
    calls = []

    def score_photos(model, ps, scales):
        calls.append((tuple(p.index for p in ps), scales))
        return [_record(p, [_det(25, 25, 0.95), _det(300, 300, 0.6 + p.index % 2 * 0.2)]) for p in ps], 0.1

    monkeypatch.setattr(cli, "score_photos", score_photos)
    return cli, ckpt, run_dir, calls


def test_test_end_to_end_reads_dev_thresholds_and_writes_before_printing(stubbed, monkeypatch, capsys):
    cli, ckpt, run_dir, calls = stubbed
    sha = sha256_file(ckpt)
    thresholds = {TAU_SYN: 0.48, TAU_W: 0.7, "fppi_0.1": 0.9, "fppi_1.0": 0.6}
    _write_dev(run_dir, sha, thresholds=thresholds)
    capsys.readouterr()
    real_write, printed_before_write = cli.write_json_atomic, []

    def write(path, obj):
        printed_before_write.append(capsys.readouterr().out)
        real_write(path, obj)

    monkeypatch.setattr(cli, "write_json_atomic", write)
    cli.main(["--checkpoint", str(ckpt), "--split", "test"])

    assert calls == [((600, 601), PYRAMID_SCALES)]
    assert "found" not in printed_before_write[0] and "FPPI" not in printed_before_write[0]
    assert "found" in capsys.readouterr().out
    out = json.loads((run_dir / "detect_test.json").read_text())
    assert out["thresholds"] == thresholds and out["checkpoint_sha256"] == sha
    assert out["thresholds_from"]["sha256"] == sha256_file(run_dir / "detect_dev.json")
    assert out["scales"] == list(PYRAMID_SCALES) and out["report"]["photo_indices"] == [600, 601]
    assert "windowed_vs_whole" not in out["report"]["at"][TAU_SYN]  # test has no windowed comparison
    assert not (run_dir / "detect_test.json.tmp").exists()
    with pytest.raises(SystemExit) as e:  # and only once
        cli.main(["--checkpoint", str(ckpt), "--split", "test"])
    assert e.value.code == 2


def test_dev_end_to_end_at_scale_1(stubbed, tmp_path):
    cli, ckpt, run_dir, calls = stubbed
    sha = sha256_file(ckpt)
    (run_dir / "presence_val.json").write_text(json.dumps({"checkpoint_sha256": sha, "tau": 0.48}))
    _real_dev(tmp_path, run_dir, sha, tau=0.48, photo=(0, 1), sign=(0, 0), presence=(0.9, 0.9), correct=(True, False),
              hit=(True, True))
    cli.main(["--checkpoint", str(ckpt), "--split", "dev", "--scales", "1"])
    assert calls == [((0, 1), (1.0,))]
    out = json.loads((run_dir / "detect_dev_scale1.json").read_text())
    assert not (run_dir / "detect_dev.json").exists()
    assert out["scales"] == [1.0] and out["thresholds_from"] is None
    w = out["report"]["at"][TAU_SYN]["windowed_vs_whole"]
    assert (w["n"], w["whole_found"], w["window_found"]) == (2, 2, 1)
    kept0 = out["photos"][0]["kept"]
    assert [k["found"] for k in kept0] == [0, None] and kept0[1]["fp_kind"] == "background"
