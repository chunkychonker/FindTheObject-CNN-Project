"""scripts/grade_v3b.py: it grades exactly the files it names, and refuses mismatched ones.

The grading itself is tested in test_detect_grade.py; here grade_dev / grade_test are stubbed, and the
summary is built from a real detect_report on a hand-made photo.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from findobject.data.gtsdb import RealPhoto, RealSign
from findobject.eval.detect_report import TAU_SYN, TAU_W, PhotoRecord, detect_report
from findobject.eval.photo_match import match_photo
from findobject.eval.photo_truth import photo_truth
from findobject.inference.detection import PYRAMID_SCALES, Detection
from findobject.utils.files import sha256_file

SCRIPTS = Path(__file__).parents[1] / "scripts"
RUNS = ["r0", "r1", "r2"]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report():
    photo = RealPhoto(path=Path("x"), index=0, signs=(RealSign((10, 10, 40, 40), 1),))
    truth = photo_truth(photo)
    kept = (Detection(x=25, y=25, score=0.9, class_id=1, class_prob=0.9, scale=1.0),
            Detection(x=300, y=300, score=0.6, class_id=1, class_prob=0.9, scale=0.5))
    rec = PhotoRecord(index=0, truth=truth, kept=kept, match=match_photo(kept, truth), best_raw={0: 0.9})
    return detect_report([rec], {TAU_SYN: 0.5, TAU_W: 0.7}, {}, rng_seed=0, n_resamples=20)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cli = _script("grade_v3b")
    monkeypatch.setattr(cli, "EVAL_ROOT", tmp_path / "eval")
    monkeypatch.setattr(cli, "OUT_DIR", tmp_path / "results")
    seen = {}
    monkeypatch.setattr(cli, "grade_dev", lambda reports, scale1: seen.update(dev=(reports, scale1)) or
                        {"W1": {"pass": True, "whole_found": 0.97}})
    monkeypatch.setattr(cli, "grade_test", lambda dev, test: seen.update(test=(dev, test)) or
                        {"W6": {"pass": False, "mean_test_fppi": 0.9}})
    return cli, tmp_path, seen


def _write(tmp_path, run, name, sha="a" * 64, scales=PYRAMID_SCALES, tag=None):
    path = tmp_path / "eval" / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _report() | {"tag": tag or f"{run}/{name}"}
    path.write_text(json.dumps({"checkpoint_sha256": sha, "scales": list(scales), "seconds_per_photo": 0.2,
                                "report": report}))
    return path


def test_dev_grades_the_three_dev_files_and_seed_0s_scale_1(env, capsys):
    cli, tmp_path, seen = env
    paths = [_write(tmp_path, r, "detect_dev.json") for r in RUNS]
    paths.append(_write(tmp_path, "r0", "detect_dev_scale1.json", scales=(1.0,)))
    cli.main(["--runs", *RUNS, "--split", "dev"])
    reports, scale1 = seen["dev"]
    assert [r["tag"] for r in reports] == [f"{r}/detect_dev.json" for r in RUNS]
    assert scale1["tag"] == "r0/detect_dev_scale1.json"
    out = json.loads((tmp_path / "results" / "grades_dev.json").read_text())
    assert out["grades"]["W1"]["pass"] is True and out["runs"] == RUNS
    assert out["files"] == {str(p): {"sha256": sha256_file(p)} for p in paths}
    s = out["summary"]["r0"]["at"][TAU_SYN]
    assert s["found"]["all"]["k"] == 1 and s["fppi"]["n_fp"] == 1 and s["fp_by_scale"] == {"0.5": 1, "1.0": 0}
    assert "r0 (scale 1)" in out["summary"]
    assert "W1: PASS" in capsys.readouterr().out


def test_test_grades_dev_against_test(env):
    cli, tmp_path, seen = env
    for r in RUNS:
        _write(tmp_path, r, "detect_dev.json")
        _write(tmp_path, r, "detect_test.json")
    cli.main(["--runs", *RUNS, "--split", "test"])
    dev, test = seen["test"]
    assert [t["tag"] for t in test] == [f"{r}/detect_test.json" for r in RUNS]
    assert json.loads((tmp_path / "results" / "grades_test.json").read_text())["grades"]["W6"]["pass"] is False


def test_refuses_missing_mismatched_or_wrong_scale_files(env):
    cli, tmp_path, _ = env
    for r in RUNS:
        _write(tmp_path, r, "detect_dev.json")
    with pytest.raises(FileNotFoundError, match="detect_dev_scale1"):
        cli.main(["--runs", *RUNS, "--split", "dev"])
    _write(tmp_path, "r0", "detect_dev_scale1.json", sha="b" * 64, scales=(1.0,))
    with pytest.raises(ValueError, match="different checkpoints"):
        cli.main(["--runs", *RUNS, "--split", "dev"])
    _write(tmp_path, "r1", "detect_dev.json", scales=(1.0,))
    with pytest.raises(ValueError, match="used scales"):
        cli.main(["--runs", *RUNS, "--split", "dev"])
    assert not (tmp_path / "results").exists()  # nothing written on a refusal


def test_needs_exactly_three_runs(env):
    cli, _, _ = env
    with pytest.raises(SystemExit):
        cli.main(["--runs", "r0", "r1", "--split", "dev"])
