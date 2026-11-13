"""notebooks/nbhelpers.py: finding runs and files, and launching scripts, as the notebooks rely on."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import nbhelpers  # noqa: E402  (lives in notebooks/, not in the package)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Point nbhelpers at an empty repo root, with outputs/checkpoints and scripts/ folders."""
    (tmp_path / "outputs/checkpoints").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(nbhelpers, "ROOT", tmp_path)
    monkeypatch.setattr(nbhelpers, "CHECKPOINTS", tmp_path / "outputs/checkpoints")
    return tmp_path


def _runs(root: Path, *names: str) -> None:
    for name in names:
        (root / "outputs/checkpoints" / name).mkdir()


def test_find_run_picks_the_newest_of_that_exact_name(fake_root):
    _runs(fake_root, "peak-20261101-120000", "peak-20261105-090000", "peak_decoy-20261112-000000")
    assert nbhelpers.find_run("peak").name == "peak-20261105-090000"
    assert nbhelpers.find_run("peak_decoy").name == "peak_decoy-20261112-000000"


def test_find_run_ignores_names_that_only_share_a_prefix(fake_root):
    _runs(fake_root, "peak_decoy-20261112-000000", "peak_s1-20261112-000000")
    with pytest.raises(FileNotFoundError, match="no run named 'peak'"):
        nbhelpers.find_run("peak")


def test_find_run_ignores_plain_files(fake_root):
    (fake_root / "outputs/checkpoints/peak-20261101-120000").write_text("not a run folder")
    with pytest.raises(FileNotFoundError):
        nbhelpers.find_run("peak")


def test_require_passes_when_everything_exists(fake_root):
    (fake_root / "a.npz").write_text("")
    nbhelpers.require("a.npz", fake_root / "scripts", how="unused")


def test_require_lists_every_missing_path_and_how_to_make_them(fake_root):
    (fake_root / "here.npz").write_text("")
    with pytest.raises(FileNotFoundError) as err:
        nbhelpers.require("here.npz", "gone1.npz", "gone2.npz", how="python scripts/make_it.py")
    msg = str(err.value)
    assert "gone1.npz" in msg and "gone2.npz" in msg and "here.npz" not in msg
    assert "python scripts/make_it.py" in msg


def test_run_script_disabled_prints_the_command_and_runs_nothing(fake_root, capsys):
    marker = fake_root / "ran"
    (fake_root / "scripts/touch.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    nbhelpers.run_script("touch", "--flag", 3, enabled=False)
    out = capsys.readouterr().out
    assert "$ python scripts/touch.py --flag 3" in out and "not run" in out
    assert not marker.exists()


def test_run_script_runs_from_the_root_and_streams_output(fake_root, capsys):
    (fake_root / "scripts/hello.py").write_text("import os, sys\nprint('cwd', os.getcwd())\nprint('args', sys.argv[1:])\n")
    nbhelpers.run_script("hello", "a", 1)
    out = capsys.readouterr().out
    assert f"cwd {fake_root.resolve()}" in out or f"cwd {fake_root}" in out
    assert "args ['a', '1']" in out


def test_run_script_raises_on_a_nonzero_exit(fake_root):
    (fake_root / "scripts/fail.py").write_text("import sys\nsys.exit(2)\n")
    with pytest.raises(subprocess.CalledProcessError) as err:
        nbhelpers.run_script("fail")
    assert err.value.returncode == 2


def test_start_background_refuses_a_second_copy_while_the_first_runs(fake_root, capsys):
    (fake_root / "scripts/wait.py").write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    log = fake_root / "logs/wait.log"
    pid = nbhelpers.start_background("wait", log=log)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            nbhelpers.start_background("wait", log=log)
        nbhelpers.background_status(log)
        assert "still running" in capsys.readouterr().out
    finally:
        os.killpg(pid, signal.SIGTERM)  # the job leads its own session, so this stops it and nothing else
        os.waitpid(pid, 0)
    nbhelpers.background_status(log)
    assert "finished" in capsys.readouterr().out


def test_background_status_without_a_started_job_raises(fake_root):
    with pytest.raises(FileNotFoundError):
        nbhelpers.background_status(fake_root / "never.log")
