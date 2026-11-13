"""Notebook-side plumbing: locate the repo's files and launch its scripts from a notebook cell.

Every notebook in notebooks/ imports this. It holds no project logic: data generation, training and
scoring live in src/findobject (the package) and scripts/ (the command-line programs). This module only
finds paths and starts those scripts, so a notebook can drive the whole project without a terminal.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the repo root: notebooks/ sits directly inside it
CHECKPOINTS = ROOT / "outputs/checkpoints"
EVAL = ROOT / "outputs/eval"
NB_OUT = ROOT / "outputs/notebooks"  # the only folder quick mode writes to (gitignored with all of outputs/)

_RUN_DIR = re.compile(r"-\d{8}-\d{6}$")  # scripts/train.py names runs <run_name>-YYYYMMDD-HHMMSS


def require(*paths: Path | str, how: str) -> None:
    """Check that every path exists (relative paths are taken from ROOT).

    Raises FileNotFoundError listing all missing paths plus `how` (the command that makes them), so a
    notebook stops at its first cell instead of failing halfway through with a confusing error.
    """
    missing = [str(p) for p in paths if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {', '.join(missing)}\nmake them with: {how}")


def find_run(run_name: str) -> Path:
    """The newest outputs/checkpoints/<run_name>-<YYYYMMDD-HHMMSS>/ folder.

    The name must match exactly: "peak" does not pick up "peak_decoy-...". Timestamps sort as text, so
    the last name is the newest. Raises FileNotFoundError if no run has that name.
    """
    runs = sorted(p for p in CHECKPOINTS.glob(f"{run_name}-*")
                  if p.is_dir() and _RUN_DIR.sub("", p.name) == run_name)
    if not runs:
        raise FileNotFoundError(f"no run named {run_name!r} in {CHECKPOINTS}; train one with "
                                f"notebook 03 or: python scripts/train.py --config configs/<config>.yaml")
    return runs[-1]


def read_json(path: Path | str) -> dict:
    """Parse a JSON file (relative paths are taken from ROOT). Raises FileNotFoundError if missing."""
    return json.loads((ROOT / path).read_text())


def run_script(script: str, *args: object, enabled: bool = True) -> None:
    """Run scripts/<script>.py with this notebook's Python, from ROOT, printing its output as it comes.

    Always prints the command first. enabled=False stops there, so a reader sees exactly what would run.
    Raises subprocess.CalledProcessError if the script exits non-zero; exit code 2 from evaluate.py or
    evaluate_presence.py means the test guard refused to score a test split a second time.
    """
    cmd = [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)]  # -u: don't buffer, so output streams
    print("$ python " + shlex.join(cmd[2:]))
    if not enabled:
        print("  (not run: set the flag in this notebook's Settings cell to True to run it)")
        return
    proc = subprocess.Popen(cmd, cwd=ROOT, env=_script_env(), text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def start_background(script: str, *args: object, log: Path) -> int:
    """Start scripts/<script>.py detached from this notebook, writing all output to `log`; return its PID.

    For hours-long jobs (full training). The job runs in its own session, so restarting the kernel or
    closing VS Code does not stop it; shutting the Mac down does, and sleep pauses it. The PID is kept in
    <log>.pid. Raises RuntimeError if the PID recorded there is still running, so running the cell twice
    never starts a second copy.
    """
    pid_file = log.with_suffix(".pid")
    if pid_file.exists() and _alive(int(pid_file.read_text())):
        raise RuntimeError(f"already running (PID {pid_file.read_text()}); check it with background_status")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)]
    with open(log, "w") as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=_script_env(), stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    pid_file.write_text(str(proc.pid))
    print(f"started PID {proc.pid}: python {shlex.join(cmd[2:])}\nlog: {log}")
    return proc.pid


def background_status(log: Path, lines: int = 15) -> None:
    """Print whether the job started by start_background(log=log) is still running, then its last lines.

    A finished job's success or failure shows in those lines (scripts/train.py ends with "done."; a crash
    ends with a traceback). Raises FileNotFoundError if no job was started with this log.
    """
    pid = int(log.with_suffix(".pid").read_text())
    print(f"PID {pid}: {'still running' if _alive(pid) else 'finished'}\n--- last {lines} lines of {log} ---")
    print("\n".join(log.read_text().splitlines()[-lines:]))


def _alive(pid: int) -> bool:
    """True if a process with this PID exists (signal 0 checks without sending anything)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, but belongs to another user
        return True
    return True


def _script_env() -> dict[str, str]:
    """This kernel's environment, minus the notebook's inline plotting backend: a plain script that
    inherited it would draw figures nowhere."""
    return {**os.environ, "MPLBACKEND": "Agg"}
