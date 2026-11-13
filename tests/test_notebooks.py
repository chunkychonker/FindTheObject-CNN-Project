"""The notebooks in notebooks/: committed clean and commented, and (opt-in) runnable top to bottom.

The fast tests read the .ipynb files as JSON. The execution test needs the datasets and trained runs, takes
a few minutes, and is skipped by default; run it with:  pytest -m notebooks
"""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
NB_IDS = [p.stem for p in NOTEBOOKS]
EXECUTE_TIMEOUT_S = 900  # per cell; quick mode's slowest cell (notebook 03's short training run) takes ~70 s


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def test_notebooks_are_numbered_00_to_08_with_no_gaps():
    assert [p.stem[:2] for p in NOTEBOOKS] == [f"{i:02d}" for i in range(9)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_every_code_cell_starts_with_a_comment(path):
    bad = [i for i, c in enumerate(_cells(path)) if c["cell_type"] == "code" and not _source(c).lstrip().startswith("#")]
    assert not bad, f"code cells {bad} don't start with a '#' comment saying what they do"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_committed_without_outputs(path):
    dirty = [i for i, c in enumerate(_cells(path)) if c["cell_type"] == "code" and (c["outputs"] or c["execution_count"])]
    assert not dirty, f"code cells {dirty} have outputs or execution counts; clear them before committing"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_flags_that_write_or_score_test_default_to_off(path):
    # The Settings cell is the first code cell. Anything that writes recorded results, or scores test, is
    # behind a flag there, and a fresh notebook must start with every such flag off.
    settings = next(_source(c) for c in _cells(path) if c["cell_type"] == "code")
    for flag in ("RUN_FOR_REAL", "SCORE_TEST", "REBUILD", "FULL_RUN"):
        for line in settings.splitlines():
            if line.startswith(f"{flag} ="):
                assert line.split("=", 1)[1].split("#")[0].strip() == "False", line


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_notebook_runs_top_to_bottom(path):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=EXECUTE_TIMEOUT_S, kernel_name="python3",
                                     resources={"metadata": {"path": str(NB_DIR)}})
    client.execute()  # raises CellExecutionError, naming the cell, if any cell fails; the file is not rewritten
