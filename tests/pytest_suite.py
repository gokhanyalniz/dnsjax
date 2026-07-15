"""Pytest bridge over the standalone test scripts.

The ``tests/test_*.py`` files are standalone scripts and remain the
source of truth, each runnable directly as
``uv run python tests/test_X.py``.  They rely on module-level singleton
/ JAX setup performed by their own ``__main__`` runners, so pytest must
never **import** them -- collection would execute the module top level
with the singletons unconfigured.  This module is the only file
pytest collects (``[tool.pytest.ini_options] python_files`` in
``pyproject.toml``): every case shells one script out as a subprocess
in its default invocation, asserts a zero exit code, and surfaces the
output tail on failure.  Adding a test script means adding one row to
``_SCRIPTS``.

Markers:

- ``mpi``: the script launches solver runs via ``mpirun`` (even at
  ``-np 1``); skipped automatically when ``mpirun`` is not on PATH.
- ``slow``: full solver integration runs / dt sweeps (minutes each).

Usage::

    uv run pytest                  # everything available
    uv run pytest -m "not slow"    # the offline (no-solver-run) loop
    uv run pytest -k padding       # a single script
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

_MPI = (
    pytest.mark.mpi,
    pytest.mark.skipif(
        shutil.which("mpirun") is None, reason="mpirun not on PATH"
    ),
)
_SLOW = (pytest.mark.slow,)

# One row per invocation: (script, extra args, marks, timeout in s).
# ``test_resume`` appears twice: the MPI-free ``--unit-only`` subset
# and the full run (offline units + mpirun integration).
_SCRIPTS: list[tuple[str, tuple[str, ...], tuple, int]] = [
    ("test_annular.py", (), (), 1800),
    ("test_banded_solver.py", (), (), 1800),
    ("test_banded_solver_sharded.py", (), (), 1800),
    ("test_cartesian.py", (), (), 1800),
    ("test_cnab2.py", (), (), 1800),
    ("test_cylindrical.py", (), (), 1800),
    ("test_integration.py", (), (), 1800),
    ("test_localized_rolls.py", (), (), 1800),
    ("test_mean_mask.py", (), (), 1800),
    ("test_padding.py", (), (), 1800),
    ("test_forcing.py", ("--unit-only",), (), 1800),
    ("test_probes.py", ("--unit-only",), (), 1800),
    ("test_quasi_keplerian.py", (), (), 1800),
    ("response/test_ensemble.py", (), (), 1800),
    ("response/test_lim.py", (), (), 1800),
    ("response/test_operator_tools.py", (), (), 1800),
    ("response/test_probes_reader.py", (), (), 1800),
    ("response/test_ssi.py", (), (), 1800),
    ("test_resume.py", ("--unit-only",), (), 1800),
    ("test_snapshot.py", (), (), 1800),
    ("test_snapshot_export.py", (), (), 1800),
    ("test_snapshot_import.py", (), (), 1800),
    ("test_snapshot_perturb.py", (), (), 1800),
    ("test_transient_growth.py", (), (), 1800),
    ("test_viscoelastic.py", (), (), 1800),
    ("test_temporal_order.py", (), _SLOW, 3600),
    ("test_forcing.py", (), _MPI, 1800),
    ("test_laminar_smoke.py", (), _MPI + _SLOW, 3600),
    ("test_probes.py", (), _MPI, 1800),
    ("test_random_smoke.py", (), _MPI + _SLOW, 3600),
    ("test_resume.py", (), _MPI + _SLOW, 3600),
    ("test_rolls_smoke.py", (), _MPI + _SLOW, 3600),
]


def _case_id(script: str, args: tuple[str, ...]) -> str:
    suffix = "".join(a.replace("--", "-") for a in args)
    return Path(script).stem + suffix


@pytest.mark.parametrize(
    ("script", "args", "timeout"),
    [
        pytest.param(
            script, args, timeout, marks=marks, id=_case_id(script, args)
        )
        for script, args, marks, timeout in _SCRIPTS
    ],
)
def test_script(script: str, args: tuple[str, ...], timeout: int) -> None:
    """Run one standalone script; PASS is its zero exit code."""
    result = subprocess.run(
        [sys.executable, str(_TESTS_DIR / script), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join(
            (result.stdout or "").splitlines()[-50:]
            + (result.stderr or "").splitlines()[-30:]
        )
        pytest.fail(
            f"{script} {' '.join(args)} exited with code "
            f"{result.returncode}\n{tail}",
            pytrace=False,
        )
