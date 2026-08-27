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

Output streams live: each case prints a banner, then tees the
script's stdout/stderr through as it arrives (``tests/_live.py``,
which also sets ``PYTHONUNBUFFERED=1`` in the child), and pytest runs
with ``-s`` by default (``addopts`` in ``pyproject.toml``) so the
stream reaches the terminal immediately -- a tailed run shows
progress and can be aborted early.  Failures still end with the
compact output tail in the pytest summary.

Markers:

- ``mpi``: the script launches solver runs via ``mpirun`` (even at
  ``-np 1``); skipped automatically when ``mpirun`` is not on PATH.
- ``slow``: full solver integration runs / dt sweeps / the
  transient-growth literature anchors (minutes each).

Usage::

    uv run pytest                            # everything available
    uv run pytest -m "not slow and not mpi"  # offline: no solver runs
    uv run pytest -m "not slow"              # + the two quick mpirun
                                             #   rows (probes, forcing)
    uv run pytest -k padding                 # a single script
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

sys.path.insert(0, str(_TESTS_DIR))

from _live import run_live  # noqa: E402

_MPI = (
    pytest.mark.mpi,
    pytest.mark.skipif(
        shutil.which("mpirun") is None, reason="mpirun not on PATH"
    ),
)
_SLOW = (pytest.mark.slow,)
# ``--unit-only`` fallback rows: they exist so the unit halves still
# run where mpirun is absent; with mpirun present the full (mpi) rows
# re-run those same units, so the fallback would be pure duplication.
_NO_MPI_ONLY = (
    pytest.mark.skipif(
        shutil.which("mpirun") is not None,
        reason="the full (mpi) row covers the unit subset",
    ),
)

# One row per invocation: (script, extra args, marks, timeout in s).
# Scripts appearing twice: ``test_resume`` (offline ``--unit-only``
# subset + full mpirun run), ``test_forcing``/``test_probes``/
# ``test_seeding``/``test_twin_postprocess`` (the no-mpirun unit
# fallback + the full run),
# ``test_transient_growth`` (offline ``--fast`` structure checks +
# the slow full run with the literature anchors), and
# ``test_laminar_smoke`` (single-device + the ``--np 2`` mesh row
# whose reason is spelled out at that entry).
_SCRIPTS: list[tuple[str, tuple[str, ...], tuple, int]] = [
    ("test_adaptive.py", (), (), 1800),
    ("test_annular.py", (), (), 1800),
    ("test_banded_solver.py", (), (), 1800),
    ("test_banded_solver_sharded.py", (), (), 1800),
    ("test_bootstrap.py", (), (), 1800),
    ("test_cartesian.py", (), (), 1800),
    ("test_cnab2.py", (), (), 1800),
    ("test_cylindrical.py", (), (), 1800),
    # Offline (no mpirun), but it runs five full solver steps in
    # subprocesses, so it is a slow row.
    ("test_imm_continuity.py", (), _SLOW, 3600),
    ("test_integration.py", (), (), 1800),
    ("test_localized_rolls.py", (), (), 1800),
    ("test_mean_mask.py", (), (), 1800),
    # Four in-process (forced CPU device) subprocesses, no mpirun.
    ("test_mean_mode.py", (), (), 1800),
    ("test_monochromatic.py", (), (), 1800),
    ("test_padding.py", (), (), 1800),
    ("test_param_surface.py", (), (), 1800),
    ("test_driving.py", ("--unit-only",), _NO_MPI_ONLY, 1800),
    ("test_forcing.py", ("--unit-only",), _NO_MPI_ONLY, 1800),
    ("test_probes.py", ("--unit-only",), _NO_MPI_ONLY, 1800),
    ("test_seeding.py", ("--unit-only",), _NO_MPI_ONLY, 1800),
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
    ("test_transient_growth.py", ("--fast",), (), 1800),
    ("test_twin_analysis.py", (), (), 1800),
    ("test_twin_postprocess.py", ("--unit-only",), _NO_MPI_ONLY, 1800),
    ("test_twin_unit.py", (), (), 1800),
    ("test_viscoelastic.py", (), (), 1800),
    ("test_viscoelastic_pipe.py", (), (), 1800),
    ("test_temporal_order.py", (), _SLOW, 3600),
    ("test_energy_budget.py", (), _MPI + _SLOW, 3600),
    ("test_driving.py", (), _SLOW, 2400),
    ("test_forcing.py", (), _MPI, 1800),
    ("test_laminar_smoke.py", (), _MPI + _SLOW, 3600),
    # Multi-device, and specifically on the np1 (k_x / axial) axis:
    # every entry runs at the runner default --np 1 otherwise, which
    # is how a total np1 > 1 failure of res.consistent_imm reached
    # production in the cylindrical and annular families -- the
    # geometries build wavenumber arrays that are unsharded on that
    # axis, so a spec mismatch there is invisible single-device.
    ("test_laminar_smoke.py", ("--np", "2"), _MPI + _SLOW, 3600),
    ("test_probes.py", (), _MPI, 1800),
    ("test_random_smoke.py", (), _MPI + _SLOW, 3600),
    # _MPI for the cross-process seed-agreement row alone; the seven
    # solver/twin launches are all a handful of steps each.
    ("test_seeding.py", (), _MPI, 1800),
    ("test_resume.py", (), _MPI + _SLOW, 3600),
    ("test_rolls_smoke.py", (), _MPI + _SLOW, 3600),
    ("test_transient_growth.py", (), _SLOW, 3600),
    # No _MPI: single-process throughout (a lone process needs no
    # launcher).  ~20 min at the default single seed -- the
    # per-rung 50-advective-unit parent spin-ups dominate.
    ("test_twin_budget.py", (), _SLOW, 3600),
    ("test_twin_driver.py", (), _MPI + _SLOW, 3600),
    # ~2.5 min: three short twin members plus the reconstructions of
    # them, the last on a (2, 1) mesh -- not a _SLOW row.
    ("test_twin_postprocess.py", (), _MPI, 1800),
]


def _case_id(script: str, args: tuple[str, ...]) -> str:
    suffix = "".join(a.replace("--", "-") for a in args)
    return Path(script).stem + suffix


def _tail(stdout: str, stderr: str) -> str:
    return "\n".join(stdout.splitlines()[-50:] + stderr.splitlines()[-30:])


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
    invocation = " ".join([f"tests/{script}", *args])
    print(f"\n=== running {invocation} [timeout {timeout}s] ===", flush=True)
    try:
        result = run_live(
            [sys.executable, str(_TESTS_DIR / script), *args],
            cwd=_REPO_ROOT,
            timeout=timeout,
            echo=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{invocation} timed out after {timeout}s\n"
            f"{_tail(exc.output or '', exc.stderr or '')}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"{invocation} exited with code {result.returncode}\n"
            f"{_tail(result.stdout or '', result.stderr or '')}",
            pytrace=False,
        )
