"""The pytest bridge shared by the core and lab suites.

The ``test_*.py`` files of both suites are standalone scripts and remain
the source of truth, each runnable directly as
``uv run python <dir>/test_X.py``.  They rely on module-level singleton
/ JAX setup performed by their own ``__main__`` runners, so pytest must
never **import** them -- collection would execute the module top level
with the singletons unconfigured.  Each suite's ``pytest_suite.py`` is
the only file pytest collects there (``[tool.pytest.ini_options]
python_files`` in ``pyproject.toml``): it lists its scripts, one row per
invocation, and :func:`script_test` turns the rows into one test that
shells each script out as a subprocess, asserts a zero exit code, and
surfaces the output tail on failure.  Adding a test script means adding
one row to its suite's ``_SCRIPTS``.

Output streams live: each case prints a banner, then tees the
script's stdout/stderr through as it arrives (``tests/_live.py``,
which also sets ``PYTHONUNBUFFERED=1`` in the child), and pytest runs
with ``-s`` by default (``addopts`` in ``pyproject.toml``) so the
stream reaches the terminal immediately -- a tailed run shows
progress and can be aborted early.  Failures still end with the
compact output tail in the pytest summary.  Every script runs from the
repository root.

Markers:

- ``mpi``: the script launches solver runs via ``mpirun`` (even at
  ``-np 1``); skipped automatically when ``mpirun`` is not on PATH.
- ``slow``: full solver integration runs / dt sweeps / the
  published transient-growth values (minutes each).

Usage::

    uv run pytest                            # both suites, all rows
    uv run pytest tests                      # the core suite alone
    uv run pytest lab/tests                  # the lab suite alone
    uv run pytest -m "not slow and not mpi"  # offline: no mpirun launch
    uv run pytest -m "not slow"              # + the quick mpirun rows
    uv run pytest -k padding                 # a single script
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from _live import run_live

#: The repository root, every script's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

MPI = (
    pytest.mark.mpi,
    pytest.mark.skipif(
        shutil.which("mpirun") is None, reason="mpirun not on PATH"
    ),
)
SLOW = (pytest.mark.slow,)
# ``--unit-only`` fallback rows: they exist so the unit halves still
# run where mpirun is absent; with mpirun present the full (mpi) rows
# re-run those same units, so the fallback would be pure duplication.
NO_MPI_ONLY = (
    pytest.mark.skipif(
        shutil.which("mpirun") is not None,
        reason="the full (mpi) row covers the unit subset",
    ),
)

#: One row per invocation: (script, extra args, marks, timeout in s),
#: the script relative to its suite's directory.
Row = tuple[str, tuple[str, ...], tuple, int]


def _case_id(script: str, args: tuple[str, ...]) -> str:
    suffix = "".join(a.replace("--", "-") for a in args)
    return Path(script).stem + suffix


def _tail(stdout: str, stderr: str) -> str:
    return "\n".join(stdout.splitlines()[-50:] + stderr.splitlines()[-30:])


def script_test(tests_dir: Path, rows: list[Row]) -> Callable:
    """The parametrized test running each of *rows* from *tests_dir*.

    Bind the result to a ``test_*`` name in the suite's
    ``pytest_suite.py``, where pytest collects it.
    """

    @pytest.mark.parametrize(
        ("script", "args", "timeout"),
        [
            pytest.param(
                script, args, timeout, marks=marks, id=_case_id(script, args)
            )
            for script, args, marks, timeout in rows
        ],
    )
    def test_script(script: str, args: tuple[str, ...], timeout: int) -> None:
        """Run one standalone script; PASS is its zero exit code."""
        path = tests_dir / script
        invocation = " ".join([str(path.relative_to(REPO_ROOT)), *args])
        print(
            f"\n=== running {invocation} [timeout {timeout}s] ===", flush=True
        )
        try:
            result = run_live(
                [sys.executable, str(path), *args],
                cwd=REPO_ROOT,
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

    return test_script
