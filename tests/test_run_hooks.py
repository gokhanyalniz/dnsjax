#!/usr/bin/env python3
r"""The run-loop hooks of an extension section (``RunHooks``).

A parameter section whose ``runtime`` names a factory takes part in
the solver's run: :mod:`dnsjax.__main__` builds its
:class:`~dnsjax.extensions.RunHooks` once the section is configured
and calls them at the fixed points that class's docstring lists.  The
core's own sections have no runtime, so this script registers one,
``[hookcheck]`` (``tests/_run_hooks.py``), whose hooks log every call,
and drives four short plane-Couette runs (four steps from a fixed
random IC, a snapshot every two):

1. **order**: ``start`` once, right after the factory, on the initial
   state; then, per step, ``sample`` and ``modify`` of that step's
   ``it`` on the same state, ``it`` running from 0 to the last step's;
   the snapshot of ``it = 2`` flushes between that iteration's two;
   ``finish`` once, after them, on the final ``it``; and a flush (the
   final snapshot's, then shutdown's) after it, ending the log.
2. **inert**: a configured section whose ``modify`` returns its input
   leaves ``stats.dat`` byte-identical to a run without it; that
   stream also goes through the core's own reader
   (:func:`dnsjax.analysis.read_dat`): the header's names, one value
   per row for each, ``t`` increasing.
3. **modify**: halving the state at ``it = 2`` leaves the snapshot of
   ``it = 2`` and the ``stats.dat`` rows up to it unchanged -- every
   saved state precedes a modification -- and changes every row after.
4. **abort**: a ``sample`` that reports a non-finite value at
   ``it = 2`` aborts the run with exit code 3
   (:data:`dnsjax.run_status.EXIT_NON_FINITE`) and a ``FATAL`` line
   naming it, before that iteration's snapshot is written.

Run::

    uv run python tests/test_run_hooks.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from _live import report, run_live  # noqa: E402
from _run_hooks import LOG  # noqa: E402

from dnsjax.run_status import EXIT_NON_FINITE  # noqa: E402

#: The solver child: registers ``[hookcheck]`` by importing its module,
#: then runs the production entry point on the remaining arguments.
CHILD = (
    "import sys; "
    f"sys.path.insert(0, {str(TESTS)!r}); "
    "import _run_hooks; "
    "from dnsjax.__main__ import main; "
    "sys.exit(main(sys.argv[1:]))"
)
DT = 0.005
STEPS = 4
#: The iteration the modify and abort cases act at: a snapshot one.
AT = 2
RUN_ARGS = [
    "--phys.system",
    "plane-couette",
    "--res.nx",
    "8",
    "--res.ny",
    "17",
    "--res.nz",
    "8",
    "--init.random_seed",
    "3",
    "--step.dt",
    str(DT),
    # Half a step short of STEPS * DT, so rounding in t cannot add one.
    "--stop.max_sim_time",
    str((STEPS - 0.5) * DT),
    "--outs.it_stats",
    "1",
    "--outs.it_snapshot",
    str(AT),
    "--stop.check_laminarization",
    "False",
]


def _run(rundir: Path, extra: list[str], timeout: float):
    rundir.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    return run_live(
        [sys.executable, "-c", CHILD, *RUN_ARGS, *extra],
        cwd=rundir,
        env=env,
        timeout=timeout,
    )


def _log(rundir: Path) -> list[dict]:
    text = (rundir / LOG).read_text()
    return [json.loads(line) for line in text.splitlines()]


def _snapshot_at(rundir: Path, it: int) -> np.ndarray:
    """The stored spectral state of the snapshot written at *it*."""
    from dnsjax.analysis import read_state
    from dnsjax.snapshot_meta import read_snapshot_meta

    for path in sorted(rundir.glob("state*.tar")):
        if read_snapshot_meta(path)["it"] == it:
            spectral = read_state(
                path, return_physical=False, return_spectral=True
            ).spectral
            return np.stack([np.asarray(c) for c in spectral])
    raise AssertionError(f"no snapshot of it = {it} in {rundir}")


def check_order(log: list[dict]) -> str | None:
    """The hooks' calls, against the order ``RunHooks`` documents."""
    hooks = [row["hook"] for row in log]
    if hooks[:2] != ["init", "start"] or hooks.count("start") != 1:
        return f"the log does not open with one start: {hooks[:4]}"
    if log[0]["norm2"] != log[1]["norm2"] or log[1]["t"] != 0.0:
        return "start did not see the initial state at t = 0"
    steps = [row for row in log if row["hook"] in ("sample", "modify")]
    want = [(h, it) for it in range(STEPS) for h in ("sample", "modify")]
    got = [(row["hook"], row["it"]) for row in steps]
    if got != want:
        return f"sample/modify calls {got}, expected {want}"
    for sample, modify in zip(steps[::2], steps[1::2], strict=True):
        if sample["norm2"] != modify["norm2"]:
            return f"sample and modify of it = {sample['it']} saw two states"
    if steps[0]["norm2"] != log[1]["norm2"]:
        return "the first sample did not see the initial state"
    i_sample = log.index(steps[2 * AT])
    i_modify = log.index(steps[2 * AT + 1])
    if "flush" not in hooks[i_sample:i_modify]:
        return f"no flush (the snapshot's) between it = {AT}'s two calls"
    finishes = [i for i, h in enumerate(hooks) if h == "finish"]
    if len(finishes) != 1 or log[finishes[0]]["it"] != STEPS:
        return f"finish calls at {[log[i]['it'] for i in finishes]}"
    if finishes[0] < log.index(steps[-1]):
        return "finish came before the last step's modify"
    tail = hooks[finishes[0] + 1 :]
    if not tail or set(tail) != {"flush"}:
        return f"after finish the log holds {tail}, not flushes alone"
    return None


def run_checks(timeout: float) -> list[tuple[str, str | None]]:
    """One ``(name, reason or None)`` per case, ``read_dat`` its own."""
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_hooks_"))
    results: list[tuple[str, str | None]] = []
    try:
        plain = tmp / "plain"
        res = _run(plain, [], timeout)
        if res.returncode != 0:
            reason = f"the run without the section exited {res.returncode}"
            return [(name, reason) for name in ("order", "inert", "modify")]
        if (plain / LOG).exists():
            return [("inert", "an unconfigured section ran its hooks")]

        inert = tmp / "inert"
        res = _run(inert, ["--hookcheck.on", "True"], timeout)
        if res.returncode != 0:
            results.append(("order", f"the run exited {res.returncode}"))
            results.append(("inert", f"the run exited {res.returncode}"))
        else:
            results.append(("order", check_order(_log(inert))))
            same = (inert / "stats.dat").read_bytes() == (
                plain / "stats.dat"
            ).read_bytes()
            results.append(
                ("inert", None if same else "stats.dat changed by the hooks")
            )
            results.append(("read_dat", _check_read_dat(plain / "stats.dat")))

        scaled = tmp / "modify"
        res = _run(
            scaled,
            ["--hookcheck.on", "True", "--hookcheck.scale_at", str(AT)],
            timeout,
        )
        if res.returncode != 0:
            results.append(("modify", f"the run exited {res.returncode}"))
        else:
            ref = np.loadtxt(plain / "stats.dat")
            new = np.loadtxt(scaled / "stats.dat")
            if not np.array_equal(
                _snapshot_at(scaled, AT), _snapshot_at(plain, AT)
            ):
                reason = f"the snapshot of it = {AT} holds the modified state"
            elif not np.array_equal(ref[: AT + 1], new[: AT + 1]):
                reason = f"stats.dat rows up to it = {AT} changed"
            elif np.any(np.all(ref[AT + 1 :] == new[AT + 1 :], axis=1)):
                reason = f"a stats.dat row after it = {AT} is unchanged"
            else:
                reason = None
            results.append(("modify", reason))

        failed = tmp / "abort"
        res = _run(
            failed,
            ["--hookcheck.on", "True", "--hookcheck.fail_at", str(AT)],
            timeout,
        )
        snaps = [
            path for path in failed.glob("state*.tar") if _meta_it(path) >= AT
        ]
        if res.returncode != EXIT_NON_FINITE:
            reason = f"exit {res.returncode}, not {EXIT_NON_FINITE}"
        elif f"FATAL: hookcheck: a non-finite test value at it = {AT}" not in (
            res.stdout
        ):
            reason = "no FATAL line naming the hook's message"
        elif snaps:
            reason = f"snapshots written at or after it = {AT}: {snaps}"
        else:
            reason = None
        results.append(("abort", reason))
    except Exception as exc:  # noqa: BLE001
        results.append(("run-hooks", f"{type(exc).__name__}: {exc}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def _check_read_dat(path: Path) -> str | None:
    """The core's ``.dat`` reader on a stream the solver wrote."""
    from dnsjax.analysis import read_dat

    names = path.read_text().splitlines()[0].lstrip("#").split()
    rows = np.loadtxt(path, ndmin=2)
    cols = read_dat(path)
    if list(cols) != names:
        return f"read_dat names {list(cols)}, the header says {names}"
    if any(c.shape != (rows.shape[0],) for c in cols.values()):
        return f"read_dat shapes {[c.shape for c in cols.values()]}"
    if rows.shape[0] < 2 or not np.all(np.diff(cols["t"]) > 0):
        return f"read_dat t is {cols['t']}"
    return None


def _meta_it(path: Path) -> int:
    from dnsjax.snapshot_meta import read_snapshot_meta

    return read_snapshot_meta(path)["it"]


if __name__ == "__main__":
    results = run_checks(timeout=600.0)
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
