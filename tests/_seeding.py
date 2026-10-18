r"""Shared fixtures of the seed-resolution tests, core and lab.

:func:`no_entropy` disables the entropy source the way a stripped
platform does, so a spurious draw fails instead of passing silently;
:func:`run` launches a solver (or other entry point) child;
:data:`RUN_ARGS` is the smallest plane-Couette run that exercises the
random IC on both mesh axes.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from _live import run_live

BIN = Path(sys.executable).parent

# Smallest plane-Couette box that still exercises the random IC on both
# mesh axes; two steps is enough for stats.dat to differ between seeds.
RUN_ARGS = [
    "--phys.system",
    "plane-couette",
    "--res.nx",
    "8",
    "--res.nz",
    "8",
    "--res.ny",
    "27",
    "--step.dt",
    "0.005",
    "--stop.max_sim_time",
    "0.015",
    "--outs.it_stats",
    "1",
]


@contextlib.contextmanager
def no_entropy():
    """``os.urandom`` unavailable, the way a stripped platform reports it.

    ``draw_seed`` catches ``OSError`` and ``NotImplementedError``; this
    raises the former, which is what a sandbox denying ``/dev/urandom``
    produces.
    """
    original = os.urandom

    def fail(_n):
        raise OSError("Function not implemented")

    os.urandom = fail
    try:
        yield
    finally:
        os.urandom = original


def run(cmd, cwd, timeout):
    """An entry-point child under ``run_live``'s quiet startup.

    ``DNSJAX_QUIET_STARTUP=1`` suppresses the resolved-parameter dump
    but not the seed line, which is deliberately outside that gate.
    Pinning it to ``0`` here would hide a regression that moved the
    line inside it, so leave the default and parse what a quiet run
    prints.
    """
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    return run_live(cmd, cwd=cwd, env=env, timeout=timeout)
