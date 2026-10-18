r"""Localized-rolls IC smoke tests: nonlinear integration.

Starts each core rolls-builder variant (kolmogorov for the
triply-periodic builder, plane-couette for the Cartesian one, pipe,
taylor-couette) from the deterministic localized-rolls IC
(``init.localized_rolls``) at a transitional Reynolds number on a small
domain, and integrates a short time (``t = 0.25``, the runner's
``--max-sim-time`` default, whose comment says why) -- verifying the
run completes with no error, NaN, or blow-up.  The rolls drive the full
**nonlinear** path (the laminar smoke test cannot, since ``u' = 0``
there), like ``tests/test_random_smoke.py`` but with a deterministic
IC.

The runner: ``tests/_rolls_smoke.py``; the criteria, shared with the
random smoke: exit 0 and the four of ``tests/_smoke.py``.  The lab's
flows have a table of their own, ``lab/tests/test_rolls_smoke.py``.
The triply-periodic builder's `$y$` factor is a localization rather
than a wall profile (``ic/localized_rolls.py``), but ``kolmogorov``
runs the same default ``iterative-cn`` corrector as the rest, so all
five criteria apply to it unchanged.

Includes a **forced multi-device, padding-inducing** case
(``plane-couette-mpi-pad``: ``mpirun --oversubscribe -np 2`` with
``np1 = 2`` and ``nx`` chosen so ``nx // 2`` is not divisible by
``np1``) that exercises the rolls' sharded per-device spectral build
with spectral `$k_x$` padding -- the regression guard for the
no-replication construction.

Usage (single device)::

    uv run python tests/test_rolls_smoke.py

Faster subset (one system, coarse / short)::

    uv run python tests/test_rolls_smoke.py \
        --systems plane-couette --res 16 --max-sim-time 0.2

The whole standard suite at two devices via MPI::

    uv run python tests/test_rolls_smoke.py --np 2
"""

from __future__ import annotations

from _rolls_smoke import main

# ── configuration ────────────────────────────────────────────────────

# Same Reynolds numbers / boxes as test_random_smoke.  Each entry but
# the trailing ``*-mpi-pad`` one runs at the suite's ``--np`` (default
# 1).
SYSTEMS: list[dict] = [
    {
        "name": "kolmogorov",
        "args": [
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "620",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "plane-couette",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # plane-poiseuille needs no entry: the Cartesian rolls builder
        # is shared with plane-couette (above), rolls construction is
        # unit-tested in test_localized_rolls.py, and per-flow
        # nonlinear integration lives in test_random_smoke.py.
        "name": "pipe",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
        ],
    },
    {
        # The legacy primitive (v, p) path on the pipe, from a
        # *resolved* rolls IC: every other entry here runs the default
        # reconstruction scheme, so this one is the legacy path's
        # resolved-IC coverage under the localized-spot builder.
        "name": "pipe-legacy-imm",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--res.consistent_imm",
            "False",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "taylor-couette",
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "400",
            "--phys.re2",
            "-400",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # 2 devices on the kx axis: nx // 2 = 17 is not divisible by
        # np1 = 2 (padded to 18).
        "name": "plane-couette-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 34, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
]


if __name__ == "__main__":
    main(SYSTEMS)
