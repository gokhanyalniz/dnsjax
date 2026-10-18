r"""Laminar smoke tests for the core wall-bounded flows.

Runs each core wall-bounded system from its laminar state for a few
time steps at low resolution against the criteria of the runner,
``tests/_laminar_smoke.py``: a single corrector iteration, a stepping
error of `$O(10^{-18})$` or less, a perturbation energy of
`$O(10^{-32})$` or less, a roundoff-sized ``corrector.dat`` error on
every step, and a ``steps.dat`` CFL that matches the laminar base flow
in the default moving frame.  The lab's flows have a table of their
own, ``lab/tests/test_laminar_smoke.py``.

The active CFL column is `$\Delta t \, \max|U - U_{grid}| \, n / l$`
and the others are roundoff-sized.  ``phys.u_grid`` defaults to the
laminar bulk velocity, so the active CFL is `$\max|U_x| = 1$` for
plane-Couette (`$U_{grid} = 0$`), `$2/3$` of it for plane-Poiseuille
(`$U_{grid} = 2/3$`), `$1/2$` of it for the pipe (`$U_{grid} = 1/2$`),
and the azimuthal `$\max|U_\theta/r| = 1$` for Taylor-Couette
(`$U_{grid} = 0$`), doubled on its ``geo.m0 = 2`` wedge (where each
maximum sits: the runner's ``CFL_*`` constants).

Caveat: the laminar state has ``u' = 0``, so every `$\omega'$`/
`$u'$`-proportional term vanishes -- this checks the base-flow fixed
point, time stepping, and the CFL diagnostic, but **not** the
rotational nonlinear term (a wrong ``rhs.py``/advection change can
still report ``err = 0``); ``tests/test_random_smoke.py`` drives that
path.

The suite also guards the ``dnsjax`` console script (the why of each
check: its runner function): a ``--help`` preflight (exit 0, usage
text, **no files created**), a ``bare-launch`` pair running the
console script with **no** ``mpirun`` at all (it must meet the
``err`` / `$E'$` thresholds of the launched rows, and refuse to hold
several CPU devices in one process, naming the launch that works), and
one smoke case launched through the installed ``dnsjax`` script under
``mpirun`` (the CLI parse and the distributed init through the console
entry point; same :func:`dnsjax.__main__.main` as
``python -m dnsjax``).  Console-script entries are skipped with a
notice when the script is not installed (``uv sync``).  A
``plane-couette-nz-pad`` case, whose ``nz = 6`` yields the odd natural
``nz_padded = 9``, steps as-is where ``np1`` divides it (asserting that
no rounding note appears) and is rounded up for ``np1`` divisibility
otherwise (asserting the startup diagnostic, e.g. 9 -> 10 at
``np1 = 2``).

Usage (single device)::

    uv run python tests/test_laminar_smoke.py

Usage (two devices via MPI)::

    uv run python tests/test_laminar_smoke.py --np 2

With ``--np N`` (N > 1), each test invokes
``mpirun -np N python -m dnsjax --dist.np0 NP0 --dist.np1 NP1 ...``
where ``NP0 * NP1 == N``.  ``--np0`` sets ``NP0`` (default 1, a 1D
mesh along ``np1``); ``--np 4 --np0 2`` runs the ``(2, 2)`` mesh.
"""

from __future__ import annotations

from _laminar_smoke import (
    CFL_GRID_LAMINAR_PIPE,
    CFL_GRID_LAMINAR_POISEUILLE,
    CFL_TH_LAMINAR_TC,
    CFL_X_LAMINAR,
    main,
)

# ── configuration ────────────────────────────────────────────────────


def _nz_pad_note(np1: int) -> str | None:
    """Expected rounding note for the nz-pad case (natural size 9).

    ``None`` where ``np1`` divides 9: the odd dealiasing pad runs
    unrounded and no note must appear.
    """
    if 9 % np1 == 0:
        return None
    rounded = ((9 + np1 - 1) // np1) * np1
    return f"nz_padded rounded from 9 to {rounded}"


SYSTEMS: list[dict] = [
    {
        "name": "plane-couette",
        "cfl_active": CFL_X_LAMINAR,
        "args": [
            "--phys.system",
            "plane-couette",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        # An odd dealiasing pad (9 - 6 = 3), allowed: pad parity is
        # unconstrained.  The expectation per layout: ``_nz_pad_note``.
        # The laminar CFL checks are nz-agnostic for plane-Couette
        # (streamwise-active).
        "name": "plane-couette-nz-pad",
        "cfl_active": CFL_X_LAMINAR,
        "expect_stdout": _nz_pad_note,
        "args": [
            "--phys.system",
            "plane-couette",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "6",
            "--res.ny",
            "27",
        ],
    },
    {
        # The base case, launched through the installed console script.
        "name": "plane-couette-console-script",
        "cfl_active": CFL_X_LAMINAR,
        "console": True,
        "args": [
            "--phys.system",
            "plane-couette",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-poiseuille",
        "cfl_active": CFL_GRID_LAMINAR_POISEUILLE,
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-poiseuille-cbv",
        "cfl_active": CFL_GRID_LAMINAR_POISEUILLE,
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--phys.driving",
            "constant_bulk_velocity",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        # The spanwise-blocking projection is implemented once per
        # geometry family, so one entry per family covers the branch:
        # this Cartesian one (also standing in for plane-poiseuille)
        # and taylor-couette-block-spanwise (annular).
        "name": "plane-couette-block-spanwise",
        "cfl_active": CFL_X_LAMINAR,
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "pipe",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "args": [
            "--phys.system",
            "pipe",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        "name": "pipe-cbv",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "args": [
            "--phys.system",
            "pipe",
            "--phys.driving",
            "constant_bulk_velocity",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        # Cylindrical azimuthal wedge (geo.m0 > 1): the axisymmetric
        # laminar pipe profile is a fixed point on any wedge, exercising
        # the m0-scaled cylindrical Fourier / parity path.
        "name": "pipe-wedge",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "args": [
            "--phys.system",
            "pipe",
            "--geo.m0",
            "3",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        "name": "taylor-couette",
        "cfl_active": CFL_TH_LAMINAR_TC,
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        # Annular azimuthal wedge (geo.m0 = 2): the circular-Couette
        # profile is a laminar fixed point on the half-annulus wedge,
        # and the azimuthal CFL doubles (lz = pi).
        "name": "taylor-couette-wedge",
        "cfl_active": 2 * CFL_TH_LAMINAR_TC,
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--geo.m0",
            "2",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        "name": "taylor-couette-block-spanwise",
        "cfl_active": CFL_TH_LAMINAR_TC,
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        # The legacy res.consistent_imm = False path: the primitive
        # (v, p) scheme must leave the laminar fixed point exactly
        # fixed too.  Every entry above covers the default
        # reconstruction scheme, whose every term is linear in u' and
        # so vanishes identically at u' = 0: err = 0 there is the cheap
        # structural proof that it adds nothing spurious (the teeth are
        # the nonlinear smoke's: the module docstring's caveat).  One
        # legacy entry per geometry covers the three implementations.
        "name": "plane-couette-legacy-imm",
        "cfl_active": CFL_X_LAMINAR,
        "args": [
            "--phys.system",
            "plane-couette",
            "--res.consistent_imm",
            "False",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "taylor-couette-legacy-imm",
        "cfl_active": CFL_TH_LAMINAR_TC,
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "-50",
            "--geo.eta",
            "0.5",
            "--res.consistent_imm",
            "False",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
    {
        # The pipe's legacy primitive scheme: same cheap structural
        # proof that it leaves u' = 0 fixed.
        "name": "pipe-legacy-imm",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "args": [
            "--phys.system",
            "pipe",
            "--res.consistent_imm",
            "False",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nz",
            "4",
            "--res.ntheta",
            "4",
            "--res.nr",
            "27",
        ],
    },
]


if __name__ == "__main__":
    main(SYSTEMS, console_checks=True)
