r"""Snapshot-figure helpers (JAX-free).

The friction Reynolds number of the snapshot being drawn and the wall
units that follow from it, which ``snapshot_figure.py`` labels a plane
with: the one piece of per-flow physics a figure needs, kept here for a
second figure script to share.

Sibling import: the scripts are run as ``python scripts/<name>.py``,
which puts ``scripts/`` on ``sys.path``, so ``from _figure_common
import ...`` resolves.  Nothing here imports JAX or the solver runtime.
"""

from __future__ import annotations

import numpy as np

#: The laminar wall velocity gradient `$dU/dy$` at the bottom and top
#: walls, per Cartesian base flow: `$U = y$` gives `$(1, 1)$`,
#: `$U = 1 - y^2$` gives `$(2, -2)$`.  A snapshot stores the
#: perturbation about that profile, so the *total* wall shear -- and
#: hence `$Re_\tau$` -- needs it added back, sign included.
LAMINAR_WALL_SHEAR = {
    "plane-couette": (1.0, 1.0),
    "plane-poiseuille": (2.0, -2.0),
}


def friction_reynolds(stats, params) -> float:
    r"""`$Re_\tau = u_\tau h/\nu$` from the snapshot's own wall shear.

    `$u_\tau = \sqrt{\nu\,|\mathrm{d}U/\mathrm{d}y|_w}$` in code units
    (`$h = 1$`, `$\nu = 1/Re$`), averaged over both walls: at each wall
    the stored ``tau'_s,*`` perturbation stress (signed, `$\nu\,
    \partial_y u'_s$`) plus the laminar one of
    :data:`LAMINAR_WALL_SHEAR`.  Cartesian base flows only.
    """
    system = str(params.phys.system)
    if system not in LAMINAR_WALL_SHEAR:
        raise ValueError(
            f"friction_reynolds supports {sorted(LAMINAR_WALL_SHEAR)}; "
            f"got {system!r}"
        )
    re = float(params.phys.re)
    lam_b, lam_t = LAMINAR_WALL_SHEAR[system]
    lo = abs(lam_b / re + float(stats["tau'_s,b"]))
    hi = abs(lam_t / re + float(stats["tau'_s,t"]))
    return float(np.sqrt(0.5 * (lo + hi)) * re)


def y_plus(y: float, re_tau: float) -> float:
    r"""Wall distance of *y* in wall units, from the nearer wall."""
    return float(min(y + 1.0, 1.0 - y) * re_tau)
