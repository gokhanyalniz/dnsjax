r"""Snapshot-figure helpers (JAX-free).

The friction Reynolds number of the snapshot being drawn, and the wall
units that follow from it -- what ``snapshot_figure.py`` labels a plane
with.  It sits here rather than in that script because it is the one
piece of per-flow physics a figure needs, and a second figure script
would want it too.

Sibling import: the scripts are run as ``python scripts/<name>.py``,
which puts ``scripts/`` on ``sys.path``, so ``from _figure_common
import ...`` resolves.  Nothing here imports JAX or the solver runtime.
"""

from __future__ import annotations

import numpy as np

#: Magnitude of the laminar wall velocity gradient, per Cartesian base
#: flow: `$U = y$` gives 1, `$U = 1 - y^2$` gives 2.  A snapshot stores
#: the perturbation about that profile, so the *total* wall shear --
#: and hence `$Re_\tau$` -- needs it added back.
LAMINAR_WALL_SHEAR = {"plane-couette": 1.0, "plane-poiseuille": 2.0}


def friction_reynolds(stats, params) -> float:
    r"""`$Re_\tau = u_\tau h/\nu$` from the snapshot's own wall shear.

    `$u_\tau = \sqrt{\nu\,|\mathrm{d}U/\mathrm{d}y|_w}$` in code units
    (`$h = 1$`, `$\nu = 1/Re$`), averaged over both walls: the stored
    ``tau'_s,*`` perturbation stresses plus the laminar contribution of
    :data:`LAMINAR_WALL_SHEAR`.  Cartesian base flows only.
    """
    system = str(params.phys.system)
    if system not in LAMINAR_WALL_SHEAR:
        raise ValueError(
            f"friction_reynolds supports {sorted(LAMINAR_WALL_SHEAR)}; "
            f"got {system!r}"
        )
    re = float(params.phys.re)
    lam = LAMINAR_WALL_SHEAR[system] / re
    lo = abs(lam + float(stats["tau'_s,b"]))
    hi = abs(-lam + float(stats["tau'_s,t"]))
    return float(np.sqrt(0.5 * (lo + hi)) * re)


def y_plus(y: float, re_tau: float) -> float:
    r"""Wall distance of *y* in wall units, from the nearer wall."""
    return float(min(y + 1.0, 1.0 - y) * re_tau)
