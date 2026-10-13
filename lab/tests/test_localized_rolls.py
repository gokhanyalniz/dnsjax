r"""IC construction self-test for the lab's flows.

The lab's cases of ``tests/test_localized_rolls.py`` (the workers and
the checks: ``tests/_localized_rolls.py`` and the core script's
docstring):

- Dean and the viscoelastic pipe build both ICs; a total-field state
  carries its laminar profile, and the viscoelastic one its
  conformation block too, whose wall condition is ``grad^2 c = 0``, not
  a Dirichlet zero -- so no-slip is checked on the velocity block;
- viscoelastic Dean runs the random half only (``random_only``), its
  random state carrying the finiteness, determinism and cross-device
  checks; both 9-component flows exercise the builders'
  component-count-agnostic path;
- the curved pipe is measured on its solver's continuity row, the
  straight divergence of `$(u_s, h u_r, h u_\theta)$`
  (:func:`_metric_weighted`), rolls and random field alike.  Its rolls
  are held to roundoff on that row (``DIV_TOL_CURVED``): the straight
  rolls miss it by `$O(\kappa)$`, well inside the truncation bound.
  Its toroidal map gives `$u_\theta$` the `$O(\kappa)$` mean its
  continuity needs, so the spot's mean mode is exactly zero on `$u_s$`
  and zero to roundoff on `$(h u_r, h u_\theta)$` (``MEAN_TOL_CURVED``).

Run directly::

    uv run python lab/tests/test_localized_rolls.py
"""

from __future__ import annotations

import argparse
import sys

import _repo  # noqa: F401 -- the core suite's helpers
import numpy as np
from _live import report
from _localized_rolls import run_case, run_worker

sys.stdout.reconfigure(line_buffering=True)

# Curved pipe: its rolls are the pipe's mapped exactly onto toroidal
# continuity, and ``fd_order = 4`` differentiates the ``m0 = 1`` quartic
# profile exactly, so they meet the continuity row to roundoff -- bound
# it relative to the transverse pair (measured 9.9e-15 here, flat in
# kappa to 0.9, growing with the radial operator to 1.3e-13 at
# nr = 41).  The truncation bound would pass the straight rolls, which
# miss the row by 1.9e-2 of the pair.
DIV_TOL_CURVED = 1e-12
# ... and the (0, 0) mode of its (h u_r, h u_theta), relative to the
# same pair: measured <= 1.9e-16 up to kappa = 0.9 (4e-18 at the
# default).
MEAN_TOL_CURVED = 2e-15


def _metric_weighted(true: np.ndarray) -> np.ndarray:
    r"""Curved pipe: `$(u_s, h u_r, h u_\theta)$` of a true-mode state.

    The variables whose *straight* divergence is the solver's toroidal
    continuity row (``CurvedCylindricalFlow.divergence_defect``), with
    the exact truncating `$h$` product the solver applies.
    """
    from dnsjax.analysis._core import geometry_info
    from dnsjax.geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
    )
    from dnsjax.parameters import params

    rs, *_ = build_cylindrical_grid(
        params.res.ny,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    info = geometry_info(params)
    out = true.copy()
    for c in (1, 2):
        out[c] = info.metric.product(true[c], np.asarray(rs), info, 1)
    return out


def _curved_divergence(div: float, true: np.ndarray):
    """The rolls on the toroidal continuity row, at roundoff."""
    pscale = float(np.max(np.abs(true[1:3])))
    return (
        "toroidal divergence at roundoff",
        div < DIV_TOL_CURVED * pscale,
        f"max|div|={div:.2e} scale={pscale:.2e}",
    )


def _curved_mean(state, zero_amp, true_of):
    """The spot's mean mode: exact on u_s, roundoff on (h u_r, h u_theta).

    The map onto toroidal continuity gives u_theta the O(kappa) mean its
    m = +-1 modes need (``generate_curved_pipe_rolls``).  u_s, which
    alone carries the flux and the wall shear, is held bit-exactly; the
    pair continuity is straight in, to roundoff.
    """
    same_s = np.array_equal(state[0, :, 0, 0], zero_amp[0, :, 0, 0])
    spot = true_of(state - zero_amp)
    hmean = float(np.max(np.abs(_metric_weighted(spot)[1:3, :, 0, 0])))
    pscale = float(np.max(np.abs(spot[1:3])))
    return [
        ("spot adds nothing to the (0, 0) mode of u_s", same_s, ""),
        (
            "spot's (h u_r, h u_theta) (0, 0) mode at roundoff",
            hmean < MEAN_TOL_CURVED * pscale,
            f"max|h u(0,0)|={hmean:.2e} scale={pscale:.2e}",
        ),
    ]


#: ``name -> case`` (the keys: ``tests/_localized_rolls.py``).
CASES: dict[str, dict] = {
    "curved-pipe": {
        "weight": _metric_weighted,
        "rolls_divergence": _curved_divergence,
        "mean_check": _curved_mean,
    },
    "dean": {"geo": {"eta": 0.5}},
    # Re := Wi/El is derived (the annulus additionally takes its
    # geometry from ``geo.delta``); the rheology defaults (spec) are
    # left alone.
    "viscoelastic-dean": {"phys": {"re": None}, "random_only": True},
    "viscoelastic-pipe": {"phys": {"re": None}},
}


def main() -> None:
    if "--system" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--system", required=True)
        p.add_argument("--np0", type=int, default=1)
        p.add_argument("--np1", type=int, default=1)
        p.add_argument("--out", required=True)
        a = p.parse_args()
        sys.exit(run_worker(CASES[a.system], a.system, a.np0, a.np1, a.out))
    results = [
        (name, run_case(__file__, case, name)) for name, case in CASES.items()
    ]
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
