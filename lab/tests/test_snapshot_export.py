r"""Snapshot-export API tests for the lab's flows.

The lab's cases of ``tests/test_snapshot_export.py`` (the machinery:
``tests/_snapshot_export.py``), checked against the solver as the core
cases are:

- the **curved pipe**, the row with a metric,
  `$h = 1 + \kappa r\cos\theta$`.  Its references are the solver's own
  metric-weighted quantities with the weight divided out as
  ``get_stats`` does: the curl of the carried
  `$(h u_s, u_r, u_\theta)$`, and the continuity row
  `$h^2\nabla\cdot\mathbf{u}$` its corrector drives to zero.  They
  agree to roundoff because the fixture runs on the default
  ``phys.curvature_padding`` grid; on the 3/2 rule alone the solver's
  collocation of `$1/h$` would show at about `$10^{-9}$` here.  Its
  ``integrate`` checks pin where `$h$` enters the volume element.
- the **viscoelastic** flows: the conformation read-back (components
  3..8) against the solver's stored physical tensor slice -- the pipe's
  on the ``m0 = 2`` wedge, where the analysis package's tensor parity
  classes must be right.

Run directly::

    uv run python lab/tests/test_snapshot_export.py
"""

from __future__ import annotations

import os
import sys

import _repo  # noqa: F401 -- the core suite's helpers
import numpy as np
from _snapshot_export import VOL_TOL, main, solver_divergence

#: Curved-pipe curvature: the straight-pipe forms miss it by 3-5e-2 in
#: the curl and divergence checks.
KAPPA = 0.1


def _curved(state, probe, outdir):
    # The solver's metric-weighted quantities, with the weight divided
    # out by its own dealiased collocation (as get_stats does): the
    # curl of the carried (h u_s, u_r, u_th) is (om_s, h om_r, h om_th),
    # and its continuity row -- the straight divergence of the carried
    # state less divergence_defect -- is h^2 div u.
    import jax.numpy as jnp
    from dnsjax_lab.flows.wall_bounded import curved_pipe as cp
    from dnsjax_lab.flows.wall_bounded.curved_pipe import flow, get_stats

    from dnsjax.geometries.wall_bounded._base import from_pm_basis
    from dnsjax.geometries.wall_bounded._cylindrical_stepping import _curl_fn
    from dnsjax.geometries.wall_bounded.cylindrical import fourier
    from dnsjax.operators import phys_to_spec_2d, spec_to_phys_2d

    def _over_h(f, power):
        return phys_to_spec_2d(spec_to_phys_2d(f) * flow.inv_h_phys**power)

    carried = from_pm_basis(cp.to_solver_basis(state))
    omega_h = _curl_fn(carried, fourier, flow)
    omega = jnp.concatenate([omega_h[:1], _over_h(omega_h[1:], 1)])
    probe_w = cp.to_solver_basis(probe)
    row = solver_divergence(probe_w, flow, fourier) - flow.divergence_defect(
        probe_w, fourier
    )
    return get_stats, omega, _over_h(row[None], 2)[0]


def _curved_volume(system, st, info, ones) -> None:
    """The metric rides with s: the toroidal volume and cross-section."""
    from dnsjax.analysis import integrate

    # cos(theta) integrates to kappa pi L_s R^3 / 3 over the volume,
    # and to zero over each cross-section, whose element carries no h.
    r_max = float(np.asarray(st.physical_coords[0])[-1])
    cos_th = np.cos(st.physical_coords[1])[None, :, None] * ones
    got = float(integrate(cos_th, st.params, st.physical_coords))
    want = info.metric.curvature * np.pi * info.length[2] * r_max**3 / 3
    assert abs(got - want) / want < VOL_TOL, (
        f"{system}: volume metric {got} != {want}"
    )
    section = integrate(
        cos_th, st.params, st.physical_coords, directions=("r", "theta")
    )
    assert np.abs(section).max() < VOL_TOL, f"{system}: section metric"


def _viscoelastic_pipe(state, probe, outdir):
    # 9-component state in the *cylindrical* family: the velocity
    # divergence takes the pipe's parity-reduced path (on the [:3]
    # slice), and the ground truth for the tensor is the stored
    # conformation slice itself -- which on the m0 = 2 wedge is what
    # pins the analysis package's tensor parity classes.
    from dnsjax_lab.flows.wall_bounded.viscoelastic_pipe import flow, get_stats

    from dnsjax.geometries.wall_bounded._base import to_pm_basis
    from dnsjax.geometries.wall_bounded.cylindrical import fourier

    # ``probe`` is already the physical velocity triad rescaled per
    # component, so the 3-component velocity map is the right one (the
    # 9-component ``to_solver_basis`` would need the tensor).
    div_true = solver_divergence(to_pm_basis(probe), flow, fourier)
    np.save(os.path.join(outdir, "extra_true.npy"), np.asarray(state[3:]))
    # No omega dump: ``save_snapshot`` writes this system's **9**
    # components, and the velocity curl is already pinned node-for-node
    # by the core's 3-component ``pipe`` rows (same ``_curl_fn``, same
    # parity-reduced ``D1``).  What only this row can pin is the
    # *tensor* read-back on an even wedge.
    return get_stats, None, div_true


def _viscoelastic_dean(state, probe, outdir):
    # 9-component state; the velocity curl path is the annular one
    # (pinned by the core's taylor-couette case).  The ground truth
    # here is the stored conformation slice itself (format 6: the
    # native components 3..8 are the physical tensor (c_zz, c_rz,
    # c_thz, c_rr, c_thth, c_rth), read one-to-one; the solver state
    # axes (r, m, k_ax) ARE the snapshot layout read_state returns).
    from dnsjax_lab.flows.wall_bounded.viscoelastic_dean import get_stats

    np.save(os.path.join(outdir, "extra_true.npy"), np.asarray(state[3:]))
    return get_stats, None, None


#: re = wi/el = 1 (derived) for both sPTT flows.
_SPTT = {"wi": 5.0, "el": 5.0}
_CONF = {"random_conformation_amplitude": 10.0}

CASES: list[dict] = [
    {
        "system": "viscoelastic-pipe",
        "m0": 2,
        "phys": _SPTT,
        "init": _CONF,
        "fixtures": _viscoelastic_pipe,
        "exp_re": 1.0,
    },
    {
        # The one row with a metric (no wedge: curvature breaks the
        # symmetry).
        "system": "curved-pipe",
        "phys": {"re": 100.0},
        "geo": {"curvature": KAPPA},
        "fixtures": _curved,
        "extra_check": _curved_volume,
    },
    {
        "system": "viscoelastic-dean",
        "phys": _SPTT,
        "geo": {"eta": 0.5},
        "init": _CONF,
        "fixtures": _viscoelastic_dean,
        "exp_re": 1.0,
    },
]


if __name__ == "__main__":
    sys.exit(main(__file__, CASES))
