r"""Laminar smoke tests for the lab's wall-bounded flows.

The lab's table of ``tests/test_laminar_smoke.py`` (the runner:
``tests/_laminar_smoke.py``): quasi-Keplerian (full circle and the
``geo.m0 = 2`` wedge, whose azimuthal CFL doubles) is a base-flow
system checked like the core ones.  The total-field flows bring checks
of their own (each row's ``check``):

Dean flow is force-driven and integrates the **total** field, so it is
checked differently: started from the *analytical* laminar profile, its
``E'`` (the perturbation kinetic energy of the deviation from that
profile) stays tiny, the corrector still converges (``err``
`$O(10^{-14})$`), the energy balance `$I \approx D$` holds, and the
total energy is near-steady.  Its azimuthal ``CFL_th`` is the active
column (radial / axial are roundoff), positive and constant.

The two viscoelastic flows (total-field, 9-component) share a check,
differing only in which CFL column is active -- ``CFL_z`` for the
axially driven pipe (the same tight analytic value the Newtonian pipe
asserts), ``CFL_th`` for the azimuthally driven Dean.  At
`$\epsilon = \kappa = 0$` the analytical laminar velocity + sPTT
equilibrium conformation pair is the *exact* discrete fixed point, so
``E'`` stays tiny, the corrector converges (its error floor sits at FD
truncation, `$O(10^{-10})$`, not roundoff, because the conformation
carries magnitude `$O(10)$` -- hence a relaxed error threshold), the
polymer energy balance `$I \approx D_s - W_p$` holds, the energy is
near-steady, and ``steps.dat`` carries the extra ``TrC_max`` column.

``curved-pipe`` is deliberately **absent**: at its own curvature it has
no laminar fixed point to sit at (its laminar state is the
two-dimensional Dean solution, which is numerical), and at
``geo.curvature = 0`` every curvature term is switched off at trace
time, so the entry would pin the straight pipe a second time.  What it
would have covered is covered better elsewhere -- the `$\kappa = 0$`
identity with ``pipe`` to `$10^{-15}$` through five nonlinear steps in
``lab/tests/test_curved_pipe.py``, and the CLI path at finite curvature
by the ``curved-pipe`` entry of ``lab/tests/test_random_smoke.py``.

Usage::

    uv run python lab/tests/test_laminar_smoke.py
    uv run python lab/tests/test_laminar_smoke.py --np 2
"""

from __future__ import annotations

import math
import re

import _repo  # noqa: F401 -- the core suite's helpers
from _laminar_smoke import (
    CFL_GRID_LAMINAR_PIPE,
    D_PATTERN,
    E_PATTERN,
    EP_PATTERN,
    ERR_PATTERN,
    ERR_THRESHOLD,
    I_PATTERN,
    SMOKE_DT,
    SMOKE_NZ,
    main,
)

# ── configuration ────────────────────────────────────────────────────

# Quasi-Keplerian laminar azimuthal CFL: dt * max(U_theta/r) * nz/lz.
# Omega(r) = A0 + B0/r^2 is monotone decreasing (B0 > 0), so the maximum
# of U_theta/r is Omega(r1) = 1/r1 = (1-eta)/eta at the inner wall node
# (eta = 0.71).  The full circle (m0 = 1) uses lz = 2*pi; the m0 = 2
# wedge halves lz, doubling the azimuthal CFL.
_QK_ETA = 0.71
CFL_TH_LAMINAR_QK = (
    SMOKE_DT * ((1 - _QK_ETA) / _QK_ETA) * SMOKE_NZ / (2 * math.pi)
)
CFL_TH_LAMINAR_QK_WEDGE = 2 * CFL_TH_LAMINAR_QK

# E' is the deviation energy ||u - U_lam||^2 / 2 from the analytical
# laminar Dean profile.  Observed ~7e-24 at ny=27 with the default
# fd_order=8 (~2e-17 at fd_order=4).  The energy balance
# |I-D|/D (~8e-6) and total-energy drift (~3e-8) have generous margins,
# still catching a wrong forcing sign/magnitude (O(1) departures).
DEAN_EP_THRESHOLD = 5e-13
DEAN_IB_TOL = 1e-3
DEAN_E_DRIFT_TOL = 1e-4

# Viscoelastic diagnostics (total-field, 9-component).  The
# conformation carries magnitude O(10) (TrC ~ 70 at wi = 5), so the
# corrector error and E' floor sit at FD truncation (~3e-13 / ~1e-27
# at the default fd_order=8; ~1e-10 / ~1e-17 at fd_order=4),
# well above the perturbation flows' roundoff but far below the
# corrector tolerance.  The polymer energy balance is I = D_s - W_p
# (solvent dissipation minus polymer work; W_p < 0 as polymers dissipate).
DS_PATTERN = re.compile(r"D_s\s*=\s*([\d.eE+\-]+)")
WP_PATTERN = re.compile(r"W_p\s*=\s*([\d.eE+\-]+)")
VE_ERR_THRESHOLD = 1e-7
VE_EP_THRESHOLD = 1e-12
VE_BALANCE_TOL = 5e-3
VE_E_DRIFT_TOL = 1e-4


def check_dean(stdout: str, name: str) -> tuple[float, float]:
    """Validate a Dean (analytical near-steady laminar) run.

    Dean integrates the total field; the error metric is ``E'``, the
    perturbation kinetic energy of the deviation from the analytical
    laminar profile, alongside corrector convergence (``err``), the
    energy balance ``I ~= D``, and a near-steady total energy ``E``.
    Returns ``(last_err, last_ep)`` for the PASS summary.
    """
    last_err: float | None = None
    ep_vals: list[float] = []
    e_vals: list[float] = []
    i_vals: list[float] = []
    d_vals: list[float] = []
    for line in stdout.splitlines():
        m = ERR_PATTERN.search(line)
        if m:
            last_err = float(m.group(1))
        for pat, acc in (
            (EP_PATTERN, ep_vals),
            (E_PATTERN, e_vals),
            (I_PATTERN, i_vals),
            (D_PATTERN, d_vals),
        ):
            m = pat.search(line)
            if m:
                acc.append(float(m.group(1)))

    if last_err is None or not (ep_vals and e_vals and i_vals and d_vals):
        raise AssertionError(f"{name}: could not parse Dean diagnostics")

    if last_err > ERR_THRESHOLD:
        raise AssertionError(
            f"{name}: stepping error {last_err:.3e} > {ERR_THRESHOLD:.0e}"
        )

    last_ep = ep_vals[-1]
    if last_ep > DEAN_EP_THRESHOLD:
        raise AssertionError(
            f"{name}: deviation energy from laminar {last_ep:.3e} "
            f"> {DEAN_EP_THRESHOLD:.0e}"
        )

    last_i, last_d = i_vals[-1], d_vals[-1]
    if last_i <= 0.0 or last_d <= 0.0:
        raise AssertionError(
            f"{name}: non-positive I={last_i:.3e} or D={last_d:.3e}"
        )
    if abs(last_i - last_d) > DEAN_IB_TOL * last_d:
        raise AssertionError(
            f"{name}: energy balance off: I={last_i:.6e}, D={last_d:.6e}"
        )

    if abs(e_vals[-1] - e_vals[0]) > DEAN_E_DRIFT_TOL * e_vals[0]:
        raise AssertionError(
            f"{name}: energy drift {e_vals[0]:.6e} -> {e_vals[-1]:.6e}"
        )

    return last_err, last_ep


def check_viscoelastic(stdout: str, name: str) -> tuple[float, float]:
    r"""Validate a viscoelastic laminar run (`$\epsilon=\kappa=0$`).

    The analytical laminar pair is the exact discrete steady state, so
    ``E'`` (velocity deviation energy) is FD-truncation-tiny and the
    corrector converges; additionally the polymer energy balance
    ``I = D_s - W_p`` holds and the total energy is near-steady.  Returns
    ``(last_err, last_ep)`` for the PASS summary.
    """
    last_err: float | None = None
    ep_vals: list[float] = []
    e_vals: list[float] = []
    i_vals: list[float] = []
    ds_vals: list[float] = []
    wp_vals: list[float] = []
    for line in stdout.splitlines():
        m = ERR_PATTERN.search(line)
        if m:
            last_err = float(m.group(1))
        for pat, acc in (
            (EP_PATTERN, ep_vals),
            (E_PATTERN, e_vals),
            (I_PATTERN, i_vals),
            (DS_PATTERN, ds_vals),
            (WP_PATTERN, wp_vals),
        ):
            m = pat.search(line)
            if m:
                acc.append(float(m.group(1)))

    if last_err is None or not (
        ep_vals and e_vals and i_vals and ds_vals and wp_vals
    ):
        raise AssertionError(
            f"{name}: could not parse viscoelastic diagnostics"
        )

    if last_err > VE_ERR_THRESHOLD:
        raise AssertionError(
            f"{name}: stepping error {last_err:.3e} > {VE_ERR_THRESHOLD:.0e}"
        )

    last_ep = ep_vals[-1]
    if last_ep > VE_EP_THRESHOLD:
        raise AssertionError(
            f"{name}: deviation energy from laminar {last_ep:.3e} "
            f"> {VE_EP_THRESHOLD:.0e}"
        )

    # Polymer energy balance I = D_s - W_p at the near-steady state.
    last_i, last_ds, last_wp = i_vals[-1], ds_vals[-1], wp_vals[-1]
    balance = last_ds - last_wp
    if last_i <= 0.0:
        raise AssertionError(f"{name}: non-positive I={last_i:.3e}")
    if abs(last_i - balance) > VE_BALANCE_TOL * abs(last_i):
        raise AssertionError(
            f"{name}: energy balance off: I={last_i:.6e}, "
            f"D_s - W_p={balance:.6e}"
        )

    if abs(e_vals[-1] - e_vals[0]) > VE_E_DRIFT_TOL * e_vals[0]:
        raise AssertionError(
            f"{name}: energy drift {e_vals[0]:.6e} -> {e_vals[-1]:.6e}"
        )

    return last_err, last_ep


SYSTEMS: list[dict] = [
    {
        "name": "quasi-keplerian",
        "cfl_active": CFL_TH_LAMINAR_QK,
        "args": [
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "100",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
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
        "name": "quasi-keplerian-wedge",
        "cfl_active": CFL_TH_LAMINAR_QK_WEDGE,
        "args": [
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "100",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
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
        "name": "dean",
        "cfl_active": None,
        "check": check_dean,
        "args": [
            "--phys.system",
            "dean",
            "--geo.eta",
            "0.5",
            "--phys.re",
            "100",
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
        "name": "dean-block-spanwise",
        "cfl_active": None,
        "check": check_dean,
        "args": [
            "--phys.system",
            "dean",
            "--geo.eta",
            "0.5",
            "--phys.re",
            "100",
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
        # Viscoelastic (sPTT) Dean flow: total-field 9-component state.
        # At epsilon = kappa = 0 the analytical laminar pair (azimuthal
        # velocity + pointwise sPTT-equilibrium conformation) is the
        # *exact* discrete steady state, so E' (velocity deviation) stays
        # ~1e-18 and the corrector converges in a single step.  A modest
        # wi = el (=> Re = 1) keeps the conformation magnitude O(10) so
        # the corrector error is FD-truncation-, not overflow-, limited.
        # Checked differently from the perturbation flows (own branch):
        # near-steady energy and the polymer energy balance I = D_s - W_p.
        "name": "viscoelastic-dean",
        "cfl_active": None,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-dean",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
        # Viscoelastic (sPTT) pipe flow: the axially driven twin of
        # viscoelastic-dean, and the only 9-component flow whose axis
        # regularity is carried by the parity-reduced FD matrices -- so
        # this exercises the per-slot conformation parity signs, the
        # single-wall H_c BC row, and the uniform axial body force
        # Pi_z = 4/Re through a real run.  At epsilon = kappa = 0 the
        # analytical laminar pair (Hagen-Poiseuille velocity +
        # pointwise sPTT-equilibrium conformation) is the *exact*
        # discrete steady state.  Checked on the viscoelastic branch
        # (polymer balance I = D_s - W_p), but with the pipe's tight
        # analytic axial CFL rather than Dean's azimuthal one.
        "name": "viscoelastic-pipe",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-pipe",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
        # Viscoelastic pipe on an azimuthal wedge (geo.m0 = 2): the
        # tensor spin shifts m_eff = m + s ride the *physical*
        # m = m0 * harmonic, and so does the axis parity that selects
        # each slot's ghost sign -- the wedge is the only configuration
        # that can tell a physical-m selector from a harmonic-index one.
        "name": "viscoelastic-pipe-wedge",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-pipe",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
        # Viscoelastic azimuthal wedge (geo.m0 = 2): the axisymmetric
        # laminar state is a fixed point on the half-annulus wedge,
        # exercising the m0-scaled azimuthal wavenumbers through both
        # the velocity u_pm and the conformation tensor-spin operators
        # (m_eff = m + s with m = m0 * harmonic).
        "name": "viscoelastic-dean-wedge",
        "cfl_active": None,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-dean",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
        # Viscoelastic pipe flag-on: the spin quad carrying a
        # **total-field** 9-component state, so the mean-plane packing
        # must reproduce the Pi_z-vs-viscosity balance (as Dean's does
        # for force_theta) while the polymer stress rides the sources.
        "name": "viscoelastic-pipe-legacy-imm",
        "cfl_active": CFL_GRID_LAMINAR_PIPE,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-pipe",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
        # Quasi-Keplerian flag-on: the same annular machinery at a very
        # different (re1, r_omega, eta) corner.
        "name": "quasi-keplerian-legacy-imm",
        "cfl_active": CFL_TH_LAMINAR_QK,
        "args": [
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "100",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
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
        # Dean flag-on: a **total-field** flow, so the laminar state is
        # the mean-mode balance between the azimuthal body force
        # ``force_theta`` and viscosity.  That balance lives entirely in
        # the two packed k^2 = 0 slots, which is exactly what the
        # reformulation re-plumbs -- a packing error would show as a
        # drifting Ub_th here even though no perturbation exists.
        "name": "dean-legacy-imm",
        "cfl_active": None,
        "check": check_dean,
        "args": [
            "--phys.system",
            "dean",
            "--geo.eta",
            "0.5",
            "--phys.re",
            "100",
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
        # Viscoelastic Dean flag-on: the same composition guard in the
        # annular geometry, whose flag-on pass is the `(Phi, omega_r)`
        # pair rather than the pipe's spin quad.
        "name": "viscoelastic-dean-legacy-imm",
        "cfl_active": None,
        "step_columns": ("TrC_max",),
        "check": check_viscoelastic,
        "err_threshold": VE_ERR_THRESHOLD,
        "args": [
            "--phys.system",
            "viscoelastic-dean",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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
    main(SYSTEMS, console_checks=False)
