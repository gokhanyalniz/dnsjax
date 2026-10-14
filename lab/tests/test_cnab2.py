r"""CN/AB2 scheme guards for the lab's flows.

The lab's cases of ``tests/test_cnab2.py`` (the worker:
``tests/_cnab2.py``):

- **Dean** (total field, ``base_flow = 0``): without the mean
  coupling ``l_bf`` is identically 0, and with it ``l_bf == L_mf`` --
  the mean-flow coupling is the whole coupling; the regime where the
  split and unsplit correctors differ most.
- **The viscoelastic split** (9-component total field, both
  geometries): ``_l_bf`` is FFT-free (jaxpr); with the mean coupling
  off it is exactly the polymer-stress divergence (velocity) + linear
  relaxation (conformation) to machine precision; and a mean-only
  state at `$\epsilon = 0$` has a vanishing explicit conformation
  remainder (``get_rhs`` conf == ``_l_bf`` conf), validating the
  conformation mean advection + stretching jointly (the identity
  `$I$` in the relaxation is gated to the mean mode -- a spectral
  subtlety ``_get_rhs_core`` gets for free in physical space).  See
  ``_check_viscoelastic_split``.

Run::

    uv run python lab/tests/test_cnab2.py
    uv run python lab/tests/test_cnab2.py --system dean
"""

from __future__ import annotations

import argparse

import _repo  # noqa: F401 -- the core suite's helpers
from _cnab2 import LX, SPLIT_RTOL, count_ffts, run_cases, worker


def _check_viscoelastic_split(gmod, fmod, state, fourier_, flow_) -> None:
    r"""Viscoelastic CN/AB2 ``_l_bf`` split guards (both geometries).

    The 9-component split differs from the perturbation flows: the
    velocity slice adds the polymer-stress divergence, the conformation
    slice is mean advection / stretching (gated by
    ``implicit_mean_coupling``) plus the always-implicit linear
    relaxation.  Checks: (a) ``_l_bf`` is FFT-free; (b) with the mean
    coupling off, ``_l_bf`` is exactly the polymer divergence (velocity)
    + linear relaxation (conformation), to machine precision
    (``SPLIT_RTOL``); (c) a mean-only state
    at `$\epsilon = 0$` has a vanishing explicit conformation remainder
    (``get_rhs`` conf == ``_l_bf`` conf), validating the mean advection
    and stretching jointly; (d) the moving-frame term is
    `$i k_z U_{\mathrm{grid}} u$` on all 9 slots in **both**
    ``get_rhs`` and ``_l_bf``.  The velocity mean-flow coupling reuses
    the geometry's own ``_l_bf``, already pinned by that geometry's
    Newtonian entry (``dean`` / ``pipe``).

    Why (d) is here and not covered by the Newtonian branch's
    ``get_rhs - _l_bf - self_adv`` check: this function *replaces* that
    check for the viscoelastic flows, (b) pins only ``_l_bf``'s copy of
    the frame term, and (c) runs on a mean-only state where the term
    vanishes identically -- so without (d) the one in ``_get_rhs_core``
    is pinned by nothing.  A regression dropping it would leave
    ``iterative-cn`` silently frame-free and ``cnab2`` with a spurious
    `$-i k_z U_{\mathrm{grid}} u$` in the explicit AB2 remainder, both
    of which still run.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dnsjax.parameters import derived_params, params

    coef = (1.0 - params.phys.beta) / (params.phys.re * params.phys.wi)
    faclin = (1.0 - 3.0 * params.phys.epsilon) / params.phys.wi

    rhs = np.asarray(gmod._get_rhs(state, fourier_, flow_))
    scale = float(np.max(np.abs(rhs)))
    assert np.isfinite(scale) and scale > 0

    # (a) _l_bf is FFT-free (the premise of the 1-FFT/step scheme).
    lbf_jaxpr = jax.make_jaxpr(lambda s: gmod._l_bf(s, fourier_, flow_))(
        state
    ).jaxpr
    lbf_ffts, _, _ = count_ffts(lbf_jaxpr)
    assert lbf_ffts == 0, f"viscoelastic _l_bf not FFT-free ({lbf_ffts})"
    print(f"{params.phys.system}: _l_bf is FFT-free")

    # l_bf with the instantaneous mean-flow coupling off (the velocity
    # mean-flow coupling itself is pinned by the ``dean`` entry).
    assert params.step.implicit_mean_coupling
    params.step.implicit_mean_coupling = False
    try:
        l_off = np.asarray(gmod._l_bf(state, fourier_, flow_))
    finally:
        params.step.implicit_mean_coupling = True

    # (b) Mean coupling off => l_bf == polymer divergence (velocity) +
    # linear relaxation (conformation) + the moving-frame convective
    # term, to machine precision.  The frame term is *always* implicit
    # (both slices, mode-diagonal) and is nonzero whenever the flow's
    # ``phys.u_grid`` is -- which is the case for the axially driven
    # viscoelastic pipe (default 1/2) but not for the azimuthally
    # driven Dean (default 0).
    frame = np.asarray(
        (1j * derived_params.u_grid) * fourier_.kz * state[:9]
        if derived_params.u_grid != 0
        else jnp.zeros_like(state[:9])
    )
    cs = gmod.spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    div_r, div_th, div_z = gmod._div_c(*cs, fourier_, flow_)
    vel_div = (
        coef
        * np.asarray(
            jnp.array([div_z, div_r + 1j * div_th, div_r - 1j * div_th])
        )
        + frame[:3]
    )
    d_pd = float(np.max(np.abs(l_off[:3] - vel_div)))
    assert d_pd <= SPLIT_RTOL * scale, (
        f"polymer-divergence reference off by {d_pd:.3e} (scale {scale:.3e})"
    )

    # Linear relaxation in the solver spin basis: -(1 - 3 eps)/Wi
    # (c - I), with I_spin = (c_zz, c_+-) = (1, 2), other components
    # 0 -- and the identity supported at the mean mode only (constant
    # field).
    ident_mm = np.asarray(jnp.where(fourier_.mean_mask, 1.0, 0.0))
    i_spin = np.zeros((6, *state.shape[1:]), dtype=np.asarray(state).dtype)
    i_spin[0] = ident_mm
    i_spin[3] = 2.0 * ident_mm
    # The conformation is slots 3..8 (a pipe state carries the velocity
    # pass's two spin-quad differences after it).
    relax = -faclin * (np.asarray(state[3:9]) - i_spin) + frame[3:]
    conf_scale = float(np.max(np.abs(np.asarray(state[3:9]))))
    d_rel = float(np.max(np.abs(l_off[3:] - relax)))
    assert d_rel <= SPLIT_RTOL * conf_scale, (
        f"linear-relaxation reference off by {d_rel:.3e} (scale {conf_scale})"
    )
    print(
        f"{params.phys.system}: l_bf(mean off) == polymer div + linear relax"
    )

    # (c) Mean-only conformation invariant at eps = 0: the explicit
    # conformation remainder get_rhs - l_bf vanishes for a mean-only
    # state (no fluctuation advection/stretching, no nonlinear
    # relaxation), so get_rhs conf == l_bf conf.  Velocity = laminar
    # profile (u_r == 0 exactly, so the dropped u_r d_r c term is truly
    # zero); conformation = 2x the laminar tensor -- a valid mean-mode
    # (Hermitian-partner) tensor that is *not* the equilibrium, so the
    # remainder scale is O(1/Wi) rather than machine zero.
    lam = gmod.to_spin_basis(fmod.init_state())
    mstate = jnp.concatenate([lam[:3], 2.0 * lam[3:]])
    rhs_m = np.asarray(gmod._get_rhs(mstate, fourier_, flow_))[3:]
    lbf_m = np.asarray(gmod._l_bf(mstate, fourier_, flow_))[3:]
    cscale = float(np.max(np.abs(rhs_m)))
    d_mo = float(np.max(np.abs(rhs_m - lbf_m)))
    assert d_mo <= 1e-10 * cscale, (
        f"mean-only conformation invariant off by {d_mo:.3e} "
        f"(conf scale {cscale:.3e})"
    )
    print(
        f"{params.phys.system}: mean-only conf get_rhs == l_bf "
        f"(rel {d_mo / cscale:.2e})"
    )

    # (d) The frame term, in get_rhs *and* _l_bf.  Difference each
    # against itself at u_grid = 0 (the term is a trace-time branch on
    # ``derived_params.u_grid``, and neither function is jitted here),
    # and compare against the analytic i k_z U_grid u on all 9 slots.
    # Probe a nonzero speed even where the flow ships 0 -- the annular
    # twin defaults to 0, but the code path is shared, so a flow that
    # later turns it on inherits a checked one.
    u_probe = derived_params.u_grid or 0.5
    saved = derived_params.u_grid
    try:
        derived_params.u_grid = 0.0
        rhs_0 = np.asarray(gmod._get_rhs(state, fourier_, flow_))
        lbf_0 = np.asarray(gmod._l_bf(state, fourier_, flow_))
        derived_params.u_grid = u_probe
        rhs_u = np.asarray(gmod._get_rhs(state, fourier_, flow_))
        lbf_u = np.asarray(gmod._l_bf(state, fourier_, flow_))
    finally:
        derived_params.u_grid = saved
    frame_probe = np.asarray((1j * u_probe) * fourier_.kz * state[:9])
    fscale = float(np.max(np.abs(frame_probe)))
    assert fscale > 0, "frame probe is identically zero"
    d_frhs = float(np.max(np.abs((rhs_u - rhs_0) - frame_probe)))
    d_flbf = float(np.max(np.abs((lbf_u - lbf_0) - frame_probe)))
    assert d_frhs <= SPLIT_RTOL * fscale, (
        f"get_rhs frame term != i k_z U_grid u by {d_frhs:.3e} "
        f"(scale {fscale:.3e}, u_grid probe {u_probe})"
    )
    assert d_flbf <= SPLIT_RTOL * fscale, (
        f"_l_bf frame term != i k_z U_grid u by {d_flbf:.3e} "
        f"(scale {fscale:.3e}, u_grid probe {u_probe})"
    )
    print(
        f"{params.phys.system}: frame term == i k_z U_grid u in get_rhs "
        f"and _l_bf (rel {max(d_frhs, d_flbf) / fscale:.2e}; "
        f"u_grid probe {u_probe})"
    )


# ── worker (one system, singletons owned by this process) ────────


#: Re = wi/el is derived (no explicit re); eps = kappa = 0 so the
#: mean-only conformation invariant is exact (no nonlinear relaxation,
#: no diffusion).  The annular delta defaults to 11; the pipe has no
#: radius parameter at all.
_SPTT = {"el": 20.0, "wi": 20.0, "beta": 0.8, "epsilon": 0.0, "kappa": 0.0}

#: ``system -> case`` (the keys: :func:`_cnab2.worker`).
CASES: dict[str, dict] = {
    "viscoelastic-pipe": {
        "stepper": (
            "dnsjax_lab.geometries.wall_bounded.cylindrical_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
        "phys": _SPTT,
        "geo": {"lx": LX},
        "split_check": _check_viscoelastic_split,
    },
    "dean": {
        "stepper": (
            "dnsjax.geometries.wall_bounded.annular:build_annular_stepper"
        ),
        "geo": {"lx": LX, "lz": 5.0, "eta": 0.5},
    },
    "viscoelastic-dean": {
        "stepper": (
            "dnsjax_lab.geometries.wall_bounded.annular_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
        "phys": _SPTT,
        "geo": {"lx": LX},
        "split_check": _check_viscoelastic_split,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(CASES), default=None)
    parser.add_argument("--worker", choices=list(CASES), default=None)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, CASES[args.worker])
        return
    run_cases(__file__, [args.system] if args.system else list(CASES))


if __name__ == "__main__":
    main()
