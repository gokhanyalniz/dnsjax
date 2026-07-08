#!/usr/bin/env python3
r"""CN/AB2 scheme guards (offline, in-process, no ``mpirun``).

Pins the structural claims of ``step.scheme == "cnab2"`` (see
``step_cnab2`` / ``_cnab2_lbf_core`` in ``timestep.py``) that the
subprocess smoke tests cannot check directly:

- **Split exactness**: the FFT-free base-flow coupling equals the
  base-flow part of the full nonlinear RHS to machine precision,
  ``get_rhs(u) - l_bf(u) == get_rhs(u; U = 0)`` (checked with
  ``step.implicit_mean_coupling`` off) -- the explicit AB2 forcing
  really is the pure self-advection `$u' \times \omega'$`.  The
  zero-base-flow oracle reuses the geometry ``_get_rhs`` on a
  shallow flow copy with zeroed ``base_flow_padded`` /
  ``curl_base_flow_padded`` (identical transforms, so the difference
  isolates exactly the coupling terms).  For the total-field Dean
  flow (``base_flow = 0``) the check is ``l_bf == 0`` identically.
- **Mean-flow coupling oracle**: with ``implicit_mean_coupling`` on
  (the default), ``l_bf`` gains exactly `$L_{mf} = \mathbf{u}
  \times \bar{\boldsymbol{\omega}} + \bar{\mathbf{u}} \times
  \boldsymbol{\omega}$` -- checked against a manually-written cross
  product with the mean profiles indexed directly off the single
  device's ``[:, :, 0, 0]`` mode (validating the ``psum``-based
  ``extract_mean_mode``, the profile broadcast, and the
  cylindrical/annular basis conversion).  For Dean this *is* the
  whole coupling (``l_bf == L_mf``).  Cartesian additionally checks
  the mean of the spectral curl equals the ``D1`` derivative of the
  mean profile (curl is mode-diagonal, so mean-of-curl ==
  curl-of-mean).
- **Carry-seed independence**: the AB2-history output of
  ``step_cnab2(state, carry)`` (the second element) is independent of
  the ``carry`` argument -- the ``__main__`` priming call
  ``step_cnab2(state, zeros)`` seeds the true
  `$N_{nl}(u^0)$` regardless of the dummy carry.
- **FFT-count guards** (jaxpr traversal): outside the ``lax.cond``
  fallback branch, one ``step_cnab2`` costs exactly one nonlinear
  RHS evaluation's FFTs (**1 FFT eval/step**); the implicit-coupling
  Picard ``while_loop`` body is **FFT-free**; the iterative-CN
  fallback (with its FFTs) exists only under a ``cond``.
  ``predict_and_fully_correct`` -- wall-bounded: the module stepper is
  the **unsplit** corrector (``step.split_corrector`` defaults off) --
  2 RHS evaluations plus one per corrector iteration, no ``cond``.
  Rebuilt with ``split_corrector = True`` (``_split_core`` in
  ``timestep.py``) it has the split shape: FFT-free coupling tail and
  solver loops as 0-FFT bodies, the outer refresh paying one RHS
  evaluation, and the unsplit-corrector fallback under a ``cond``.
  Both gate values are built explicitly here (the default is off, so
  the split shape is only reached via the forced-on rebuild).
  Triply-periodic ``step_cnab2`` (no ``l_bf_fn``) has no loop and no
  cond -- exactly one RHS evaluation total -- and its corrector is
  always unsplit.
- **Split-corrector equivalence** (wall-bounded): at a tight
  ``corrector_tolerance`` (1e-12; read at trace time, so no operator
  rebuild) the split and unsplit correctors converge the same step
  to the same CN fixed point (states agree to ~1e-10 relative) with
  the split outer loop genuinely entered (``num_c >= 1``) -- the FFT
  savings change the iteration path, never the converged answer.
- **Viscoelastic-dean split** (9-component total field, distinct
  split): ``_l_bf`` is FFT-free (jaxpr); with the mean coupling off
  it is exactly the polymer-stress divergence (velocity) + linear
  relaxation (conformation) to machine precision; and a mean-only
  state at `$\epsilon = 0$` has a vanishing explicit conformation
  remainder (``get_rhs`` conf == ``_l_bf`` conf), validating the
  conformation mean advection + stretching jointly (the identity
  `$I$` in the relaxation is gated to the mean mode -- a spectral
  subtlety ``_get_rhs_core`` gets for free in physical space).  See
  ``_check_viscoelastic_split``.

Each system runs in its own subprocess (import-time singletons), on a
single forced-CPU device, mirroring ``tests/test_localized_rolls.py``::

    uv run python tests/test_cnab2.py                  # all systems
    uv run python tests/test_cnab2.py --system pipe    # one system
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import subprocess
import sys

# Small but nontrivial resolutions (nz = 8: the complex-FFT axis
# rejects nz = 6 in the 3/2-rule padding).  Wall-bounded ny is the FD
# axis (odd is fine); triply-periodic ny is a Fourier axis (even).
NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
AMP, SMOOTH, SEED = 0.1, 0.4, 1
# Machine-precision bound for the split, relative to max|rhs|.
SPLIT_RTOL = 5e-13

SYSTEMS = [
    "plane-couette",
    "pipe",
    "taylor-couette",
    "dean",
    "viscoelastic-dean",
    "kolmogorov",
]

FLOW_MODULES = {
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "pipe": "dnsjax.flows.wall_bounded.pipe",
    "taylor-couette": "dnsjax.flows.wall_bounded.taylor_couette",
    "dean": "dnsjax.flows.wall_bounded.dean",
    "viscoelastic-dean": "dnsjax.flows.wall_bounded.viscoelastic_dean",
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
}

GEO_MODULES = {
    "plane-couette": "dnsjax.geometries.wall_bounded.cartesian",
    "pipe": "dnsjax.geometries.wall_bounded.cylindrical",
    "taylor-couette": "dnsjax.geometries.wall_bounded.annular",
    "dean": "dnsjax.geometries.wall_bounded.annular",
    "viscoelastic-dean": (
        "dnsjax.geometries.wall_bounded.annular_viscoelastic"
    ),
    "kolmogorov": "dnsjax.geometries.triply_periodic.triply_periodic",
}

# Geometry stepper builders (re-invoked to rebuild the steppers with
# ``step.split_corrector`` off for the gate-off checks).
STEPPER_BUILDERS = {
    "plane-couette": "build_cartesian_stepper",
    "pipe": "build_cylindrical_stepper",
    "taylor-couette": "build_annular_stepper",
    "dean": "build_annular_stepper",
    "viscoelastic-dean": "build_viscoelastic_stepper",
}


# ── parameter / JAX setup (forced single CPU device) ─────────────


def _configure(system: str) -> None:
    """Configure JAX and the dnsjax parameter singletons (1 CPU device).

    Must run before importing ``sharding`` / geometry modules.
    """
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    phys: dict = {"system": system, "re": 100.0}
    geo: dict = {"lx": LX, "lz": LZ}
    if system == "taylor-couette":
        phys.update(re1=100.0, re2=0.0)
        geo["eta"] = 0.5
    elif system == "dean":
        geo["eta"] = 0.5
    elif system == "viscoelastic-dean":
        # Re = wi/el is derived (no explicit re); eps = kappa = 0 so the
        # mean-only conformation invariant is exact (no nonlinear
        # relaxation, no diffusion).  delta defaults to 11.
        phys = {
            "system": system,
            "el": 20.0,
            "wi": 20.0,
            "beta": 0.8,
            "epsilon": 0.0,
            "kappa": 0.0,
        }
        geo = {"lx": LX}

    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY_PERIODIC if system == "kolmogorov" else NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            step={"scheme": "cnab2"},
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)


# ── jaxpr FFT-count traversal ────────────────────────────────────


def _sub_jaxprs(eqn) -> list:
    """All nested jaxprs of *eqn* (pjit/while/cond/scan bodies)."""
    subs = []
    for val in eqn.params.values():
        vals = val if isinstance(val, (tuple, list)) else (val,)
        for v in vals:
            if hasattr(v, "jaxpr"):  # ClosedJaxpr
                subs.append(v.jaxpr)
            elif hasattr(v, "eqns"):  # Jaxpr
                subs.append(v)
    return subs


def _count_ffts(jaxpr, in_cond: bool = False) -> tuple[int, int, list]:
    """Walk *jaxpr*; return ``(ffts_outside_cond, ffts_inside_cond,
    loop_body_fft_counts)``.

    ``ffts_outside_cond`` counts once-per-call FFT ops (not inside a
    ``cond`` branch and not repeated by a loop); ``ffts_inside_cond``
    counts FFT ops under any ``cond`` (the iterative-cn fallback);
    ``loop_body_fft_counts`` lists the per-iteration FFT count of every
    ``while``/``scan`` sub-jaxpr (cond and body) not inside a ``cond``,
    including loops nested inside other loop bodies (the split
    corrector's FFT-free coupling tail lives inside the outer
    corrector loop).
    """
    outside = inside = 0
    loops: list[int] = []
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "fft":
            if in_cond:
                inside += 1
            else:
                outside += 1
            continue
        child_in_cond = in_cond or name == "cond"
        is_loop = name in ("while", "scan")
        for sub in _sub_jaxprs(eqn):
            o, i, w = _count_ffts(sub, child_in_cond)
            if child_in_cond:
                inside += o + i
            else:
                inside += i
                if is_loop:
                    # Per-iteration cost: not a once-per-call FFT.
                    loops.append(o)
                    loops.extend(w)
                else:
                    outside += o
                    loops.extend(w)
    return outside, inside, loops


# ── viscoelastic-dean split checks (9-component, total field) ────


def _check_viscoelastic_split(gmod, fmod, state, fourier_, flow_) -> None:
    r"""Viscoelastic-dean CN/AB2 ``_l_bf`` split guards.

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
    and stretching jointly.  The velocity mean-flow coupling reuses the
    annular ``_l_bf`` already pinned by the ``dean`` entry.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dnsjax.parameters import params

    coef = (1.0 - params.phys.beta) / (params.phys.re * params.phys.wi)
    faclin = (1.0 - 3.0 * params.phys.epsilon) / params.phys.wi

    rhs = np.asarray(gmod._get_rhs(state, fourier_, flow_))
    scale = float(np.max(np.abs(rhs)))
    assert np.isfinite(scale) and scale > 0

    # (a) _l_bf is FFT-free (the premise of the 1-FFT/step scheme).
    lbf_jaxpr = jax.make_jaxpr(lambda s: gmod._l_bf(s, fourier_, flow_))(
        state
    ).jaxpr
    lbf_ffts, _, _ = _count_ffts(lbf_jaxpr)
    assert lbf_ffts == 0, f"viscoelastic _l_bf not FFT-free ({lbf_ffts})"
    print("viscoelastic-dean: _l_bf is FFT-free")

    # l_bf with the instantaneous mean-flow coupling off (the velocity
    # mean-flow coupling itself is pinned by the ``dean`` entry).
    assert params.step.implicit_mean_coupling
    params.step.implicit_mean_coupling = False
    try:
        l_off = np.asarray(gmod._l_bf(state, fourier_, flow_))
    finally:
        params.step.implicit_mean_coupling = True

    # (b) Mean coupling off => l_bf == polymer divergence (velocity) +
    # linear relaxation (conformation), to machine precision.
    cs = gmod._spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    div_r, div_th, div_z = gmod._div_c(*cs, fourier_, flow_)
    vel_div = coef * np.asarray(
        jnp.array([div_z, div_r + 1j * div_th, div_r - 1j * div_th])
    )
    d_pd = float(np.max(np.abs(l_off[:3] - vel_div)))
    assert d_pd <= SPLIT_RTOL * scale, (
        f"polymer-divergence oracle off by {d_pd:.3e} (scale {scale:.3e})"
    )

    # Linear relaxation in the spin basis: -(1 - 3 eps)/Wi (c - I), with
    # I_spin = (c_zz, c_+-) = (1, 2), other components 0 -- and the
    # identity supported at the mean mode only (constant field).
    ident_mm = np.asarray(jnp.where(fourier_.mean_mask, 1.0, 0.0))
    i_spin = np.zeros((6, *state.shape[1:]), dtype=np.asarray(state).dtype)
    i_spin[0] = ident_mm
    i_spin[3] = 2.0 * ident_mm
    relax = -faclin * (np.asarray(state[3:]) - i_spin)
    conf_scale = float(np.max(np.abs(np.asarray(state[3:]))))
    d_rel = float(np.max(np.abs(l_off[3:] - relax)))
    assert d_rel <= SPLIT_RTOL * conf_scale, (
        f"linear-relaxation oracle off by {d_rel:.3e} (scale {conf_scale})"
    )
    print("viscoelastic-dean: l_bf(mean off) == polymer div + linear relax")

    # (c) Mean-only conformation invariant at eps = 0: the explicit
    # conformation remainder get_rhs - l_bf vanishes for a mean-only
    # state (no fluctuation advection/stretching, no nonlinear
    # relaxation), so get_rhs conf == l_bf conf.  Velocity = laminar
    # profile (u_r == 0 exactly, so the dropped u_r d_r c term is truly
    # zero); conformation = 2x the laminar tensor -- a valid mean-mode
    # (Hermitian-partner) tensor that is *not* the equilibrium, so the
    # remainder scale is O(1/Wi) rather than machine zero.
    lam = fmod._laminar_state
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
        "viscoelastic-dean: mean-only conf get_rhs == l_bf "
        f"(rel {d_mo / cscale:.2e})"
    )


# ── worker (one system, singletons owned by this process) ────────


def _worker(system: str) -> None:
    _configure(system)

    import jax.numpy as jnp
    import numpy as np

    fmod = importlib.import_module(FLOW_MODULES[system])
    gmod = importlib.import_module(GEO_MODULES[system])

    from dnsjax.random_field import generate_random_state

    state = generate_random_state(AMP, SMOOTH, SEED)
    fourier_, flow_ = gmod.fourier, fmod.flow
    wall_bounded = system != "kolmogorov"

    # -- split exactness ------------------------------------------
    if system == "viscoelastic-dean":
        _check_viscoelastic_split(gmod, fmod, state, fourier_, flow_)
    elif wall_bounded:
        from dnsjax.parameters import derived_params, params

        rhs = np.asarray(gmod._get_rhs(state, fourier_, flow_))
        scale = np.max(np.abs(rhs))
        assert np.isfinite(scale) and scale > 0

        # l_bf with / without the instantaneous mean-flow coupling
        # (the param is read at trace time; _l_bf runs un-jitted here,
        # so flipping it takes effect immediately).
        assert params.step.implicit_mean_coupling
        l_bf_on = np.asarray(gmod._l_bf(state, fourier_, flow_))
        params.step.implicit_mean_coupling = False
        try:
            l_bf = np.asarray(gmod._l_bf(state, fourier_, flow_))
        finally:
            params.step.implicit_mean_coupling = True

        # Manual L_mf oracle in the coupling basis -- Cartesian
        # (u, v, w) or cylindrical/annular (u_z, u_r, u_theta) -- with
        # the mean profiles read directly off the single device's
        # (0, 0) mode entry.
        if system == "plane-couette":
            basis = state
        else:
            basis = jnp.array(
                [
                    state[0],
                    (state[1] + state[2]) / 2,
                    -1j * (state[1] - state[2]) / 2,
                ]
            )
        omega = gmod._curl_fn(basis, fourier_, flow_)
        u_np, om_np = np.asarray(basis), np.asarray(omega)
        u_m = u_np[:, :, :1, :1]
        om_m = om_np[:, :, :1, :1]
        l_mf = np.stack(
            [
                (u_np[1] * om_m[2] - u_np[2] * om_m[1])
                + (u_m[1] * om_np[2] - u_m[2] * om_np[1]),
                (u_np[2] * om_m[0] - u_np[0] * om_m[2])
                + (u_m[2] * om_np[0] - u_m[0] * om_np[2]),
                (u_np[0] * om_m[1] - u_np[1] * om_m[0])
                + (u_m[0] * om_np[1] - u_m[1] * om_np[0]),
            ]
        )
        if system != "plane-couette":
            l_mf = np.stack(
                [l_mf[0], l_mf[1] + 1j * l_mf[2], l_mf[1] - 1j * l_mf[2]]
            )
        diff_mf = np.max(np.abs(l_bf_on - (l_bf + l_mf)))
        assert diff_mf <= SPLIT_RTOL * scale, (
            f"{system}: mean-flow coupling != manual L_mf oracle "
            f"(max diff {diff_mf:.3e}, scale {scale:.3e})"
        )
        print(f"{system}: L_mf matches oracle to {diff_mf / scale:.2e}")

        if system == "plane-couette":
            # Curl is mode-diagonal: the mean of the spectral curl is
            # the D1 derivative of the mean profile,
            # curl(u, v, w)|_mean = (D1 w, 0, -D1 u).
            D1 = np.asarray(flow_.D1)
            prof, om_prof = u_m[:, :, 0, 0], om_m[:, :, 0, 0]
            np.testing.assert_allclose(om_prof[0], D1 @ prof[2], atol=1e-13)
            np.testing.assert_allclose(om_prof[2], -(D1 @ prof[0]), atol=1e-13)
            assert np.max(np.abs(om_prof[1])) <= 1e-15
            print(f"{system}: mean of curl == curl of mean (D1 oracle)")

        if system == "dean":
            # Total-field flow: base_flow = 0 => without the mean
            # coupling l_bf is identically 0; with it, l_bf == L_mf.
            assert np.max(np.abs(l_bf)) == 0.0, (
                f"dean: l_bf != 0 (max {np.max(np.abs(l_bf)):.3e})"
            )
            print(f"{system}: l_bf == 0 exactly (total-field, L_mf off)")
        else:
            # Zero-base-flow oracle: also zero the moving-frame speed
            # (pipe / plane-Poiseuille default to the bulk frame), so
            # the oracle is the pure self-advection u' x omega' and
            # the check covers the frame term's rhs/l_bf cancellation.
            flow0 = copy.copy(flow_)
            flow0.base_flow_padded = jnp.zeros_like(flow_.base_flow_padded)
            flow0.curl_base_flow_padded = jnp.zeros_like(
                flow_.curl_base_flow_padded
            )
            u_grid = derived_params.u_grid
            derived_params.u_grid = 0
            try:
                self_adv = np.asarray(gmod._get_rhs(state, fourier_, flow0))
            finally:
                derived_params.u_grid = u_grid
            diff = np.max(np.abs(rhs - l_bf - self_adv))
            assert diff <= SPLIT_RTOL * scale, (
                f"{system}: split not exact: max|get_rhs - l_bf - "
                f"self_adv| = {diff:.3e} (scale {scale:.3e}, "
                f"u_grid {u_grid})"
            )
            print(
                f"{system}: split exact to {diff / scale:.2e} "
                f"(rel; u_grid {u_grid})"
            )

    # -- carry-seed independence ----------------------------------
    # (step_cnab2 donates both arguments; pass copies so ``state``
    # stays alive for the jaxpr section below.)
    carry_a = fmod.step_cnab2(jnp.copy(state), jnp.zeros_like(state))[1]
    carry_b = fmod.step_cnab2(jnp.copy(state), jnp.copy(state))[1]
    assert np.array_equal(np.asarray(carry_a), np.asarray(carry_b)), (
        f"{system}: step_cnab2 carry output depends on the carry seed"
    )
    print(f"{system}: carry output seed-independent (bitwise)")

    # -- FFT-count guards -----------------------------------------
    import jax

    rhs_jaxpr = jax.make_jaxpr(lambda s: gmod._get_rhs(s, fourier_, flow_))(
        state
    ).jaxpr
    rhs_ffts, _, _ = _count_ffts(rhs_jaxpr)
    assert rhs_ffts > 0

    carry = jnp.zeros_like(state)
    step_jaxpr = jax.make_jaxpr(fmod.step_cnab2)(state, carry).jaxpr
    outside, inside, whiles = _count_ffts(step_jaxpr)

    assert outside == rhs_ffts, (
        f"{system}: step_cnab2 hot path has {outside} FFT ops, expected "
        f"exactly one RHS evaluation ({rhs_ffts})"
    )
    if wall_bounded:
        assert whiles and all(w == 0 for w in whiles), (
            f"{system}: implicit-coupling corrector loop not FFT-free "
            f"(per-body FFT counts {whiles})"
        )
        assert inside > 0, (
            f"{system}: no FFTs under cond -- iterative-cn fallback "
            "branch missing?"
        )
        print(
            f"{system}: 1 RHS eval/step ({rhs_ffts} FFT ops), FFT-free "
            f"corrector loop, fallback under cond ({inside} FFT ops)"
        )
    else:
        assert not whiles and inside == 0, (
            f"{system}: triply-periodic step_cnab2 should have no "
            f"corrector loop/cond (whiles={whiles}, cond FFTs={inside})"
        )
        print(f"{system}: plain 1-eval AB2 step ({rhs_ffts} FFT ops)")

    # iterative-cn: 2 evaluations + one per corrector iteration (the
    # 2 + c FFT-eval model).  ``step.split_corrector`` defaults off, so
    # the module-level stepper is the *unsplit* corrector for every
    # system: no inside FFTs, no fallback ``cond`` -- one RHS
    # evaluation per corrector ``while_loop`` iteration (the forced-on
    # split shape is checked below, on a rebuild).
    pfc_jaxpr = jax.make_jaxpr(fmod.predict_and_fully_correct)(state).jaxpr
    p_outside, p_inside, p_whiles = _count_ffts(pfc_jaxpr)
    p_nonzero = [w for w in p_whiles if w]
    assert (
        p_outside == 2 * rhs_ffts and p_inside == 0 and p_nonzero == [rhs_ffts]
    ), (
        f"{system}: predict_and_fully_correct FFT counts unexpected "
        f"(outside={p_outside}, inside={p_inside}, whiles={p_whiles}, "
        f"rhs={rhs_ffts})"
    )
    kind = "unsplit iterative-cn" if wall_bounded else "iterative-cn"
    print(
        f"{system}: {kind} (default) = 2 evals + "
        f"1 eval/corrector-iter ({p_inside} FFT ops)"
    )

    # -- split corrector: equivalence + gate-off structure ---------
    if wall_bounded:
        from dnsjax.parameters import params

        # Tight tolerance so both correctors genuinely iterate to the
        # shared CN fixed point.  It is read at trace time, so build
        # *fresh* steppers for both gate values (the module-level
        # stepper was already traced above with the default tolerance
        # baked in; no operator rebuild is involved either way).
        # Force the split path explicitly: its default is now
        # per-system (off for base-flow systems), and a direct
        # assignment before ``build`` is safe here -- no intervening
        # ``update_parameters`` re-resolves it (see the parameter
        # layering note in the root CLAUDE.md).
        saved = (
            params.step.corrector_tolerance,
            params.step.max_corrector_iterations,
            params.step.split_corrector,
        )
        params.step.corrector_tolerance = 1e-12
        params.step.max_corrector_iterations = 30
        build = getattr(gmod, STEPPER_BUILDERS[system])
        try:
            params.step.split_corrector = True
            split_pfc = build(fmod.flow)[3]
            s_split, err_split, c_split = split_pfc(jnp.copy(state))
            params.step.split_corrector = False
            unsplit_pfc = build(fmod.flow)[3]
            s_plain, err_plain, c_plain = unsplit_pfc(jnp.copy(state))
        finally:
            (
                params.step.corrector_tolerance,
                params.step.max_corrector_iterations,
                params.step.split_corrector,
            ) = saved

        assert int(c_split) >= 1, (
            f"{system}: split corrector loop never entered "
            "(equivalence check vacuous)"
        )
        assert float(err_split) <= 1e-12 and float(err_plain) <= 1e-12, (
            f"{system}: corrector(s) not converged "
            f"(split {float(err_split):.2e}, "
            f"unsplit {float(err_plain):.2e})"
        )
        scale = float(np.max(np.abs(np.asarray(s_plain))))
        diff = float(np.max(np.abs(np.asarray(s_split - s_plain))))
        assert diff <= 1e-10 * scale, (
            f"{system}: split != unsplit fixed point "
            f"(max diff {diff:.3e}, scale {scale:.3e})"
        )
        print(
            f"{system}: split == unsplit fixed point to "
            f"{diff / scale:.2e} rel (outer iters {int(c_split)} vs "
            f"{int(c_plain)})"
        )

        # Forced-on rebuild has the split-corrector shape: the outer
        # refresh contributes FFTs, the coupling tail / solver loops
        # are 0-FFT bodies, and the unsplit fallback sits under a
        # ``cond`` (the only split-shape coverage, as no system
        # defaults to the split corrector).
        on_jaxpr = jax.make_jaxpr(split_pfc)(state).jaxpr
        n_outside, n_inside, n_whiles = _count_ffts(on_jaxpr)
        assert (
            n_outside == 2 * rhs_ffts
            and n_inside > 0
            and [w for w in n_whiles if w] == [rhs_ffts]
        ), (
            f"{system}: split (forced-on) FFT counts unexpected "
            f"(outside={n_outside}, inside={n_inside}, "
            f"whiles={n_whiles}, rhs={rhs_ffts})"
        )
        print(f"{system}: split_corrector=True has the split shape")

        # Gate-off rebuild has the legacy unsplit corrector shape:
        # no fallback cond, one RHS evaluation per loop iteration.
        off_jaxpr = jax.make_jaxpr(unsplit_pfc)(state).jaxpr
        o_outside, o_inside, o_whiles = _count_ffts(off_jaxpr)
        assert (
            o_outside == 2 * rhs_ffts
            and o_inside == 0
            and [w for w in o_whiles if w] == [rhs_ffts]
        ), (
            f"{system}: gate-off predict_and_fully_correct FFT counts "
            f"unexpected (outside={o_outside}, inside={o_inside}, "
            f"whiles={o_whiles}, rhs={rhs_ffts})"
        )
        print(f"{system}: split_corrector=False restores the legacy shape")


# ── driver ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, default=None)
    parser.add_argument("--worker", choices=SYSTEMS, default=None)
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker)
        return

    print(
        "CN/AB2 scheme guards: offline, 1 forced CPU device per system "
        "(structural jaxpr / split checks; device-agnostic, no GPU "
        "path -- real GPU runs use `python -m dnsjax --dist.platform "
        "cuda`).",
        flush=True,
    )
    systems = [args.system] if args.system else SYSTEMS
    for system in systems:
        print(f"=== {system} ===", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, "--worker", system],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"{system}: worker failed")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
