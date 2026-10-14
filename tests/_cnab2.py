r"""Workers of the CN/AB2 scheme tests, core and lab.

The per-case worker (:func:`worker`) both suites' ``test_cnab2.py``
share, and the jaxpr FFT counter (:func:`count_ffts`); what the worker
checks: the core script's docstring.  A flow whose RHS splits
differently from the base-flow form (a total field carries none; a
state with more than the velocity has terms of its own) checks the
rest of the worker's claims and brings its split check as the case's
``split_check``.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys

from _live import run_live

# Small but nontrivial resolutions (nz = 8: the complex-FFT axis
# rejects nz = 6 in the 3/2-rule padding).  Wall-bounded ny is the FD
# axis (odd is fine); triply-periodic ny is a Fourier axis (even).
NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
AMP, SMOOTH, SEED = 0.1, 0.4, 1
# The wall-normal pair of the random IC, at the model defaults
# (``init.random_wall_smoothness`` / ``random_wall_confinement``).
WALL_SMOOTH, WALL_CONF = 0.4, 0.14
# Machine-precision bound for the split, relative to max|rhs|.
SPLIT_RTOL = 5e-13


def _family(system: str) -> str:
    """The geometry module path (below ``dnsjax.geometries``) of a flow."""
    from dnsjax.flows.registry import spec_for

    geometry = spec_for(system).geometry
    if geometry == "triply-periodic":
        return "triply_periodic.triply_periodic"
    return f"wall_bounded.{geometry}"


# ── parameter / JAX setup (forced single CPU device) ─────────────


def configure(system: str, case: dict) -> None:
    """Configure JAX and the dnsjax parameter singletons (1 CPU device).

    *system* at ``phys.re = 100`` in an ``LX`` x ``LZ`` box, unless the
    *case* gives its own ``phys`` / ``geo``.  Must run before importing
    ``sharding`` / geometry modules.
    """
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.flows.registry import periodic_systems
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    phys = {"system": system, **case.get("phys", {"re": 100.0})}
    geo = case.get("geo", {"lx": LX, "lz": LZ})
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY_PERIODIC if system in periodic_systems else NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            step={"scheme": "cnab2"},
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())


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


def count_ffts(jaxpr, in_cond: bool = False) -> tuple[int, int, list]:
    """Count the FFT ops of *jaxpr* by where they run.

    Returns ``(ffts_outside_cond, ffts_inside_cond,
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
            o, i, w = count_ffts(sub, child_in_cond)
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


# ── worker ───────────────────────────────────────────────────────


def worker(system: str, case: dict) -> None:
    """The split, carry-seed, FFT-count and split-corrector checks.

    *case*: the flow's ``stepper`` builder (``"module:attr"``, absent
    for a flow with no corrector to rebuild), optional ``phys`` /
    ``geo``, and ``split_check`` -- ``(gmod, fmod, state, fourier,
    flow)``, in place of the split checks below, for a flow whose RHS
    splits differently.
    """
    configure(system, case)

    import jax.numpy as jnp
    import numpy as np

    from dnsjax.flows.registry import (
        cartesian_systems,
        periodic_systems,
        spec_for,
    )

    spec = spec_for(system)
    fmod = importlib.import_module(spec.flow_module)
    gmod_name, _, builder = case.get("stepper", "").partition(":")
    gmod = importlib.import_module(
        gmod_name or f"dnsjax.geometries.{_family(system)}"
    )

    from dnsjax.ic.random_field import generate_random_state

    # ICs are built in physical components; ``_get_rhs`` / ``_l_bf``
    # and the steppers work in the geometry's solver basis (the same
    # single crossing ``__main__`` performs).
    to_solver = getattr(fmod, "to_solver_basis", lambda s: s)
    state = to_solver(
        generate_random_state(AMP, SMOOTH, WALL_SMOOTH, WALL_CONF, SEED)
    )
    fourier_, flow_ = gmod.fourier, fmod.flow
    wall_bounded = system not in periodic_systems
    cartesian = system in cartesian_systems

    # -- split exactness ------------------------------------------
    if "split_check" in case:
        case["split_check"](gmod, fmod, state, fourier_, flow_)
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

        # Manual L_mf reference in the coupling basis -- Cartesian
        # (u, v, w) or cylindrical/annular (u_z, u_r, u_theta) -- with
        # the mean profiles read directly off the single device's
        # (0, 0) mode entry.
        if cartesian:
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
        if not cartesian:
            l_mf = np.stack(
                [l_mf[0], l_mf[1] + 1j * l_mf[2], l_mf[1] - 1j * l_mf[2]]
            )
        diff_mf = np.max(np.abs(l_bf_on - (l_bf + l_mf)))
        assert diff_mf <= SPLIT_RTOL * scale, (
            f"{system}: mean-flow coupling != manual L_mf reference "
            f"(max diff {diff_mf:.3e}, scale {scale:.3e})"
        )
        print(f"{system}: L_mf matches reference to {diff_mf / scale:.2e}")

        if cartesian:
            # Curl is mode-diagonal: the mean of the spectral curl is
            # the D1 derivative of the mean profile,
            # curl(u, v, w)|_mean = (D1 w, 0, -D1 u).
            D1 = np.asarray(flow_.D1.dense)
            prof, om_prof = u_m[:, :, 0, 0], om_m[:, :, 0, 0]
            np.testing.assert_allclose(om_prof[0], D1 @ prof[2], atol=1e-13)
            np.testing.assert_allclose(om_prof[2], -(D1 @ prof[0]), atol=1e-13)
            assert np.max(np.abs(om_prof[1])) <= 1e-15
            print(f"{system}: mean of curl == curl of mean (D1 reference)")

        if spec.total_field:
            # Total-field flow: base_flow = 0 => without the mean
            # coupling l_bf is identically 0; with it, l_bf == L_mf.
            assert np.max(np.abs(l_bf)) == 0.0, (
                f"{system}: l_bf != 0 (max {np.max(np.abs(l_bf)):.3e})"
            )
            print(f"{system}: l_bf == 0 exactly (total-field, L_mf off)")
        else:
            # Zero-base-flow reference: also zero the moving-frame speed
            # (pipe / plane-Poiseuille default to the bulk frame), so
            # the reference is the pure self-advection u' x omega' and
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
    # Seeds have the RHS's shape: the flow's physical components (a
    # pipe state also carries 2 solver slots).
    n_phys = spec.n_components
    carry_a = fmod.step_cnab2(jnp.copy(state), jnp.zeros_like(state[:n_phys]))[
        1
    ]
    carry_b = fmod.step_cnab2(jnp.copy(state), jnp.copy(state[:n_phys]))[1]
    assert np.array_equal(np.asarray(carry_a), np.asarray(carry_b)), (
        f"{system}: step_cnab2 carry output depends on the carry seed"
    )
    print(f"{system}: carry output seed-independent (bitwise)")

    # -- FFT-count guards -----------------------------------------
    import jax

    rhs_jaxpr = jax.make_jaxpr(lambda s: gmod._get_rhs(s, fourier_, flow_))(
        state
    ).jaxpr
    rhs_ffts, _, _ = count_ffts(rhs_jaxpr)
    assert rhs_ffts > 0

    carry = jnp.zeros_like(state[:n_phys])
    step_jaxpr = jax.make_jaxpr(fmod.step_cnab2)(state, carry).jaxpr
    outside, inside, whiles = count_ffts(step_jaxpr)

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
    p_outside, p_inside, p_whiles = count_ffts(pfc_jaxpr)
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
        # Force the split path explicitly (it is opt-in, default off
        # for every system).  A direct assignment before ``build`` is
        # safe here: unlike ``solver.backend`` / ``geo.grid_type``, no
        # intervening ``update_parameters`` re-resolves it (see that
        # function's docstring).
        saved = (
            params.step.corrector_tolerance,
            params.step.max_corrector_iterations,
            params.step.split_corrector,
        )
        params.step.corrector_tolerance = 1e-12
        params.step.max_corrector_iterations = 30
        build = getattr(gmod, builder)
        try:
            params.step.split_corrector = True
            split_pfc = build(fmod.flow)[1]
            s_split, err_split, c_split, *_ = split_pfc(jnp.copy(state))
            params.step.split_corrector = False
            unsplit_pfc = build(fmod.flow)[1]
            s_plain, err_plain, c_plain, *_ = unsplit_pfc(jnp.copy(state))
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
        n_outside, n_inside, n_whiles = count_ffts(on_jaxpr)
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

        # Gate-off rebuild has the default unsplit corrector shape:
        # no fallback cond, one RHS evaluation per loop iteration.
        off_jaxpr = jax.make_jaxpr(unsplit_pfc)(state).jaxpr
        o_outside, o_inside, o_whiles = count_ffts(off_jaxpr)
        assert (
            o_outside == 2 * rhs_ffts
            and o_inside == 0
            and [w for w in o_whiles if w] == [rhs_ffts]
        ), (
            f"{system}: gate-off predict_and_fully_correct FFT counts "
            f"unexpected (outside={o_outside}, inside={o_inside}, "
            f"whiles={o_whiles}, rhs={rhs_ffts})"
        )
        print(f"{system}: split_corrector=False has the unsplit shape")


# ── driver ───────────────────────────────────────────────────────


def run_cases(script: str, systems: list[str]) -> None:
    """Run *script*'s worker for each of *systems*, one process each."""
    print(
        "CN/AB2 scheme guards: offline, 1 forced CPU device per system "
        "(structural jaxpr / split checks; device-agnostic, no GPU "
        "path -- real GPU runs use `python -m dnsjax --dist.platform "
        "cuda`).",
        flush=True,
    )
    for system in systems:
        print(f"=== {system} ===", flush=True)
        result = run_live([sys.executable, script, "--worker", system])
        if result.returncode != 0:
            raise SystemExit(f"{system}: worker failed")
    print("ALL PASSED")
