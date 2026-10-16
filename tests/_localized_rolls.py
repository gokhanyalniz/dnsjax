r"""Workers of the localized-rolls and random-field IC construction tests.

The machinery both suites' ``test_localized_rolls.py`` share: each
script lists its cases (``name -> case``) and runs them through
:func:`run_case`, which builds every case on three meshes, each in its
own worker subprocess (:func:`run_worker`), and compares their true
modes.  The checks and their tolerances: the core script's docstring.

A case is a dict of optional keys: ``system`` (the flow, the case name
unless given), ``phys`` / ``geo`` (overrides of ``phys.re = 100`` in an
``LX`` x ``LZ`` box; a ``None`` drops a default), ``random_only`` (run
the random-field half only, the random state carrying the rolls
state's checks), ``weight`` (the map onto the variables whose straight
divergence is the flow's continuity row, for :func:`max_divergence`),
``rolls_divergence`` (``(max|div|, true) -> (label, ok, detail)``, in
place of the default bound) and ``mean_check`` (``(state, zero_amp,
true_of) -> [(label, ok, detail), ...]``, in place of the exact-zero
mean-mode check).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import run_live

# Small resolutions chosen so the multi-device cases exercise spectral
# padding: nx // 2 = 5 (odd -> k_x padded for np1 = 2); nz - 1 = 11 (odd
# -> k_z / m padded for np0 = 2).
NX, NY, NZ = 10, 15, 12
# Kolmogorov's ``ny`` is a *Fourier* mode count, not a wall-normal grid
# size, so it must be even -- ``validate_parameters`` refuses an odd
# one, and ``NY = 15`` above is a grid size (see ``dnsjax.harmonics``).
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
# ``L / WAVELENGTH = 5/3`` is deliberately **not** an integer: that is
# the only regime in which a roll factor has a nonzero DC bin, so it is
# what makes the mean-mode check below a real guard.  At an integral
# ratio (the model defaults among them) the leak this checks for is
# machine-zero by coincidence, so the check could not fail.
AMP, WIDTH, WAVELENGTH = 0.1, 1.5, 3.0  # physical width / cross-roll wave


# Configurations (np0, np1) to build at; (1, 1) is the reference.
CONFIGS = [(1, 1), (1, 2), (2, 1)]

DIV_TOL = 5e-2  # loose truncation-level discrete-divergence bound
# Triply-periodic: every direction is spectral, so the divergence
# cancels analytically rather than to FD truncation order -- bound it
# *relative to the field scale* at machine precision (measured ~8e-17).
DIV_TOL_SPECTRAL = 1e-13

# Random-field IC guard: amplitude / smoothness / seed, and the
# *relative* divergence bound over the whole field, k_z = 0 included
# (the builders solve continuity on every mode, so this is machine-zero).
RAND_AMP, RAND_SMOOTH, RAND_SEED = 0.2, 0.4, 7
# The wall-normal pair, at the model defaults: the divergence bound
# below is what proves the scale-dependent wall window is structurally
# inert (it reshapes a column, it does not touch the closure).
RAND_WALL_SMOOTH, RAND_WALL_CONF = 0.4, 0.14
RAND_DIV_TOL = 1e-11

# Cross-device-count comparison tolerance for the random state (a
# ``random_only`` case's; the rolls state is compared bit-exactly).
# The random builders draw per column in host NumPy -- bit-exact by
# construction -- but their final rescaling divides by a *device-side
# global reduction* (``get_norm2_*`` over the sharded mode axes),
# whose summation order, and so whose last ulp, depends on the mesh.
# That single factor multiplies the whole block: measured <= 2 ulp
# relative, with the laminar mean-mode column -- added after the
# scaling -- bit-identical.  The structural property, exact discrete
# continuity, is bit-exact at every configuration and is checked as
# such above.
RAND_CROSS_TOL = 1e-13


# ── parameter / JAX setup (forced CPU devices) ───────────────────


def configure(case: dict, name: str, np0: int, np1: int) -> None:
    """Configure JAX and the parameter singletons on forced CPU devices.

    *np0* x *np1* devices, for the case *name* of a script's table:
    its flow (``system``, the name unless given) at ``phys.re = 100`` in
    an ``LX`` x ``LZ`` box, with the case's own ``phys`` / ``geo``
    overrides.  Must run before importing ``sharding`` / the geometry
    modules.
    """
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={np0 * np1}"
    )
    os.environ["NPROC"] = "1"

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.flows.registry import periodic_systems
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    system = case.get("system", name)
    phys = {"system": system, "re": 100.0, **case.get("phys", {})}
    phys = {k: v for k, v in phys.items() if v is not None}
    geo = {"lx": LX, "lz": LZ, **case.get("geo", {})}
    # Kolmogorov's ``ny`` is a Fourier mode count (even), not a grid size.
    periodic = system in periodic_systems
    update_parameters(
        Parameters(
            dist={"np0": np0, "np1": np1},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY_PERIODIC if periodic else NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())


# ── per-geometry discrete divergence (host numpy) ────────────────


def max_divergence(true: np.ndarray, weight=None) -> float:
    r"""Max |discrete divergence| of the true-mode state (host numpy).

    The FD matrices are built with the run's **resolved** wall-normal
    grid selection (``geo.grid_type`` etc.), not the builder defaults
    -- pipe resolves to ``half-cgl``, whose `$D_1$` differs from the
    default grid's, and a mismatched reference would silently measure
    the wrong operator.  The whole field is measured, including the
    `$k_z = 0$` plane: the random-field builders solve continuity
    on every mode (see :mod:`dnsjax.ic.random_field`).

    A case's *weight* maps the true-mode state onto the variables whose
    straight divergence is its flow's continuity row, when that is not
    the velocity itself (a metric-carrying flow's).
    """
    from dnsjax.flows.registry import (
        cartesian_systems,
        cylindrical_systems,
        periodic_systems,
    )
    from dnsjax.operators import complex_harmonics, real_harmonics
    from dnsjax.parameters import derived_params, params

    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    fd = params.res.fd_order
    grid_args = (
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    system = params.phys.system
    kx_real = np.asarray(real_harmonics(nx)) * (2 * np.pi / params.geo.lx)
    if system in periodic_systems:
        # Triply-periodic: components (u_x, u_y, u_z) over
        # [k_y, k_z, k_x], every derivative a multiply.  ``ly`` is the
        # geometry's own fixed shear-direction period, not a parameter.
        from dnsjax.geometries.triply_periodic.triply_periodic import ly

        ky = np.asarray(complex_harmonics(ny)) * (2 * np.pi / ly)
        kz = np.asarray(complex_harmonics(nz)) * (2 * np.pi / params.geo.lz)
        div = 1j * (
            kx_real[None, None, :] * true[0]
            + ky[:, None, None] * true[1]
            + kz[None, :, None] * true[2]
        )
        return float(np.max(np.abs(div)))

    if system in cartesian_systems:
        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_grid,
        )

        _, d1, _, _ = build_cartesian_grid(ny, fd, *grid_args)
        d1 = np.asarray(d1)
        kz = np.asarray(complex_harmonics(nz)) * (2 * np.pi / params.geo.lz)
        duy = np.einsum("ij,jzx->izx", d1, true[1])
        div = (
            duy
            + 1j * kz[None, :, None] * true[2]
            + 1j * kx_real[None, None, :] * true[0]
        )
        return float(np.max(np.abs(div)))

    # pipe / annular: native (u_z, u_r, u_theta) over (r, m, k_z,ax),
    # keyed on the geometry list.
    m = params.geo.m0 * np.asarray(complex_harmonics(nz))
    if system in cylindrical_systems:
        from dnsjax.geometries.wall_bounded.cylindrical import (
            build_cylindrical_grid,
        )

        _, d1_even, d1_odd, _, _, _, inv_r = build_cylindrical_grid(
            ny, fd, *grid_args
        )
        d1_even, d1_odd = np.asarray(d1_even), np.asarray(d1_odd)
        inv_r = np.asarray(inv_r)
    else:  # annular
        from dnsjax.geometries.wall_bounded.annular import build_annular_grid

        r1, r2 = derived_params.r_inner, derived_params.r_outer
        _, d1, _, _, inv_r = build_annular_grid(
            ny, fd, *grid_args, r_inner=r1, r_outer=r2
        )
        d1, inv_r = np.asarray(d1), np.asarray(inv_r)

    if weight is not None:
        true = weight(true)
    div = np.zeros_like(true[0])
    for im, mv in enumerate(m):
        if system in cylindrical_systems:
            d1v = d1_even if (mv + 1) % 2 == 0 else d1_odd
        else:
            d1v = d1
        ur, uth, uz = true[1][:, im, :], true[2][:, im, :], true[0][:, im, :]
        div_perp = (
            d1v @ ur + inv_r[:, None] * ur + 1j * mv * inv_r[:, None] * uth
        )
        div[:, im, :] = div_perp + 1j * kx_real[None, :] * uz
    return float(np.max(np.abs(div)))


# ── worker ───────────────────────────────────────────────────────


def run_worker(case: dict, name: str, np0: int, np1: int, out_npy: str) -> int:
    """Build *case*'s ICs on an *np0* x *np1* mesh and check them.

    The true-mode state goes to *out_npy* for the parent's cross-mesh
    comparison.  Exit status: 1 on any failed check.
    """
    configure(case, name, np0, np1)

    from dnsjax.flows.registry import cylindrical_systems, periodic_systems
    from dnsjax.parameters import params

    system = params.phys.system
    periodic = system in periodic_systems
    weight = case.get("weight")
    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Strip mesh padding -> true modes (padding is at the global axis end).
    def _true(state: np.ndarray) -> np.ndarray:
        return state[:3, :, : nz - 1, : nx // 2]

    true: np.ndarray | None = None
    if not case.get("random_only"):
        from dnsjax.ic.localized_rolls import generate_localized_rolls

        state1 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))
        state2 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))
        true = _true(state1)

        check("finite", bool(np.all(np.isfinite(state1))))
        check("determinism (build twice)", np.array_equal(state1, state2))

        # Exact no-slip at the wall nodes (axis 1 = y / r), on the
        # **velocity** block: a state carrying more than the velocity
        # (a total-field flow's laminar conformation, say) has wall
        # values of its own boundary conditions, not a Dirichlet zero.
        # ``[:3]`` is the whole state for every velocity-only flow.  A
        # triply-periodic flow has no wall node to check: its axis 1
        # is ``k_y``, and the spot is periodic there.
        if not periodic:
            vel1 = state1[:3]
            if system in cylindrical_systems:
                wall = float(np.max(np.abs(vel1[:, -1])))  # outer r = 1
            else:
                wall = max(
                    float(np.max(np.abs(vel1[:, 0]))),
                    float(np.max(np.abs(vel1[:, -1]))),
                )
            check(
                "exact no-slip at walls",
                wall < 1e-12,
                f"max|u|_wall={wall:.2e}",
            )

        div = max_divergence(true, weight)
        if "rolls_divergence" in case:
            check(*case["rolls_divergence"](div, true))
        elif periodic:
            # Fully spectral: the cancellation is analytic, so this is
            # a machine-precision statement, not a truncation bound.
            tscale = float(np.max(np.abs(true)))
            check(
                "divergence at machine precision",
                div < DIV_TOL_SPECTRAL * tscale,
                f"max|div|={div:.2e} scale={tscale:.2e}",
            )
        else:
            check(
                "divergence truncation-level",
                div < DIV_TOL,
                f"max|div|={div:.2e}",
            )

        # Reality (Kolmogorov only): the periodic family stores the
        # ``k_x = 0`` plane under the *joint* conjugate symmetry
        # ``f(k_y, k_z, 0) = conj(f(-k_y, -k_z, 0))``.  The rolls get it
        # from separability -- every 1-D factor is the transform of a
        # real signal -- where the random-field builder needs
        # ``_periodic_hermitian_raw``; this is what would catch a factor
        # that stopped being real.  The index flip below pairs
        # `$q \leftrightarrow -q$`, which every stored Fourier axis
        # supports because its mode count is even (``NY_PERIODIC``).
        if periodic:
            nky, nkz = ny - 1, nz - 1
            fy = np.array([0] + [nky - i for i in range(1, nky)])
            fz = np.array([0] + [nkz - i for i in range(1, nkz)])
            plane = true[:, :, :, 0]
            defect = float(
                np.max(np.abs(plane - np.conj(plane[:, fy][:, :, fz])))
            )
            check(
                "reality on the k_x = 0 plane",
                defect < DIV_TOL_SPECTRAL * float(np.max(np.abs(plane))),
                f"max defect={defect:.2e}",
            )

        # The spot contributes **nothing** to the (0, 0) mean mode, so
        # it moves neither the bulk velocity nor the wall shear.  Stated
        # against an ``amplitude = 0`` build so it reads the same for
        # the perturbation flows (whose mean column is then identically
        # zero) and the total-field ones (whose mean column carries the
        # analytical laminar profile).  Bit-exact, not a tolerance --
        # the roll factors' DC bins are hard zeros (``_zero_dc``).
        zero_amp = np.asarray(generate_localized_rolls(0.0, WIDTH, WAVELENGTH))
        if "mean_check" in case:
            for result in case["mean_check"](state1, zero_amp, _true):
                check(*result)
        else:
            same_mean = np.array_equal(
                state1[:, :, 0, 0], zero_amp[:, :, 0, 0]
            )
            moved = float(
                np.max(np.abs(state1[:, :, 0, 0] - zero_amp[:, :, 0, 0]))
            )
            check(
                "spot adds nothing to the (0, 0) mean mode",
                same_mean,
                f"max|delta u(0,0)|={moved:.2e}",
            )

    # ── random-field IC: exact discrete continuity on the *whole*
    # field, k_z = 0 plane included.  Unlike the analytic rolls
    # (truncation-level), the random builders *solve* continuity per mode
    # -- u_z for k_z != 0, u_theta for k_z = 0 (m != 0), u_r = 0 at the
    # mean -- so the residual is machine-zero, a sharp guard on the
    # per-geometry divergence expression each builder inverts.
    from dnsjax.ic.random_field import generate_random_state

    rand = np.asarray(
        generate_random_state(
            RAND_AMP, RAND_SMOOTH, RAND_WALL_SMOOTH, RAND_WALL_CONF, RAND_SEED
        )
    )
    rand_true = _true(rand)
    scale = float(np.max(np.abs(rand_true)))
    rdiv = max_divergence(rand_true, weight)
    check(
        "random IC divergence-free (full field)",
        rdiv < RAND_DIV_TOL * scale,
        f"max|div|={rdiv:.2e} scale={scale:.2e}",
    )

    if true is None:
        # Rolls-less flow: the random state carries the finiteness,
        # determinism and cross-device checks instead -- and it carries
        # them for *all* its components (the conformation block
        # included), not only the velocity block continuity is solved
        # on.
        check("finite", bool(np.all(np.isfinite(rand))))
        rand2 = np.asarray(
            generate_random_state(
                RAND_AMP,
                RAND_SMOOTH,
                RAND_WALL_SMOOTH,
                RAND_WALL_CONF,
                RAND_SEED,
            )
        )
        check("determinism (build twice)", np.array_equal(rand, rand2))
        true = rand[:, :, : nz - 1, : nx // 2]
    np.save(out_npy, true)

    print(f"\n[{system} {np0}x{np1}] {passed} passed, {failed} failed.")
    return 1 if failed else 0


# ── driver ───────────────────────────────────────────────────────


def _run_config(
    script: str, system: str, np0: int, np1: int, out_npy: str
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path(script).resolve()),
        "--system",
        system,
        "--np0",
        str(np0),
        "--np1",
        str(np1),
        "--out",
        out_npy,
    ]
    return run_live(cmd, timeout=300)


def run_case(script: str, case: dict, system: str) -> str | None:
    """Run all configs for one system; ``None`` on a full pass.

    On failure returns the one-line reason for the summary, quoting
    the worker's own first ``FAIL:`` line where it has one -- the
    parent otherwise only knows the exit code.
    """
    print(f"=== {system} ===")
    bad: list[str] = []
    ref: np.ndarray | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for np0, np1 in CONFIGS:
            out = str(Path(tmp) / f"true_{np0}x{np1}.npy")
            proc = _run_config(script, system, np0, np1, out)
            tag = f"np0={np0} np1={np1}"
            if proc.returncode != 0:
                print(f"  FAIL: {tag} worker exit {proc.returncode}")
                print(proc.stdout[-1500:])
                print(proc.stderr[-1500:])
                inner = next(
                    (
                        ln.strip()
                        for ln in proc.stdout.splitlines()
                        if ln.strip().startswith("FAIL:")
                    ),
                    "",
                )
                bad.append(
                    f"{tag} worker exit {proc.returncode}"
                    + (f" ({inner})" if inner else "")
                )
                continue
            arr = np.load(out)
            if ref is None:
                ref = arr
            else:
                same = arr.shape == ref.shape and (
                    np.allclose(arr, ref, rtol=RAND_CROSS_TOL, atol=0.0)
                    if case.get("random_only")
                    else np.array_equal(arr, ref)
                )
                print(
                    f"  {'PASS' if same else 'FAIL'}: {tag} matches (1, 1) "
                    "true modes"
                )
                if not same:
                    bad.append(f"{tag} differs from the (1, 1) true modes")
    return "; ".join(bad) if bad else None
