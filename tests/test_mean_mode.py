r"""Constraint-respecting `$(k_x, k_z) = (0, 0)$` perturbations.

Covers :mod:`dnsjax.ic.mean_mode` -- the mean-mode conservation laws a
Cartesian perturbation must respect -- at three levels:

1. **The constraint rows are the right discretization.**  Profiles that
   satisfy the relations analytically (`$\sin(k\pi(y+1)/2)$` for case A,
   `$\sin(m\pi y)$` and `$1-y^2$` for case B) score below
   ``COMPAT_TOL`` on every ``(grid_type, ny, fd_order)`` in the table
   the tolerance is read off, while a profile that genuinely violates a
   relation scores `$O(1)$` -- the separation the single tolerance
   rests on.  A compatible profile's *correction* also falls with
   ``ny``, which is what makes "these rows discretize those relations"
   an empirical statement rather than an assertion.
2. **The projector is a projector.**  Machine-level residual, exact
   idempotence, exactly preserved no-slip (the kernel's window factor),
   and no amplification in the ensemble it conditions -- including at
   an extreme ``random_smoothness``, where the smoothed case-B rows go
   near-degenerate and only ``_KERNEL_FLOOR`` keeps the solve honest.
3. **The generated IC satisfies them.**  A real random Cartesian IC
   with ``init.random_mean_flow`` on, across both flows, both driving
   knobs, tilt, both grid types and two ``fd_order``s: a real, wall-
   vanishing `$(0,0)$` column with no wall-normal component, machine-
   level residuals per tilted direction -- which include the
   vanishing wall curvature in **every** direction under **every**
   driving, the statement that the pair starts at one mean pressure
   gradient -- and an exactly zero bulk in each direction whose mean
   is held.  Plus device-count independence over ``(np0, np1)``, and
   the per-flow deferral for every flow whose laws are not
   established.

The rolls' own claim -- that their `$(0,0)$` content is a cubic, which
no compatibility-satisfying profile can be -- is checked here too; the
guard that they *stay* mean-free is
``tests/test_localized_rolls.py``.

Run directly::

    uv run python tests/test_mean_mode.py
    uv run python tests/test_mean_mode.py --unit-only   # skip the ICs
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _live import report, run_live  # noqa: E402

# ── Configuration ────────────────────────────────────────────────

#: Resolution of the in-process IC builds (small: the mean mode is one
#: column, and every other mode is unaffected by this feature).
NX, NY, NZ = 8, 33, 8

#: ``(system, driving, block_mean_spanwise, tilt, grid_type, fd_order)``
#: -- both flows, every reachable driving combination (plane-couette
#: refuses ``constant_bulk_velocity``), both grids, both wall cases.
CASES = [
    ("plane-couette", "constant_pressure_gradient", False, 0.0, "cgl", 6),
    ("plane-couette", "constant_pressure_gradient", True, 0.0, "cgl", 6),
    ("plane-couette", "constant_pressure_gradient", True, 30.0, "cgl", 4),
    ("plane-couette", "constant_pressure_gradient", False, 0.0, "tanh", 6),
    ("plane-poiseuille", "constant_pressure_gradient", False, 0.0, "cgl", 6),
    ("plane-poiseuille", "constant_bulk_velocity", False, 0.0, "cgl", 6),
    ("plane-poiseuille", "constant_bulk_velocity", True, 0.0, "cgl", 4),
    ("plane-poiseuille", "constant_bulk_velocity", True, 30.0, "cgl", 6),
    ("plane-poiseuille", "constant_bulk_velocity", True, 30.0, "tanh", 6),
]

#: Device meshes the `$(0,0)$` column must come out bit-identical on.
MESHES = [(1, 1), (1, 2), (2, 1)]

#: Machine-level bound for a residual the projector zeroed.  Loose
#: against roundoff in the ``D_2`` wall rows, whose entries grow like
#: ``ny^4`` on a wall-clustered grid (``mean_mode``'s table).
EXACT_TOL = 1e-11

#: A genuinely violating profile must score at least this, so the
#: single ``COMPAT_TOL`` separates the two populations by decades.
VIOLATION_FLOOR = 0.1

#: Relative agreement of the `$(0,0)$` column across device meshes.
CROSS_TOL = 1e-13

RAND_AMP, RAND_SMOOTH, RAND_SEED = 0.1, 0.4, 3
# The wall-normal pair, at the shipped defaults.  ``RAND_WALL_CONF``
# is inert at the (0, 0) column the projector acts on -- the window is
# scaled by |k| -- so it only shapes the surrounding modes here.
RAND_WALL_SMOOTH, RAND_WALL_CONF = 0.4, 0.14


def _grid(kind: str, ny: int, order: int):
    """``(y, D1, D2, w)`` for a Cartesian wall-normal grid."""
    from dnsjax.fd import (
        build_diff_matrices,
        build_integration_weights,
        clenshaw_curtis_weights,
        tanh_two_sided_grid,
    )

    if kind == "cgl":
        y = -np.cos(np.arange(ny) * np.pi / (ny - 1))
        w = clenshaw_curtis_weights(ny)
    else:
        y = tanh_two_sided_grid(ny, 2.0)
        w = build_integration_weights(y, order)
    D1, D2 = build_diff_matrices(y, order)
    return y, D1, D2, w


def _compatible(y, case_b: bool):
    """Analytically compatible profiles on *y* for the given case.

    Restricted to modes the grid resolves (>= 8 points per
    wavelength): an unresolved profile is not a fair witness for "the
    discrete rows track the continuum relations", and scores up to
    ``0.3`` for exactly that reason (``mean_mode``'s table).
    """
    ny = len(y)
    kmax = max(1, ny // 8)
    if not case_b:
        return [np.sin(k * np.pi * (y + 1) / 2) for k in range(1, kmax + 1)]
    return [np.sin(m * np.pi * y) for m in range(1, max(1, kmax // 2) + 1)]


# ── 1. The constraint rows ───────────────────────────────────────


def test_tolerance_table(check) -> None:
    """Compatible profiles pass, violating ones fail, by decades."""
    from dnsjax.ic.mean_mode import COMPAT_TOL, constraint_residuals

    worst_ok, least_bad = 0.0, np.inf
    for kind in ("cgl", "tanh"):
        for ny in (17, 25, 33, 65, 129):
            for order in (4, 6, 8):
                y, D1, D2, w = _grid(kind, ny, order)
                for case_b in (False, True):
                    for d in _compatible(y, case_b):
                        worst_ok = max(
                            worst_ok,
                            constraint_residuals(
                                d, D1, D2, w, fixed_bulk=case_b
                            ).max(),
                        )
                # 1 - y^2 violates every row of both cases -- the
                # curvature pair (-2 at both walls), the pressure
                # gradient (its wall shears are +-2) and the bulk --
                # the witness the tolerance is set against.
                pois = 1.0 - y**2
                least_bad = min(
                    least_bad,
                    constraint_residuals(
                        pois, D1, D2, w, fixed_bulk=False
                    ).min(),
                    constraint_residuals(
                        pois, D1, D2, w, fixed_bulk=True
                    ).min(),
                )
    check(
        "compatible profiles score below COMPAT_TOL",
        worst_ok <= COMPAT_TOL,
        f"worst={worst_ok:.2e} <= {COMPAT_TOL:.0e}",
    )
    check(
        "violating profiles score O(1)",
        least_bad >= VIOLATION_FLOOR,
        f"least={least_bad:.2e} >= {VIOLATION_FLOOR:g}",
    )

    # The derivation's sharpest anchor: the even quartic
    # (1-y^2)(5-y^2) is *exact* on both curvature rows -- which are
    # literally the same rows in both cases -- and O(1) on each of the
    # two rows case B adds, so the four rows really are independent
    # statements.  A polynomial of degree <= fd_order is reproduced
    # exactly by D1/D2, so this can be a machine-level check; a
    # transcendental witness would carry the CGL near-wall D2 roundoff
    # floor (~2e-10 at ny=65) instead.  Its wall curvatures are 0, its
    # wall shears -+8 (so Delta Pi != 0) and its bulk 6.4.
    y, D1, D2, w = _grid("cgl", 65, 6)
    quartic = (1.0 - y**2) * (5.0 - y**2)
    res_a = constraint_residuals(quartic, D1, D2, w, fixed_bulk=False)
    res_b = constraint_residuals(quartic, D1, D2, w, fixed_bulk=True)
    check(
        "(1-y^2)(5-y^2): the curvature rows are case-independent",
        max(res_a.max(), res_b[:2].max()) < EXACT_TOL
        and np.allclose(res_a, res_b[:2], atol=EXACT_TOL),
        f"A {res_a.max():.1e}, B {res_b[:2].max():.1e}",
    )
    check(
        "(1-y^2)(5-y^2) violates case B's pressure-gradient row",
        abs(res_b[2] - 1.0) < 1e-12,
        f"{res_b[2]:.6f} (= (16/2)/8)",
    )
    check(
        "(1-y^2)(5-y^2) violates case B's bulk row",
        abs(res_b[3] - 0.64) < 1e-12,
        f"{res_b[3]:.6f} (= 6.4/(2*5))",
    )
    # And the laminar shape violates all four, in both cases.
    res = constraint_residuals(1.0 - y**2, D1, D2, w, fixed_bulk=True)
    check(
        "1 - y^2 violates every case-B row",
        res[:3].min() > 1.0 - 1e-12 and abs(res[3] - 2.0 / 3.0) < 1e-12,
        np.array2string(res, precision=3),
    )


def test_correction_converges(check) -> None:
    """The correction to a compatible profile falls with ``ny``.

    The empirical content of "these discrete rows are the relations":
    feed the projector a profile that satisfies them in the continuum
    and the only thing it removes is truncation.
    """
    from dnsjax.ic.mean_mode import (
        constraint_rows,
        project_profile,
        smoothing_kernel,
    )
    from dnsjax.ic.random_field import _wall_normal_filter

    for case_b in (False, True):
        sizes = (17, 33, 65)
        corr = []
        for ny in sizes:
            y, D1, D2, w = _grid("cgl", ny, 6)
            K = smoothing_kernel(1.0 - y**2, _wall_normal_filter(y, 0.6))
            C = constraint_rows(D1, D2, w, fixed_bulk=case_b)
            d = np.sin(np.pi * y) if case_b else np.sin(np.pi * (y + 1) / 2)
            corr.append(float(np.max(np.abs(project_profile(d, C, K) - d))))
        label = "B" if case_b else "A"
        check(
            f"case {label}: correction falls with ny",
            corr[-1] < corr[0] / 100.0,
            f"ny={sizes} -> " + ", ".join(f"{c:.1e}" for c in corr),
        )


def test_rolls_cubic_is_inadmissible(check) -> None:
    """No nonzero cubic satisfies the compatibility conditions.

    The reason ``ic/localized_rolls`` has no admissible mean mode to
    keep: its `$(0,0)$` content is `$-G'(y)$` with `$G = (1-y^2)^2$`.
    """
    from dnsjax.ic.mean_mode import constraint_residuals

    y, D1, D2, w = _grid("cgl", 65, 6)
    g_prime = -4.0 * y * (1.0 - y**2)
    check(
        "the rolls' (0,0) shape is the cubic -4y(1-y^2)",
        np.max(np.abs(D1 @ (1.0 - y**2) ** 2 - g_prime)) < 1e-12,
        "",
    )
    # Every cubic vanishing at both walls, spanned by y(1-y^2) (odd,
    # the roll shape) and (1-y^2) (even).
    worst = np.inf
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for b in (-1.0, -0.5, 0.5, 1.0):
            d = a * y * (1.0 - y**2) + b * (1.0 - y**2)
            worst = min(
                worst,
                constraint_residuals(d, D1, D2, w, fixed_bulk=False).max(),
            )
    check(
        "no nonzero wall-vanishing cubic satisfies case A",
        worst >= VIOLATION_FLOOR,
        f"least residual over 20 cubics = {worst:.2e}",
    )


# ── 2. The projector ─────────────────────────────────────────────


def test_projector_properties(check) -> None:
    """Residual, idempotence, no-slip, and non-amplification."""
    from dnsjax.ic.mean_mode import (
        constraint_residuals,
        constraint_rows,
        project_profile,
        smoothing_kernel,
    )
    from dnsjax.ic.random_field import _wall_normal_filter

    y, D1, D2, w = _grid("cgl", 65, 6)
    win = 1.0 - y**2
    for smooth in (0.4, 0.95, 0.99):
        F = _wall_normal_filter(y, 1.0 - smooth)
        K = smoothing_kernel(win, F)
        for case_b in (False, True):
            tag = f"s={smooth} case {'B' if case_b else 'A'}"
            C = constraint_rows(D1, D2, w, fixed_bulk=case_b)
            rng = np.random.default_rng(0)
            raw = win * (F @ rng.standard_normal(len(y)))
            out = project_profile(raw, C, K)
            res = constraint_residuals(out, D1, D2, w, fixed_bulk=case_b).max()
            # ``constraint_residuals`` is *relative* to the profile it
            # is handed, and at an extreme smoothness case B legitimately
            # removes all but ~2 % of the draw (``keep``) -- the same
            # absolute roundoff then reads inflated by ``1/keep``, and
            # measurably so: the residual tracks ``1/keep`` across these
            # six rows.  Bound it against what survives, so the check
            # stays tight where the projector keeps the profile
            # (``keep ~ 0.8`` at the default smoothness) rather than
            # being loosened wholesale.
            keep = float(out @ out) / float(raw @ raw)
            check(
                f"{tag}: residual at machine level",
                res * min(keep, 1.0) < EXACT_TOL,
                f"{res:.2e} (keep={keep:.3f})",
            )
            # Idempotent as algebra; the recompute differs only by
            # roundoff, so this is a scaled bound, not an equality
            # (unlike the no-slip check below, which is exact by
            # construction -- the kernel's window factor).
            again = project_profile(out, C, K)
            drift = float(np.max(np.abs(again - out)))
            check(
                f"{tag}: idempotent",
                drift <= 1e-13 * float(np.max(np.abs(out))),
                f"max|P^2-P| = {drift:.2e}",
            )
            check(
                f"{tag}: no-slip preserved exactly",
                out[0] == 0.0 and out[-1] == 0.0,
                f"walls = {out[0]:.1e}, {out[-1]:.1e}",
            )
            # E||d'||^2 <= E||d||^2: the projection is orthogonal in
            # the ensemble's own metric (module docstring).
            rng = np.random.default_rng(11)
            num = den = 0.0
            for _ in range(200):
                d = win * (F @ rng.standard_normal(len(y)))
                pd = project_profile(d, C, K)
                den += float(d @ d)
                num += float(pd @ pd)
            check(
                f"{tag}: does not amplify",
                num <= den,
                f"E|d'|^2/E|d|^2 = {num / den:.4f}",
            )


# ── 3. Per-flow surface ──────────────────────────────────────────


def test_per_flow_surface(check) -> None:
    """Cartesian defaults on; every other flow rejects the knob."""
    from dnsjax.flows.registry import (
        all_systems,
        cartesian_systems,
        internalize_stored,
        spec_for,
    )
    from dnsjax.parameters import (
        Geometry,
        Initiation,
        Parameters,
        Physics,
        Resolution,
        params,
        update_parameters,
        validate_parameters,
    )

    key = ("init", "random_mean_flow")
    for system in all_systems():
        spec = spec_for(system)
        want = system in cartesian_systems
        check(
            f"{system}: surface carries the knob = {want}",
            (key in spec.field_map) is want
            and (key in spec.deferred_map) is not want,
            "",
        )

    update_parameters(
        Parameters(
            phys=Physics(system="plane-couette", re=400.0),
            geo=Geometry(lx=4.0, lz=4.0),
            res=Resolution(nx=NX, ny=NY, nz=NZ),
        )
    )
    validate_parameters()
    check(
        "plane-couette defaults off",
        params.init.random_mean_flow is False,
    )

    for system, re in (("pipe", 2000.0), ("kolmogorov", 400.0)):
        update_parameters(
            Parameters(
                phys=Physics(system=system, re=re),
                res=Resolution(nx=NX, ny=NY, nz=NZ),
                init=Initiation(random_mean_flow=False),
            )
        )
        validate_parameters()  # the inert default must pass
        params.init.random_mean_flow = True
        try:
            validate_parameters()
            ok, why = False, "accepted"
        except ValueError as exc:
            ok, why = "random_mean_flow" in str(exc), str(exc)[:40]
        check(f"{system}: direct assignment rejected", ok, why)
        params.init.random_mean_flow = False

    # Snapshots written *before* the deferral recorded the field on
    # every wall-bounded and periodic surface.  ``externalize`` skips
    # deferred fields, so new ones do not -- but the old ones must stay
    # loadable, which is why ``internalize_stored`` resolves a deferred
    # key instead of treating it as unknown.
    stored = {
        "init": {
            "random_field": True,
            "random_amplitude": 0.1,
            "random_seed": 1,
            "random_mean_flow": False,
        }
    }
    for system in all_systems():
        try:
            got = internalize_stored(stored, system)["init"]
            ok = got.get("random_mean_flow") is False
            why = str(got.get("random_mean_flow", "<dropped>"))
        except ValueError as exc:
            ok, why = False, str(exc)[:50]
        check(f"{system}: a pre-deferral snapshot still loads", ok, why)
    try:
        internalize_stored({"init": {"no_such_field": 1}}, "pipe")
        ok, why = False, "accepted"
    except ValueError as exc:
        ok, why = "no_such_field" in str(exc), str(exc)[:40]
    check("a genuinely unknown stored key still raises", ok, why)


# ── 4. The generated initial condition (subprocess) ──────────────


def _configure_jax() -> None:
    """Enable float64 before JAX initializes any array.

    The worker's ``XLA_FLAGS`` device count and platform come from
    :func:`_run_worker`; ``res.double_precision`` only records the
    *intent*, so x64 has to be turned on here or every check below
    silently measures float32 roundoff instead.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


def _worker_matrix() -> int:
    """Build a random IC per :data:`CASES` and check its (0,0) column."""
    _configure_jax()

    from dnsjax.parameters import (
        Geometry,
        Initiation,
        Parameters,
        Physics,
        Resolution,
        derived_params,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  {detail}")
        passed, failed = passed + bool(ok), failed + (not ok)

    for system, driving, block, tilt, kind, order in CASES:
        phys = dict(
            system=system, re=400.0, block_mean_spanwise_velocity=block
        )
        if system == "plane-poiseuille":
            phys["driving"] = driving
        update_parameters(
            Parameters(
                phys=Physics(**phys),
                geo=Geometry(lx=4.0, lz=4.0, tilt_degree=tilt, grid_type=kind),
                res=Resolution(
                    nx=NX,
                    ny=NY,
                    nz=NZ,
                    fd_order=order,
                    double_precision=True,
                ),
                init=Initiation(
                    random_mean_flow=True,
                    random_amplitude=RAND_AMP,
                    random_smoothness=RAND_SMOOTH,
                    random_seed=RAND_SEED,
                ),
            )
        )
        validate_parameters()
        padded_res.set_padded_resolution(params)

        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_grid,
        )
        from dnsjax.ic.mean_mode import constraint_residuals
        from dnsjax.ic.random_field import generate_random_state

        state = np.asarray(
            generate_random_state(
                RAND_AMP,
                RAND_SMOOTH,
                RAND_WALL_SMOOTH,
                RAND_WALL_CONF,
                RAND_SEED,
                True,
            )
        )
        col = state[:, :, 0, 0]
        _, D1, D2, w = build_cartesian_grid(
            NY, order, None, kind, params.geo.grid_stretch
        )
        D1, D2, w = np.asarray(D1), np.asarray(D2), np.asarray(w)
        c, s = derived_params.cos_tilt, derived_params.sin_tilt
        r = col.real
        tag = (
            f"{system[6:]} drive={driving[9:]} block={block:d} "
            f"tilt={tilt:g} {kind} p={order}"
        )
        print(f"=== {tag} ===")
        check(
            "mean-mode column is real",
            np.all(col.imag == 0.0),
            f"max|Im| = {np.max(np.abs(col.imag)):.1e}",
        )
        check(
            "wall-normal mean mode is zero",
            np.all(col[1] == 0.0),
            f"max|v00| = {np.max(np.abs(col[1])):.1e}",
        )
        wall = max(abs(r[0, 0]), abs(r[0, -1]), abs(r[2, 0]), abs(r[2, -1]))
        check("no-slip exact at both walls", wall == 0.0, f"{wall:.1e}")
        check(
            "mean mode is not trivially zero",
            np.max(np.abs(r)) > 1e-3,
            f"max|u00| = {np.max(np.abs(r)):.3f}",
        )

        bulk_s = system == "plane-poiseuille" and (
            driving == "constant_bulk_velocity"
        )
        for name, d, held in (
            ("streamwise", r[0] * c + r[2] * s, bulk_s),
            ("spanwise", -r[0] * s + r[2] * c, block),
        ):
            res = constraint_residuals(d, D1, D2, w, fixed_bulk=held).max()
            check(
                f"{name}: conservation laws hold",
                res < EXACT_TOL,
                f"residual {res:.1e}",
            )
            if held:
                bulk = abs(float(w @ d)) / 2.0
                check(
                    f"{name}: bulk velocity unchanged",
                    bulk < 1e-14,
                    f"|Ub'| = {bulk:.1e}",
                )
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


def _worker_column(np0: int, np1: int, out: str) -> int:
    """Save the `$(0,0)$` column of one fixed IC on an (np0, np1) mesh."""
    _configure_jax()

    from dnsjax.parameters import (
        Distribution,
        Geometry,
        Initiation,
        Parameters,
        Physics,
        Resolution,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    update_parameters(
        Parameters(
            phys=Physics(
                system="plane-poiseuille",
                re=400.0,
                driving="constant_bulk_velocity",
                block_mean_spanwise_velocity=True,
            ),
            geo=Geometry(lx=4.0, lz=4.0),
            res=Resolution(nx=NX, ny=NY, nz=NZ, double_precision=True),
            init=Initiation(
                random_mean_flow=True,
                random_amplitude=RAND_AMP,
                random_smoothness=RAND_SMOOTH,
                random_seed=RAND_SEED,
            ),
            dist=Distribution(np0=np0, np1=np1),
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)

    from dnsjax.ic.random_field import generate_random_state

    state = np.asarray(
        generate_random_state(
            RAND_AMP,
            RAND_SMOOTH,
            RAND_WALL_SMOOTH,
            RAND_WALL_CONF,
            RAND_SEED,
            True,
        )
    )
    np.save(out, state[:, :, 0, 0])
    return 0


def _run_worker(args: list[str], devices: int, timeout: int = 900):
    """Run this file as a worker with *devices* forced CPU devices."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    env["JAX_PLATFORMS"] = "cpu"
    return run_live(
        [sys.executable, str(Path(__file__).resolve()), "--worker", *args],
        timeout=timeout,
        env=env,
    )


def _check_device_independence() -> str | None:
    """The `$(0,0)$` column must be bit-identical at any (np0, np1)."""
    with tempfile.TemporaryDirectory(prefix="mean_mode_") as tmp:
        cols = []
        for np0, np1 in MESHES:
            path = Path(tmp) / f"col_{np0}_{np1}.npy"
            print(f"=== (0,0) column at (np0, np1) = ({np0}, {np1}) ===")
            proc = _run_worker(
                ["column", str(np0), str(np1), str(path)], np0 * np1
            )
            if proc.returncode != 0:
                return f"worker ({np0},{np1}) exit {proc.returncode}"
            cols.append(np.load(path))
        # A tolerance, not ``array_equal``: the per-mode fill is keyed
        # by the *global* index and so is bit-identical, but the final
        # ``amplitude / get_norm(...)`` scaling reduces over devices,
        # and that sum's order is device-count dependent (the same
        # convention as ``RAND_CROSS_TOL`` in test_localized_rolls).
        scale = float(np.max(np.abs(cols[0])))
        for (np0, np1), col in zip(MESHES[1:], cols[1:], strict=True):
            drift = float(np.max(np.abs(col - cols[0])))
            if drift > CROSS_TOL * scale:
                return (
                    f"({np0},{np1}) differs from (1,1) by "
                    f"{drift:.2e} (> {CROSS_TOL:.0e} x {scale:.2e})"
                )
    print(f"  PASS: (0,0) column agrees across meshes (< {CROSS_TOL:.0e})")
    return None


def main() -> None:
    if "--worker" in sys.argv:
        kind = sys.argv[sys.argv.index("--worker") + 1]
        if kind == "matrix":
            sys.exit(_worker_matrix())
        sys.exit(
            _worker_column(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        )

    unit_only = "--unit-only" in sys.argv
    results: list[tuple[str, str | None]] = []
    passed = failed = 0
    failed_names: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1
            failed_names.append(name)

    for fn in (
        test_tolerance_table,
        test_correction_converges,
        test_rolls_cubic_is_inadmissible,
        test_projector_properties,
        test_per_flow_surface,
    ):
        print(f"=== {fn.__name__} ===")
        before = failed
        fn(check)
        results.append(
            (
                fn.__name__,
                None if failed == before else f"{failed - before} checks",
            )
        )

    if not unit_only:
        print("=== random IC: the (0,0) column of every case ===")
        proc = _run_worker(["matrix"], 1)
        results.append(
            (
                "generated IC obeys the laws",
                None if proc.returncode == 0 else f"exit {proc.returncode}",
            )
        )
        results.append(
            ("device-count independence", _check_device_independence())
        )

    for name in failed_names:
        print(f"  (failed check: {name})")
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
