#!/usr/bin/env python3
r"""Curved (toroidal) pipe guards (offline, subprocess per config).

Six checks, cheapest first.  Each runs in its own subprocess because
the geometry singletons are captured at import, and `$\kappa$` is one
of the things they capture.

``reference``
    The toroidal equation set, against Webster & Humphrey,
    *Phys. Fluids* **9**, 407 (1997), Eqs. (2)-(5) -- the reference the
    implementation is written from.  Their primitive-variable
    convective and viscous terms are reproduced by
    `$\nabla(|u|^2/2) - u\times\omega$` and `$\nabla(\nabla\cdot u) -
    \nabla\times\nabla\times u$` in the toroidal metric, evaluated with
    independent finite differences on an analytic field.  This is what
    lets the solver take the rotational form and never transcribe a
    curvature term: it pins the claim that curvature enters *only*
    through `$\nabla$`, `$\nabla\cdot$` and `$\nabla\times$`.
    Pure NumPy, no solver import.

``metric``
    The `$m \pm 1$` structure (the handover's own test): multiplying a
    single azimuthal mode by `$h$` produces exactly `$m$` and
    `$m \pm 1$` with neighbour amplitude `$\kappa/2$` at `$r = 1$`; the
    shift **truncates** rather than wraps at `$|m| = M$`; and the
    `$1/h$` harmonics the flux read contracts with match the closed
    form `$q^{|m|}/\sqrt{1-\epsilon^2}$` to machine precision.

``defect``
    The algebraic identity behind
    :meth:`~dnsjax.geometries.wall_bounded.cylindrical_curved.CurvedCylindricalFlow.divergence_defect`,
    on the code's own grid and parity-reduced operators: for **any**
    state,

    .. math::
        r\,(\nabla_0\cdot w) + \kappa\,[h A + B]
        = h\,\partial_r(r h w_r) + h\,\partial_\theta(h w_\theta)
          + r\,\partial_s w_s ,

    whose right-hand side is `$r h$` times the toroidal divergence.
    Discretised with the *scheme's* radial form, so it also pins the
    choice of `$r D_1 x + x$` over `$D_1(r x)$`.

``straight``
    `$\kappa = 0$` must reproduce ``pipe``.  Five nonlinear steps from
    the same random initial condition, comparing the **total** field
    (this flow's state against ``pipe``'s perturbation plus its base
    flow) and the reported diagnostics.  It exercises the whole path --
    RHS, influence matrix, driving, both basis crossings -- and is the
    one check that would catch a curvature term that is wrong in a way
    that does not vanish with `$\kappa$`.

``continuity``
    The stepped state satisfies the **toroidal** discrete divergence.
    Unlike the straight pipe, where the reconstruction makes it
    machine-eps by construction, here it holds at the corrector's fixed
    point, so the residual tracks ``step.corrector_tolerance`` -- which
    is what this check measures, over three decades of it.

``dean``
    The physics, at a Dean number where it is unambiguous: the
    streamwise maximum moves to the **outer** wall, the core flow is
    directed outward and returns along the walls, the solution is
    symmetric about the `$\theta = 0$` plane, and the flux under a
    fixed pressure gradient is *reduced* against the straight pipe.

Usage::

    uv run python tests/test_curved_pipe.py
    uv run python tests/test_curved_pipe.py --only straight
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _live import report  # noqa: E402

KAPPA = 0.037  # the shipped default


# ── reference: Webster & Humphrey (2)-(5) vs the rotational form ──


def _check_reference() -> str:
    """Independent FD check of the toroidal equation set."""
    import numpy as np

    kap = 0.17  # large, so every curvature term is well above roundoff

    def h(x):
        return 1.0 + kap * x[0] * np.cos(x[1])

    def d(f, i, x, st=1e-4):
        x = np.asarray(x, float)
        e = np.zeros(3)
        e[i] = st
        return (f(x - 2 * e) - 8 * f(x - e) + 8 * f(x + e) - f(x + 2 * e)) / (
            12 * st
        )

    def U(x):
        r, th, s = x
        return np.array(
            [
                (r * r + 0.3) * np.cos(2 * th + 0.3) * np.sin(1.3 * s),
                (0.7 + r) * np.sin(th - 0.4) * np.cos(0.8 * s + 0.2),
                (1 - r * r) * (0.5 + np.cos(th + 0.2)) * np.sin(0.6 * s),
            ]
        )

    def co(i):
        return lambda y: U(y)[i]

    def curl(A, x):
        r, hx = x[0], h(x)
        ar = lambda y: A(y)[0]  # noqa: E731
        has = lambda y: h(y) * A(y)[2]  # noqa: E731
        rat = lambda y: y[0] * A(y)[1]  # noqa: E731
        return np.array(
            [
                (d(has, 1, x) - d(rat, 2, x)) / (r * hx),
                (d(ar, 2, x) - d(has, 0, x)) / hx,
                (d(rat, 0, x) - d(ar, 1, x)) / r,
            ]
        )

    def grad(f, x):
        return np.array([d(f, 0, x), d(f, 1, x) / x[0], d(f, 2, x) / h(x)])

    def div(A, x):
        r, hx = x[0], h(x)
        return (
            d(lambda y: y[0] * h(y) * A(y)[0], 0, x)
            + d(lambda y: h(y) * A(y)[1], 1, x)
            + r * d(lambda y: A(y)[2], 2, x)
        ) / (r * hx)

    def lap(f, x):
        r, hx = x[0], h(x)
        return (
            d(lambda y: y[0] * h(y) * d(f, 0, y), 0, x)
            + d(lambda y: (h(y) / y[0]) * d(f, 1, y), 1, x)
            + d(lambda y: (y[0] / h(y)) * d(f, 2, y), 2, x)
        ) / (r * hx)

    X = np.array([0.63, 0.9, 0.4])
    r, th, _ = X
    hx = h(X)
    ur, ut, us = U(X)
    ct, st_ = np.cos(th), np.sin(th)

    # W&H (3)-(5), left-hand sides: (u.grad)u with every curvature term.
    def adv(i):
        return (
            ur * d(co(i), 0, X)
            + (ut / r) * d(co(i), 1, X)
            + (us / hx) * d(co(i), 2, X)
        )

    wh_adv = np.array(
        [
            adv(0) - ut * ut / r - (kap * ct / hx) * us * us,
            adv(1) + ur * ut / r + (kap * st_ / hx) * us * us,
            adv(2) + (kap * us / hx) * (ur * ct - ut * st_),
        ]
    )
    rot = grad(lambda y: 0.5 * float(np.dot(U(y), U(y))), X) - np.cross(
        U(X), curl(U, X)
    )
    e_adv = float(np.abs(wh_adv - rot).max())

    # W&H (3)-(5), viscous brackets.
    lp = np.array([lap(co(i), X) for i in range(3)])
    dsu = np.array([d(co(i), 2, X) for i in range(3)])
    dtu = np.array([d(co(i), 1, X) for i in range(3)])
    wh_vis = np.array(
        [
            lp[0]
            - 2 * dtu[1] / r**2
            - ur / r**2
            + (kap * st_ / (r * hx)) * ut
            + (kap * kap * ct / hx**2) * (ut * st_ - ur * ct)
            - 2 * kap * ct * dsu[2] / hx**2,
            lp[1]
            + 2 * dtu[0] / r**2
            - ut / r**2
            - (kap * st_ / (r * hx)) * ur
            - (kap * kap * st_ / hx**2) * (ut * st_ - ur * ct)
            + 2 * kap * st_ * dsu[2] / hx**2,
            lp[2]
            + (2 * kap / hx**2) * (ct * dsu[0] - st_ * dsu[1])
            - (kap * kap / hx**2) * us,
        ]
    )
    vec_lap = grad(lambda y: div(U, y), X) - curl(lambda y: curl(U, y), X)
    e_vis = float(np.abs(wh_vis - vec_lap).max())

    assert e_adv < 1e-9, f"W&H advective terms differ by {e_adv:.2e}"
    assert e_vis < 1e-6, f"W&H viscous terms differ by {e_vis:.2e}"
    return f"advective {e_adv:.1e}, viscous {e_vis:.1e} (nested-FD roundoff)"


# ── in-process workers (one subprocess each) ─────────────────────


def _configure(kappa: float, **over) -> None:
    """Configure JAX and the parameter singletons for one check."""
    import os

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

    res = {
        "nx": 8,
        "ny": 24,
        "nz": 12,
        "fd_order": 6,
        "double_precision": True,
    }
    res.update(over.pop("res", {}))
    step = {"dt": 0.004, "corrector_tolerance": 1e-12}
    step.update(over.pop("step", {}))
    phys = {
        "system": over.pop("system", "curved-pipe"),
        "re": 200.0,
        "u_grid": 0.0,
    }
    phys.update(over.pop("phys", {}))
    geo = {"lx": 4.0}
    if phys["system"] == "curved-pipe":
        geo["curvature"] = kappa
    geo.update(over.pop("geo", {}))
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res=res,
            step=step,
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)


def _check_metric(kappa: float) -> str:
    r"""`$h\,e^{im\theta}$` is exactly `$m, m\pm1$` with weight
    `$\kappa r/2$`, truncating at the top mode; and the `$1/h$`
    harmonics match the closed form."""
    _configure(kappa)
    import numpy as np
    from jax import numpy as jnp

    import dnsjax.flows.wall_bounded.curved_pipe as cp
    from dnsjax.geometries.wall_bounded.cylindrical_curved import (
        _inv_h_harmonics,
    )
    from dnsjax.harmonics import complex_harmonics
    from dnsjax.operators import pad_harmonics
    from dnsjax.parameters import params
    from dnsjax.sharding import sharding

    flow = cp.flow
    m_vals = np.asarray(
        pad_harmonics(
            complex_harmonics(params.res.nz),
            params.res.nz,
            sharding.nz_spec_pad,
        )
    )
    n_true = params.res.nz - 1
    top = params.res.nz // 2 - 1
    rs = np.asarray(flow.rs)
    worst_nb, worst_zero, worst_trunc = 0.0, 0.0, 0.0
    for j in range(n_true):
        one = (
            jnp.zeros((len(rs), len(m_vals), 1), dtype=sharding.complex_type)
            .at[:, j, 0]
            .set(1.0)
        )
        out = np.asarray(flow.chi_mul(one))[:, :, 0]
        for k in range(len(m_vals)):
            want = 0.0
            if k < n_true and abs(int(m_vals[k]) - int(m_vals[j])) == 1:
                want = 0.5  # chi = r cos(theta) -> r/2 on each neighbour
            got = out[:, k] / np.where(rs > 0, rs, 1.0)
            err = float(np.abs(got - want).max())
            if want:
                worst_nb = max(worst_nb, err)
            elif k < n_true and abs(int(m_vals[k])) != abs(int(m_vals[j])):
                worst_zero = max(worst_zero, err)
        # the top mode must LEAVE the set, not wrap to -M
        if int(m_vals[j]) == int(m_vals[top]):
            worst_trunc = float(np.abs(out[:, top + 1]).max())

    # the wall neighbour amplitude the handover quotes
    wall_amp = 0.5 * kappa * rs[-1]

    # exact 1/h harmonics vs an FFT of 1/h
    nth = 4096
    th = 2 * np.pi * np.arange(nth) / nth
    err_c = 0.0
    for r in (rs[len(rs) // 3], rs[-1]):
        num = np.fft.fft(1.0 / (1.0 + kappa * r * np.cos(th))) / nth
        ana = _inv_h_harmonics(np.array([r]), np.arange(6))[0]
        err_c = max(err_c, float(np.abs(num[:6] - ana).max()))

    assert worst_nb < 1e-14, f"neighbour weight off by {worst_nb:.2e}"
    assert worst_zero < 1e-14, f"non-neighbour leakage {worst_zero:.2e}"
    assert worst_trunc == 0.0, f"top mode wrapped: {worst_trunc:.2e}"
    assert err_c < 1e-14, f"1/h harmonics off by {err_c:.2e}"
    return (
        f"neighbour {worst_nb:.1e}, leakage {worst_zero:.1e}, "
        f"truncation exact, 1/h {err_c:.1e}; kappa/2 at the wall = "
        f"{wall_amp:.6f}"
    )


def _check_defect(kappa: float) -> str:
    r"""The closed form of `$\nabla_0\cdot w$` against the toroidal
    divergence, on the code's grid and operators."""
    _configure(kappa)
    from jax import numpy as jnp

    import dnsjax.flows.wall_bounded.curved_pipe as cp
    from dnsjax.geometries.wall_bounded._base import from_pm_basis
    from dnsjax.geometries.wall_bounded._cylindrical_stepping import (
        _parity_y_matvec,
    )
    from dnsjax.geometries.wall_bounded.cylindrical import fourier
    from dnsjax.ic.random_field import generate_random_state

    flow = cp.flow
    state = cp.to_solver_basis(generate_random_state(0.2, 0.4, 0.4, 0.14, 3))
    w = from_pm_basis(state)
    psv = 1 - fourier.m_is_even * 2
    r_col = flow.rs[:, None, None]
    im = 1j * fourier.m

    def dr(x):
        return r_col * _parity_y_matvec(flow.D1_pos, flow.D1_ghost, x, psv) + x

    def hmul(x):
        return x + kappa * flow.chi_mul(x)

    straight = dr(w[1]) + im * w[2] + r_col * 1j * fourier.kz * w[0]
    toroidal = (
        hmul(dr(hmul(w[1])))
        + hmul(im * hmul(w[2]))
        + r_col * 1j * fourier.kz * w[0]
    )
    defect = flow.divergence_defect(state, fourier) * r_col
    err = float(jnp.abs(straight - defect - toroidal).max())
    scale = float(jnp.abs(straight).max())
    assert err < 1e-12 * max(scale, 1.0), f"defect identity off by {err:.2e}"
    return f"|r div0 w - r g - r h div_c u| = {err:.1e} (scale {scale:.1e})"


def _steps(module, state, n: int):
    """Advance *n* steps, copying (the steppers donate)."""
    from jax import numpy as jnp

    for _ in range(n):
        state, err, nc, _aux = module.predict_and_fully_correct(
            jnp.copy(state)
        )
    return state, float(err), int(nc)


def _check_straight(_kappa: float, out: str = "", system: str = "") -> str:
    """One half of the kappa = 0 comparison; writes its total field."""
    import numpy as np
    from jax import numpy as jnp

    _configure(0.0, system=system)
    import importlib

    from dnsjax.ic.random_field import generate_random_state

    mod = importlib.import_module(
        "dnsjax.flows.wall_bounded."
        + ("pipe" if system == "pipe" else "curved_pipe")
    )
    state = generate_random_state(0.15, 0.4, 0.4, 0.14, 7)
    s, err, nc = _steps(mod, mod.to_solver_basis(state), 5)
    total = np.asarray(mod.from_solver_basis(s))
    stats = mod.get_stats(mod.from_solver_basis(s))
    if system == "pipe":
        from dnsjax.geometries.wall_bounded.cylindrical import fourier
        from dnsjax.parameters import params

        prof = 1.0 - mod.flow.rs**2
        zero = jnp.zeros((params.res.ny,) + total.shape[2:])
        total = total + np.asarray(
            jnp.stack(
                [
                    jnp.where(fourier.mean_mask, prof[:, None, None], 0.0),
                    zero,
                    zero,
                ]
            )
        )
        keys = {"E": "E", "I": "I", "Ub": "Ub'_z"}
    else:
        keys = {"E": "E", "I": "I", "Ub": "Ub_s"}
    np.save(out, total)
    vals = {k: float(stats[v]) for k, v in keys.items()}
    if system == "pipe":
        vals["Ub"] += 0.5  # pipe reports the perturbation bulk
    Path(out + ".json").write_text(json.dumps({"err": err, "nc": nc, **vals}))
    return f"{system}: err={err:.1e} nc={nc}"


def _check_continuity(kappa: float, tol: float = 1e-9) -> str:
    r"""The toroidal discrete divergence of a stepped state, against
    ``step.corrector_tolerance``."""
    _configure(kappa, step={"corrector_tolerance": tol})
    from jax import numpy as jnp

    import dnsjax.flows.wall_bounded.curved_pipe as cp
    import dnsjax.geometries.wall_bounded._cylindrical_stepping as cs
    from dnsjax.geometries.wall_bounded._base import from_pm_basis
    from dnsjax.geometries.wall_bounded.cylindrical import fourier
    from dnsjax.ic.random_field import generate_random_state

    flow = cp.flow
    s, err, _nc = _steps(
        cp,
        cp.to_solver_basis(generate_random_state(0.15, 0.4, 0.4, 0.14, 7)),
        4,
    )
    w = from_pm_basis(s)
    psv = 1 - fourier.m_is_even * 2
    r_col = flow.rs[:, None, None]

    def dr(x):
        return (
            r_col * cs._parity_y_matvec(flow.D1_pos, flow.D1_ghost, x, psv) + x
        )

    def hmul(x):
        return x + kappa * flow.chi_mul(x)

    res = (
        hmul(dr(hmul(w[1])))
        + hmul(1j * fourier.m * hmul(w[2]))
        + r_col * 1j * fourier.kz * w[0]
    )
    scale = float(jnp.abs(dr(w[1])).max())
    rel = float(jnp.abs(res).max()) / max(scale, 1e-30)
    # The defect is read on the corrector iterate, so continuity holds
    # at the fixed point and the residual is bounded by how far the
    # accepted iterate is from it -- not by machine epsilon, as it is
    # on the straight pipe.  Measured: 0.7, 2.8 and 88 times the
    # tolerance at 1e-6, 1e-9 and 1e-12 (kappa = 0.037, nr = 24).
    assert rel < 1e4 * tol + 1e-15, (
        f"continuity residual {rel:.2e} at tolerance {tol:.0e}"
    )
    return f"tol={tol:.0e} err={err:.1e} -> relative divergence {rel:.2e}"


def _check_dean(kappa: float) -> str:
    """Steady-state structure at a Dean number where it is unambiguous."""
    _configure(
        kappa,
        phys={"re": 100.0, "u_grid": 0.0},
        res={"nx": 4, "ny": 32, "nz": 16},
        step={"dt": 0.01, "corrector_tolerance": 1e-10},
    )
    import numpy as np

    import dnsjax.flows.wall_bounded.curved_pipe as cp
    from dnsjax.harmonics import complex_harmonics
    from dnsjax.operators import pad_harmonics
    from dnsjax.parameters import params
    from dnsjax.sharding import sharding

    s, err, _nc = _steps(cp, cp.to_solver_basis(cp.init_state()), 6000)
    u = np.asarray(cp.from_solver_basis(s))
    stats = cp.get_stats(cp.from_solver_basis(s))
    rs = np.asarray(cp.flow.rs)
    nth = 64
    th = 2 * np.pi * np.arange(nth) / nth
    m = np.asarray(
        pad_harmonics(
            complex_harmonics(params.res.nz),
            params.res.nz,
            sharding.nz_spec_pad,
        )
    )
    basis = np.exp(1j * np.outer(th, m))
    us = (basis @ u[0, :, :, 0].T).real.T
    ur = (basis @ u[1, :, :, 0].T).real.T
    i_r, i_th = np.unravel_index(np.argmax(us), us.shape)
    bulk = float(stats["Ub_s"])
    # reflection symmetry about theta = 0 (u_r even, u_theta odd)
    half = nth // 2
    sym = float(np.abs(ur[:, 1:half] - ur[:, -1:-half:-1]).max())
    core = len(rs) // 2
    outward, inward = float(ur[core, 0]), float(ur[core, half])

    assert np.degrees(th[i_th]) < 1.0 or np.degrees(th[i_th]) > 359.0, (
        f"streamwise maximum at theta={np.degrees(th[i_th]):.0f} deg, "
        "not the outer wall"
    )
    assert outward > 0 > inward, (
        f"core flow not outward/inward: {outward:+.3f}/{inward:+.3f}"
    )
    assert bulk < 0.5, f"flux not reduced by curvature: Ub={bulk:.4f}"
    assert sym < 1e-9 * max(float(np.abs(ur).max()), 1e-30) + 1e-14, (
        f"theta-reflection symmetry broken by {sym:.2e}"
    )
    return (
        f"De={params.phys.re * np.sqrt(kappa):.0f}: max u_s at "
        f"r={rs[i_r]:.2f}, theta={np.degrees(th[i_th]):.0f} deg (outer "
        f"wall); core u_r {outward:+.3f} out / {inward:+.3f} in; "
        f"Ub={bulk:.4f} < 0.5; symmetry {sym:.1e}"
    )


# ── driver ───────────────────────────────────────────────────────

WORKERS = {
    "reference": lambda a: _check_reference(),
    "metric": lambda a: _check_metric(a.kappa),
    "defect": lambda a: _check_defect(a.kappa),
    "straight": lambda a: _check_straight(a.kappa, a.out, a.system),
    "continuity": lambda a: _check_continuity(a.kappa, a.tol),
    "dean": lambda a: _check_dean(a.kappa),
}

#: ``(label, check, extra worker args)``, cheapest first.
CASES: list[tuple[str, str, list[str]]] = [
    ("reference (W&H Eqs. 2-5)", "reference", []),
    ("metric coupling", "metric", []),
    ("divergence defect", "defect", []),
    ("continuity @ 1e-6", "continuity", ["--tol", "1e-6"]),
    ("continuity @ 1e-9", "continuity", ["--tol", "1e-9"]),
    ("continuity @ 1e-12", "continuity", ["--tol", "1e-12"]),
    ("straight limit: pipe", "straight", ["--system", "pipe"]),
    ("straight limit: curved-pipe", "straight", ["--system", "curved-pipe"]),
    ("Dean structure", "dean", []),
]


def main(only: str | None, kappa: float) -> int:
    import tempfile

    passed, failures = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for label, check, extra in CASES:
            if only and only not in label and only != check:
                continue
            args = [
                sys.executable,
                __file__,
                "--worker",
                check,
                "--kappa",
                str(kappa),
                *extra,
            ]
            if check == "straight":
                args += ["--out", f"{tmp}/{extra[1]}.npy"]
            proc = subprocess.run(args, capture_output=True, text=True)
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            note = tail[-1] if tail else "(no output)"
            if proc.returncode != 0:
                print(f"  FAIL  {label}: {note}")
                failures.append((label, note))
                continue
            print(f"  PASS  {label}: {note}")
            passed += 1
        # the straight-limit comparison needs both halves
        both = [f"{tmp}/pipe.npy", f"{tmp}/curved-pipe.npy"]
        if all(Path(p).exists() for p in both):
            import numpy as np

            a, b = (np.load(p) for p in both)
            ja, jb = (json.loads(Path(p + ".json").read_text()) for p in both)
            rel = float(np.abs(a - b).max() / np.abs(a).max())
            stat = max(
                abs(ja[k] - jb[k]) / max(abs(ja[k]), 1e-30)
                for k in ("E", "I", "Ub")
            )
            ok = rel < 1e-12 and stat < 1e-9
            print(
                f"  {'PASS' if ok else 'FAIL'}  straight limit: "
                f"|pipe - curved(kappa=0)| = {rel:.2e} relative, "
                f"diagnostics agree to {stat:.1e}"
            )
            passed += ok
            if not ok:
                failures.append(("straight limit", f"{rel:.2e} / {stat:.1e}"))
    return report(passed, failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--only")
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--system", default="curved-pipe")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.worker:
        print(WORKERS[args.worker](args))
    else:
        sys.exit(main(args.only, args.kappa))
