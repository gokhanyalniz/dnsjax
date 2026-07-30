"""GPU diagnostic: is the corrector's loop-invariant work worth hoisting?

The ``iterative-cn`` corrector is a ``lax.while_loop`` whose body
re-executes on every pass, yet much of each pass's dense-`$y$` work
depends only on `$u^n$` / `$N^n$` -- **Cartesian 3 of 5 matvec columns,
annular 6 of 12, cylindrical 13 of 17** (the ``SITES`` tables below
carry the per-call-site split with source lines).

A CPU probe of the *optimized* HLO already established that XLA's LICM
does **not** hoist any of it, and why: ``jnp.stack`` batches an
invariant column together with a varying one, so the batched dot is
dataflow-varying, and every value downstream of a *slice* of it
inherits that -- including the matvecs whose own operands are wholly
invariant.  Capturing the invariant work therefore needs the batches
**split** (more, smaller GEMMs, plus a cross-geometry ``correct_fn``
signature change), which is a trade rather than a free win: with
``T(C) = h + C g`` for a C-column matvec, splitting one mixed site pays
only if ``(K - 1) g > h`` at the natural pass count ``K``.  ``h`` and
``g`` are hardware, so the trade cannot be decided on a CPU box.

What this measures, in the order the numbers matter:

Part 0  The **natural corrector pass count** ``K`` (and CFL) at
        realistic ``dt``, after a short spin-up.  Everything below
        scales with ``K - 1``: at ``K = 1`` the whole lever is worth
        exactly zero and the remaining parts are academic.
Part A  **Per-pass wall clock**: the real step timed with the corrector
        forced to exactly ``K = 1, 2, 3, ...`` passes (a fresh
        ``build_*_stepper`` per ``K`` -- the iteration cap is baked in
        at trace time), then a least-squares fit ``t_step = a + b K``.
        ``b`` is one corrector pass, in situ, with all fusion.
Part B  The **matvec cost curve** ``T(C)`` at the corrector's real
        shapes ``(N_y, C, N_1, N_2)``, for the primitive the geometry
        actually uses (``apply_y_matrix``; ``_parity_y_matvec`` for
        cylindrical, which is two GEMMs -- full + near-axis ghost).
        The fit gives the fixed overhead ``h`` and the per-column
        slope ``g`` that decide the split.
Part C  The **projection**: applies the measured ``T(C)`` to every
        call site of the selected geometry/flag, comparing
        ``K T(C_tot)`` against ``T(C_inv) + K T(C_var)`` site by site,
        and reports the total saving as a fraction of the measured
        step.
Part D  The **HLO invariance census** on the *GPU-optimized* module
        (the CPU numbers were 4-8 % of the body, all elementwise --
        the GPU fuses differently, so it is re-measured here).  Also
        lists every dot-bearing fusion's shape, which cross-checks the
        hand-derived column counts of ``SITES``: the trailing dot
        dimension is ``C x N_1 x N_2 x 2``.

Run **on a GPU** (single device, no mpirun)::

    .venv/bin/python scripts/corrector_invariance_probe.py \
        --dist.platform cuda --system plane-couette

    # the three geometries x both formulations (6 processes; the
    # per-flow singletons and the iteration cap are import-time state)
    for sys in plane-couette taylor-couette pipe; do \
      for cimm in 0 1; do \
        .venv/bin/python scripts/corrector_invariance_probe.py \
          --dist.platform cuda --system $sys --consistent-imm $cimm; \
      done; done | tee corrector_probe.log

Each run ends with a greppable ``SUMMARY`` line, so a sweep can be
collapsed with ``grep '^SUMMARY' corrector_probe.log``.  Paste the
full stdout back for the decision.

``--cpu-smoke`` runs every part on CPU at tiny resolution to validate
the harness on a GPU-less box (the timings are meaningless there; the
structure, the fits and the census are not).
"""

from __future__ import annotations

import argparse
import re
import time

import jax
import numpy as np

# x64 before any dnsjax module creates arrays (the solver's precision
# is what the GEMM curve has to be measured in).
jax.config.update("jax_enable_x64", True)

from dnsjax.bootstrap import configure_jax_platform  # noqa: E402
from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
    validate_parameters,
)

# ── the call-site tables ─────────────────────────────────────────────
#
# One row per dense-`$y$` matvec *call site* reachable from the
# corrector's ``_imm_iteration_*``: (label, matrix, C_total,
# C_invariant, source).  "Invariant" means the column depends only on
# ``velocity_n`` / ``nonlin_n`` (the step's `$u^n$` and `$N^n$`), so it
# is recomputed identically on every corrector pass; "varying" means it
# reads the running iterate ``velocity_j`` / ``nonlin_j``.
#
# A cylindrical row is one ``_parity_y_matvec`` (a full GEMM plus the
# few-row near-axis ghost GEMM), which is why Part B times that
# primitive there.  The flag-on totals are 5 / 12 / 17 columns with
# 3 / 6 / 13 invariant -- the counts the refactor question is about.
#
# Hand-derived; the source column is the check, and Part D's dot-shape
# census is the independent one.

SITES: dict[tuple[str, bool], list[tuple[str, str, int, int, str]]] = {
    ("cartesian", True): [
        ("_to_solver   D2 v^n", "D2", 1, 1, "cartesian.py:1342"),
        ("S_phi        D1 div_h", "D1", 1, 0, "cartesian.py:1814"),
        ("Hk^- matvec  D2 [phi_n, om_n]", "D2", 2, 2, "cartesian.py:1826"),
        ("_from_solver D1 v_new", "D1", 1, 0, "cartesian.py:1306"),
    ],
    ("cartesian", False): [
        ("D1 [v_n, Nv_j, Nv_n]", "D1", 3, 2, "cartesian.py:1521"),
        ("D1 pP", "D1", 1, 0, "cartesian.py:1560"),
        ("Hk^- matvec  D2 velocity_n", "D2", 3, 3, "cartesian.py:1565"),
    ],
    ("annular", True): [
        ("D1 d1_in  [u_r^n | 4 varying]", "D1", 5, 1, "annular.py:2067"),
        ("D2 d2_in  [u_r^n | u_th^it]", "D2", 2, 1, "annular.py:2069"),
        ("D2 pair_n [phi_n, om_n]", "D2", 2, 2, "annular.py:2096"),
        ("D1 pair_n [phi_n, om_n]", "D1", 2, 2, "annular.py:2097"),
        ("D1 ur_new -> chi", "D1", 1, 0, "annular.py:2156"),
    ],
    ("annular", False): [
        ("D1 all_v  [3 vel^n, 2 N^n | 2 N^j]", "D1", 7, 5, "annular.py:1710"),
        ("D1 pP", "D1", 1, 0, "annular.py:1749"),
        ("D2 vel_n_stack", "D2", 3, 3, "annular.py:1758"),
    ],
    ("cylindrical", True): [
        (
            "D1 d1_in  [u_+^n, u_-^n, u_z^n | 2 N]",
            "D1",
            5,
            3,
            "cylindrical.py:2585",
        ),
        ("D2 pair_n [u_+^n, u_-^n]", "D2", 2, 2, "cylindrical.py:2592"),
        ("D1 C_z", "D1", 1, 0, "cylindrical.py:2632"),
        ("D2 quad   [phi_pm^n, om_pm^n]", "D2", 4, 4, "cylindrical.py:2668"),
        ("D1 quad   [phi_pm^n, om_pm^n]", "D1", 4, 4, "cylindrical.py:2670"),
        ("D1 ur_new", "D1", 1, 0, "cylindrical.py:2710"),
    ],
    ("cylindrical", False): [
        (
            "D1 all_vparity [2 vel^n, 2 N^n | 2 N^j]",
            "D1",
            6,
            4,
            "cylindrical.py:2280",
        ),
        ("D1 pP_and_vel  [3 vel^n | pP]", "D1", 4, 3, "cylindrical.py:2325"),
        ("D2 vel_n_stack", "D2", 3, 3, "cylindrical.py:2339"),
    ],
}

FAMILY = {
    "plane-couette": "cartesian",
    "plane-poiseuille": "cartesian",
    "taylor-couette": "annular",
    "dean": "annular",
    "pipe": "cylindrical",
}


# ── setup ────────────────────────────────────────────────────────────


def _configure_system(
    system: str,
    ny: int,
    nx: int,
    nz: int,
    order: int,
    consistent_imm: bool,
    dt: float,
) -> None:
    """Set the global ``params`` for *system* and derive the singletons.

    The same direct-assignment + one layering call idiom as
    ``scripts/pallas_solve_profile.py``; ``res.consistent_imm`` and the
    backend go through the layering call so nothing re-materializes
    over them.
    """
    params.phys.system = system
    params.phys.re = 400.0
    params.res.nx = nx
    params.res.ny = ny
    params.res.nz = nz
    params.res.fd_order = order
    params.res.double_precision = True
    params.step.dt = dt
    if system == "taylor-couette":
        params.phys.re1 = 100.0
        params.phys.re2 = 0.0
        params.geo.eta = 0.5
    elif system == "dean":
        params.geo.eta = 0.5
    update_parameters(
        Parameters(
            res={"consistent_imm": consistent_imm},
            solver={"backend": "pallas"},
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()


def _geom_module(system: str):
    """The geometry module (holds the matvec primitives + builder)."""
    from dnsjax.geometries.wall_bounded import annular, cartesian, cylindrical

    return {
        "cartesian": cartesian,
        "annular": annular,
        "cylindrical": cylindrical,
    }[FAMILY[system]]


def _import_flow(system: str):
    """Import the flow module (builds the singletons + the stepper)."""
    if system == "plane-couette":
        from dnsjax.flows.wall_bounded import plane_couette as m
    elif system == "plane-poiseuille":
        from dnsjax.flows.wall_bounded import plane_poiseuille as m
    elif system == "pipe":
        from dnsjax.flows.wall_bounded import pipe as m
    elif system == "taylor-couette":
        from dnsjax.flows.wall_bounded import taylor_couette as m
    elif system == "dean":
        from dnsjax.flows.wall_bounded import dean as m
    else:
        raise SystemExit(f"unsupported system: {system}")
    return m


def _builder(system: str, geom):
    """The geometry's ``build_*_stepper(flow)``."""
    name = {
        "cartesian": "build_cartesian_stepper",
        "annular": "build_annular_stepper",
        "cylindrical": "build_cylindrical_stepper",
    }[FAMILY[system]]
    return getattr(geom, name)


# ── helpers ──────────────────────────────────────────────────────────


def _ms(sec: float) -> str:
    return f"{sec * 1e3:9.4f} ms"


def _fit_line(xs, ys) -> tuple[float, float, float]:
    """Least-squares ``(intercept, slope, R^2)`` of *ys* on *xs*."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mat = np.stack([np.ones_like(x), x], axis=1)
    (a, b), *_ = np.linalg.lstsq(mat, y, rcond=None)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), r2


def _bench_step(step, state, n: int, warmup: int = 3):
    """Time a donating step by chaining the state; returns (t, num_c)."""
    from jax import numpy as jnp

    s = jnp.copy(state)
    for _ in range(warmup):
        s, _err, _c = step(s)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    cc = 0
    for _ in range(n):
        s, _err, c = step(s)
        cc = c
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n, int(cc)


def _bench(fn, args_list, warmup: int = 2) -> float:
    """Per-call wall time over a batch of **distinct** operands."""
    f = jax.jit(fn)
    for a in args_list[: max(1, warmup)]:
        jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    outs = [f(*a) for a in args_list]
    jax.block_until_ready(outs[-1])
    return (time.perf_counter() - t0) / len(args_list)


def _make_state(m):
    """A random initial state in the geometry's carried basis."""
    from dnsjax.random_field import generate_random_state

    to_solver = getattr(m, "to_solver_basis", None)
    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_seed,
        params.init.random_mean_flow,
    )
    return state if to_solver is None else to_solver(state)


# ── Part 0: the natural pass count ───────────────────────────────────


def _part_0(m, state, spinup: int, dts: list[float]):
    """Spin up, then report ``K`` and the CFL at each ``dt``.

    ``set_dt`` rebuilds the ``dt``-dependent flow leaves on device with
    no recompilation, so the whole sweep runs in one process.
    """
    from jax import numpy as jnp

    print("\n" + "-" * 72)
    print("PART 0 -- natural corrector pass count K (the lever scales")
    print("          with K - 1: at K = 1 there is nothing to hoist)")
    print("-" * 72)
    dt0 = float(params.step.dt)
    s = jnp.copy(state)
    for _ in range(spinup):
        s, _e, _c = m.predict_and_fully_correct(s)
    jax.block_until_ready(s)
    tol = float(params.step.corrector_tolerance)
    print(f"  spun up {spinup} steps at dt = {dt0}")
    print(
        f"  corrector_tolerance = {tol:g}; err/tol is the margin -- "
        "K rises\n  the moment it crosses 1"
    )
    print(
        f"  {'dt':>10}  {'K':>4}  {'error':>11}  {'err/tol':>8}  measurements"
    )
    out = {}
    for dt in dts:
        m.set_dt(dt)
        m.reset_ab2_kappa()
        sd = jnp.copy(s)
        sd, err, c, meas = m.predict_and_fully_correct_measured(sd)
        passes = 1 + int(c)
        info = "  ".join(
            f"{k}={float(v):.4g}" for k, v in sorted(meas.items())
        )
        print(
            f"  {dt:10.4g}  {passes:4d}  {float(err):11.4e}  "
            f"{float(err) / tol:8.2f}  {info}"
        )
        out[dt] = passes
    m.set_dt(dt0)
    m.reset_ab2_kappa()
    return out, s


# ── Part A: the per-pass wall clock ──────────────────────────────────


def _forced_stepper(build, flow, passes: int):
    """Rebuild the stepper with the corrector pinned to *passes*.

    ``corrector_tolerance`` / ``max_corrector_iterations`` are read
    inside the traced ``_step_core``, so they are baked in at trace
    time: a fresh ``build_*_stepper`` is what makes them take effect.
    One correction always runs before the loop, so ``passes = 1 +
    max_corrector_iterations``, and ``passes == 1`` is a tolerance the
    first error can never exceed.  Assigned directly (post-setup, with
    no further ``update_parameters`` pass to overwrite them).
    """
    if passes < 2:
        params.step.corrector_tolerance = 1e30
        params.step.max_corrector_iterations = 1
    else:
        params.step.corrector_tolerance = 1e-300
        params.step.max_corrector_iterations = passes - 1
    return build(flow)[3]


def _part_a(build, flow, state, ks: list[int], steps: int):
    print("\n" + "-" * 72)
    print("PART A -- step time vs forced corrector passes")
    print("-" * 72)
    print(f"  {'passes':>7}  {'step':>12}  {'K reported':>11}")
    times = []
    for k in ks:
        step = _forced_stepper(build, flow, k)
        t, c = _bench_step(step, state, steps)
        times.append(t)
        print(f"  {k:7d}  {_ms(t)}  {1 + c:11d}")
    a, b, r2 = _fit_line(ks, times)
    print(
        f"\n  fit  t_step = {a * 1e3:.4f} ms + {b * 1e3:.4f} ms x K"
        f"   (R^2 = {r2:.5f})"
    )
    print(f"  one corrector pass b = {_ms(b)}")
    return a, b, times


# ── Part B: the matvec cost curve ────────────────────────────────────


def _matvec_fn(geom, flow, family: str, kind: str):
    """``(x) -> matvec(x)`` for the primitive the geometry uses."""
    from dnsjax.geometries.wall_bounded._base import apply_y_matrix

    if family == "cylindrical":
        pos = flow.D1_pos if kind == "D1" else flow.D2_pos
        ghost = flow.D1_ghost if kind == "D1" else flow.D2_ghost

        def f(x):
            # ``parity_sign`` is a scalar here: the real one is a
            # per-mode +/-1 array multiplying the few-row ghost block,
            # a negligible elementwise term either way.
            return geom._parity_y_matvec(pos, ghost, x, 1.0, component_axis=1)

        return f

    mat = flow.D1 if kind == "D1" else flow.D2
    return lambda x: apply_y_matrix(mat, x, component_axis=1)


def _part_b(geom, flow, sharding, family: str, cmax: int, reps: int):
    """Time the corrector's matvec at ``C = 1 .. cmax`` columns."""
    ny, n1, n2 = sharding.spec_shape
    print("\n" + "-" * 72)
    print("PART B -- matvec cost T(C) at the corrector's real shapes")
    print(
        f"          field (N_y, C, N_1, N_2) = ({ny}, C, {n1}, {n2})"
        f"  {sharding.complex_type.__name__}"
    )
    if family == "cylindrical":
        print("          primitive: _parity_y_matvec (full + ghost GEMM)")
    else:
        print("          primitive: apply_y_matrix")
    print("-" * 72)
    rng = np.random.default_rng(7)
    spec = jax.NamedSharding(sharding.mesh, sharding.spec_vector_shard)
    curves: dict[str, dict[int, float]] = {}
    print(f"  {'C':>3}  {'D1':>12}  {'D2':>12}   (per call)")
    for kind in ("D1", "D2"):
        curves[kind] = {}
    for c in range(1, cmax + 1):
        row = {}
        for kind in ("D1", "D2"):
            fn = _matvec_fn(geom, flow, family, kind)
            args = []
            for _ in range(reps):
                arr = rng.standard_normal((ny, c, n1, n2)) + 1j * (
                    rng.standard_normal((ny, c, n1, n2))
                )
                args.append((jax.device_put(arr, spec),))
            row[kind] = _bench(fn, args)
            curves[kind][c] = row[kind]
            del args
        print(f"  {c:3d}  {_ms(row['D1'])}  {_ms(row['D2'])}")
    for kind in ("D1", "D2"):
        cs = sorted(curves[kind])
        h, g, r2 = _fit_line(cs, [curves[kind][c] for c in cs])
        print(
            f"\n  {kind}: T(C) = h + C g with h = {_ms(h)} (fixed), "
            f"g = {_ms(g)} (per column), R^2 = {r2:.5f}"
        )
        print(
            f"      splitting one mixed site pays iff (K - 1) g > h,"
            f"  i.e. K > {1 + h / g if g > 0 else float('inf'):.2f}"
        )
    return curves


# ── Part C: the projection ───────────────────────────────────────────


def _t_of(curve: dict[int, float], c: int) -> float:
    """Measured ``T(C)``; ``T(0) = 0`` (the site disappears)."""
    if c <= 0:
        return 0.0
    if c in curve:
        return curve[c]
    cs = sorted(curve)
    h, g, _ = _fit_line(cs, [curve[k] for k in cs])
    return h + c * g


def _part_c(curves, family: str, cimm: bool, k_nat: int, step_at):
    """*step_at(k)* is the measured step time at ``k`` passes.

    The percentage is quoted against the step of the ``K`` the table
    assumes -- at ``K = 1`` the table is the ``K = 2`` hypothetical, and
    a ``K = 2`` saving measured against the (cheaper) ``K = 1`` step
    would overstate it by the ratio of the two.
    """
    sites = SITES.get((family, cimm))
    print("\n" + "-" * 72)
    print(f"PART C -- projection at the natural K = {k_nat}")
    print("-" * 72)
    if not sites:
        print("  no site table for this configuration")
        return 0.0, k_nat
    if k_nat < 2:
        print("  K = 1: the corrector body runs once per step, so there")
        print("  is no re-execution to hoist.  Projection is zero by")
        print("  construction; the table below is what a K > 1 run would")
        print("  save.")
    k = max(k_nat, 2)
    print(
        f"  {'site':38} {'mat':>3} {'tot':>4} {'inv':>4} "
        f"{'now':>11} {'split':>11} {'save':>11}"
    )
    tot_now = tot_split = 0.0
    fit_now = fit_split = 0.0
    fits = {
        kind: _fit_line(sorted(c), [c[k2] for k2 in sorted(c)])[:2]
        for kind, c in curves.items()
    }
    for label, kind, c_tot, c_inv, src in sites:
        curve = curves[kind]
        c_var = c_tot - c_inv
        now = k * _t_of(curve, c_tot)
        split = _t_of(curve, c_inv) + k * _t_of(curve, c_var)
        tot_now += now
        tot_split += split
        h, g = fits[kind]
        fit_now += k * (h + c_tot * g)
        fit_split += (h + c_inv * g if c_inv else 0.0) + k * (
            h + c_var * g if c_var else 0.0
        )
        print(
            f"  {label:38} {kind:>3} {c_tot:4d} {c_inv:4d} "
            f"{now * 1e3:9.4f}ms {split * 1e3:9.4f}ms "
            f"{(now - split) * 1e3:9.4f}ms"
        )
        print(f"    {src}")
    save = tot_now - tot_split
    t_k = step_at(k)
    print(
        f"\n  matvec total per step:  now {_ms(tot_now)}   "
        f"split {_ms(tot_split)}"
    )
    print(
        f"  projected saving:       {_ms(save)}"
        + (
            f"  = {100.0 * save / t_k:.2f}% of the K = {k} step ({_ms(t_k)})"
            if t_k > 0
            else ""
        )
    )
    print(
        f"  same from the h + C g fit (noise-smoothed): "
        f"{_ms(fit_now - fit_split)}"
    )
    print("  (the wholly-invariant sites are only *blocked* by the mixed")
    print("   ones upstream -- a slice of a mixed batch is dataflow-")
    print("   varying -- so their share is realizable only if the mixed")
    print("   batches are split too.)")
    return save, k


# ── Part D: the HLO invariance census ────────────────────────────────

_HEADER = re.compile(r"^(ENTRY\s+)?%?([\w.\-]+)\s*\(.*\{\s*$")
_INSTR = re.compile(
    r"^\s*(ROOT\s+)?%([\w.\-]+)\s*=\s*(.*?)\s+([a-z][\w\-]*)\("
)
_OPERAND = re.compile(r"%([\w.\-]+)")
_CHEAP = {
    "get-tuple-element",
    "parameter",
    "constant",
    "tuple",
    "bitcast",
    "copy",
    "reshape",
    "iota",
}


def _split_computations(text: str):
    comps: dict[str, list[str]] = {}
    entry = None
    cur = None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m and (line.startswith("%") or line.startswith("ENTRY")):
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        if line.startswith("}"):
            cur = None
            continue
        if cur is not None:
            comps[cur].append(line)
    return comps, entry


def _parse(body: list[str]):
    out, order = {}, []
    for line in body:
        m = _INSTR.match(line)
        if not m:
            continue
        is_root, name, shape, opcode = m.groups()
        ops = _OPERAND.findall(line[m.end() :].split("), ")[0])
        out[name] = (bool(is_root), opcode, ops, shape, line.strip())
        order.append(name)
    return out, order


def _cost(shape: str) -> int:
    """Output element count -- a compute / traffic proxy."""
    tot = 0
    for dims in re.findall(r"\[([\d,]+)\]", shape):
        n = 1
        for v in dims.split(","):
            n *= int(v)
        tot += n
    return tot


def _body_cost(comps, name: str, seen=None) -> int:
    seen = seen if seen is not None else set()
    if name in seen or name not in comps:
        return 0
    seen.add(name)
    instrs, order = _parse(comps[name])
    tot = 0
    for n in order:
        _, opcode, _, shape, line = instrs[n]
        if opcode in ("parameter", "get-tuple-element", "constant"):
            continue
        tot += _cost(shape)
        for sub in re.findall(r"(?:calls|body|to_apply)=%([\w.\-]+)", line):
            tot += _body_cost(comps, sub, seen)
    return tot


def _part_d(step, state, dump: str | None):
    """Invariance census of the corrector while body, on this backend."""
    print("\n" + "-" * 72)
    print("PART D -- optimized-HLO invariance census (this backend)")
    print("-" * 72)
    try:
        compiled = jax.jit(step).lower(state).compile()
        text = compiled.as_text()
    except Exception as exc:  # noqa: BLE001
        # XLA refuses to serialize a module past 2 GiB of protobuf,
        # which production resolutions reach.  The census is
        # *structural* -- which ops are loop-invariant does not depend
        # on the mode-plane size -- so a smaller grid answers the same
        # question, and the parts above still stand.
        print(
            f"  optimized HLO unavailable "
            f"({type(exc).__name__}: {str(exc)[:120]})"
        )
        print("  re-run with e.g. --nx 32 --nz 32 for Part D alone.")
        return 0.0, False
    if dump:
        with open(dump, "w") as fh:
            fh.write(text)
        print(f"  optimized HLO written to {dump}")
    comps, entry = _split_computations(text)
    if entry is None:
        print("  no ENTRY computation found")
        return 0.0, False
    e_instrs, e_order = _parse(comps[entry])
    whiles = [
        (n, re.search(r"body=%([\w.\-]+)", e_instrs[n][4]).group(1))
        for n in e_order
        if e_instrs[n][1] == "while"
        and re.search(r"body=%([\w.\-]+)", e_instrs[n][4])
    ]
    if not whiles:
        print("  no while in ENTRY (corrector unrolled?)")
        return 0.0, False
    # The banded solves carry their own small ``while``s; the corrector
    # is the one with by far the largest body, so its share is the one
    # the verdict quotes.
    best = (0, 0.0, "")
    for wname, bname in whiles:
        instrs, order = _parse(comps[bname])
        root = next((n for n in order if instrs[n][0]), None)
        if root is None:
            continue
        root_ops = instrs[root][2]
        gte = {}
        for n, (_, opcode, _ops, _, line) in instrs.items():
            if opcode == "get-tuple-element":
                m = re.search(r"index=(\d+)", line)
                if m:
                    gte[n] = int(m.group(1))
        inv_slots = {
            slot for slot, op in enumerate(root_ops) if gte.get(op, -1) == slot
        }
        inv = {n for n, i in gte.items() if i in inv_slots}
        changed = True
        while changed:
            changed = False
            for n in order:
                if n in inv or instrs[n][1] in (
                    "get-tuple-element",
                    "parameter",
                ):
                    continue
                if all(o in inv for o in instrs[n][2]):
                    inv.add(n)
                    changed = True
        tot_c = inv_c = tot_n = inv_n = 0
        inv_rows, dot_rows = [], []
        for n in order:
            _, opcode, _, shape, line = instrs[n]
            if opcode in _CHEAP:
                continue
            c = _cost(shape)
            kinds = set()
            for sub in re.findall(r"calls=%([\w.\-]+)", line):
                c += _body_cost(comps, sub)
                body_txt = "\n".join(comps.get(sub, []))
                for k in ("dot", "reduce", "convolution", "custom-call"):
                    if f"{k}(" in body_txt:
                        kinds.add(k)
            # On the GPU the dense-y GEMMs are not `dot` at all: XLA
            # rewrites them to a cuBLAS `custom-call` returning
            # (result, workspace).  Tagging those as elementwise would
            # hide exactly the ops this probe is about.
            if opcode == "custom-call":
                kinds.add("gemm" if "gemm" in line.lower() else "custom-call")
            tot_c += c
            tot_n += 1
            tag = ",".join(sorted(kinds)) or "elementwise"
            if kinds & {"dot", "gemm"} or opcode == "dot":
                dot_rows.append((shape.split("{")[0].strip(), n in inv))
            if n in inv:
                inv_c += c
                inv_n += 1
                inv_rows.append((opcode, shape.split("{")[0].strip(), tag))
        share = inv_c / max(tot_c, 1)
        if tot_c > best[0]:
            best = (tot_c, share, wname)
        print(
            f"  while %{wname} body=%{bname}: {len(order)} instrs, "
            f"{len(root_ops)} carry slots ({len(inv_slots)} invariant)"
        )
        print(f"    expensive ops {tot_n}, loop-INVARIANT {inv_n}")
        print(
            f"    cost proxy (out elems): invariant {inv_c}/{tot_c}"
            f"  = {100.0 * share:.1f}%"
        )
        for opcode, shape, tag in inv_rows:
            print(f"      inv  {opcode:10} {shape:28} [{tag}]")
        print(
            f"    GEMM ops in the body (dot / cuBLAS custom-call): "
            f"{len(dot_rows)}"
        )
        print(
            "     (trailing dim = C x N_1 x N_2 x 2, so it reads off "
            "C -- the\n      cross-check on PART C's C_total column)"
        )
        for shape, is_inv in dot_rows:
            print(
                f"      dot  {shape:32} {'INVARIANT' if is_inv else 'varying'}"
            )
    print(
        f"\n  corrector while = %{best[2]} (largest body): "
        f"{100.0 * best[1]:.1f}% of its cost proxy is loop-invariant"
    )
    return best[1], True


# ── env banner + main ────────────────────────────────────────────────


def _print_env(system: str, cimm: bool) -> None:
    import jaxlib

    print("=" * 72)
    print("Corrector loop-invariance probe")
    print(f"  jax     {jax.__version__}")
    print(f"  jaxlib  {jaxlib.__version__}")
    print(f"  devices {jax.devices()}")
    print(f"  backend {jax.default_backend()}")
    print(
        f"  system  {system}  family {FAMILY[system]}  consistent_imm {cimm}"
    )
    print(
        f"  res     ny={params.res.ny} nx={params.res.nx} "
        f"nz={params.res.nz} fd_order={params.res.fd_order}"
    )
    print(
        f"  step    dt={params.step.dt} scheme={params.step.scheme} "
        f"implicitness={params.step.implicitness}"
    )
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        default="plane-couette",
        help="plane-couette/plane-poiseuille/pipe/taylor-couette/dean",
    )
    ap.add_argument(
        "--consistent-imm",
        type=int,
        default=1,
        choices=[0, 1],
        help="res.consistent_imm (1 = the v-omega formulation)",
    )
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--fd-order", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument(
        "--dt-sweep",
        default="",
        help="extra dt values for Part 0, comma-separated "
        "(set_dt rebuild, no recompile)",
    )
    ap.add_argument(
        "--spinup",
        type=int,
        default=20,
        help="steps run before the pass count is read (a fresh random "
        "IC is not a developed field)",
    )
    ap.add_argument(
        "--steps", type=int, default=10, help="timed steps per forced K"
    )
    ap.add_argument(
        "--passes",
        default="1,2,3,4,6",
        help="forced corrector pass counts for the Part A fit",
    )
    ap.add_argument(
        "--cmax", type=int, default=8, help="max matvec columns in Part B"
    )
    ap.add_argument(
        "--gemm-reps",
        type=int,
        default=4,
        help="distinct operands per timed matvec batch (each is "
        "N_y x C x N_1 x N_2 complex -- keep small)",
    )
    ap.add_argument(
        "--hlo-out", default=None, help="file for the optimized HLO dump"
    )
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="GPU-less self-check: every part on CPU at tiny resolution "
        "(timings meaningless; structure/fits/census exercised)",
    )
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend (default cpu; use cuda for the real numbers)",
    )
    args = ap.parse_args()

    if args.cpu_smoke:
        args.platform = "cpu"
        args.ny, args.nx, args.nz = 17, 8, 8
        args.spinup, args.steps, args.cmax, args.gemm_reps = 2, 2, 3, 2
        args.passes = "1,2,3"

    cimm = bool(args.consistent_imm)
    if args.system not in FAMILY:
        raise SystemExit(f"unsupported system: {args.system}")

    # Backend first: the geometry / sharding singletons capture it.
    configure_jax_platform(args.platform)
    _configure_system(
        args.system,
        args.ny,
        args.nx,
        args.nz,
        args.fd_order,
        cimm,
        args.dt,
    )
    geom = _geom_module(args.system)
    m = _import_flow(args.system)
    flow = m.flow
    from dnsjax.sharding import sharding

    _print_env(args.system, cimm)

    state = _make_state(m)
    dts = [args.dt] + [float(v) for v in args.dt_sweep.split(",") if v.strip()]
    k_map, state = _part_0(m, state, args.spinup, dts)
    k_nat = k_map[args.dt]

    ks = [int(v) for v in args.passes.split(",") if v.strip()]
    build = _builder(args.system, geom)
    _a, b, times = _part_a(build, flow, state, ks, args.steps)

    def step_at(k: int) -> float:
        """Measured step time at *k* passes (the fit fills the gaps)."""
        return times[ks.index(k)] if k in ks else _a + b * k

    t_step = step_at(k_nat)

    curves = _part_b(
        geom,
        flow,
        sharding,
        FAMILY[args.system],
        args.cmax,
        args.gemm_reps,
    )
    save, k_used = _part_c(curves, FAMILY[args.system], cimm, k_nat, step_at)

    # The census wants the production cap back (it only sets the trip
    # count constant, but a pinned one is a misleading dump).
    params.step.corrector_tolerance = 1e-5
    params.step.max_corrector_iterations = 10
    step = build(flow)[3]
    share, hlo_ok = _part_d(step, state, args.hlo_out)

    # Nothing re-executes at K = 1, so the realized saving is zero
    # there and the Part C table is the K = 2 hypothetical.
    real = k_nat >= 2
    save_r = save if real else 0.0
    elem = max(k_nat - 1, 0) * share * b
    print("\n" + "-" * 72)
    print("VERDICT")
    print("-" * 72)
    print(
        f"  natural passes K            {k_nat}"
        + ("" if real else "   <- nothing to hoist at K = 1")
    )
    print(f"  step                        {_ms(t_step)}")
    print(
        f"  one corrector pass b        {_ms(b)}"
        f"   ({100.0 * b / t_step:.1f}% of the step)"
    )
    print(
        f"  matvec saving if split      {_ms(save_r)}"
        f"   ({100.0 * save_r / t_step:.2f}% of the step)  [measured]"
    )
    if not real:
        t_k = step_at(k_used)
        print(
            f"    (the Part C table is the K = {k_used} hypothetical: "
            f"{_ms(save)} = {100.0 * save / t_k:.2f}% of the "
            f"K = {k_used} step, {(share * b + save) / t_k * 100:.2f}% "
            f"with the elementwise share)"
        )
    print(
        f"  invariant elementwise       {_ms(elem)}"
        f"   ({100.0 * elem / t_step:.2f}% of the step)  [HLO proxy "
        f"{100.0 * share:.1f}% of the body x (K-1) b]"
    )
    print(
        f"  upper bound on the refactor {_ms(save_r + elem)}"
        f"   ({100.0 * (save_r + elem) / t_step:.2f}% of the step)"
    )
    print(
        f"SUMMARY system={args.system} cimm={int(cimm)} "
        f"ny={args.ny} nx={args.nx} nz={args.nz} K={k_nat} "
        f"step_ms={t_step * 1e3:.4f} pass_ms={b * 1e3:.4f} "
        f"gemm_save_ms={save_r * 1e3:.4f} elem_share={share:.4f} "
        f"bound_pct={100.0 * (save_r + elem) / t_step:.3f} "
        # Part D can bail out (the 2 GiB HLO limit), and a zero share
        # then reads exactly like a measured zero.  A sweep collapsed
        # with ``grep '^SUMMARY'`` never sees the stdout that says so.
        f"hlo={'ok' if hlo_ok else 'UNAVAILABLE'}"
    )
    if args.cpu_smoke:
        print("\n--cpu-smoke PASS: harness runs end-to-end.")


if __name__ == "__main__":
    main()
