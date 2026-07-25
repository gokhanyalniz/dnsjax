"""GPU diagnostic: where does the Pallas banded solve's time go?

The mode-inner ``.solve`` contract was bit-identical yet gave **0%**
speedup on a real GPU.  This script measures *why*, answering two
questions for the ``pallas`` backend on real hardware:

  H1  Is the per-solve time dominated by the **unavoidable** complex ->
      real split/recombine around the Triton kernel (which the contract
      change did **not** remove), rather than the layout transpose
      (which it did)?  The split is mandatory: the f64 Triton kernel
      cannot ingest ``c128`` (JAX has no zero-copy complex<->real
      bitcast), so every solve must materialise a real buffer and
      recombine.  If the split/recombine is a large share of the solve,
      no transpose removal could ever help -- the transpose was fused
      *into* that mandatory copy (same bytes), which is exactly why
      removing it changed nothing.

  H2  Is the banded solve even the bottleneck of a corrector step, or do
      the FFTs / FD matvecs / influence-matrix apply dominate?  If the
      solve is a small share of the step, no solve-kernel optimisation
      can move the wall clock.

Part A  micro-breakdown of one ``Lk`` solve: full solve vs kernel-only
        vs split-only vs recombine-only, each with effective HBM
        bandwidth (compare to the device peak: H100 HBM3 ~3.35 TB/s).
Part B  full ``predict_and_fully_correct`` step time vs the isolated
        ``Lk`` and ``Hk`` solve times -- the solve's share of the step.
        Then (Cartesian) a **stage-level ``_imm_iteration`` breakdown**
        that sizes the influence-matrix boundary correction -- the part
        the openpipeflow wiki calls "negligible" -- against the rest of
        the implicit CN update (assembly GEMMs + banded solves), and the
        corrected **cnab2 composition** (wall-bounded cnab2 still runs a
        ``2 + c``-apply FFT-free IMM corrector, so its ``FFT RHS`` count
        drops ``2 + c -> 1`` but its IMM count does not).
Part C  optimized-HLO op census around the Triton custom call (static
        evidence that no separate transpose copy is left) and an
        optional ``jax.profiler`` trace for the per-kernel breakdown.

Run **on a GPU** (single device, no mpirun)::

    .venv/bin/python scripts/pallas_solve_profile.py
    .venv/bin/python scripts/pallas_solve_profile.py --system pipe \
        --ny 128 --nx 192 --nz 192 --trace /tmp/dnsjax_trace \
        --hlo-out /tmp/dnsjax_hlo.txt

Solve-kernel tile sweep (each config is a fresh process -- the tile is
baked into the operator at construction, so it cannot vary in-process)::

    for m0 in 1 2 4; do for m1 in 16 32; do \
      .venv/bin/python scripts/pallas_solve_profile.py --dist.platform cuda \
        --pallas-block-m0 $m0 --pallas-block-m1 $m1 --steps 40; done; done

Resolution x tile sweep to pick sane tile defaults across a range.
``--solve-only`` times just the ``Lk``/``Hk`` solves (the tile affects
only the solve; the FFT-bound step is tile-independent and costly at
large planes) and prints a greppable ``SUMMARY`` line -- collect them
with ``grep '^SUMMARY'`` instead of pasting full dumps::

    for ny in 48 96 128; do for nz in 64 128 256 512; do \
      for m0 in 1 2; do \
        .venv/bin/python scripts/pallas_solve_profile.py --solve-only \
          --dist.platform cuda --ny $ny --nx $nz --nz $nz \
          --pallas-block-m0 $m0; \
      done; done; done | grep '^SUMMARY'

A full run also ends with a ``SUMMARY`` line (adds ``imm``/``step``/
``cnab2`` too), so the same grep works for the step-level confirmations.

On a GPU-less box it prints the HLO census only (timings need real
hardware) so the harness can be sanity-checked before the cluster;
``--cpu-smoke`` additionally exercises Parts B/C once on CPU at tiny
resolution (numerics only) to validate the harness end-to-end.
**Paste the full stdout back** for diagnosis.
"""

from __future__ import annotations

import argparse
import time

import jax

# x64 must be set before any dnsjax module creates arrays (f64 is the
# whole point of the question).
jax.config.update("jax_enable_x64", True)

from dnsjax.bootstrap import configure_jax_platform  # noqa: E402
from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
    validate_parameters,
)

# H100 80GB HBM3 peak HBM bandwidth (~3.35 TB/s) for the roofline %.
HBM_PEAK = 3.35e12

GBPS = 1e9


# ── setup ────────────────────────────────────────────────────────────


def _configure_system(
    system: str,
    ny: int,
    nx: int,
    nz: int,
    order: int,
    solver_overrides: dict | None = None,
):
    """Set the global ``params`` for *system* and derive singletons.

    Mutates ``params`` directly (the ``pallas`` backend, f64, the given
    resolution), then triggers the derived-parameter recompute with an
    empty :class:`Parameters` merge (the ``test_*`` / tiling-diagnostic
    idiom).  Per-geometry required fields (Taylor-Couette / Dean) use the
    test-suite-standard values; the physics is irrelevant to timing.

    *solver_overrides* merges extra ``[solver]`` fields into the same
    layering call -- the Pallas kernel-tile knobs (``pallas_block_m0``
    / ``pallas_block_m1``) are baked into the operator at construction
    (geometry import), so a knob sweep needs a subprocess per value
    (the ``test_*`` subprocess-per-config idiom).
    """
    params.phys.system = system
    params.phys.re = 400.0
    params.res.nx = nx
    params.res.ny = ny
    params.res.nz = nz
    params.res.fd_order = order
    params.res.double_precision = True
    if system == "taylor-couette":
        params.phys.re1 = 100.0
        params.phys.re2 = 0.0
        params.geo.eta = 0.5
    elif system == "dean":
        params.geo.eta = 0.5
    # Through the layering call (not a direct assignment), so the
    # per-family backend re-resolution in ``update_parameters`` cannot
    # overwrite it.
    solver = {"backend": "pallas"}
    if solver_overrides:
        solver.update(solver_overrides)
    update_parameters(Parameters(solver=solver))
    # Recompute the 3/2-rule padded sizes for the FFT dealiasing (the
    # step path needs these; ``update_parameters`` does not set them --
    # ``__main__`` calls this separately).
    padded_res.set_padded_resolution(params)
    validate_parameters()


def _geom_module(system: str):
    """The geometry module (holds ``_get_rhs`` / ``_imm_iteration``)."""
    from dnsjax.geometries.wall_bounded import annular, cartesian, cylindrical

    if system in ("plane-couette", "plane-poiseuille"):
        return cartesian
    if system == "pipe":
        return cylindrical
    if system in ("taylor-couette", "dean"):
        return annular
    raise SystemExit(f"unsupported system: {system}")


def _import_flow(system: str):
    """Import the flow module for *system* and return its singletons.

    The import builds the geometry ``fourier`` / ``flow`` singletons and
    the jitted stepper, exactly as ``__main__`` does.
    """
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
        raise SystemExit(f"unsupported system for this diagnostic: {system}")
    return m


# ── timing helpers ───────────────────────────────────────────────────


def _bench(fn, args_list, warmup: int = 3) -> float:
    """Median-ish throughput: queue all calls, block on the last.

    *args_list* must hold **distinct** operands so XLA cannot CSE the
    repeated calls; the per-call wall time is the total drained-stream
    time over the batch.
    """
    f = jax.jit(fn)
    for a in args_list[: max(1, warmup)]:
        jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    outs = [f(*a) for a in args_list]
    jax.block_until_ready(outs[-1])
    return (time.perf_counter() - t0) / len(args_list)


def _bench_step(step, state, n: int, warmup: int = 3):
    """Time the donating corrector step by chaining ``state``.

    Starts from a copy: the step donates its state argument, and the
    caller's ``state`` is reused by later benchmark sections.
    """
    import jax.numpy as jnp

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
    return (time.perf_counter() - t0) / n, int(cc), s


def _bench_step_cnab2(step_cnab2, state, n: int, warmup: int = 3):
    """Time the CN/AB2 step by chaining ``(state, rhs_prev)``.

    Seeds the AB2 history with ``N(u^0)`` via a priming call (the same
    discarded-priming bootstrap the driver uses), then chains the
    carry.  Starts from copies (the step donates both arguments).
    """
    import jax.numpy as jnp

    s = jnp.copy(state)
    # seed rhs_prev = N(u^0); step_cnab2 returns (state, carry,
    # error, num_c)
    _, rp, _, _ = step_cnab2(jnp.copy(s), jnp.zeros_like(s))
    for _ in range(warmup):
        s, rp, _err, _c = step_cnab2(s, rp)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    cc = 0
    for _ in range(n):
        s, rp, _err, c = step_cnab2(s, rp)
        cc = c  # device scalar; host-convert once after the loop
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n, int(cc), s


def _ms(sec: float) -> str:
    return f"{sec * 1e3:8.3f} ms"


def _bw(nbytes: int, sec: float) -> str:
    return f"{nbytes / sec / GBPS:7.1f} GB/s"


# ── input builders ───────────────────────────────────────────────────


def _make_complex(shape, seed, sharding, spec):
    """A distinct device complex array on *spec* (mode-inner field).

    *spec* is a bare :class:`PartitionSpec`; it is wrapped in a
    :class:`NamedSharding` on ``sharding.mesh`` so the placement does not
    rely on an Explicit mesh being active.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jax.device_put(a, jax.NamedSharding(sharding.mesh, spec))


# ── Part A: solve micro-breakdown ────────────────────────────────────


def _part_a(flow, sharding, reps: int) -> None:
    import jax.numpy as jnp
    from jax import lax

    from dnsjax.solvers import (
        PerModeBandedPallasOperator,
        _pallas_banded_solve,
    )

    op = flow.Lk_op
    print("\n" + "-" * 72)
    print("PART A -- one Lk solve, broken down (the H1 test)")
    print("-" * 72)
    if not isinstance(op, PerModeBandedPallasOperator):
        print(
            f"  Lk_op is {type(op).__name__}, not the Pallas operator -- "
            "run with the pallas backend.  Skipping."
        )
        return

    L, U = op.L, op.U  # mode-inner (N, p, Nkz, Nkx) / (N, p+1, Nkz, Nkx)
    N, p, Nkz, Nkx = L.shape
    pp1 = U.shape[1]
    spec = sharding.spec_scalar_shard
    print(f"  operator: N(y)={N} p={p} mode-plane (Nkz,Nkx)=({Nkz},{Nkx})")

    # Distinct complex RHS fields, and their pre-split real buffers.
    zs = [
        _make_complex((N, Nkz, Nkx), 100 + i, sharding, spec)
        for i in range(reps)
    ]
    bs = [
        jax.block_until_ready(
            jax.jit(lambda z: jnp.stack([z.real, z.imag], axis=1))(z)
        )
        for z in zs
    ]

    t_full = _bench(lambda z: op.solve(z), [(z,) for z in zs])
    t_kern = _bench(
        lambda b: _pallas_banded_solve(L, U, b, p), [(b,) for b in bs]
    )
    t_split = _bench(
        lambda z: jnp.stack([z.real, z.imag], axis=1), [(z,) for z in zs]
    )
    t_recomb = _bench(
        lambda x: lax.complex(x[:, 0], x[:, 1]), [(b,) for b in bs]
    )

    # Essential HBM traffic (f64 = 8 B), per call.
    m = N * Nkz * Nkx
    kern_bytes = (m * p + m * pp1 + m * 2 + m * 2) * 8  # L,U read; b r; x w
    split_bytes = m * 16 + m * 2 * 8  # read c128, write 2xf64
    recomb_bytes = m * 2 * 8 + m * 16  # read 2xf64, write c128
    full_bytes = kern_bytes + split_bytes + recomb_bytes

    def _pct(t):
        return f"({100 * t / t_full:4.1f}% of full)"

    print(
        f"  full   op.solve(z)            {_ms(t_full)}   "
        f"{_bw(full_bytes, t_full)}"
    )
    print(
        f"  kernel _pallas_banded_solve   {_ms(t_kern)}   "
        f"{_bw(kern_bytes, t_kern)}   {_pct(t_kern)}"
    )
    print(
        f"  split  stack([re, im])        {_ms(t_split)}   "
        f"{_bw(split_bytes, t_split)}   {_pct(t_split)}"
    )
    print(
        f"  recomb lax.complex(x0, x1)    {_ms(t_recomb)}   "
        f"{_bw(recomb_bytes, t_recomb)}   {_pct(t_recomb)}"
    )

    plumb = (t_split + t_recomb) / t_full
    summ = t_kern + t_split + t_recomb
    marginal = t_full - t_kern  # true fused cost of split+recombine
    kern_frac = t_kern / t_full
    kern_peak = (kern_bytes / t_kern) / HBM_PEAK
    print(
        f"\n  isolated split+recombine = {100 * plumb:4.1f}% of full, but "
        f"sum(pieces)/full = {summ / t_full:4.2f}."
    )
    print(
        "  A ratio >1 means the isolated ops OVER-count (each pays a launch "
        "+ a full\n  HBM round-trip it does NOT pay when fused into the "
        "solve).  The TRUE cost of"
    )
    print(
        f"  the split+recombine fused in the solve is full - kernel = "
        f"{_ms(marginal)} ({100 * marginal / t_full:4.1f}% of full)."
    )
    print(
        f"  kernel = {100 * kern_frac:4.1f}% of the solve, running at "
        f"{kern_bytes / t_kern / GBPS:.0f} GB/s = {100 * kern_peak:.0f}% "
        "of H100 HBM3 peak."
    )
    if kern_frac >= 0.70:
        print(
            "  => The solve is KERNEL-bound; the plumbing (and the removed "
            "transpose,\n     fused into it) is ~free -- THAT is why the "
            "contract change gave 0%, not\n     because the split "
            "dominates.  The kernel sits far below peak BW, so it\n     is "
            "limited by the sequential banded recurrence / occupancy, not "
            "bandwidth.\n     A faster SOLVE needs the kernel itself: tune "
            "pallas_block_m0/m1, or parallelize along Ny.\n     But first: "
            "Part B -- is the solve even the step's bottleneck?"
        )
    else:
        print(
            "  => The plumbing is a real share of the solve; the split/"
            "recombine round-trips\n     (mandatory: the kernel cannot "
            "ingest c128) are worth attacking -- carry\n     the field "
            "split-real, or batch the launches.  See Part B for whether the"
            "\n     solve matters to the step at all."
        )


# ── Part B: solve share of a corrector step ──────────────────────────


def _imm_stage_breakdown(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Cartesian: split one ``_imm_iteration`` into its stages.

    Sizes the **influence-matrix boundary correction** (Stages 4-7 of
    ``_imm_iteration`` in ``cartesian.py``) -- the openpipeflow-wiki
    "negligible" per-step overhead -- against the rest of the implicit
    Crank-Nicolson update (assembly ``D1``/``D2`` GEMMs + the ``Lk``/
    ``Hk`` banded solves), which is the work openpipeflow *also* runs
    every corrector iteration and the wiki does **not** count as
    influence-matrix overhead.  Stages mirror ``_imm_iteration``
    exactly; the constant-bulk / block-spanwise branch (Stages 8-9) is
    compiled out for the default plane-Couette driving and omitted.
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import apply_y_matrix

    c = params.step.implicitness
    dt = params.step.dt
    nu = 1.0 / params.phys.re
    ikx = 1j * fourier.kx
    ikz = 1j * fourier.kz
    mean_mask = fourier.mean_mask

    def poisson_rhs(velocity_n, nonlin_j, nonlin_n):
        # Stages 1-2 up to the zero-BC pressure RHS f_hat_P.
        u_n, v_n, w_n = velocity_n[0], velocity_n[1], velocity_n[2]
        dy_stack = apply_y_matrix(
            flow.D1,
            jnp.stack([v_n, nonlin_j[1], nonlin_n[1]], axis=1),
            component_axis=1,
        )
        dy_v_n = dy_stack[:, 0]
        dy_Nv_j = dy_stack[:, 1]
        dy_Nv_n = dy_stack[:, 2]
        d_hat_n = ikx * u_n + dy_v_n + ikz * w_n
        div_Nj = ikx * nonlin_j[0] + dy_Nv_j + ikz * nonlin_j[2]
        div_Nn = ikx * nonlin_n[0] + dy_Nv_n + ikz * nonlin_n[2]
        Lk_d = geom._lk_matvec(d_hat_n, flow, fourier)
        f_hat = (
            d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d
        )
        return f_hat.at[0].set(0.0).at[-1].set(0.0)

    def helmholtz_rhs(velocity_n, nonlin_j, nonlin_n, pP):
        # Stage 3 up to the Helmholtz RHS R_stack (before the solve).
        grad_pP = jnp.stack([ikx * pP, apply_y_matrix(flow.D1, pP), ikz * pP])
        hk_minus = jax.vmap(geom._hk_minus_matvec, in_axes=(0, None, None))(
            velocity_n, flow, fourier
        )
        r = hk_minus - grad_pP + c * nonlin_j + (1 - c) * nonlin_n
        r = r.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return r.at[1].set(jnp.where(mean_mask, 0.0, r[1]))

    def influence_correct(velocity_j, arb_stack):
        # Stages 4-7 + finalize: the influence-matrix correction.
        # Transcribes the *ungated* 2x2 IMM.  On Cartesian
        # ``res.consistent_imm`` selects a different algorithm
        # entirely (the v-omega_y scheme: no pressure, three solves
        # per mode instead of four -- ``cartesian._imm_iteration``),
        # so this profiler rejects the flag rather than silently
        # characterising something it does not transcribe.
        assert not params.res.consistent_imm, (
            "pallas_solve_profile transcribes the default 2x2 IMM; "
            "res.consistent_imm selects a different scheme"
        )
        u_arb, v_arb, w_arb = arb_stack[0], arb_stack[1], arb_stack[2]
        d_wall = jnp.einsum("bj, jzx -> zxb", flow.D1_bnd, v_arb)
        d_wall = d_wall.at[..., 1].set(
            jnp.where(mean_mask[0], 0.0, d_wall[..., 1])
        )
        alpha = -jnp.einsum("zxab, zxb -> zxa", flow.M_inv, d_wall)
        alpha1 = alpha[..., 0][None]
        alpha2 = alpha[..., 1][None]
        v_new = v_arb + alpha1 * flow.v1 + alpha2 * flow.v2
        v_new = jnp.where(mean_mask, 0.0, v_new)
        q_new = alpha1 * flow.q1 + alpha2 * flow.q2
        u_new = u_arb - ikx * q_new
        w_new = w_arb - ikz * q_new
        return jnp.array([u_new, v_new, w_new]) - velocity_j

    # Realistic intermediates for the later stages (predictor call:
    # velocity_n = velocity_j = state, nonlin_n = nonlin_j = nonlin).
    lk_solve = jax.jit(lambda f: flow.Lk_op.solve(f))
    hk_solve = jax.jit(lambda r: flow.Hk_op.solve(r))
    f_hat_P = jax.block_until_ready(
        jax.jit(poisson_rhs)(state, nonlin, nonlin)
    )
    pP = jax.block_until_ready(lk_solve(f_hat_P))
    R_stack = jax.block_until_ready(
        jax.jit(helmholtz_rhs)(state, nonlin, nonlin, pP)
    )
    arb = jax.block_until_ready(hk_solve(R_stack))

    t_prhs = _bench(poisson_rhs, [(state, nonlin, nonlin)] * reps)
    t_hrhs = _bench(helmholtz_rhs, [(state, nonlin, nonlin, pP)] * reps)
    t_infl = _bench(influence_correct, [(state, arb)] * reps)

    t_sum = t_prhs + t_lk + t_hrhs + t_hk + t_infl
    print("\n  IMM stage breakdown (Cartesian; each runs n times/step):")
    print(f"    Poisson RHS asm (D1 GEMM+div+_lk+CN)  {_ms(t_prhs)}")
    print(f"    Lk banded solve                       {_ms(t_lk)}")
    print(f"    Helmholtz RHS asm (grad+_hk GEMM+CN)  {_ms(t_hrhs)}")
    print(f"    Hk banded solve (3 fields)            {_ms(t_hk)}")
    print(
        f"    influence correct + recombine         {_ms(t_infl)}  "
        "<- openpipeflow-wiki 'negligible' part"
    )
    print(
        f"    {'-' * 52}\n"
        f"    sum of stages                         {_ms(t_sum)}  "
        f"(vs _imm_iteration {_ms(t_imm)})"
    )
    print(
        f"\n    influence correct = {100 * t_infl / t_imm:.1f}% of "
        f"_imm_iteration, {100 * n * t_infl / t_step:.1f}% of the step."
    )
    print(
        "    => THAT is the wiki's 'negligible' influence-matrix overhead,"
        " and it\n       IS negligible here too.  The rest of the IMM is "
        "the implicit CN\n       update (assembly GEMMs + Lk/Hk banded "
        "solves) openpipeflow also\n       runs every corrector iteration"
        " -- the wiki does not count THAT."
    )


def _part_b(geom, m, flow, sharding, reps: int, steps: int) -> None:
    from dnsjax.random_field import generate_random_state
    from dnsjax.solvers import PerModeBandedPallasOperator

    print("\n" + "-" * 72)
    print("PART B -- where the corrector step's time goes (the H2 test)")
    print("-" * 72)

    fourier = m.fourier
    # ICs are physical; the steppers work in the geometry's solver
    # basis (the same single crossing ``__main__`` performs).
    to_solver = getattr(m, "to_solver_basis", lambda x: x)
    state = to_solver(
        generate_random_state(
            params.init.random_amplitude,
            params.init.random_smoothness,
            params.init.random_seed,
            params.init.random_mean_flow,
        )
    )
    t_step, c, _ = _bench_step(m.predict_and_fully_correct, state, steps)
    n = 2 + c  # RHS evals AND IMM applies per step (predict + correct loop)
    print(
        f"  predict_and_fully_correct     {_ms(t_step)}   "
        f"(c={c} corrector iters => n={n} RHS evals + {n} IMM applies)"
    )

    # Phase costs: one nonlinear RHS (FFT-heavy) and one IMM apply (FD
    # matvecs + banded solves + influence matrix + reshard transposes).
    def f_rhs(s):
        return geom._get_rhs(s, fourier, flow)

    def f_imm(s, r):
        return geom._imm_iteration(s, s, r, r, fourier, flow)

    rhs = jax.block_until_ready(jax.jit(f_rhs)(state))
    t_rhs = _bench(f_rhs, [(state,)] * reps)
    t_imm = _bench(f_imm, [(state, rhs)] * reps)
    print(f"  _get_rhs       (FFTs)         {_ms(t_rhs)}")
    print(f"  _imm_iteration (lin. alg.)    {_ms(t_imm)}")

    # Banded-solve sub-cost inside one IMM apply (Lk + stacked Hk).
    t_solve = None
    t_lk = t_hk = None
    lk, hk = flow.Lk_op, flow.Hk_op
    if isinstance(lk, PerModeBandedPallasOperator):
        N, _p, Nkz, Nkx = lk.L.shape
        sspec, vspec = (sharding.spec_scalar_shard, sharding.spec_vector_shard)
        zs = [
            _make_complex((N, Nkz, Nkx), 200 + i, sharding, sspec)
            for i in range(reps)
        ]
        z3s = [
            _make_complex((3, N, Nkz, Nkx), 300 + i, sharding, vspec)
            for i in range(reps)
        ]
        t_lk = _bench(lambda z: lk.solve(z), [(z,) for z in zs])
        t_hk = _bench(lambda z: hk.solve(z), [(z,) for z in z3s])
        t_solve = t_lk + t_hk
        print(
            f"    of which banded solve Lk+Hk {_ms(t_solve)}  "
            f"(Lk {_ms(t_lk)}, Hk {_ms(t_hk)})"
        )

    # Step composition (each phase runs n times); 'other' is the
    # unmodelled remainder (norm, predictor/corrector arithmetic, the
    # lax.while_loop carry + dispatch overhead -- i.e. non-kernel work).
    fft = n * t_rhs / t_step
    imm = n * t_imm / t_step
    other = 1.0 - fft - imm
    print("\n  step composition (each phase runs n times):")
    print(f"    FFT / nonlinear RHS        {100 * fft:5.1f}%")
    print(f"    IMM linear algebra         {100 * imm:5.1f}%")
    if t_solve is not None:
        solve = n * t_solve / t_step
        print(f"        banded solve           {100 * solve:5.1f}%")
        print(f"        matvec+transpose+infl. {100 * (imm - solve):5.1f}%")
    print(
        f"    other (norm/predict/loop)  {100 * other:5.1f}%   "
        "<- non-kernel remainder"
    )
    print(
        f"    [trust: FFT+IMM = {100 * (fft + imm):.0f}% of the step; the "
        "rest is 'other'.\n     Isolated timing != the fused step, so treat "
        "these as indicative.]"
    )

    shares = {
        "FFT / nonlinear transforms (cuFFT)": fft,
        "IMM linear algebra (FD matvec GEMMs + reshard transposes)": imm,
        "non-kernel overhead (while_loop carry / dispatch)": other,
    }
    top = max(shares, key=shares.get)
    print(
        f"\n  => H2: the banded solve is a small slice; the step's largest "
        f"bucket is\n     {top}.\n     That -- not the Pallas kernel -- is "
        "the target for a faster step."
    )

    # Stage-level IMM breakdown (Cartesian only): where the non-solve
    # part of _imm_iteration goes, and how small the influence-matrix
    # correction really is (the openpipeflow-wiki claim, measured).
    if (
        params.phys.system in ("plane-couette", "plane-poiseuille")
        and t_lk is not None
    ):
        _imm_stage_breakdown(
            geom,
            flow,
            fourier,
            rhs,
            state,
            reps,
            t_imm,
            t_lk,
            t_hk,
            t_step,
            n,
        )

    t_cnab2 = None
    # Scheme A/B.  cnab2 does ONE FFT RHS eval per step, but -- for
    # wall-bounded flows -- STILL runs a (2 + c)-apply IMM corrector
    # (the FFT-free base-flow-coupling loop, ``_cnab2_lbf_core``): the
    # predictor + first correction + c coupling iterations, each a full
    # ``_imm_iteration``, plus a (2 + c) FFT-free ``_l_bf`` re-eval.  So
    # cnab2 cuts the FFT-RHS count 2+c -> 1 but pays the SAME IMM count
    # as iterative-cn; with IMM ~ FFT on the GPU, that is why the step
    # speedup is modest, not ~3x.
    if hasattr(m, "step_cnab2"):
        t_cnab2, cc, _ = _bench_step_cnab2(m.step_cnab2, state, steps)
        n_cn = 2 + cc  # IMM applies AND _l_bf evals per cnab2 step
        t_lbf = None
        if hasattr(geom, "_l_bf"):
            t_lbf = _bench(
                lambda s: geom._l_bf(s, fourier, flow), [(state,)] * reps
            )
        print(f"\n  scheme A/B (same state, {steps} steps):")
        print(
            f"    iterative-cn  {_ms(t_step)}   "
            f"({n} FFT RHS evals + {n} IMM applies per step)"
        )
        print(
            f"    cnab2         {_ms(t_cnab2)}   "
            f"(1 FFT RHS eval + {n_cn} IMM applies + {n_cn} FFT-free "
            f"_l_bf evals, c={cc})"
        )
        print(
            f"    => step speedup {t_step / t_cnab2:.2f}x; explicit "
            "nonlinear, so dt is advective-CFL-limited."
        )
        if t_lbf is not None:
            print(f"\n  cnab2 composition (c={cc}):")
            print(f"    1  FFT RHS eval              {_ms(t_rhs)}")
            print(f"    {n_cn}  IMM applies              {_ms(n_cn * t_imm)}")
            print(f"    {n_cn}  _l_bf evals (FFT-free)   {_ms(n_cn * t_lbf)}")
            print(
                "    => cnab2 removes (1 + c) of iterative-cn's FFT RHS "
                "evals but keeps\n       the same (2 + c) IMM applies; "
                "with IMM ~ FFT here, the FFT-count\n       cut only "
                "reaches ~half the step -- the IMM is the other lever."
            )

    return {
        "step": t_step,
        "imm": t_imm,
        "lk": t_lk,
        "hk": t_hk,
        "solve": t_solve,
        "cnab2": t_cnab2,
    }


# ── solve-only sweep: isolated solve timing + greppable summary ───────


def _summary_line(system: str, args, flow, times: dict) -> None:
    """Emit one ``SUMMARY`` line for a resolution x tile sweep.

    Machine-parseable (``grep '^SUMMARY'``); ``times`` values are
    seconds (``None`` -> ``NA``).  ``progs`` is the per-field Triton
    program count `$(\\lceil N_{kz}/m_0\\rceil)(\\lceil N_{kx}/m_1
    \\rceil)$` on the *stored* (whole-tile-padded) mode plane -- the
    occupancy metric to compare against the device SM count (H100:
    132).
    """
    from dnsjax.solvers import PerModeBandedPallasOperator

    so = params.solver
    m0, m1 = so.pallas_block_m0, so.pallas_block_m1
    if isinstance(flow.Lk_op, PerModeBandedPallasOperator):
        _, _, pnz, pnx = flow.Lk_op.L.shape
        progs = (pnz // m0) * (pnx // m1)
        plane = f"{pnz}x{pnx}"
    else:
        progs, plane = 0, "NA"

    def fmt(v):
        return "NA" if v is None else f"{v * 1e3:.3f}"

    print(
        "SUMMARY "
        f"sys={system} ny={args.ny} nx={args.nx} nz={args.nz} "
        f"pad_plane={plane} progs={progs} m0={m0} m1={m1} "
        f"lk_ms={fmt(times.get('lk'))} hk_ms={fmt(times.get('hk'))} "
        f"solve_ms={fmt(times.get('solve'))} imm_ms={fmt(times.get('imm'))} "
        f"step_ms={fmt(times.get('step'))} cnab2_ms={fmt(times.get('cnab2'))}"
    )


def _solve_sweep(system: str, args, flow, sharding, reps: int) -> None:
    r"""Time ONLY the isolated ``Lk``/``Hk`` banded solves.

    The tile choice affects *only* the solve (the step is FFT-bound and
    tile-independent), and the full FFT/step is expensive at large mode
    planes, so this skips Parts A/C and the full step/cnab2: it builds
    the operators (cheap, `$O(N_y\,p\,\text{modes})$`) and times the two
    solves the IMM runs.  Cheap at arbitrarily large ``nx``/``nz`` --
    the driver for a resolution x tile sweep deciding sane tile
    defaults.  Emits a ``SUMMARY`` line.
    """
    from dnsjax.solvers import PerModeBandedPallasOperator

    print("\n" + "-" * 72)
    print("SOLVE-ONLY -- isolated Lk/Hk banded solve timing")
    print("-" * 72)
    lk, hk = flow.Lk_op, flow.Hk_op
    if not isinstance(lk, PerModeBandedPallasOperator):
        print(f"  needs the pallas backend; got {type(lk).__name__}.")
        return
    N, _p, pnz, pnx = lk.L.shape
    sspec = sharding.spec_scalar_shard
    vspec = sharding.spec_vector_shard
    zs = [
        _make_complex((N, pnz, pnx), 400 + i, sharding, sspec)
        for i in range(reps)
    ]
    z3s = [
        _make_complex((3, N, pnz, pnx), 500 + i, sharding, vspec)
        for i in range(reps)
    ]
    t_lk = _bench(lambda z: lk.solve(z), [(z,) for z in zs])
    t_hk = _bench(lambda z: hk.solve(z), [(z,) for z in z3s])
    m0, m1 = params.solver.pallas_block_m0, params.solver.pallas_block_m1
    progs = (pnz // m0) * (pnx // m1)
    print(
        f"  padded plane {pnz}x{pnx}, tile {m0}x{m1} -> {progs} "
        "programs/field\n  (H100 = 132 SMs; want >=~132 for one wave, "
        "several x for latency hiding)"
    )
    print(f"  Lk solve (1 field)   {_ms(t_lk)}")
    print(f"  Hk solve (3 fields)  {_ms(t_hk)}")
    print(f"  Lk+Hk                {_ms(t_lk + t_hk)}")
    _summary_line(
        system, args, flow, {"lk": t_lk, "hk": t_hk, "solve": t_lk + t_hk}
    )


# ── Part C: HLO census + optional profiler trace ─────────────────────


def _hlo_census(label, jitted, args, hlo_out) -> None:
    keys = [
        "transpose",
        "copy",
        "bitcast",
        "fusion",
        "custom-call",
        "convert",
        "triton",
    ]
    try:
        txt = jitted.lower(*args).compile().as_text()
    except Exception as e:  # noqa: BLE001
        print(f"  [{label}] compile/as_text failed: {type(e).__name__}: {e}")
        return
    low = txt.lower()
    print(f"  [{label}] optimized-HLO op census:")
    print("      " + "  ".join(f"{k}={low.count(k)}" for k in keys))
    if hlo_out:
        with open(hlo_out, "a") as f:
            f.write(f"\n===== {label} =====\n{txt}\n")
        print(f"      (full HLO appended to {hlo_out})")


def _part_c(flow, m, sharding, trace_dir, hlo_out) -> None:
    import jax.numpy as jnp

    print("\n" + "-" * 72)
    print("PART C -- optimized-HLO census (static) + optional trace")
    print("-" * 72)
    print(
        "  Census is for the ACTIVE backend: on GPU the 'triton' /"
        " 'custom-call'\n  counts are the solve; a near-absence of a "
        "standalone 'transpose' feeding\n  it is the point (the transpose "
        "fused into the mandatory split copy).  On\n  CPU it shows the "
        "pure-JAX fallback instead (its own mode-outer moveaxes)."
    )

    from dnsjax.random_field import generate_random_state
    from dnsjax.solvers import PerModeBandedPallasOperator

    state = getattr(m, "to_solver_basis", lambda x: x)(
        generate_random_state(
            params.init.random_amplitude,
            params.init.random_smoothness,
            params.init.random_seed,
            params.init.random_mean_flow,
        )
    )
    _hlo_census(
        "corrector-step",
        jax.jit(m.predict_and_fully_correct),
        (state,),
        hlo_out,
    )

    op = flow.Lk_op
    if isinstance(op, PerModeBandedPallasOperator):
        N, _p, Nkz, Nkx = op.L.shape
        z = _make_complex(
            (N, Nkz, Nkx), 7, sharding, sharding.spec_scalar_shard
        )
        _hlo_census(
            "Lk.solve", jax.jit(lambda zz: op.solve(zz)), (z,), hlo_out
        )

    if trace_dir:
        if jax.default_backend() != "gpu":
            print("  --trace ignored (no GPU backend).")
            return
        print(f"\n  capturing a profiler trace of 20 steps to {trace_dir} ...")
        s = jnp.copy(state)  # the step donates its state argument
        for _ in range(3):  # warm up before tracing
            s, _e, _c = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        with jax.profiler.trace(trace_dir):
            for _ in range(20):
                s, _e, _c = m.predict_and_fully_correct(s)
            jax.block_until_ready(s)
        print(
            f"  trace written.  View per-kernel times with:\n"
            f"    tensorboard --logdir {trace_dir}\n"
            "  (Profile tab -> trace_viewer), or open the .pb in "
            "https://ui.perfetto.dev .\n"
            "  Look at the kernel-category split: triton custom-call "
            "(the solve) vs\n  cuFFT vs cublas/GEMM (matvec) vs "
            "copy/fusion (the split/recombine)."
        )


# ── env banner + main ────────────────────────────────────────────────


def _print_env() -> None:
    import jaxlib

    print("=" * 72)
    print("Pallas banded-solve profile")
    print(f"  jax     {jax.__version__}")
    print(f"  jaxlib  {jaxlib.__version__}")
    try:
        import triton

        print(f"  triton  {triton.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  triton  (import failed: {e})")
    print(f"  devices {jax.devices()}")
    for dv in jax.devices():
        print(f"    - {dv} kind={getattr(dv, 'device_kind', '?')}")
    print(f"  default_backend  {jax.default_backend()}")
    print(
        f"  pallas tile      m0={params.solver.pallas_block_m0} "
        f"m1={params.solver.pallas_block_m1}"
    )
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        default="plane-couette",
        help="plane-couette/plane-poiseuille/pipe/taylor-couette/dean",
    )
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nz", type=int, default=128)
    ap.add_argument("--fd-order", type=int, default=8)
    ap.add_argument(
        "--reps",
        type=int,
        default=8,
        help="distinct RHS fields per timed batch",
    )
    ap.add_argument(
        "--steps", type=int, default=20, help="corrector steps timed in Part B"
    )
    ap.add_argument(
        "--steps-only",
        type=int,
        default=0,
        help="run N steps and exit (skips A/B/C); a clean "
        "driver to wrap in an external profiler, e.g. "
        "`nsys profile --stats=true`",
    )
    ap.add_argument(
        "--solve-only",
        action="store_true",
        help="time ONLY the isolated Lk/Hk banded solves and emit a "
        "SUMMARY line (skips Parts A/C + the full step/cnab2).  Cheap at "
        "large nx/nz -- the tile-vs-resolution sweep driver.",
    )
    ap.add_argument(
        "--trace", default=None, help="dir for a jax.profiler trace (Part C)"
    )
    ap.add_argument(
        "--hlo-out",
        default=None,
        help="file to append the full optimized HLO to",
    )
    # Pallas kernel-tile knobs (baked into the operator at construction;
    # sweep with one subprocess per value).  None -> model default.
    ap.add_argument("--pallas-block-m0", type=int, default=None)
    ap.add_argument("--pallas-block-m1", type=int, default=None)
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="GPU-less self-check: run Parts B/C once on CPU at tiny "
        "resolution (timings meaningless) to validate the harness "
        "before the cluster run.",
    )
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend (default cpu).  Use cuda to run the profile on "
        "a real GPU (Parts A/B need real hardware).",
    )
    args = ap.parse_args()

    if args.cpu_smoke:
        # Force a small CPU run: numerics are exercised (the pure-JAX
        # banded sweep replaces the Triton kernel), timings are not.
        args.platform = "cpu"
        args.ny, args.nx, args.nz = 16, 8, 8
        args.reps, args.steps = 2, 2

    solver_overrides = {}
    for field in ("pallas_block_m0", "pallas_block_m1"):
        val = getattr(args, field)
        if val is not None:
            solver_overrides[field] = val

    # Select the backend before importing any geometry / sharding module
    # (they capture the platform at import, all deferred into main here).
    configure_jax_platform(args.platform)

    _configure_system(
        args.system,
        args.ny,
        args.nx,
        args.nz,
        args.fd_order,
        solver_overrides or None,
    )
    geom = _geom_module(args.system)
    m = _import_flow(args.system)
    flow = m.flow
    from dnsjax.sharding import sharding

    _print_env()

    if args.cpu_smoke:
        # Exercise the full Part B (incl. the IMM stage breakdown and
        # cnab2 composition) and Part C on CPU so the added code is
        # validated on the GPU-less dev box.  Timings are meaningless.
        print(
            "\n--cpu-smoke: exercising Parts B/C + solve-only on CPU at "
            f"ny={args.ny} nx={args.nx} nz={args.nz} "
            "(numerics only; ignore the timings).\n"
        )
        times = _part_b(geom, m, flow, sharding, args.reps, args.steps)
        _part_c(flow, m, sharding, None, args.hlo_out)
        _solve_sweep(args.system, args, flow, sharding, args.reps)
        _summary_line(args.system, args, flow, times)
        print("\n--cpu-smoke PASS: harness runs end-to-end.")
        return

    if args.solve_only:
        if jax.default_backend() != "gpu":
            print("--solve-only needs a GPU backend (real solve timing).")
            return
        _solve_sweep(args.system, args, flow, sharding, args.reps)
        return

    if args.steps_only:
        # Minimal steady-state stepping for an external profiler (nsys):
        # warm up (compile), then run N steps so the capture is dominated
        # by the corrector kernels.
        from dnsjax.random_field import generate_random_state

        state = generate_random_state(
            params.init.random_amplitude,
            params.init.random_smoothness,
            params.init.random_seed,
            params.init.random_mean_flow,
        )
        s = state  # chained (donated) from here on; state unused after
        for _ in range(3):
            s, _e, _c = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        for _ in range(args.steps_only):
            s, _e, _c = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        print(
            f"ran {args.steps_only} steps (steps-only mode); wrap this "
            "invocation in a profiler."
        )
        return

    gpu = jax.default_backend() == "gpu"
    if not gpu:
        print(
            "No GPU backend -> timings skipped (they need real hardware).\n"
            "Running the HLO census only; launch on the cluster for A/B.\n"
        )
        _part_c(flow, m, sharding, None, args.hlo_out)
        return

    _part_a(flow, sharding, args.reps)
    times = _part_b(geom, m, flow, sharding, args.reps, args.steps)
    _part_c(flow, m, sharding, args.trace, args.hlo_out)
    _summary_line(args.system, args, flow, times)
    print("\n" + "=" * 72)
    print(
        "Done.  Paste the full stdout back.  Key numbers: Part A "
        "'split+recombine\n% of the solve' (H1) and Part B 'c*(Lk+Hk) / "
        "step' (H2)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
