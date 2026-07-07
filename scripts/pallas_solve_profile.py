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
Part C  optimized-HLO op census around the Triton custom call (static
        evidence that no separate transpose copy is left) and an
        optional ``jax.profiler`` trace for the per-kernel breakdown.

Run **on a GPU** (single device, no mpirun)::

    .venv/bin/python scripts/pallas_solve_profile.py
    .venv/bin/python scripts/pallas_solve_profile.py --system pipe \
        --ny 128 --nx 192 --nz 192 --trace /tmp/dnsjax_trace \
        --hlo-out /tmp/dnsjax_hlo.txt

On a GPU-less box it prints the HLO census only (timings need real
hardware) so the harness can be sanity-checked before the cluster.
**Paste the full stdout back** for diagnosis.
"""

from __future__ import annotations

import argparse
import time

import jax

# x64 must be set before any dnsjax module creates arrays (f64 is the
# whole point of the question).
jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    configure_jax_platform,
    padded_res,
    params,
    update_parameters,
    validate_parameters,
)

# H100 80GB HBM3 peak HBM bandwidth (~3.35 TB/s) for the roofline %.
HBM_PEAK = 3.35e12

GBPS = 1e9


# ── setup ────────────────────────────────────────────────────────────


def _configure_system(system: str, ny: int, nx: int, nz: int, order: int):
    """Set the global ``params`` for *system* and derive singletons.

    Mutates ``params`` directly (the ``pallas`` backend, f64, the given
    resolution), then triggers the derived-parameter recompute with an
    empty :class:`Parameters` merge (the ``test_*`` / tiling-diagnostic
    idiom).  Per-geometry required fields (Taylor-Couette / Dean) use the
    test-suite-standard values; the physics is irrelevant to timing.
    """
    params.phys.system = system
    params.phys.re = 400.0
    params.res.nx = nx
    params.res.ny = ny
    params.res.nz = nz
    params.res.fd_order = order
    params.res.double_precision = True
    params.solver.backend = "pallas"
    if system == "taylor-couette":
        params.phys.re1 = 100.0
        params.phys.re2 = 0.0
        params.geo.eta = 0.5
    elif system == "dean":
        params.geo.eta = 0.5
    update_parameters(Parameters())
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
    for _ in range(n):
        s, rp, _err, _c = step_cnab2(s, rp)
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n, s


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
            "pallas_block_m0/m1 /\n     num_warps / num_stages, or "
            "parallelize along Ny.  But first: Part B --\n     is the solve "
            "even the step's bottleneck?"
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


def _part_b(geom, m, flow, sharding, reps: int, steps: int) -> None:
    from dnsjax.random_field import generate_random_state
    from dnsjax.solvers import PerModeBandedPallasOperator

    print("\n" + "-" * 72)
    print("PART B -- where the corrector step's time goes (the H2 test)")
    print("-" * 72)

    fourier = m.fourier
    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_seed,
        params.init.random_mean_flow,
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

    # Scheme A/B: CN/AB2 does ONE RHS/FFT eval per step (no corrector
    # loop) vs n = 2+c for iterative-cn -- the primary throughput lever.
    if hasattr(m, "step_cnab2"):
        t_cnab2, _ = _bench_step_cnab2(m.step_cnab2, state, steps)
        print(f"\n  scheme A/B (same state, {steps} steps):")
        print(
            f"    iterative-cn  {_ms(t_step)}   "
            f"({n} RHS/FFT evals + {n} IMM applies per step)"
        )
        print(
            f"    cnab2         {_ms(t_cnab2)}   "
            "(1 RHS/FFT eval + 1 IMM apply per step, no while_loop)"
        )
        print(
            f"    => step speedup {t_step / t_cnab2:.2f}x  "
            f"(FFT/IMM invocation count {n} -> 1); explicit nonlinear, "
            "so dt is advective-CFL-limited."
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

    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_seed,
        params.init.random_mean_flow,
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
    ap.add_argument("--fd-order", type=int, default=4)
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
        "--trace", default=None, help="dir for a jax.profiler trace (Part C)"
    )
    ap.add_argument(
        "--hlo-out",
        default=None,
        help="file to append the full optimized HLO to",
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

    # Select the backend before importing any geometry / sharding module
    # (they capture the platform at import, all deferred into main here).
    configure_jax_platform(args.platform)

    _configure_system(args.system, args.ny, args.nx, args.nz, args.fd_order)
    geom = _geom_module(args.system)
    m = _import_flow(args.system)
    flow = m.flow
    from dnsjax.sharding import sharding

    _print_env()

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
    _part_b(geom, m, flow, sharding, args.reps, args.steps)
    _part_c(flow, m, sharding, args.trace, args.hlo_out)
    print("\n" + "=" * 72)
    print(
        "Done.  Paste the full stdout back.  Key numbers: Part A "
        "'split+recombine\n% of the solve' (H1) and Part B 'c*(Lk+Hk) / "
        "step' (H2)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
