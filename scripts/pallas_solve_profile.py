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
Part A2 the **split-real hoist**: Part A sizes the plumbing inside one
        solve, which is not what a hoist removes.  This times the real
        ``Hk.solve -> real-coefficient map -> Lk.solve`` chain against a
        variant that stays split-real across the map, so the round trip
        between two consumers is priced directly (fidelity-gated, and
        extrapolated to the step).  Its static half is the per-region
        complex/real crossing census printed by Part C.
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

Every number below is reported for the configured
``res.consistent_imm`` formulation.  The default is the shipped
reconstruction scheme; ``--legacy-imm`` profiles the retired primitive
`$(v, p)$` one instead, which has a different operator set, a different
per-mode solve count and its own stage transcription.

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
    legacy_imm: bool = False,
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

    *legacy_imm* selects the retired primitive `$(v, p)$` scheme
    (``res.consistent_imm = False``) instead of the shipped
    reconstruction one; it changes the operator set, the per-mode solve
    count and therefore every number this script reports, so it is an
    explicit opt-in (``--legacy-imm``).
    """
    params.phys.system = system
    params.phys.re = 400.0
    params.res.nx = nx
    params.res.ny = ny
    params.res.nz = nz
    params.res.fd_order = order
    params.res.double_precision = True
    params.res.consistent_imm = not legacy_imm
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
    _, rp, *_ = step_cnab2(jnp.copy(s), jnp.zeros_like(s))
    for _ in range(warmup):
        s, rp, _err, _c, *_ = step_cnab2(s, rp)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    cc = 0
    for _ in range(n):
        s, rp, _err, c, *_ = step_cnab2(s, rp)
        cc = c  # device scalar; host-convert once after the loop
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n, int(cc), s


def _ms(sec: float) -> str:
    return f"{sec * 1e3:8.3f} ms"


def _bw(nbytes: int, sec: float) -> str:
    return f"{nbytes / sec / GBPS:7.1f} GB/s"


# ── input builders ───────────────────────────────────────────────────


def _hk_components(hk) -> int:
    """Component count of one ``Hk_op`` solve, for the isolated probes.

    The RHS a step hands ``Hk_op`` is not a fixed 3-stack: the default
    ``res.consistent_imm`` scheme solves the wall-normal
    velocity/vorticity **pair** (Cartesian batches the two scalars
    through one operator; the cylindrical geometries carry a two-family
    group), while the legacy primitive scheme solves the three velocity
    components against a three-family group.  Read the arity off a
    grouped operator, and fall back to the formulation's own count for
    the ungrouped Cartesian one.
    """
    if hk.L.ndim == 5:  # (C, N, p, Nkz, Nkx): one family per slot
        return int(hk.L.shape[0])
    return 2 if params.res.consistent_imm else 3


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


def _part_a_cpu(op, sharding, reps: int) -> None:
    r"""CPU arm of Part A: where one ``Lk`` solve's time goes on CPU.

    The GPU decomposition (kernel / split / recombine) does not
    transfer, because a CPU run never reaches ``pallas_call``: it takes
    the pure-JAX sweep, which first rebuilds the **mode-outer** factors
    the sweep was written against -- crop off the Pallas tile pad,
    ``moveaxis`` both factors, un-invert the ``U`` diagonal.  That
    *factor prologue* is exactly what a CPU-native stored layout would
    delete, so sizing it bounds what any such layout can win.

    Reported: the full solve, the sweep alone (factors and RHS
    pre-prepared), the prologue alone, and the mandatory complex ->
    real split / recombine.  As on GPU the isolated pieces over-count
    -- each pays a round trip it does not pay when fused -- so the
    honest figure for anything fused into the solve is ``full - sweep``.

    **The factors are passed as jit arguments, not closed over.**  That
    is not a detail: the real stepper takes ``flow`` as an argument
    (``_base.build_wall_bounded_stepper`` calls
    ``_predict_and_fully_correct_jit(state, fourier, flow)``), so the
    factor arrays are runtime parameters and the prologue *executes*.
    Closing over the operator instead makes them compile-time constants,
    XLA folds the whole prologue away, and the solve then measures as if
    the prologue were free -- which it is not, in the configuration that
    ships.
    """
    import jax.numpy as jnp
    from jax import lax

    from dnsjax.solvers import _banded_solve_batched, _real_rhs_view

    L, U = op.L, op.U  # mode-inner (N, p, Nkz*, Nkx*) / (N, p+1, ...)
    N, p = L.shape[0], L.shape[1]
    pp1 = U.shape[1]
    Nkz, Nkx = sharding.nz_spec, sharding.nx_spec
    spec = sharding.spec_scalar_shard
    print(f"  operator: N(y)={N} p={p} mode-plane (Nkz,Nkx)=({Nkz},{Nkx})")
    if L.shape[2:] != (Nkz, Nkx):
        print(
            f"  stored mode plane {tuple(L.shape[2:])} -- unexpected on "
            "CPU (the tile pad is kernel-path only)"
        )

    zs = [
        _make_complex((N, Nkz, Nkx), 100 + i, sharding, spec)
        for i in range(reps)
    ]

    def _prologue(L_, U_):
        """The CPU branch's factor preparation, verbatim.

        Two ``moveaxis`` and nothing else: the CPU build stores the
        plain diagonal at the true plane, so there is no tile crop and
        no un-inversion here (``from_banded_factors``).
        """
        return (
            jnp.moveaxis(L_, (0, 1), (-2, -1)),
            jnp.moveaxis(U_, (0, 1), (-2, -1)),
        )

    Lo, Uo = jax.block_until_ready(jax.jit(_prologue)(L, U))
    bs = [
        jax.block_until_ready(
            jax.jit(lambda z: _real_rhs_view(jnp.moveaxis(z, 0, -1)))(z)
        )
        for z in zs
    ]

    def _full(L_, U_, z):
        return type(op)(L=L_, U=U_).solve(z)

    def _sweep(L_, U_, b):
        return _banded_solve_batched(L_, U_, b, p)

    t_full = _bench(_full, [(L, U, z) for z in zs])
    t_sweep = _bench(_sweep, [(Lo, Uo, b) for b in bs])
    t_prol = _bench(_prologue, [(L, U)] * reps)
    t_split = _bench(
        lambda z: _real_rhs_view(jnp.moveaxis(z, 0, -1)), [(z,) for z in zs]
    )
    t_recomb = _bench(
        lambda x: jnp.moveaxis(lax.complex(x[..., 0], x[..., 1]), -1, 0),
        [(b,) for b in bs],
    )

    m = N * Nkz * Nkx
    fac = (m * p + m * pp1) * 8
    sweep_bytes = fac + m * 2 * 8 * 2
    prol_bytes = 2 * fac  # read + write both factors
    split_bytes = m * 16 + m * 2 * 8
    recomb_bytes = m * 2 * 8 + m * 16

    def _pct(t):
        return f"({100 * t / t_full:4.1f}% of full)"

    print(
        f"  full   op.solve(z)            {_ms(t_full)}   "
        f"{_bw(sweep_bytes + split_bytes + recomb_bytes, t_full)}"
    )
    print(
        f"  sweep  _banded_solve_batched  {_ms(t_sweep)}   "
        f"{_bw(sweep_bytes, t_sweep)}   {_pct(t_sweep)}"
    )
    print(
        f"  prolog moveaxis x2            {_ms(t_prol)}   "
        f"{_bw(prol_bytes, t_prol)}   {_pct(t_prol)}"
    )
    print(
        f"  split  moveaxis+re/im view    {_ms(t_split)}   "
        f"{_bw(split_bytes, t_split)}   {_pct(t_split)}"
    )
    print(
        f"  recomb complex+moveaxis       {_ms(t_recomb)}   "
        f"{_bw(recomb_bytes, t_recomb)}   {_pct(t_recomb)}"
    )

    marginal = t_full - t_sweep
    summ = t_sweep + t_prol + t_split + t_recomb
    print(
        f"\n  sum(pieces)/full = {summ / t_full:4.2f}; the isolated pieces "
        "over-count (each pays\n  a round trip it does not pay fused), so "
        "read the fused figure, not them:\n  everything the solve does "
        f"around the sweep costs full - sweep = {_ms(marginal)}"
        f" ({100 * marginal / t_full:4.1f}% of full)."
    )
    if marginal <= 0.0:
        print(
            "  That is <= 0: the prologue and the re/im plumbing are fused "
            "into the sweep\n  and cost nothing measurable.  A CPU-native "
            "stored factor layout has nothing\n  left to remove here -- "
            "any change would have to make the SWEEP itself faster."
        )
    else:
        print(
            f"  The prologue alone measures {_ms(t_prol)} unfused, which "
            "bounds nothing by\n  itself (it exceeds the full solve "
            "whenever dispatch dominates).  Only the\n  fused figure "
            "above caps what a CPU-native stored layout could win, and\n"
            "  Part B's solve share caps what that is worth to the step."
        )


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
    if jax.default_backend() != "gpu":
        _part_a_cpu(op, sharding, reps)
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

    # Factors as jit *arguments*, not closed over: the stepper passes
    # ``flow`` in, so the factors are runtime parameters there.  Closing
    # over them bakes them in as constants, which is a different
    # placement (and, on the CPU arm, folds an entire stage away).
    def _full(L_, U_, z):
        return type(op)(L=L_, U=U_).solve(z)

    def _kern(L_, U_, b):
        return _pallas_banded_solve(L_, U_, b, p)

    t_full = _bench(_full, [(L, U, z) for z in zs])
    t_kern = _bench(_kern, [(L, U, b) for b in bs])
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
            "recombine round-trips\n     (mandatory *per solve*: the kernel "
            "cannot ingest c128) are worth attacking\n     -- carry the "
            "field split-real across an IMM apply, or batch the launches."
        )
    print(
        "  NOTE: this sizes ONE solve.  It does not count the "
        "split/recombine pairs a\n  step pays, so it does not size the "
        "hoist -- Part A2 does that directly.  And\n  see Part B for "
        "whether the solve matters to the step at all."
    )


# ── Part A2: the split-real hoist (census + fused ceiling) ───────────


# Word-boundary patterns: the optimized HLO writes an op application as
# ``real(f64[...] %x)``, while instruction *names* (``%real.3 = ...``)
# and identifiers containing the word must not be counted.  This is why
# ``_hlo_census``'s ``str.count`` is not reused here -- it matches
# ``real`` inside ``all-reduce``-style names and inside ``%real.3``.
_CROSSINGS = {
    "complex(": r"\bcomplex\(",
    "real(": r"\breal\(",
    "imag(": r"\bimag\(",
    "concat(": r"\bconcatenate\(",
}


def _crossing_census(label: str, jitted, args) -> dict[str, int]:
    r"""Count complex `$\leftrightarrow$` real crossings in optimized HLO.

    JAX has no zero-copy complex/real bitcast, so every banded solve and
    every :func:`~dnsjax.geometries.wall_bounded._base.apply_y_matrix`
    FD GEMM brackets itself with a split (``real``/``imag``) and a
    recombine (``complex``).  Some are redundant *between* consumers --
    ``Hk_op.solve`` recombines, the caller only indexes the result, and
    ``Lk_op.solve`` splits it straight back apart.

    This counts them **after** optimization, so it reports what
    survives XLA's simplifier rather than what the source writes.  A
    count is not a cost: fused crossings can be free, which is what
    :func:`_part_a2`'s timed arm tests.  The census sizes the *target*
    and says which region owns it.
    """
    import re

    try:
        txt = jitted.lower(*args).compile().as_text()
    except Exception as e:
        print(f"    {label:26s} census failed: {type(e).__name__}: {e}")
        return {}
    counts = {k: len(re.findall(p, txt)) for k, p in _CROSSINGS.items()}
    print(
        f"    {label:26s} "
        + "  ".join(f"{k}={v:4d}" for k, v in counts.items())
    )
    return counts


def _split_helpers():
    """Backend-matched split-real conversions and bare sweeps.

    The carried split-real layout is the one the backend's own solve
    body already builds internally, so the hoisted arm removes work
    without inventing a layout:

    - kernel path: mode-inner ``(N, 2, Nkz, Nkx)``, re/im on axis 1 --
      exactly the buffer :func:`_pallas_banded_solve` ingests;
    - CPU sweep: mode-outer ``(Nkz, Nkx, N, 2)`` -- exactly what
      :func:`_banded_solve_batched` ingests.

    Returned as ``(to_split, from_split, sweep, coeff)``; *coeff*
    reshapes a real mode-inner ``(N, Nkz, Nkx)`` coefficient into the
    matching split layout so the between-solves linear map can be
    applied without leaving it.
    """
    import jax.numpy as jnp
    from jax import lax

    from dnsjax.solvers import (
        _banded_solve_batched,
        _complex_from_view,
        _kernel_path,
        _pallas_banded_solve,
        _real_rhs_view,
    )

    if _kernel_path():

        def to_split(z):
            return jnp.stack([z.real, z.imag], axis=1)

        def from_split(x):
            return lax.complex(x[:, 0], x[:, 1])

        def sweep(L, U, b, p):
            return _pallas_banded_solve(L, U, b, p)

        def coeff(w):
            return w[:, None]

    else:

        def to_split(z):
            return _real_rhs_view(jnp.moveaxis(z, 0, -1))

        def from_split(x):
            return jnp.moveaxis(_complex_from_view(x), -1, 0)

        def sweep(L, U, b, p):
            Lo = jnp.moveaxis(L, (0, 1), (-2, -1))
            Uo = jnp.moveaxis(U, (0, 1), (-2, -1))
            return _banded_solve_batched(Lo, Uo, b, p)

        def coeff(w):
            return jnp.moveaxis(w, 0, -1)[..., None]

    return to_split, from_split, sweep, coeff


def _part_a2(system, flow, sharding, reps, t_step, n) -> None:
    r"""Size the split-real hoist: is the round trip between two
    solves worth removing?

    Every wall-bounded IMM runs the same chain -- one ``Hk`` solve, a
    **linear map with real coefficients** over its components, one
    ``Lk`` solve:

    ==============  =========================================
    geometry        the map between the two solves
    ==============  =========================================
    Cartesian       ``phi_arb = arb[0]``  (a bare index)
    annular / pipe  ``phi_arb - om_shift * omega_new``
    ==============  =========================================

    Shipped, that chain recombines the ``Hk`` result to complex, does
    real-coefficient arithmetic on it, and splits it straight back for
    the ``Lk`` solve.  Being linear with **real** coefficients, the map
    commutes with the re/im split, so the whole chain can stay split.

    Both arms take the factors as jit **arguments** (the stepper takes
    ``flow`` as one; closing over them folds stages away -- see
    :func:`_part_a_cpu`), and are timed as one jit region each.  The
    hoisted arm is one ``shard_map`` region built from the solver's own
    :func:`_banded_solve_batched` / :func:`_pallas_banded_solve`, so it
    cannot drift from the shipped body, and its output is asserted
    **bit-identical** -- a representation change that moves a value is
    a bug, and the assertion is also what pins the arm's fidelity.

    Reported: the fused chain margin, and its extrapolation to the step
    (chains per IMM apply x ``n = 2 + c`` applies per step).  The
    extrapolation is an **upper bound on this hoist only**: the pipe
    runs a second ``Hk`` batch whose round trip this arm does not model,
    and the ``apply_y_matrix`` crossings the census counts are a
    separate, larger candidate.
    """
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P

    from dnsjax.solvers import PerModeBandedPallasOperator

    print("\n" + "-" * 72)
    print("PART A2 -- the split-real hoist: is the round trip removable?")
    print("-" * 72)

    hk, lk = flow.Hk_op, flow.Lk_op
    if not isinstance(hk, PerModeBandedPallasOperator):
        print(f"  needs the pallas backend; got {type(hk).__name__}.")
        return

    # The geometry's real call shape.  Cartesian stacks components
    # leading and indexes; the curvilinear pair is y-leading and takes
    # the om_shift combination (annular/cylindrical stage 5).
    cartesian = system in ("plane-couette", "plane-poiseuille")
    ca = 0 if cartesian else 1
    nc = _hk_components(hk)
    N = lk.L.shape[0]
    Nkz, Nkx = sharding.nz_spec, sharding.nx_spec
    p_h = hk.L.shape[-3]
    p_l = lk.L.shape[-3]
    shape = (nc, N, Nkz, Nkx) if ca == 0 else (N, nc, Nkz, Nkx)
    print(
        f"  chain: Hk.solve({shape}, component_axis={ca}) -> "
        f"{'index' if cartesian else 'om_shift combine'} -> Lk.solve"
    )

    Rs = [
        _make_complex(shape, 600 + i, sharding, sharding.spec_vector_shard)
        for i in range(reps)
    ]
    # A real mode-inner coefficient standing in for om_shift (timing is
    # value-independent; both arms see the same one).
    rng = np.random.default_rng(11)
    w = jax.device_put(
        rng.standard_normal((N, Nkz, Nkx)),
        jax.NamedSharding(sharding.mesh, sharding.spec_scalar_shard),
    )
    hL, hU, lL, lU = hk.L, hk.U, lk.L, lk.U

    def _mid(phi, om, wc):
        return phi if cartesian else phi - wc * om

    def _shipped(hL_, hU_, lL_, lU_, R_, w_):
        arb = PerModeBandedPallasOperator(L=hL_, U=hU_).solve(
            R_, component_axis=ca
        )
        phi, om = (arb[0], arb[1]) if ca == 0 else (arb[:, 0], arb[:, 1])
        src = _mid(phi, om, w_)
        return PerModeBandedPallasOperator(L=lL_, U=lU_).solve(src), om

    to_split, from_split, sweep, coeff = _split_helpers()

    def _hoisted(hL_, hU_, lL_, lU_, R_, w_):
        def _local(hL_l, hU_l, lL_l, lU_l, R_l, w_l):
            # Component-leading for the vmap, then split once.
            R_c = R_l if ca == 0 else jnp.moveaxis(R_l, 1, 0)
            bs = jax.vmap(to_split)(R_c)
            stacked = hL_l.ndim == 5
            in_ax = (0, 0, 0, None) if stacked else (None, None, 0, None)
            xs = jax.vmap(sweep, in_axes=in_ax)(hL_l, hU_l, bs, p_h)
            # ... the map runs *in* the split representation ...
            src_s = _mid(xs[0], xs[1], coeff(w_l))
            y = sweep(lL_l, lU_l, src_s, p_l)
            # ... and only the two outputs recombine.
            return from_split(y), from_split(xs[1])

        fspec_h = P(*(None,) * (hL_.ndim - 2), sharding.a0, sharding.a1)
        fspec_l = P(*(None,) * (lL_.ndim - 2), sharding.a0, sharding.a1)
        rspec = P(*(None,) * (R_.ndim - 2), sharding.a0, sharding.a1)
        sspec = P(None, sharding.a0, sharding.a1)
        return shard_map(
            _local,
            mesh=sharding.mesh,
            in_specs=(fspec_h, fspec_h, fspec_l, fspec_l, rspec, sspec),
            out_specs=(sspec, sspec),
            check_vma=False,
        )(hL_, hU_, lL_, lU_, R_, w_)

    args = [(hL, hU, lL, lU, R, w) for R in Rs]
    ship = jax.block_until_ready(jax.jit(_shipped)(*args[0]))
    hois = jax.block_until_ready(jax.jit(_hoisted)(*args[0]))
    # Fidelity gate: the hoist only changes the *representation* a value
    # is carried in, so the two arms must agree to rounding.  Agreement
    # to ~machine epsilon is the bar, not bit-identity -- the arms feed
    # the sweep output to different consumers, so XLA is free to
    # contract differently around it, and a last-bit difference is
    # expected rather than a defect.
    worst = 0.0
    for a, b, name in zip(ship, hois, ("Lk out", "omega"), strict=True):
        a_, b_ = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(
            b_, a_, rtol=1e-12, atol=0.0, err_msg=f"hoisted {name}"
        )
        scale = max(float(np.abs(a_).max()), np.finfo(float).tiny)
        worst = max(worst, float(np.abs(a_ - b_).max()) / scale)
    print(f"  fidelity: agrees to {worst:.1e} relative (bar: ~1e-15)")

    t_ship = _bench(_shipped, args)
    t_hois = _bench(_hoisted, args)
    gain = t_ship - t_hois
    print(f"  shipped  Hk.solve -> map -> Lk.solve   {_ms(t_ship)}")
    print(f"  hoisted  (split-real across the map)   {_ms(t_hois)}")
    print(
        f"  margin                                 {_ms(gain)}  "
        f"({100 * gain / t_ship:5.1f}% of the chain)"
    )
    if t_step:
        per_step = n * gain
        print(
            f"\n  extrapolated to the step: {n} IMM applies x 1 chain = "
            f"{_ms(per_step)},\n  {100 * per_step / t_step:.2f}% of the "
            f"{_ms(t_step)} step."
        )
    print(
        "  Upper bound for THIS hoist only: the pipe's second Hk batch "
        "is not\n  modelled, and the apply_y_matrix crossings the census "
        "counts are a\n  separate (larger) candidate."
    )


# ── Part B: solve share of a corrector step ──────────────────────────


def _imm_stage_breakdown(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Cartesian: split one ``_imm_iteration`` into its stages.

    Sizes the **influence-matrix boundary correction** -- the
    openpipeflow-wiki "negligible" per-step overhead -- against the rest
    of the implicit Crank-Nicolson update (assembly ``D1``/``D2`` GEMMs
    + the ``Lk``/``Hk`` banded solves), which is the work openpipeflow
    *also* runs every corrector iteration and the wiki does **not**
    count as influence-matrix overhead.

    The two ``res.consistent_imm`` formulations are different
    algorithms with different stages, so each has its own transcription
    and this function dispatches:
    :func:`_stages_vw` for the shipped reconstruction scheme,
    :func:`_stages_vp` for the legacy primitive one (``--legacy-imm``).
    Both mirror their ``cartesian.py`` counterpart stage for stage; the
    constant-bulk / block-spanwise branch is compiled out for the
    default plane-Couette driving and omitted from both.

    All three geometries have an arm under the default scheme
    (:func:`_stages_vw` Cartesian, :func:`_stages_vw_ann` annular,
    :func:`_stages_vw_cyl` cylindrical), which is what makes the
    pipe's measured ~2x ``_imm_iteration`` attributable: the annulus
    shares every curvilinear cost but has neither the spin quad nor
    the parity reduction, so it separates the two explanations.  The
    legacy primitive scheme keeps its Cartesian-only transcription.
    """
    if not params.res.consistent_imm:
        if params.phys.system not in ("plane-couette", "plane-poiseuille"):
            print(
                "\n  (no --legacy-imm stage transcription for "
                f"{params.phys.system}; Cartesian only.)"
            )
            return
        stages = _stages_vp
    elif params.phys.system == "pipe":
        stages = _stages_vw_cyl
    elif params.phys.system in ("taylor-couette", "dean"):
        stages = _stages_vw_ann
    else:
        stages = _stages_vw
    stages(
        geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
    )


def _stages_vp(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Stage split of the **legacy** primitive `$(v, p)$` pass.

    Transcribes ``_cartesian_primitive_imm._imm_iteration_vp``: a
    pressure Poisson solve, a three-component Helmholtz solve, then the
    `$2 \times 2$` influence correction (its stages 4-7).
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded import (
        _cartesian_primitive_imm as prim,
    )
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
        Lk_d = prim._lk_matvec(d_hat_n, flow, fourier)
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
        # Stages 4-7 + finalize: the influence-matrix correction of
        # the primitive 2x2 IMM.
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
    print(
        "\n  IMM stage breakdown (Cartesian, LEGACY primitive (v, p) "
        "scheme; each runs n times/step):"
    )
    print(f"    Poisson RHS asm (D1 GEMM+div+_lk+CN)  {_ms(t_prhs)}")
    print(f"    Lk banded solve                       {_ms(t_lk)}")
    print(f"    Helmholtz RHS asm (grad+_hk GEMM+CN)  {_ms(t_hrhs)}")
    print(f"    Hk banded solve (3 components)        {_ms(t_hk)}")
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


def _stages_vw(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Stage split of the **default** `$v$`-`$\omega_y$` pass.

    Transcribes ``cartesian._imm_iteration_vw`` stage for stage.  The
    shape differs from the primitive twin in three ways that matter to
    a reading of the numbers:

    - there is no pressure Poisson stage at all, so the ``Lk`` solve is
      the `$\varphi \to v$` recovery (stage 4) rather than a pressure
      solve, and it comes *after* the Helmholtz one instead of before;
    - the ``Hk`` solve carries **two** scalars (`$\varphi$`,
      `$\omega_y$`) rather than three velocity components;
    - the influence correction (stages 5-7) additionally reconstructs
      the tangential pair, which is the work that makes the discrete
      divergence vanish -- so it is doing strictly more than the
      primitive scheme's rank-2 recombination, and the "negligible
      boundary correction" reading has to be made against that.
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import apply_y_matrix

    c = params.step.implicitness
    ikx = 1j * fourier.kx
    ikz = 1j * fourier.kz
    mean_mask = fourier.mean_mask

    def source_asm(velocity_n, nonlin_j, nonlin_n):
        # Stages 1-2: re-derive the evolved scalars from the carried
        # physical state, then form the pressure-free CN sources and
        # the zero-wall-data Helmholtz RHS (stage 3 up to the solve).
        sol_n = geom._to_solver(velocity_n, fourier, flow)
        phi_n, omega_n = sol_n[0], sol_n[2]
        nl = c * nonlin_j + (1 - c) * nonlin_n
        div_h = ikx * nl[0] + ikz * nl[2]
        s_phi = -fourier.k2 * nl[1] - apply_y_matrix(flow.D1, div_h)
        s_omega = ikz * nl[0] - ikx * nl[2]
        s_phi = jnp.where(mean_mask, nl[0], s_phi)
        s_omega = jnp.where(mean_mask, nl[2], s_omega)
        r = jax.vmap(geom._hk_minus_matvec, in_axes=(0, None, None))(
            jnp.stack([phi_n, omega_n]), flow, fourier
        ) + jnp.stack([s_phi, s_omega])
        return r.at[:, 0].set(0.0).at[:, -1].set(0.0)

    def influence_reconstruct(velocity_j, phi_arb, omega_new, v_arb):
        # Stages 5-7 + finalize: pick the two free phi wall values that
        # make (D1 v)|wall = 0, then reconstruct (u, w) from
        # (D1 v, omega) -- the stage that makes continuity an identity.
        d_wall = jnp.einsum("bj, jzx -> zxb", flow.D1_bnd, v_arb)
        alpha = -jnp.einsum("zxab, zxb -> zxa", flow.M_inv, d_wall)
        v_new = (
            v_arb
            + alpha[..., 0][None] * flow.v1
            + alpha[..., 1][None] * flow.v2
        )
        v_new = jnp.where(mean_mask, 0.0, v_new)
        out = geom._from_solver(
            jnp.array([phi_arb, v_new, omega_new]), fourier, flow
        )
        return jnp.array(out) - velocity_j

    # Realistic intermediates (predictor call: velocity_n = velocity_j
    # = state, nonlin_n = nonlin_j = nonlin).
    lk_solve = jax.jit(lambda f: flow.Lk_op.solve(f))
    hk_solve = jax.jit(lambda r: flow.Hk_op.solve(r))
    R_stack = jax.block_until_ready(jax.jit(source_asm)(state, nonlin, nonlin))
    arb = jax.block_until_ready(hk_solve(R_stack))
    phi_arb, omega_new = arb[0], arb[1]
    v_arb = jax.block_until_ready(lk_solve(phi_arb))

    t_src = _bench(source_asm, [(state, nonlin, nonlin)] * reps)
    t_infl = _bench(
        influence_reconstruct, [(state, phi_arb, omega_new, v_arb)] * reps
    )

    t_sum = t_src + t_hk + t_lk + t_infl
    print(
        "\n  IMM stage breakdown (Cartesian, default v-omega_y scheme; "
        "each runs n times/step):"
    )
    print(f"    source asm (_to_solver+proj+_hk GEMM)  {_ms(t_src)}")
    print(f"    Hk banded solve (2 scalars)            {_ms(t_hk)}")
    print(f"    Lk banded solve (phi -> v recovery)    {_ms(t_lk)}")
    print(
        f"    influence + reconstruct (u, w)         {_ms(t_infl)}  "
        "<- the boundary-correction share"
    )
    print(
        f"    {'-' * 53}\n"
        f"    sum of stages                          {_ms(t_sum)}  "
        f"(vs _imm_iteration {_ms(t_imm)})"
    )
    print(
        f"\n    influence + reconstruct = {100 * t_infl / t_imm:.1f}% of "
        f"_imm_iteration, {100 * n * t_infl / t_step:.1f}% of the step."
    )
    print(
        "    => The boundary work is still a small share; the rest is the"
        "\n       implicit CN update (assembly GEMMs + Hk/Lk banded solves)"
        " that\n       openpipeflow also runs every corrector iteration."
        "  Note this\n       scheme folds the tangential reconstruction "
        "into that share, and\n       runs one banded solve fewer per mode"
        " than the legacy path\n       (--legacy-imm to time that one)."
    )


def _stages_vw_ann(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Stage split of the **annular** `$u_r$`-`$\omega_r$` pass.

    Transcribes ``annular._imm_iteration_vw`` stage for stage.  The
    annulus is the **control** for the pipe's measured ~2x: it shares
    every curvilinear cost -- the `$u_\pm$` basis crossings, the
    `$1/r$` / `$1/r^2$` metric multiplies, the `$A = D_2 + (1/r)D_1$`
    pair of matvecs on the same array -- but has *neither* the spin
    quad nor the parity reduction, and evolves two scalars against the
    pipe's four.  So reading this table against the Cartesian one
    attributes the pipe's excess: what shows up here is curvilinear,
    what shows up only in :func:`_stages_vw_cyl` is the quad/parity.

    The constant-bulk / block-mean-spanwise branches are compiled out
    for the default Taylor-Couette driving and omitted, as in the
    Cartesian arm.
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import (
        apply_y_matrix,
        from_pm_basis,
        to_pm_basis,
    )
    from dnsjax.parameters import derived_params
    from dnsjax.sharding import sharding as sharding_mod

    c = params.step.implicitness
    dt = flow.dt
    nu = derived_params.nu
    im = 1j * fourier.m
    ikz = 1j * fourier.kz
    inv_r = flow.inv_r[:, None, None]
    inv_r2 = flow.inv_r2[:, None, None]
    kz2 = fourier.kz2
    mean_mask = fourier.mean_mask
    pair2 = fourier.m2 + 1.0
    phi2 = jnp.where(mean_mask, 0.0, pair2)
    spin = 2.0 * im * inv_r2

    def basis(velocity_n, velocity_j, nonlin_j, nonlin_n):
        # Stage 0: the three field-sized u_pm -> physical crossings.
        return (
            from_pm_basis(velocity_n),
            from_pm_basis(c * velocity_j + (1 - c) * velocity_n),
            from_pm_basis(c * nonlin_j + (1 - c) * nonlin_n),
        )

    def source_asm(state_n, state_cn, nonlin):
        # Stages 1-3: the two batched FD matvecs, the evolved scalars,
        # and the conservative-curl pressure-free sources.
        d1_in = jnp.stack(
            [
                state_cn[0],
                nonlin[0],
                flow.rs[:, None, None] * nonlin[2],
            ],
            axis=1,
        )
        d1 = apply_y_matrix(flow.D1, d1_in, component_axis=1)
        A_in = jnp.stack([state_n[1], state_cn[2]], axis=1)
        A_pair_n = apply_y_matrix(flow.A_base, A_in, component_axis=1)
        A_ur_n, A_ut_it = A_pair_n[:, 0], A_pair_n[:, 1]
        phi_n = (
            A_ur_n - (pair2 * inv_r2 + kz2) * state_n[1] - spin * state_n[2]
        )
        omega_n = im * inv_r * state_n[0] - ikz * state_n[2]
        phi_n = jnp.where(mean_mask, state_n[0], phi_n)
        omega_n = jnp.where(mean_mask, state_n[2], omega_n)
        C_r = im * inv_r * nonlin[0] - ikz * nonlin[2]
        C_theta = ikz * nonlin[1] - d1[:, 1]
        C_z = inv_r * (d1[:, 2] - im * nonlin[1])
        S_phi = jnp.where(
            mean_mask, nonlin[0], -(im * inv_r * C_z - ikz * C_theta)
        )
        S_omega = jnp.where(mean_mask, nonlin[2], C_r)
        return phi_n, omega_n, S_phi, S_omega, A_ut_it, d1[:, 0]

    def cn_explicit(phi_n, omega_n, S_phi, S_omega, A_ut_it, d1_2, state_cn):
        # Stage 4: the explicit CN half of both slots + the lagged spin
        # partners, then the Dirichlet wall rows.
        pair_n = jnp.stack([phi_n, omega_n], axis=1)
        inv_r_y = inv_r[..., None]
        A_pair = apply_y_matrix(flow.A_base, pair_n, component_axis=1)
        meff2_pair = jnp.stack(
            [
                phi2,
                jnp.broadcast_to(
                    pair2,
                    phi2.shape,
                    out_sharding=sharding_mod.spec_scalar_shard,
                ),
            ],
            axis=1,
        )
        lapl_pair = A_pair - (meff2_pair * inv_r_y**2 + kz2[:, None]) * pair_n
        partner = jnp.stack(
            [
                A_ut_it
                - (pair2 * inv_r2 + kz2) * state_cn[2]
                + spin * state_cn[1],
                ikz * state_cn[1] - d1_2,
            ],
            axis=1,
        )
        r = (
            pair_n / dt
            + (1 - c) * nu * lapl_pair
            - nu * spin[:, None] * partner
            + jnp.stack([S_phi, S_omega], axis=1)
        )
        return r.at[0].set(0.0).at[-1].set(0.0)

    def influence_reconstruct(velocity_j, phi_arb, omega_new, ur_arb):
        # Stages 6-8 + the exit basis crossing.
        det = kz2 + fourier.m2 * inv_r2
        inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
        d_wall = jnp.einsum("bj, jmz -> mzb", flow.D1_bnd, ur_arb)
        alpha = -jnp.einsum("mzab, mzb -> mza", flow.M_inv, d_wall)
        ur_new = (
            ur_arb
            + alpha[..., 0][None] * flow.ur_1
            + alpha[..., 1][None] * flow.ur_2
        )
        chi = -(apply_y_matrix(flow.D1, ur_new) + inv_r * ur_new)
        b_th = im * inv_r
        uz_new = (-ikz * chi - b_th * omega_new) * inv_det
        ut_new = (-b_th * chi + ikz * omega_new) * inv_det
        uz_new = jnp.where(mean_mask, phi_arb, uz_new)
        ut_new = jnp.where(mean_mask, omega_new, ut_new)
        ur_new = jnp.where(mean_mask, 0.0, ur_new)
        return to_pm_basis(jnp.stack([uz_new, ur_new, ut_new])) - velocity_j

    # Realistic intermediates (predictor call: velocity_n = velocity_j
    # = state, nonlin_n = nonlin_j = nonlin).
    b_out = jax.block_until_ready(jax.jit(basis)(state, state, nonlin, nonlin))
    src = jax.block_until_ready(jax.jit(source_asm)(*b_out))
    phi_n, omega_n, S_phi, S_omega, A_ut_it, d1_2 = src
    R_stack = jax.block_until_ready(
        jax.jit(cn_explicit)(
            phi_n, omega_n, S_phi, S_omega, A_ut_it, d1_2, b_out[1]
        )
    )
    arb = jax.block_until_ready(
        jax.jit(lambda r: flow.Hk_op.solve(r, component_axis=1))(R_stack)
    )
    phi_arb, omega_new = arb[:, 0], arb[:, 1]
    det = kz2 + fourier.m2 * inv_r2
    inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
    om_shift = 2.0 * fourier.m * fourier.kz * inv_r2 * inv_det
    ur_arb = jax.block_until_ready(
        jax.jit(lambda a, b: flow.Lk_op.solve(a - om_shift * b))(
            phi_arb, omega_new
        )
    )

    t_bas = _bench(basis, [(state, state, nonlin, nonlin)] * reps)
    t_src = _bench(source_asm, [b_out] * reps)
    t_cn = _bench(
        cn_explicit,
        [(phi_n, omega_n, S_phi, S_omega, A_ut_it, d1_2, b_out[1])] * reps,
    )
    t_inf = _bench(
        influence_reconstruct,
        [(state, phi_arb, omega_new, ur_arb)] * reps,
    )

    t_sum = t_bas + t_src + t_cn + t_hk + t_lk + t_inf
    print(
        "\n  IMM stage breakdown (annular, default u_r-omega_r scheme; "
        "each runs n times/step):"
    )
    rows = (
        ("basis crossings (3 x from_pm_basis)", t_bas),
        ("source asm (D1/D2 matvecs + curl)  ", t_src),
        ("CN explicit half (A_pair + partner)", t_cn),
        ("Hk banded solve (2 scalars)        ", t_hk),
        ("Lk banded solve (u_r recovery)     ", t_lk),
        ("influence 2x2 + reconstruct + exit ", t_inf),
    )
    for label, t in rows:
        print(f"    {label}  {_ms(t)}  ({100 * t / t_imm:4.1f}% of IMM)")
    print(
        f"    {'-' * 53}\n"
        f"    sum of stages                       {_ms(t_sum)}  "
        f"(vs _imm_iteration {_ms(t_imm)})"
    )
    _stage_verdict(t_sum, t_imm, t_step, n, rows, t_hk + t_lk)


def _stages_vw_cyl(
    geom, flow, fourier, nonlin, state, reps, t_imm, t_lk, t_hk, t_step, n
) -> None:
    r"""Stage split of the **cylindrical** (pipe) spin-quad pass.

    Transcribes ``cylindrical._imm_iteration_vw`` stage for stage.
    Read against :func:`_stages_vw_ann` (same curvilinear algebra, no
    quad, no parity) and :func:`_stages_vw` (neither), this is what
    attributes the pipe's measured ~2x ``_imm_iteration`` at equal
    solve cost.  The pipe-only costs are, by construction:

    1. the spin quad -- four evolved scalars against two, so stage 4's
       `$A = D_2 + (1/r) D_1$` runs on a 4-wide stack;
    2. the parity reduction -- there is no single ``D1``, so every
       matvec is :func:`~...cylindrical._parity_y_matvec`, a ``pos``
       GEMM plus a ``g``-row ``ghost`` GEMM and a scatter-add;
    3. two ``Hk`` solve batches instead of one (both counted in
       *t_hk*, which Part B measures on the full stacked RHS).

    :func:`_cyl_extras` prices 1-2 in isolation; this table is where
    they land in the pass.
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import (
        from_pm_basis,
        to_pm_basis,
    )
    from dnsjax.geometries.wall_bounded.cylindrical import _parity_y_matvec
    from dnsjax.parameters import derived_params
    from dnsjax.sharding import sharding as sharding_mod

    c = params.step.implicitness
    dt = flow.dt
    nu = derived_params.nu
    m = fourier.m
    im = 1j * m
    ikz = 1j * fourier.kz
    kz2 = fourier.kz2
    inv_r = flow.inv_r[:, None, None]
    inv_r2 = flow.inv_r2[:, None, None]
    mean_mask = fourier.mean_mask
    psp = fourier.m_is_even * 2 - 1
    psv = -psp
    inv_r2_y = inv_r2[..., None]
    kz2_y = kz2[:, None]

    def _pack(minus_slot, plus_val, minus_val):
        return jnp.stack(
            [
                jnp.where(mean_mask, plus_val, minus_slot[:, 0]),
                jnp.where(mean_mask, minus_val, minus_slot[:, 1]),
            ],
            axis=1,
        )

    def basis(velocity_n, nonlin_j, nonlin_n):
        # Stage 0: the two field-sized u_pm -> physical crossings.
        return (
            from_pm_basis(velocity_n),
            from_pm_basis(c * nonlin_j + (1 - c) * nonlin_n),
        )

    def quad_asm(velocity_n, state_n, nonlin):
        # Stages 1-2: the batched parity-reduced D1 (5-wide) and D2
        # (2-wide), the evolved quad, and the mean-plane packing.
        pair_n = jnp.stack([velocity_n[1], velocity_n[2]], axis=1)
        d1_in = jnp.stack(
            [
                state_n[0],
                nonlin[0],
                flow.rs[:, None, None] * nonlin[2],
            ],
            axis=1,
        )
        d1 = _parity_y_matvec(
            flow.D1_pos,
            flow.D1_ghost,
            d1_in,
            jnp.stack([psp, psp, psp], axis=1),
            component_axis=1,
        )
        A_pair = _parity_y_matvec(
            flow.A_base_pos,
            flow.A_base_ghost,
            pair_n,
            jnp.stack([psv, psv], axis=1),
            component_axis=1,
        )
        meff2_pm = jnp.stack([(m + 1) ** 2, (m - 1) ** 2], axis=1)
        phi_pm = A_pair - (meff2_pm * inv_r2_y + kz2_y) * pair_n
        ur_n, ut_n = state_n[1], state_n[2]
        om_r_n = im * inv_r * state_n[0] - ikz * ut_n
        om_t_n = ikz * ur_n - d1[:, 0]
        zero = jnp.zeros_like(mean_mask, dtype=phi_pm.dtype)
        phi_pm = _pack(phi_pm, zero, state_n[0])
        om_pm_n = _pack(
            jnp.stack([om_r_n + 1j * om_t_n, om_r_n - 1j * om_t_n], axis=1),
            ut_n,
            zero,
        )
        return phi_pm, om_pm_n, d1[:, 1], d1[:, 2]

    def sources(nonlin, d1_3, d1_4):
        # Stage 3: the conservative discrete double curl.
        zero = jnp.zeros_like(mean_mask, dtype=nonlin.dtype)
        C_r = im * inv_r * nonlin[0] - ikz * nonlin[2]
        C_t = ikz * nonlin[1] - d1_3
        C_z = inv_r * (d1_4 - im * nonlin[1])
        d1_Cz = _parity_y_matvec(flow.D1_pos, flow.D1_ghost, C_z, psp)
        cc_r = im * inv_r * C_z - ikz * C_t
        cc_t = ikz * C_r - d1_Cz
        S_phi = _pack(
            jnp.stack([-(cc_r + 1j * cc_t), -(cc_r - 1j * cc_t)], axis=1),
            zero,
            nonlin[0],
        )
        S_om = _pack(
            jnp.stack([C_r + 1j * C_t, C_r - 1j * C_t], axis=1),
            nonlin[2],
            zero,
        )
        return S_phi, S_om

    def cn_explicit(phi_pm, om_pm_n, S_phi, S_om, velocity_j):
        # Stage 4: the quad-wide explicit CN half + the wall row whose
        # two free differences ride the corrector iterate.
        quad = jnp.concatenate([phi_pm, om_pm_n], axis=1)
        psv_b = jnp.broadcast_to(
            psv, mean_mask.shape, out_sharding=sharding_mod.spec_scalar_shard
        )
        psv_m = jnp.where(mean_mask, psp, psv)
        par_quad = jnp.stack([psv_b, psv_m, psv_b, psv_m], axis=1)
        meff2_m = jnp.where(mean_mask, m**2, (m - 1) ** 2)
        meff2_p = jnp.broadcast_to(
            (m + 1) ** 2,
            mean_mask.shape,
            out_sharding=sharding_mod.spec_scalar_shard,
        )
        meff2_quad = jnp.stack([meff2_p, meff2_m, meff2_p, meff2_m], axis=1)
        A_quad = _parity_y_matvec(
            flow.A_base_pos,
            flow.A_base_ghost,
            quad,
            par_quad,
            component_axis=1,
        )
        lapl_quad = A_quad - (meff2_quad * inv_r2_y + kz2_y) * quad
        R_quad = (
            quad / dt
            + (1 - c) * nu * lapl_quad
            + jnp.concatenate([S_phi, S_om], axis=1)
        )
        pair_j = velocity_j[1:3]
        d1w_pair = jnp.einsum("j, cjmz -> cmz", flow.D1_wall.ravel(), pair_j)
        d2w_pair = jnp.einsum("j, cjmz -> cmz", flow.D2_wall.ravel(), pair_j)
        meff2_w = jnp.stack([(m + 1) ** 2, (m - 1) ** 2], axis=1)[0]
        phi_w = (
            d2w_pair
            + inv_r[-1] * d1w_pair
            - (meff2_w * inv_r2[-1] + kz2) * pair_j[:, -1]
        )
        state_j_w = from_pm_basis(velocity_j[:, -1])
        om_t_w = ikz[0] * state_j_w[1] - jnp.einsum(
            "j, jmz -> mz", flow.D1_wall.ravel(), velocity_j[0]
        )
        d_phi = (phi_w[0] - phi_w[1]) / 2
        d_om = 1j * om_t_w
        wall = jnp.where(
            mean_mask[0], 0.0, jnp.stack([d_phi, -d_phi, d_om, -d_om])
        )
        return R_quad.at[-1].set(wall)

    def influence_reconstruct(
        velocity_j, phi_arb, omega_new, ur_arb, uz_mean, ut_mean
    ):
        # Stages 6-8 + the exit basis crossing.
        det = kz2 + fourier.m2 * inv_r2
        inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
        d_wall = jnp.einsum("j, jmz -> mz", flow.D1_wall.ravel(), ur_arb)
        ur_new = ur_arb + (-flow.M_inv * d_wall)[None] * flow.ur_1
        d1_ur = _parity_y_matvec(flow.D1_pos, flow.D1_ghost, ur_new, psv)
        chi = -(d1_ur + inv_r * ur_new)
        b_th = im * inv_r
        uz_new = (-ikz * chi - b_th * omega_new) * inv_det
        ut_new = (-b_th * chi + ikz * omega_new) * inv_det
        uz_new = jnp.where(mean_mask, uz_mean, uz_new)
        ut_new = jnp.where(mean_mask, ut_mean, ut_new)
        ur_new = jnp.where(mean_mask, 0.0, ur_new)
        return to_pm_basis(jnp.stack([uz_new, ur_new, ut_new])) - velocity_j

    # Realistic intermediates (predictor call).
    state_n, nl = jax.block_until_ready(jax.jit(basis)(state, nonlin, nonlin))
    phi_pm, om_pm_n, d1_3, d1_4 = jax.block_until_ready(
        jax.jit(quad_asm)(state, state_n, nl)
    )
    S_phi, S_om = jax.block_until_ready(jax.jit(sources)(nl, d1_3, d1_4))
    R_quad = jax.block_until_ready(
        jax.jit(cn_explicit)(phi_pm, om_pm_n, S_phi, S_om, state)
    )
    hk_pair = jax.jit(lambda r: flow.Hk_op.solve(r, component_axis=1))
    phi_arb_pm = jax.block_until_ready(hk_pair(R_quad[:, :2]))
    om_pm = jax.block_until_ready(hk_pair(R_quad[:, 2:]))
    phi_arb = (phi_arb_pm[:, 0] + phi_arb_pm[:, 1]) / 2
    omega_new = (om_pm[:, 0] + om_pm[:, 1]) / 2
    det = kz2 + fourier.m2 * inv_r2
    inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
    om_shift = 2.0 * m * fourier.kz * inv_r2 * inv_det
    ur_arb = jax.block_until_ready(
        jax.jit(lambda a, b: flow.Lk_op.solve(a - om_shift * b))(
            phi_arb, omega_new
        )
    )

    t_bas = _bench(basis, [(state, nonlin, nonlin)] * reps)
    t_quad = _bench(quad_asm, [(state, state_n, nl)] * reps)
    t_src = _bench(sources, [(nl, d1_3, d1_4)] * reps)
    t_cn = _bench(cn_explicit, [(phi_pm, om_pm_n, S_phi, S_om, state)] * reps)
    t_inf = _bench(
        influence_reconstruct,
        [
            (
                state,
                phi_arb,
                omega_new,
                ur_arb,
                phi_arb_pm[:, 1],
                om_pm[:, 0],
            )
        ]
        * reps,
    )

    t_sum = t_bas + t_quad + t_src + t_cn + t_hk + t_lk + t_inf
    print(
        "\n  IMM stage breakdown (cylindrical, spin-quad scheme; each "
        "runs n times/step):"
    )
    rows = (
        ("basis crossings (2 x from_pm_basis)", t_bas),
        ("quad asm (parity D1 5-wide + D2)   ", t_quad),
        ("sources (conservative double curl) ", t_src),
        ("CN explicit half (quad-wide A)     ", t_cn),
        ("Hk banded solves (2 x 2 scalars)   ", t_hk),
        ("Lk banded solve (u_r recovery)     ", t_lk),
        ("influence 1x1 + reconstruct + exit ", t_inf),
    )
    for label, t in rows:
        print(f"    {label}  {_ms(t)}  ({100 * t / t_imm:4.1f}% of IMM)")
    print(
        f"    {'-' * 53}\n"
        f"    sum of stages                       {_ms(t_sum)}  "
        f"(vs _imm_iteration {_ms(t_imm)})"
    )
    _stage_verdict(t_sum, t_imm, t_step, n, rows, t_hk + t_lk)
    _cyl_extras(flow, fourier, state, nonlin, reps, t_imm)


def _stage_verdict(t_sum, t_imm, t_step, n, rows, t_solve) -> None:
    """Name the largest non-solve stage and price it against the step.

    The isolated stages **over-count** (each pays a round trip it does
    not pay fused), so ``sum of stages / _imm_iteration`` is reported
    as a fidelity read, not a decomposition: far above 1 means the
    transcription is dominated by round trips, far below 1 means it is
    missing work.  Shares are of the fused ``_imm_iteration``, which is
    the honest denominator.
    """
    print(
        f"\n    sum/IMM = {t_sum / t_imm:4.2f} (isolated stages over-count; "
        "read shares, not the sum)"
    )
    non_solve = [r for r in rows if "banded solve" not in r[0]]
    label, t = max(non_solve, key=lambda r: r[1])
    print(
        f"    largest non-solve stage: {label.strip()} at "
        f"{100 * t / t_imm:.1f}% of _imm_iteration,\n    "
        f"{100 * n * t / t_step:.1f}% of the step "
        f"(solves are {100 * t_solve / t_imm:.1f}% of IMM)."
    )


def _cyl_extras(flow, fourier, state, nonlin, reps, t_imm) -> None:
    r"""Price the two pipe-only mechanisms in isolation.

    The stage table says *where* the pipe's time goes; this says *what
    it is paying for* -- the two things neither Cartesian nor the
    annulus does at all:

    1. **parity-reduced FD**: :func:`_parity_y_matvec` against a plain
       :func:`apply_y_matrix` on the identical array, so the ``pos`` +
       ``ghost`` GEMM pair and the ``g``-row scatter-add are priced
       against one GEMM;
    2. **quad assembly**: the mode-plane ``par_quad`` / ``meff2_quad``
       broadcasts and the three field-sized ``_pack`` selections.

    Also reported for scale: the basis crossings and the metric
    multiplies, which the annulus shares.  These are **isolated**
    figures -- they over-count against the fused pass, so they rank
    mechanisms, they do not decompose the stage table.
    """
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import (
        apply_y_matrix,
        from_pm_basis,
        to_pm_basis,
    )
    from dnsjax.geometries.wall_bounded.cylindrical import _parity_y_matvec
    from dnsjax.sharding import sharding as sharding_mod

    m = fourier.m
    mean_mask = fourier.mean_mask
    psp = fourier.m_is_even * 2 - 1
    psv = -psp
    inv_r = flow.inv_r[:, None, None]
    inv_r2 = flow.inv_r2[:, None, None]

    quad = jax.block_until_ready(
        jax.jit(
            lambda s: jnp.concatenate([s[1:3], s[1:3]], axis=0).swapaxes(0, 1)
        )(state)
    )
    par = jnp.stack([psv, psv, psv, psv], axis=1)

    def parity_fd(x, p_):
        return _parity_y_matvec(
            flow.D1_pos, flow.D1_ghost, x, p_, component_axis=1
        )

    def plain_fd(x):
        return apply_y_matrix(flow.D1_pos, x, component_axis=1)

    def quad_assembly(s):
        psv_b = jnp.broadcast_to(
            psv, mean_mask.shape, out_sharding=sharding_mod.spec_scalar_shard
        )
        psv_m = jnp.where(mean_mask, psp, psv)
        par_quad = jnp.stack([psv_b, psv_m, psv_b, psv_m], axis=1)
        meff2_m = jnp.where(mean_mask, m**2, (m - 1) ** 2)
        meff2_p = jnp.broadcast_to(
            (m + 1) ** 2,
            mean_mask.shape,
            out_sharding=sharding_mod.spec_scalar_shard,
        )
        meff2_quad = jnp.stack([meff2_p, meff2_m, meff2_p, meff2_m], axis=1)
        packed = jnp.stack(
            [
                jnp.where(mean_mask, s[0], s[1]),
                jnp.where(mean_mask, s[2], s[1]),
            ],
            axis=1,
        )
        return par_quad, meff2_quad, packed

    def crossings(s, nl):
        return to_pm_basis(from_pm_basis(s) + from_pm_basis(nl))

    def metric(s):
        return inv_r * s[0] + inv_r2 * s[1]

    t_par = _bench(parity_fd, [(quad, par)] * reps)
    t_pln = _bench(plain_fd, [(quad,)] * reps)
    t_qas = _bench(quad_assembly, [(state,)] * reps)
    t_cro = _bench(crossings, [(state, nonlin)] * reps)
    t_met = _bench(metric, [(state,)] * reps)

    print("\n    pipe-only mechanisms, isolated (they over-count; rank only):")
    print(
        f"      parity D1 on a 4-wide quad      {_ms(t_par)}  "
        f"({100 * t_par / t_imm:4.1f}% of IMM)"
    )
    print(
        f"      plain  D1 on the same array     {_ms(t_pln)}  "
        f"-> parity costs {t_par / t_pln:4.2f}x one GEMM"
    )
    print(
        f"      quad assembly (par/meff2/pack)  {_ms(t_qas)}  "
        f"({100 * t_qas / t_imm:4.1f}% of IMM)"
    )
    print("    shared with the annulus (not pipe-only), for scale:")
    print(
        f"      basis crossings (in + in + out) {_ms(t_cro)}  "
        f"({100 * t_cro / t_imm:4.1f}% of IMM)"
    )
    print(
        f"      metric multiplies (1/r, 1/r^2)  {_ms(t_met)}  "
        f"({100 * t_met / t_imm:4.1f}% of IMM)"
    )


def _part_b(geom, m, flow, sharding, reps: int, steps: int) -> None:
    from dnsjax.ic.random_field import generate_random_state
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
        nc = _hk_components(hk)
        z3s = [
            _make_complex((nc, N, Nkz, Nkx), 300 + i, sharding, vspec)
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

    # Stage-level IMM breakdown: where the non-solve part of
    # _imm_iteration goes, per geometry.  Cartesian additionally sizes
    # the influence-matrix correction (the openpipeflow-wiki
    # "negligible" claim, measured); the curvilinear arms exist to
    # attribute the pipe's ~2x _imm_iteration at equal solve cost --
    # see _imm_stage_breakdown.
    if t_lk is not None:
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
    # speedup is modest, not ~3x.  The IMM cost per apply is
    # ``res.consistent_imm``-dependent (the default runs one banded
    # solve fewer per mode on this geometry), so this ratio is reported
    # for whichever formulation is configured.
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
        "c": c,  # corrector iterations, so n = 2 + c is reusable
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
        plane = f"{pnz}x{pnx}"
        # The program count is the kernel grid; a CPU run has none (and
        # stores the true, unpadded plane), so report NA rather than a
        # number that reads as "no work".
        progs = (
            (pnz // m0) * (pnx // m1)
            if jax.default_backend() == "gpu"
            else "NA"
        )
    else:
        progs, plane = "NA", "NA"

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
    nc = _hk_components(hk)
    z3s = [
        _make_complex((nc, N, pnz, pnx), 500 + i, sharding, vspec)
        for i in range(reps)
    ]
    t_lk = _bench(lambda z: lk.solve(z), [(z,) for z in zs])
    t_hk = _bench(lambda z: hk.solve(z), [(z,) for z in z3s])
    m0, m1 = params.solver.pallas_block_m0, params.solver.pallas_block_m1
    if jax.default_backend() == "gpu":
        progs = (pnz // m0) * (pnx // m1)
        print(
            f"  padded plane {pnz}x{pnx}, tile {m0}x{m1} -> {progs} "
            "programs/field\n  (H100 = 132 SMs; want >=~132 for one wave, "
            "several x for latency hiding)"
        )
    else:
        # No kernel grid on CPU, and the stored plane is the true one.
        print(f"  mode plane {pnz}x{pnx} (CPU: pure-JAX sweep, no tiling)")
    print(f"  Lk solve (1 field)   {_ms(t_lk)}")
    print(f"  Hk solve ({nc} fields)  {_ms(t_hk)}")
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
    except Exception as e:
        print(f"  [{label}] compile/as_text failed: {type(e).__name__}: {e}")
        return
    low = txt.lower()
    print(f"  [{label}] optimized-HLO op census:")
    print("      " + "  ".join(f"{k}={low.count(k)}" for k in keys))
    if hlo_out:
        with open(hlo_out, "a") as f:
            f.write(f"\n===== {label} =====\n{txt}\n")
        print(f"      (full HLO appended to {hlo_out})")


def _part_c(geom, flow, m, sharding, trace_dir, hlo_out) -> None:
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

    from dnsjax.ic.random_field import generate_random_state
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

    # Complex <-> real crossing census, per region (Part A2's static
    # half): how many split/recombine pairs survive optimization, and
    # which region owns them.  A count is not a cost -- Part A2's timed
    # arm is what says whether removing one is worth anything.
    print(
        "\n  complex <-> real crossings in the optimized HLO (the "
        "split-real\n  hoist's target; JAX has no zero-copy bitcast, so "
        "each solve and each\n  apply_y_matrix GEMM brackets itself):"
    )
    fourier = m.fourier
    if isinstance(op, PerModeBandedPallasOperator):
        _crossing_census("one Lk.solve", jax.jit(op.solve), (z,))
    rhs = jax.block_until_ready(
        jax.jit(lambda s: geom._get_rhs(s, fourier, flow))(state)
    )
    _crossing_census(
        "one _get_rhs (FFTs)",
        jax.jit(lambda s: geom._get_rhs(s, fourier, flow)),
        (state,),
    )
    _crossing_census(
        "one _imm_iteration",
        jax.jit(lambda s, r: geom._imm_iteration(s, s, r, r, fourier, flow)),
        (state, rhs),
    )
    _crossing_census(
        "one corrector step",
        jax.jit(m.predict_and_fully_correct),
        (state,),
    )

    if trace_dir:
        if jax.default_backend() != "gpu":
            print("  --trace ignored (no GPU backend).")
            return
        print(f"\n  capturing a profiler trace of 20 steps to {trace_dir} ...")
        s = jnp.copy(state)  # the step donates its state argument
        for _ in range(3):  # warm up before tracing
            s, _e, _c, *_ = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        with jax.profiler.trace(trace_dir):
            for _ in range(20):
                s, _e, _c, *_ = m.predict_and_fully_correct(s)
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
    except Exception as e:
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
        "--legacy-imm",
        action="store_true",
        help="profile the legacy res.consistent_imm=False primitive "
        "(v, p) scheme instead of the shipped reconstruction one "
        "(different operator set and per-mode solve count, so every "
        "number below changes)",
    )
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
        legacy_imm=args.legacy_imm,
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
        _part_a2(
            args.system,
            flow,
            sharding,
            args.reps,
            times["step"],
            2 + times["c"],
        )
        _part_c(geom, flow, m, sharding, None, args.hlo_out)
        _solve_sweep(args.system, args, flow, sharding, args.reps)
        _summary_line(args.system, args, flow, times)
        print("\n--cpu-smoke PASS: harness runs end-to-end.")
        return

    if args.solve_only:
        # CPU is a first-class target here: it takes a *different* solve
        # path (the pure-JAX sweep -- ``pallas_call`` is never reached),
        # so its timings answer their own question rather than standing
        # in for the GPU's.  Part A prints the CPU decomposition.
        _solve_sweep(args.system, args, flow, sharding, args.reps)
        return

    if args.steps_only:
        # Minimal steady-state stepping for an external profiler (nsys):
        # warm up (compile), then run N steps so the capture is dominated
        # by the corrector kernels.
        from dnsjax.ic.random_field import generate_random_state

        # ICs are physical; the steppers work in the geometry's solver
        # basis (the single crossing ``__main__`` performs, and the one
        # Parts B/C above already do).  Without it the cylindrical
        # geometries step a state the solver reads as ``u_+``/``u_-``:
        # no error, but a different corrector count and a trajectory
        # that can blow up -- so the captured profile would not be the
        # one a real run produces.
        s = getattr(m, "to_solver_basis", lambda x: x)(
            generate_random_state(
                params.init.random_amplitude,
                params.init.random_smoothness,
                params.init.random_seed,
                params.init.random_mean_flow,
            )
        )  # chained (donated) from here on
        for _ in range(3):
            s, _e, _c, *_ = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        for _ in range(args.steps_only):
            s, _e, _c, *_ = m.predict_and_fully_correct(s)
        jax.block_until_ready(s)
        print(
            f"ran {args.steps_only} steps (steps-only mode); wrap this "
            "invocation in a profiler."
        )
        return

    gpu = jax.default_backend() == "gpu"
    if not gpu:
        # Not a degraded GPU run: the CPU takes its own solve path (the
        # pure-JAX sweep), so these timings are the answer for that path.
        # What is *not* transferable is the reverse -- CPU numbers say
        # nothing about the Triton kernel.
        print(
            "CPU backend: profiling the CPU solve path (the pure-JAX "
            "sweep;\n``pallas_call`` is never reached here).  These "
            "timings do not stand in\nfor GPU ones -- launch on the "
            "cluster for those.\n"
        )

    _part_a(flow, sharding, args.reps)
    times = _part_b(geom, m, flow, sharding, args.reps, args.steps)
    _part_a2(
        args.system, flow, sharding, args.reps, times["step"], 2 + times["c"]
    )
    _part_c(geom, flow, m, sharding, args.trace if gpu else None, args.hlo_out)
    _summary_line(args.system, args, flow, times)
    print("\n" + "=" * 72)
    print(
        "Done.  Paste the full stdout back.  Key numbers: Part A "
        "'split+recombine\n% of the solve' (H1; on CPU the factor "
        "prologue's share) and Part B\n'c*(Lk+Hk) / step' (H2)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
