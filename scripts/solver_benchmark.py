r"""Pallas-backend validation & benchmark vs the dense reference.

Measures the production (``pallas``) solver backend against the
``dense`` reference on real hardware, and validates multi-GPU
execution (stability-check companion:
``scripts/pivot_stability_survey.py``):

A.  **Single-device matrix** (one subprocess per config x backend --
    the singletons capture the backend at import): per operator group
    (``Lk``/``Hk``/``Hc``) the operator class built, exact persistent
    factor bytes, device memory after setup and peak after stepping,
    isolated ``.solve`` times, and per-step times for **both**
    schemes (``predict_and_fully_correct`` and ``step_cnab2``), plus
    a fixed-seed parity scalar (perturbation energy + ``get_stats``
    after a few steps) compared across backends.  Every child also
    runs the split-vs-unsplit iterative-cn corrector A/B
    (``step.split_corrector`` forced on vs off, via an in-child
    stepper rebuild on the same operators and IC, so the comparison is
    independent of the default -- which is off, an opt-in; see
    ``_split_core`` in ``timestep.py``), reported per row and as a
    verdict section; a dedicated coupling-stressed
    ``dean-split-stress`` entry
    (``nz = 64``, ``dt = 0.15``, the ``TimeStepping``-docstring
    reference regime) shows the split's FFT-refresh savings, while the
    default-``dt`` entries pin the
    no-regression claim (corrector converges in ~1 iteration either
    way).
B.  **Multi-GPU section** (``mpirun ... -m dnsjax`` production runs
    from scratch dirs): multi-GPU execution of the Pallas Triton
    kernel -- correctness via JAX-free ``dnsjax.analysis`` snapshot
    diffs across device counts and vs the dense oracle (including a
    padding-inducing ``nx = 34`` plane, a ``2 x 2`` mesh, and a
    ``1 x 4`` case), and production-size timing runs parsed from the
    ``__main__`` benchmark summary.
    A preflight ladder bisects the launch stack first (task launch,
    distributed init, collectives, Explicit-mesh reshard, then real
    micro-runs over launch topology / scheme / NCCL env; see
    :func:`_preflight_launcher`) and locks the first working
    configuration -- including a single-process multi-GPU fallback
    (``--launch-mode sp``) when multi-process launches hang **or**
    when the allocation cannot host multi-task steps at all (a
    1-task-per-node allocation runs the multi-GPU sections
    single-process instead of skipping them).
C.  **CPU bench** (``--cpu-bench``): the same child measurements with
    ``JAX_PLATFORMS=cpu`` on a reduced matrix -- what CPU production
    pays per backend (the pallas pure-JAX sweep vs dense).

The driver is JAX-free (children own the devices); every child /
mpirun stdout is logged under ``{workdir}/logs``, results stream as
``@@RESULT`` JSON lines (also ``{workdir}/results.jsonl``), and the
run ends in summary tables plus a ``VERDICT`` section (parity,
multi-GPU health, stability notices).

Run **on the GPU cluster** from inside a single-node allocation with
>= 1 GPU (4 for the full mesh cases) and generous **host** memory --
the production-size children compile large XLA programs, and the
observed SLURM cgroup OOM kills (exit -9) were host memory, not HBM;
request e.g. ``--mem 64G`` or more.  Then **paste the full stdout
back**::

    salloc -N1 --gpus=4 --mem=64G ...        # or an sbatch wrapper
    .venv/bin/python scripts/solver_benchmark.py --max-gpus 4
    .venv/bin/python scripts/solver_benchmark.py --skip-mpi-timing

The driver is a **single process**: do not fan it out
(``srun -n 4 python ...`` runs four interleaved drivers fighting over
the same GPU; surplus SLURM tasks now exit at startup).  The
multi-process ``-m dnsjax`` runs are launched via ``--launcher``
(default ``auto``: ``mpirun`` when on PATH, else ``srun -n N
--overlap`` job steps inside the surrounding allocation;
site-specific step flags go through ``--srun-args``, e.g.
``--srun-args "--gpus-per-task=1"``).

On a CPU node (or the dev laptop)::

    .venv/bin/python scripts/solver_benchmark.py --cpu-bench
    .venv/bin/python scripts/solver_benchmark.py --cpu-smoke  # harness

Without a GPU and without a ``--cpu-*`` flag it prints the environment
banner and the planned matrix (with dense-size estimates) and exits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
SELF = Path(__file__).resolve()
RESULT_TAG = "@@RESULT "

# Per-system physics (the test-suite-standard values from
# tests/test_random_smoke.py) as dotted parameter paths, usable both
# as child ``params`` assignments and as ``--{path}`` CLI flags.
SYS_ARGS: dict[str, dict[str, float]] = {
    "plane-couette": {"phys.re": 330.0, "geo.lx": 5.0, "geo.lz": 5.0},
    "plane-poiseuille": {"phys.re": 660.0, "geo.lx": 5.0, "geo.lz": 5.0},
    "pipe": {"phys.re": 1800.0, "geo.lx": 5.0},
    "taylor-couette": {
        "phys.re1": 400.0,
        "phys.re2": -400.0,
        "geo.eta": 0.5,
        "geo.lx": 5.0,
    },
    "dean": {"phys.re": 1000.0, "geo.eta": 0.5, "geo.lx": 5.0},
    "viscoelastic-dean": {
        "phys.wi": 20.0,
        "phys.el": 20.0,
        "init.random_conformation_amplitude": 10.0,
        "geo.lx": 5.0,
    },
    "viscoelastic-pipe": {
        "phys.wi": 20.0,
        "phys.el": 0.02,
        "init.random_conformation_amplitude": 10.0,
        "geo.lx": 5.0,
    },
}

# (nx, ny, nz) per system and size class.
SIZES: dict[str, dict[str, tuple[int, int, int]]] = {
    "plane-couette": {"small": (64, 64, 64), "prod": (256, 192, 256)},
    "pipe": {"small": (64, 48, 64), "prod": (256, 192, 256)},
    "taylor-couette": {"small": (64, 48, 64), "prod": (256, 192, 256)},
    "viscoelastic-dean": {"small": (32, 48, 32), "prod": (128, 96, 128)},
    "viscoelastic-pipe": {"small": (32, 48, 32), "prod": (128, 96, 128)},
}

# Dense-backend operator count: Lk + Hk components (+ 6 Hc).
N_OPS = {
    "plane-couette": 2,
    "plane-poiseuille": 2,
    "pipe": 4,
    "taylor-couette": 4,
    "dean": 4,
    "viscoelastic-dean": 10,
    "viscoelastic-pipe": 10,
}

BACKENDS = ("pallas", "dense")

TILE_SWEEP = ((1, 32), (2, 16), (2, 64), (4, 32))  # + the (2, 32) default

# ── stdout parsers (formats from __main__ / test_random_smoke.py) ────

_NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
T_PATTERN = re.compile(rf"\bt\s*=\s*({_NUM})")
ERR_PATTERN = re.compile(rf"\berr\s*=\s*({_NUM})")
VALUE_PATTERN = re.compile(rf"=\s*({_NUM})")
RAN_PATTERN = re.compile(r"Ran for ([\d.]+)s with (\d+) devices?")
# The "NP x s/t" scaled variants have no numeric token directly before
# "s/t", so this matches only the plain per-device figures.
SPT_PATTERN = re.compile(rf"({_NUM}) s/t[,.]")
SPRHS_PATTERN = re.compile(rf"({_NUM}) s/rhs[,.]")
CIT_PATTERN = re.compile(rf"c/it\s*=\s*({_NUM})")
PALLAS_PATTERN = re.compile(r"^\[pallas\] .*$", re.M)


def _to_text(x: str | bytes | None) -> str:
    """Normalize captured subprocess output to text.

    ``subprocess.TimeoutExpired.stdout``/``stderr`` carry raw **bytes**
    even under ``text=True`` (on POSIX the exception is populated from
    the raw capture buffers) -- an ``isinstance(x, str)`` guard
    silently discards a hung run's entire captured output.
    """
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x


def _echo_tail(name: str, stdout: str, stderr: str) -> None:
    """Inline the end of a failed run's output into the driver stdout
    (the workdir often sits on a node-local /tmp that vanishes with
    the job, so the paste-back must carry its own diagnosis)."""
    # Separate windows per stream: a chatty stderr (jax/XLA INFO) must
    # not crowd the stdout breadcrumbs out of the tail.
    print(f"          --- {name}: stdout tail ---")
    for ln in stdout[-3000:].splitlines()[-15:]:
        print(f"          | {ln}")
    print(f"          --- {name}: stderr tail ---")
    for ln in stderr[-3000:].splitlines()[-15:]:
        print(f"          | {ln}")


# ── child mode ───────────────────────────────────────────────────────


def _import_flow(system: str):
    """Import the flow module for *system* (builds the operators)."""
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
    elif system == "viscoelastic-dean":
        from dnsjax.flows.wall_bounded import viscoelastic_dean as m
    else:
        raise SystemExit(f"unsupported system: {system}")
    return m


def _rebuild_pfc(system: str, flow):
    """Rebuild ``predict_and_fully_correct`` from the geometry stepper
    builder, picking up the *current* ``params.step.split_corrector``
    (read at stepper-construction time) -- same operators, same flow,
    no re-import.  Used for the split-vs-unsplit corrector A/B."""
    if system in ("plane-couette", "plane-poiseuille"):
        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_stepper as build,
        )
    elif system == "pipe":
        from dnsjax.geometries.wall_bounded.cylindrical import (
            build_cylindrical_stepper as build,
        )
    elif system == "viscoelastic-pipe":
        from dnsjax.geometries.wall_bounded.cylindrical_viscoelastic import (
            build_viscoelastic_stepper as build,
        )
    elif system in ("taylor-couette", "dean"):
        from dnsjax.geometries.wall_bounded.annular import (
            build_annular_stepper as build,
        )
    elif system == "viscoelastic-dean":
        from dnsjax.geometries.wall_bounded.annular_viscoelastic import (
            build_viscoelastic_stepper as build,
        )
    else:
        raise SystemExit(f"unsupported system: {system}")
    return build(flow)[3]


def _mem_stats(jax) -> dict:
    """Guarded device memory stats (absent/None on CPU)."""
    d = jax.local_devices()[0]
    fn = getattr(d, "memory_stats", None)
    ms = fn() if fn is not None else None
    if not ms:
        return {"available": False}
    return {
        "available": True,
        "bytes_in_use": int(ms.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(ms.get("peak_bytes_in_use", 0)),
    }


def _bench(jax, fn, args_list, warmup: int = 3) -> float:
    """Per-call wall time: queue all calls, block on the last.

    *args_list* must hold **distinct** operands so XLA cannot CSE the
    repeated calls (the ``pallas_solve_profile.py`` idiom).
    """
    f = jax.jit(fn)
    for a in args_list[: max(1, warmup)]:
        jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    outs = [f(*a) for a in args_list]
    jax.block_until_ready(outs[-1])
    return (time.perf_counter() - t0) / len(args_list)


def _bench_step(jax, jnp, step, state, n: int, warmup: int = 3):
    """Time the donating corrector step by chaining a copied state."""
    s = jnp.copy(state)
    c = 0
    for _ in range(warmup):
        s, _err, c = step(s)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for _ in range(n):
        s, _err, c = step(s)
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n, int(c)


def _bench_step_cnab2(jax, jnp, step_cnab2, state, n: int, warmup: int = 3):
    """Time the CN/AB2 step by chaining ``(state, rhs_prev)`` from
    copies (both arguments are donated); a discarded priming call
    seeds the AB2 history, as the ``__main__`` driver does."""
    s = jnp.copy(state)
    _, rp, _, _ = step_cnab2(jnp.copy(s), jnp.zeros_like(s))
    for _ in range(warmup):
        s, rp, _err, _c = step_cnab2(s, rp)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for _ in range(n):
        s, rp, _err, _c = step_cnab2(s, rp)
    jax.block_until_ready(s)
    return (time.perf_counter() - t0) / n


def _make_complex(jax, shape, seed, sharding, spec):
    """A distinct complex device array on *spec* (mode-inner field)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jax.device_put(a, jax.NamedSharding(sharding.mesh, spec))


def run_child(a: argparse.Namespace) -> None:
    """One (system, backend, resolution) measurement, single device."""
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    # Select the backend explicitly (the driver passes --platform;
    # CUDA_VISIBLE_DEVICES / JAX_PLATFORMS in the child env still pin the
    # concrete device) so the child's sharding banner and
    # params.dist.platform match the hardware it actually runs on.
    configure_jax_platform(a.platform)

    import jax
    import jaxlib

    params.phys.system = a.system
    for dotted, v in SYS_ARGS[a.system].items():
        section, key = dotted.split(".")
        setattr(getattr(params, section), key, v)
    params.res.nx = a.nx
    params.res.ny = a.ny
    params.res.nz = a.nz
    params.res.fd_order = a.fd_order
    params.res.double_precision = True
    params.step.dt = a.dt
    params.step.scheme = "iterative-cn"  # both steppers are built
    if a.bm0 is not None:
        params.solver.pallas_block_m0 = a.bm0
    if a.bm1 is not None:
        params.solver.pallas_block_m1 = a.bm1
    # The backend must go through the layering call, not a direct
    # ``params.solver.backend = ...`` assignment: ``update_parameters``
    # re-resolves the per-family backend default for any field not
    # recorded in ``_user_set_fields``, so a direct assignment is
    # overwritten and the dense children would silently run pallas.
    update_parameters(Parameters(solver={"backend": a.backend}))
    padded_res.set_padded_resolution(params)
    validate_parameters()

    t0 = time.perf_counter()
    m = _import_flow(a.system)
    setup_s = time.perf_counter() - t0
    flow = m.flow

    import jax.numpy as jnp

    from dnsjax.flows.registry import viscoelastic_systems
    from dnsjax.random_field import generate_random_state
    from dnsjax.sharding import sharding

    dev = jax.local_devices()[0]
    env = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", "?"),
    }

    # Operator classes + exact persistent factor bytes.
    groups = {"Lk": flow.Lk_op, "Hk": flow.Hk_op}
    if a.system in viscoelastic_systems:
        groups["Hc"] = flow.Hc_op
    operators: dict[str, str | None] = {}
    factor_bytes: dict[str, int] = {}
    for g, op in groups.items():
        if op is None:  # viscoelastic kappa == 0: no Hc group
            operators[g] = None
            factor_bytes[g] = 0
            continue
        operators[g] = type(op).__name__
        factor_bytes[g] = int(
            sum(x.nbytes for x in jax.tree_util.tree_leaves(op))
        )
    factor_bytes["total"] = sum(
        v for k, v in factor_bytes.items() if k != "total"
    )
    mem_setup = _mem_stats(jax)

    # Parity phase first, so every backend sees the identical sequence
    # from the identical fixed-seed IC.
    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        a.seed,
        False,
    )
    # The IC and the diagnostics are physical; the steppers work in
    # the geometry's solver basis (as ``__main__`` does, one crossing
    # each way).
    to_solver = getattr(m, "to_solver_basis", lambda x: x)
    from_solver = getattr(m, "from_solver_basis", lambda x: x)
    # ``jnp.copy`` unconditionally: the step donates its argument, and
    # ``state`` is benchmarked again further down.  Relying on
    # ``to_solver`` to hand back a fresh array is what broke here --
    # the Cartesian flows stopped exporting the basis maps when the
    # carried (phi, v, omega_y) basis was dropped, so the fallback
    # became the identity and the parity loop deleted ``state`` itself
    # ("Array has been deleted" at the first ``_bench_step``).
    s = jnp.copy(to_solver(state))
    for _ in range(a.parity_steps):
        s, _err, _c = m.predict_and_fully_correct(s)
    jax.block_until_ready(s)
    s_phys = from_solver(s)
    parity = {
        "steps": a.parity_steps,
        "energy": float(m.get_perturbation_energy(s_phys)),
        "stats": {k: float(v) for k, v in m.get_stats(s_phys).items()},
    }
    del s_phys
    del s

    # Isolated solve timings (distinct operands, CSE-safe).
    ncomp, N, Nkz, Nkx = state.shape
    sspec = sharding.spec_scalar_shard
    vspec = sharding.spec_vector_shard
    zs = [
        _make_complex(jax, (N, Nkz, Nkx), 100 + i, sharding, sspec)
        for i in range(a.reps)
    ]
    z3 = [
        _make_complex(jax, (3, N, Nkz, Nkx), 200 + i, sharding, vspec)
        for i in range(a.reps)
    ]
    t_lk = _bench(jax, lambda z: flow.Lk_op.solve(z), [(z,) for z in zs])
    t_hk = _bench(jax, lambda z: flow.Hk_op.solve(z), [(z,) for z in z3])
    solve_ms: dict[str, float | None] = {
        "Lk": 1e3 * t_lk,
        "Hk": 1e3 * t_hk,
        "Hc": None,
    }
    if groups.get("Hc") is not None:
        z6 = [
            _make_complex(jax, (6, N, Nkz, Nkx), 300 + i, sharding, vspec)
            for i in range(a.reps)
        ]
        solve_ms["Hc"] = 1e3 * _bench(
            jax, lambda z: flow.Hc_op.solve(z), [(z,) for z in z6]
        )
    del zs, z3

    # Step timings, both schemes, from fresh copies of the same IC.
    t_icn, corrs = _bench_step(
        jax, jnp, m.predict_and_fully_correct, state, a.steps
    )
    t_cnab2 = _bench_step_cnab2(jax, jnp, m.step_cnab2, state, a.steps)

    # Split-corrector A/B: ``t_icn`` above ran the *default* corrector
    # (``step.split_corrector`` is off by default, so that is normally
    # the unsplit corrector).  Rebuild and time the *opposite* gate
    # value on the same IC, then label both by gate, so the [3] verdict
    # always compares split vs unsplit regardless of the default (the
    # corrector counts show where the FFT-refresh savings come from).
    default_split = params.step.split_corrector
    params.step.split_corrector = not default_split
    try:
        t_other, corrs_other = _bench_step(
            jax, jnp, _rebuild_pfc(a.system, flow), state, a.steps
        )
    finally:
        params.step.split_corrector = default_split
    (t_icn_split, corrs_split), (t_icn_unsplit, corrs_unsplit) = (
        ((t_icn, corrs), (t_other, corrs_other))
        if default_split
        else ((t_other, corrs_other), (t_icn, corrs))
    )
    mem_peak = _mem_stats(jax)

    record = {
        "kind": "single",
        "status": "ok",
        "config": {
            "system": a.system,
            "backend": a.backend,
            "nx": a.nx,
            "ny": a.ny,
            "nz": a.nz,
            "fd_order": a.fd_order,
            "dt": a.dt,
            "seed": a.seed,
            "bm0": a.bm0,
            "bm1": a.bm1,
        },
        "env": env,
        "operators": operators,
        "setup_s": round(setup_s, 3),
        "factor_bytes": factor_bytes,
        "mem": {
            "after_setup": mem_setup,
            "peak": mem_peak,
        },
        "solve_ms": {
            k: (round(v, 4) if v is not None else None)
            for k, v in solve_ms.items()
        },
        "step_ms": {
            "icn": round(1e3 * t_icn, 3),
            "icn_correctors": corrs,
            "icn_split": round(1e3 * t_icn_split, 3),
            "icn_split_correctors": corrs_split,
            "icn_unsplit": round(1e3 * t_icn_unsplit, 3),
            "icn_unsplit_correctors": corrs_unsplit,
            "cnab2": round(1e3 * t_cnab2, 3),
        },
        "parity": parity,
    }
    print(RESULT_TAG + json.dumps(record), flush=True)


# ── driver: env probe + child/mpirun plumbing ────────────────────────


def _probe_env() -> dict:
    """Versions + devices via a subprocess, so the driver process
    itself never initializes a (GPU-attached) JAX client."""
    code = (
        "import json, jax, jaxlib\n"
        "d = jax.devices()\n"
        "print(json.dumps({'jax': jax.__version__,"
        " 'jaxlib': jaxlib.__version__,"
        " 'backend': jax.default_backend(),"
        " 'devices': [str(x) for x in d],"
        " 'kinds': [getattr(x, 'device_kind', '?') for x in d]}))"
    )
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        proc = subprocess.run(
            [str(PY), "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"backend": "unknown", "error": f"{type(e).__name__}: {e}"}


def _dense_gb(system: str, nx: int, ny: int, nz: int) -> float:
    """Analytic dense-backend factor estimate (f64, all groups)."""
    return N_OPS[system] * (nz - 1) * (nx // 2) * ny * ny * 8 / 2**30


def _log(workdir: Path, tag: str, text: str) -> Path:
    logdir = workdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"{tag}.log"
    path.write_text(text)
    return path


def _record(workdir: Path, rec: dict) -> None:
    with open(workdir / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _spawn_child(
    entry: dict,
    backend: str,
    args: argparse.Namespace,
    workdir: Path,
    env_extra: dict[str, str],
    tag: str,
    bm: tuple[int, int] | None = None,
) -> dict:
    """Run one child subprocess; parse its ``@@RESULT`` line."""
    cmd = [
        str(PY),
        str(SELF),
        "--child",
        "--system",
        entry["system"],
        "--backend",
        backend,
        "--nx",
        str(entry["nx"]),
        "--ny",
        str(entry["ny"]),
        "--nz",
        str(entry["nz"]),
        "--fd-order",
        str(entry.get("fd_order", 8)),
        "--dt",
        str(entry.get("dt", args.dt)),
        "--steps",
        str(args.steps),
        "--parity-steps",
        str(args.parity_steps),
        "--reps",
        str(args.reps),
        "--seed",
        str(args.seed),
    ]
    if bm is not None:
        cmd += ["--bm0", str(bm[0]), "--bm1", str(bm[1])]
    # The section's env carries the platform: JAX_PLATFORMS=cpu for the
    # CPU sections, otherwise a GPU pinned via CUDA_VISIBLE_DEVICES.  Pass
    # it through explicitly so the child records the right platform.
    cmd += ["--platform", env_extra.get("JAX_PLATFORMS", "cuda")]
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.child_timeout,
            env=env,
            cwd=REPO,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        stdout = _to_text(e.stdout)
        stderr = _to_text(e.stderr) + "\n[TIMEOUT]"
        rc = -1
        timed_out = True
    _log(
        workdir,
        tag,
        f"$ {' '.join(cmd)}\n\n{stdout}\n--- stderr ---\n{stderr}",
    )
    result = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_TAG):
            result = json.loads(line[len(RESULT_TAG) :])
    if result is None:
        cfg = {
            "system": entry["system"],
            "backend": backend,
            **{k: entry[k] for k in ("nx", "ny", "nz")},
        }
        if bm is not None:
            cfg["bm0"], cfg["bm1"] = bm
        result = {
            "kind": "single",
            "status": "crash",
            "config": cfg,
            "error": (
                f"timeout after {args.child_timeout:.0f}s"
                if timed_out
                else f"exit {rc}, no result line"
            ),
            "stderr_tail": stderr[-1200:],
        }
        _echo_tail(tag, stdout, stderr)
    result["entry"] = tag
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    # Setup check lines ([pallas] residual/growth, and any notice) --
    # scanned by the verdict's stability-notice check.
    result["pallas_lines"] = PALLAS_PATTERN.findall(stdout)
    _record(workdir, result)
    return result


def _sys_cli_flags(system: str) -> list[str]:
    flags = ["--phys.system", system]
    for dotted, v in SYS_ARGS[system].items():
        flags += [f"--{dotted}", str(v)]
    return flags


# GPU launch modes for srun steps, probed in order by the preflight.
# The right per-rank device assignment is site-dependent:
#
# - "pinned": ``--gpus-per-task=1`` binding (each task sees one GPU via
#   CUDA_VISIBLE_DEVICES) + JAX_LOCAL_DEVICE_IDS=0.  The override is
#   required because JAX's SLURM auto-detect sets ``local_device_ids =
#   [SLURM_LOCALID]`` (jax/_src/clusters/cluster.py), out of range for
#   rank > 0 under per-task narrowing (observed: backend-independent
#   rank>0 crashes).
# - "pinned-nocumem": pinned + ``NCCL_CUMEM_ENABLE=0`` -- NCCL's cuMem
#   path can fail with "Cuda failure 101 'invalid device ordinal'" on
#   some systems when visibility is narrowed per process (observed on
#   the first NCCL collective after a clean distributed init).
# - "all-visible": no binding; every task sees all job GPUs and JAX's
#   ``[SLURM_LOCALID]`` heuristic picks a distinct device per rank --
#   the textbook single-node multi-process layout, with full NCCL P2P
#   visibility.
GPU_MODES = ("pinned", "pinned-nocumem", "all-visible")

# XLA-flag variants once suspected for the first-step hang -- all
# exonerated on the cluster (2026-07-08): with JAX_LOG_COMPILES the
# big step demonstrably *compiles* (~3.3 s per rank) and hangs in its
# first execution, identically under every variant below.  Kept only
# for manual forcing via --xla-variant; the preflight now bisects
# constructs (reshard probe, launch topology, scheme, backend, NCCL
# env) instead of XLA flags.
XLA_VARIANTS = (
    ("default", ""),
    ("no-command-buffer", "--xla_gpu_enable_command_buffer="),
    ("no-comm-splitting", "--xla_gpu_enable_nccl_comm_splitting=false"),
    (
        "no-latency-hiding",
        "--xla_gpu_enable_latency_hiding_scheduler=false",
    ),
)

_PORT_SEQ = itertools.count()


def _next_port() -> int:
    """A fresh JAX coordinator port per multi-process launch.

    JAX's SLURM auto-detect derives a single job-wide port from
    SLURM_JOB_ID, so every step of the job binds the same port
    back-to-back; handing each launch its own port removes that
    coupling from the diagnosis space.
    """
    return 52000 + next(_PORT_SEQ) % 4000


def _echo_wrapper() -> list[str]:
    """Per-task launch proof + env truth, echoed to the captured
    stderr before exec'ing the real command: an empty tail after a
    timeout cannot otherwise distinguish "the step never launched its
    tasks" from "hung inside jax.distributed.initialize()"."""
    return [
        "bash",
        "-c",
        'echo "[task $SLURM_PROCID/$SLURM_NTASKS] host=$(hostname)'
        ' localid=$SLURM_LOCALID cvd=${CUDA_VISIBLE_DEVICES:-unset}"'
        ' >&2; exec "$@"',
        "bash",
    ]


def _task_wrapper(n: int, port: int) -> list[str]:
    """Detection-proof distributed bootstrap for one srun launch.

    Extends :func:`_echo_wrapper` with per-task env exports that
    remove every auto-detection variable from ``jax.distributed``:
    the step task count is forced to the known *n* (a job-level
    ``SLURM_NTASKS`` leaking into the tasks would make each rank wait
    for the wrong world size -- a silent barrier hang), and the
    coordinator address is set explicitly to this (single-node)
    host on a fresh per-run *port* (``JAX_COORDINATOR_ADDRESS``,
    read before any cluster detection; JAX's SLURM default is one
    job-wide port shared by every step of the job).  Rank identity
    stays per-task from ``SLURM_PROCID`` and is echoed for
    verification.
    """
    script = (
        f"export SLURM_NTASKS={n}; "
        # Coordinator host: the *first* node of the step, so a future
        # multi-node step agrees on one coordinator (a per-task
        # $SLURMD_NODENAME would make every rank dial its own node);
        # on a single node all fallbacks coincide.
        '_ch="$(scontrol show hostnames "$SLURM_STEP_NODELIST" '
        '2>/dev/null | head -n1)"; '
        "export JAX_COORDINATOR_ADDRESS="
        f'"${{_ch:-${{SLURMD_NODENAME:-$(hostname)}}}}:{port}"; '
        'echo "[task $SLURM_PROCID/$SLURM_NTASKS] host=$(hostname)'
        " localid=$SLURM_LOCALID cvd=${CUDA_VISIBLE_DEVICES:-unset}"
        ' coord=$JAX_COORDINATOR_ADDRESS" >&2; '
        'exec "$@"'
    )
    return ["bash", "-c", script, "bash"]


def _gpu_mode_flags(args: argparse.Namespace) -> list[str]:
    """srun GRES flags for the active GPU launch mode."""
    if args.gpu_mode in ("pinned", "pinned-nocumem") and (
        "gpu" not in args.srun_args
    ):
        return ["--gpus-per-task=1"]
    return []


def _gpu_mode_env(args: argparse.Namespace, platform: str) -> dict:
    """Step-environment overrides for the active GPU launch mode."""
    env: dict[str, str] = {}
    if args.launcher == "srun" and platform == "cuda":
        if args.gpu_mode in ("pinned", "pinned-nocumem"):
            env["JAX_LOCAL_DEVICE_IDS"] = "0"
        if args.gpu_mode == "pinned-nocumem":
            env["NCCL_CUMEM_ENABLE"] = "0"
    return env


def _resolve_launcher(choice: str) -> str:
    """Resolve ``--launcher auto`` to whichever launcher exists."""
    if choice != "auto":
        return choice
    if shutil.which("mpirun"):
        return "mpirun"
    if shutil.which("srun"):
        return "srun"
    raise SystemExit(
        "neither mpirun nor srun found on PATH; pass --launcher explicitly"
    )


def _launch_prefix(
    n: int, args: argparse.Namespace, platform: str
) -> list[str]:
    """Multi-process launch prefix for one ``n``-process dnsjax run.

    ``srun`` starts an ``n``-task job step inside the surrounding SLURM
    allocation; ``--overlap`` lets the step share the allocation with
    the driver's own step when the driver itself was launched via srun.
    ``mpirun`` is the plain OpenMPI path (``--oversubscribe`` on CPU /
    when requested).  Site-specific step flags (GPU binding, gres) go
    through ``--srun-args``.
    """
    if args.launcher == "srun":
        prefix = ["srun", "-n", str(n), "--overlap"]
        if platform == "cuda":
            prefix += _gpu_mode_flags(args)
        if args.srun_args:
            prefix += shlex.split(args.srun_args)
        return prefix
    prefix = ["mpirun"]
    if args.oversubscribe or platform == "cpu":
        prefix.append("--oversubscribe")
    prefix += ["-np", str(n)]
    return prefix


# dnsjax's exact JAX-side startup pattern, without dnsjax: x64 +
# platform configured *before* initialize (the ``__main__`` order), an
# Explicit-axis-type mesh via make_mesh/set_mesh (``sharding.py``), a
# device allocation sharded with ``out_sharding``, and a host ``float``
# read of a reduction over the sharded axis -- the replicated-result
# collective that is dnsjax's first cross-GPU operation (the
# ``[pallas]`` setup-residual read during geometry import).
_EXPLICIT_MESH_PROBE = (
    "import jax\n"
    "jax.config.update('jax_enable_x64', True)\n"
    "jax.config.update('jax_platforms', 'cuda')\n"
    "jax.distributed.initialize()\n"
    "pc = jax.process_count()\n"
    "assert pc == 2, ('process_count', pc)\n"
    "import jax.numpy as jnp\n"
    "from jax.sharding import AxisType, PartitionSpec\n"
    "mesh = jax.make_mesh((1, 2), ('np0', 'np1'),\n"
    "                     axis_types=(AxisType.Explicit,) * 2)\n"
    "jax.set_mesh(mesh)\n"
    "x = jnp.zeros((8, 64), out_sharding=PartitionSpec(None, 'np1'))\n"
    "r = float(jnp.max(x + 1.0))\n"
    "print('explicit-ok', r, flush=True)\n"
)

# Bare 2-process bootstrap: nothing but jax.distributed.initialize().
_INIT_PROBE = (
    "import jax\n"
    "jax.distributed.initialize()\n"
    "print('init-ok', jax.process_count(), jax.process_index(),"
    " flush=True)\n"
)

# A minimal 2-process jax.distributed program exercising the NCCL
# collective *patterns* the solver actually uses -- an all-gather and,
# crucially, a jit resharding between orthogonal partition specs (the
# all-to-all class behind dnsjax's FFT reshard pipeline; a launch mode
# whose communicator comes up for an all-gather can still hang there).
# The process_count assert forbids a trivial pass as two independent
# 1-process worlds (which a per-step SLURM_NTASKS mixup would create).
_COLLECTIVE_PROBE = (
    "import jax\n"
    "jax.distributed.initialize()\n"
    "pc = jax.process_count()\n"
    "assert pc == 2, ('process_count', pc)\n"
    "import numpy as np\n"
    "import jax.numpy as jnp\n"
    "from jax.sharding import Mesh, NamedSharding, PartitionSpec\n"
    "from jax.experimental import multihost_utils\n"
    "out = multihost_utils.process_allgather(jnp.ones(3))\n"
    "mesh = Mesh(np.array(jax.devices()), ('a',))\n"
    "s0 = NamedSharding(mesh, PartitionSpec('a', None))\n"
    "s1 = NamedSharding(mesh, PartitionSpec(None, 'a'))\n"
    "x = jax.device_put(jnp.ones((8, 8)), s0)\n"
    "y = jax.jit(lambda v: v * 2, out_shardings=s1)(x)\n"
    "jax.block_until_ready(y)\n"
    "print('collective-ok', pc, out.shape, flush=True)\n"
)

# The construct no rung above exercises and the first big step
# executes: ``jax.sharding.reshard`` between orthogonal specs on an
# *Explicit* mesh in complex128 -- the all-to-all behind dnsjax's FFT
# pipeline (``fft.py`` reshards #1/#2; the L4 probe reshards float32
# on an implicit mesh, L4.5 only allreduces).  Tried bare (ph1),
# inside a jitted shard_map-FFT pipeline executed twice (ph2/ph2b:
# communicator setup + reuse), and inside a ``lax.while_loop`` whose
# condition consumes a global reduction (ph3: the corrector
# fixed-point skeleton).  Each phase prints a marker, so a timeout
# tail names the construct; ``reshard-ok`` marks a full pass.
_RESHARD_PROBE = (
    "import functools\n"
    "import jax\n"
    "jax.config.update('jax_enable_x64', True)\n"
    "jax.config.update('jax_platforms', 'cuda')\n"
    "jax.distributed.initialize()\n"
    "pc = jax.process_count()\n"
    "assert pc == 2, ('process_count', pc)\n"
    "import jax.numpy as jnp\n"
    "from jax.sharding import AxisType, PartitionSpec, reshard\n"
    "mesh = jax.make_mesh((1, 2), ('np0', 'np1'),\n"
    "                     axis_types=(AxisType.Explicit,) * 2)\n"
    "jax.set_mesh(mesh)\n"
    "mid = PartitionSpec(None, 'np1', None)\n"
    "spec = PartitionSpec(None, None, 'np1')\n"
    "x = jnp.full((24, 12, 8), 1.0 + 0.5j, dtype=jnp.complex128,\n"
    "             out_sharding=mid)\n"
    "jax.block_until_ready(reshard(x, spec))\n"
    "print('reshard-ph1-ok', flush=True)\n"
    "def lfft(spec_, axis):\n"
    "    f = functools.partial(jnp.fft.fft, axis=axis, norm='forward')\n"
    "    return jax.shard_map(f, mesh=mesh, in_specs=spec_,\n"
    "                         out_specs=spec_)\n"
    "@jax.jit\n"
    "def fwd(a):\n"
    "    b = reshard(lfft(mid, 2)(a), spec)\n"
    "    b = reshard(lfft(spec, 1)(b), mid)\n"
    "    return jnp.max(jnp.abs(b))\n"
    "print('reshard-ph2-ok', float(fwd(x)), flush=True)\n"
    "print('reshard-ph2b-ok', float(fwd(x)), flush=True)\n"
    "@jax.jit\n"
    "def loop(a):\n"
    "    def cond(c):\n"
    "        return (c[2] > 1e-30) & (c[0] < 3)\n"
    "    def body(c):\n"
    "        i, v, _ = c\n"
    "        w = reshard(lfft(mid, 2)(v), spec)\n"
    "        w = reshard(lfft(spec, 1)(w), mid)\n"
    "        return i + 1, w, jnp.max(jnp.abs(w))\n"
    "    return jax.lax.while_loop(cond, body,\n"
    "                              (0, a, jnp.asarray(1.0)))\n"
    "i, v, e = loop(x)\n"
    "print('reshard-ph3-ok', int(i), float(e), flush=True)\n"
    "print('reshard-ok', flush=True)\n"
)

# NCCL environment variants bisected when the reshard probe or the
# dnsjax baseline leg hangs: each disables one transport-layer NCCL
# feature that the small passing collectives may not touch (P2P
# send/recv transport, cuMem buffer registration -- the mechanism
# whose import already hard-fails under pinned per-task visibility on
# this site -- and the low-latency protocols).  The first variant
# whose retry completes is locked into every subsequent run.
NCCL_VARIANTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("nccl-p2p-off", {"NCCL_P2P_DISABLE": "1"}),
    ("nccl-cumem-off", {"NCCL_CUMEM_ENABLE": "0"}),
    ("nccl-proto-simple", {"NCCL_PROTO": "Simple"}),
)


def _probe_step(
    cmd: list[str], env: dict, timeout: float, marker: str
) -> tuple[bool, str, str]:
    """Run one preflight probe; returns ``(ok, why, output)``."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        out = proc.stdout + "\n--- stderr ---\n" + proc.stderr
        ok = proc.returncode == 0 and (marker == "" or marker in proc.stdout)
        why = "ok" if ok else f"exit {proc.returncode}"
    except subprocess.TimeoutExpired as e:
        out = _to_text(e.stdout) + "\n--- stderr ---\n" + _to_text(e.stderr)
        ok = False
        why = f"timeout after {timeout:.0f}s"
    return ok, why, out


def _preflight_launcher(
    args: argparse.Namespace, platform: str, workdir: Path
) -> tuple[bool, str]:
    """Fail fast when a multi-task launch cannot work -- and, on cuda,
    pick a working GPU launch mode via an escalating probe ladder.

    Each rung runs with a short timeout and, on failure, echoes its
    output tail immediately -- localizing a multi-rank bootstrap
    failure to a layer within minutes instead of one ``--mpi-timeout``
    per hung run:

    - ``L1-tasks``: 2-task step launch + per-task env echo (a step
      that cannot start makes srun retry silently forever).  When
      this rung fails -- the allocation cannot host multi-task steps
      at all (e.g. 1 task per node) -- the ladder skips straight to
      the single-process L5 leg instead of giving up: that topology
      needs only one task.
    - ``L2-init-detected``: bare ``jax.distributed.initialize()`` with
      SLURM auto-detection; prints the world size each rank believes.
    - ``L3-init-explicit``: the same under :func:`_task_wrapper` --
      forced step task count + explicit fresh-port coordinator
      address, the bootstrap the real runs use.
    - ``L4-collectives``: all-gather + jit reshard (the all-to-all
      class of the FFT pipeline) under the explicit bootstrap.
    - ``L4.5-explicit-mesh``: dnsjax's exact first-collective pattern
      (Explicit mesh + sharded alloc + host reduction read).
    - ``L4.6-explicit-reshard``: :data:`_RESHARD_PROBE` (phased FFT
      reshard / while_loop-corrector skeleton); on a hang the
      :data:`NCCL_VARIANTS` are bisected on this cheap repro.
    - ``L5``: real 2-device dnsjax micro-runs through
      :func:`_run_mpi`, bisecting launch topology / scheme:
      ``mp-icn-pallas`` (the production baseline), ``sp-icn-pallas``
      (single-process multi-GPU -- the offline-test topology on real
      devices, no cross-process NCCL), ``mp-cnab2-pallas``, then
      NCCL-variant retries of the baseline and a
      ``--xla_gpu_nccl_termination_timeout_seconds`` diagnosis leg.
      Any passing leg unlocks the matrix; lock preference is
      mp-icn > mp+NCCL-variant > single-process > mp-cnab2 (recorded
      in ``args.launch_mode`` / ``args.locked_scheme`` /
      ``args.extra_run_env``).

    The first mode whose ladder yields a working leg is locked into
    ``args.gpu_mode``; pass ``--gpu-mode`` / ``--launch-mode`` to
    probe a single one.
    """
    if args.launcher != "srun":
        return True, ""

    if platform != "cuda":
        cmd = _launch_prefix(2, args, platform) + ["true"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired as e:
            err = _to_text(e.stderr)
            return False, (
                f"launcher preflight: `{' '.join(cmd)}` did not start "
                "within 120s -- this allocation cannot host a 2-task "
                "step (srun retries forever while no task slots are "
                "free for it).  Allocate with enough tasks and run "
                "the driver as a plain process.  srun said:\n" + err[-800:]
            )
        if proc.returncode != 0:
            return False, (
                f"launcher preflight: `{' '.join(cmd)}` failed "
                f"(exit {proc.returncode}):\n" + proc.stderr[-800:]
            )
        return True, ""

    modes = list(GPU_MODES) if args.gpu_mode == "auto" else [args.gpu_mode]
    fails: list[tuple[str, str]] = []
    for mode in modes:
        args.gpu_mode = mode
        # An NCCL variant is only "locked" if its mode returns below;
        # do not leak one from a mode whose legs still failed.
        args.extra_run_env = {}
        prefix = _launch_prefix(2, args, "cuda")
        base_env = dict(os.environ)
        base_env.update(_gpu_mode_env(args, "cuda"))
        base_env.setdefault("NCCL_DEBUG", "WARN")
        base_env["JAX_LOGGING_LEVEL"] = "INFO"

        # L5 micro-run + leg helper, defined before the rung loop so a
        # failed L1 (an allocation that cannot host multi-task steps
        # at all) can still probe the single-process leg below.
        micro = {
            "system": "plane-couette",
            "backend": "pallas",
            "np0": 1,
            "np1": 2,
            "nx": 8,
            "ny": 24,
            "nz": 8,
            "dt": 0.01,
            "tmax": 0.03,
            "snapshots": False,
            "timeout": 180,
            # Legs are expected to hang: trace collective launches
            # into per-rank nccl.*.log files (tails echoed on
            # failure) -- the last enqueued collective before silence
            # names the hung one.
            "env": {
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,COLL",
            },
        }

        def _leg(tag: str, _micro=micro, _mode=mode, **over) -> dict:
            run = dict(_micro)
            run["env"] = {**_micro["env"], **over.pop("env", {})}
            run.update(over, name=f"preflight-L5-{_mode}-{tag}")
            rec = _run_mpi(run, args, workdir, "cuda")
            print(
                f"[preflight/{_mode}] L5-{tag}: {rec['status']} "
                f"{rec.get('error', '')}".rstrip(),
                flush=True,
            )
            if rec["status"] != "ok":
                fails.append((f"{_mode}/L5-{tag}", rec.get("error", "")))
            return rec

        def _ok(rec: dict | None) -> bool:
            return rec is not None and rec["status"] == "ok"

        def _sp_leg() -> dict:
            # Single-process multi-GPU: 1 task whose PJRT client
            # addresses both devices (the offline-test topology on
            # real GPUs).  Identical global mesh / partitioning /
            # outputs; no cross-process NCCL transport.
            return _leg(
                "sp-icn-pallas",
                tasks=1,
                env={"JAX_LOCAL_DEVICE_IDS": "0,1"},
            )

        rungs = [
            ("L1-tasks", prefix + _echo_wrapper() + ["true"], 60, ""),
            (
                "L2-init-detected",
                prefix + _echo_wrapper() + [str(PY), "-c", _INIT_PROBE],
                90,
                "init-ok 2",
            ),
            (
                "L3-init-explicit",
                prefix
                + _task_wrapper(2, _next_port())
                + [str(PY), "-c", _INIT_PROBE],
                90,
                "init-ok 2",
            ),
            (
                "L4-collectives",
                prefix
                + _task_wrapper(2, _next_port())
                + [str(PY), "-c", _COLLECTIVE_PROBE],
                120,
                "collective-ok 2",
            ),
            (
                "L4.5-explicit-mesh",
                prefix
                + _task_wrapper(2, _next_port())
                + [str(PY), "-c", _EXPLICIT_MESH_PROBE],
                120,
                "explicit-ok",
            ),
        ]
        mode_ok = True
        for rung, cmd, timeout, marker in rungs:
            env = dict(base_env)
            env["JAX_COORDINATOR_PORT"] = str(_next_port())
            ok, why, out = _probe_step(cmd, env, timeout, marker)
            print(f"[preflight/{mode}] {rung}: {why}", flush=True)
            if not ok:
                _echo_tail(f"preflight/{mode}/{rung}", "", out)
                fails.append((f"{mode}/{rung}", why))
                mode_ok = False
                break
        if not mode_ok:
            # A failed L1 means this allocation cannot host multi-task
            # steps at all (e.g. the 1-task-per-node discipline this
            # site otherwise prefers) -- every multi-process rung is
            # then moot, but the single-process multi-GPU topology
            # needs only one task: probe it directly before giving up
            # on the mode.
            if (
                rung == "L1-tasks"
                and args.launch_mode in ("auto", "sp")
                and _ok(_sp_leg())
            ):
                args.launch_mode = "sp"
                print(
                    f"[preflight] locked in: gpu mode '{mode}', "
                    "single-process multi-GPU (multi-task steps "
                    "cannot launch in this allocation)"
                )
                return True, ""
            continue

        # L4.6: the construct the ladder has never exercised and the
        # first big step executes (Explicit-mesh reshard in
        # complex128, incl. the while_loop corrector skeleton).  A
        # hang here is a 15-line repro; the NCCL variants are then
        # bisected on it, and a working variant is locked into every
        # subsequent run.
        cmd = (
            prefix
            + _task_wrapper(2, _next_port())
            + [str(PY), "-c", _RESHARD_PROBE]
        )
        env = dict(base_env)
        env["JAX_COORDINATOR_PORT"] = str(_next_port())
        ok, why, out = _probe_step(cmd, env, 150, "reshard-ok")
        print(f"[preflight/{mode}] L4.6-explicit-reshard: {why}", flush=True)
        if not ok:
            _echo_tail(f"preflight/{mode}/L4.6-explicit-reshard", "", out)
            fails.append((f"{mode}/L4.6-explicit-reshard", why))
            for vname, venv in NCCL_VARIANTS:
                env = dict(base_env)
                env.update(venv)
                env["JAX_COORDINATOR_PORT"] = str(_next_port())
                ok, why, out = _probe_step(cmd, env, 150, "reshard-ok")
                print(
                    f"[preflight/{mode}] L4.6-explicit-reshard "
                    f"[{vname}]: {why}",
                    flush=True,
                )
                if ok:
                    args.extra_run_env.update(venv)
                    print(
                        f"[preflight] NCCL variant '{vname}' unblocks "
                        f"the reshard probe -- locked into all runs: "
                        f"{venv}"
                    )
                    break
                fails.append((f"{mode}/L4.6-explicit-reshard[{vname}]", why))

        # L5: real dnsjax micro-runs, bisecting launch topology /
        # scheme / backend (each ~1 min healthy, 180 s hung; the hang
        # under bisection sits in the first *execution* of the big
        # jitted step -- it compiles fine, and every small-program
        # collective incl. L4.6 tells its own story above).
        want = args.launch_mode
        a_rec = None
        if want in ("auto", "mp"):
            a_rec = _leg("mp-icn-pallas")
            if _ok(a_rec):
                args.launch_mode = "mp"
                print(
                    f"[preflight] locked in: gpu mode '{mode}', "
                    "multi-process launch"
                )
                return True, ""

        sp_rec = None
        if want in ("auto", "sp"):
            sp_rec = _sp_leg()
            if want == "sp":
                if _ok(sp_rec):
                    args.launch_mode = "sp"
                    print(
                        f"[preflight] locked in: gpu mode '{mode}', "
                        "single-process multi-GPU (forced)"
                    )
                    return True, ""
                continue

        cn_rec = None
        env_fixed = False
        if want in ("auto", "mp") and not _ok(a_rec):
            cn_rec = _leg("mp-cnab2-pallas", scheme="cnab2")
            if not args.extra_run_env and not _ok(cn_rec):
                for vname, venv in NCCL_VARIANTS:
                    rec = _leg(f"mp-icn-pallas-{vname}", env=venv, timeout=150)
                    if _ok(rec):
                        args.extra_run_env.update(venv)
                        env_fixed = True
                        print(
                            f"[preflight] NCCL variant '{vname}' "
                            f"unblocks the step -- locked into all "
                            f"runs: {venv}"
                        )
                        break
            if not env_fixed and not _ok(cn_rec):
                # Pure diagnosis: ask XLA to abort collectives stuck
                # > 45 s.  A recognized flag turns the silent hang
                # into an error naming the collective; an unknown
                # flag fails fast and harmlessly.
                _leg(
                    "mp-icn-pallas-nccl-abort",
                    timeout=150,
                    xla_extra=(
                        "--xla_gpu_nccl_termination_timeout_seconds=45"
                    ),
                )
        if env_fixed:
            args.launch_mode = "mp"
            print(
                f"[preflight] locked in: gpu mode '{mode}', "
                "multi-process launch + NCCL variant"
            )
            return True, ""
        if _ok(sp_rec):
            args.launch_mode = "sp"
            print(
                f"[preflight] locked in: gpu mode '{mode}', "
                "single-process multi-GPU (multi-process launches "
                "hang in the first jitted step; bisection above)"
            )
            return True, ""
        if _ok(cn_rec):
            args.launch_mode = "mp"
            args.locked_scheme = "cnab2"
            print(
                f"[preflight] locked in: gpu mode '{mode}', "
                "multi-process + cnab2 (iterative-cn hangs; the "
                "parity checks run within cnab2)"
            )
            return True, ""

    detail = "\n".join(f"  {name}: {why}" for name, why in fails)
    return False, (
        "no GPU launch mode passed the bootstrap ladder (per-rung "
        "tails echoed above).  Failures:\n" + detail
    )


def _echo_nccl_logs(rundir: Path, max_files: int = 3) -> None:
    """Echo the tail of each per-rank NCCL log of a failed run: under
    NCCL_DEBUG=INFO/SUBSYS=COLL the last enqueued collective before
    silence names the hung operation."""
    for nf in sorted(rundir.glob("nccl.*.log"))[:max_files]:
        try:
            lines = nf.read_text(errors="replace").splitlines()[-8:]
        except OSError:
            continue
        print(f"          --- {nf.name} tail ---", flush=True)
        for line in lines:
            print(f"          | {line[:200]}", flush=True)


def _run_mpi(
    run: dict, args: argparse.Namespace, workdir: Path, platform: str
) -> dict:
    """One multi-process ``-m dnsjax`` run in its own scratch dir."""
    n = run["np0"] * run["np1"]
    # ``tasks`` < n = single-process multi-GPU (the run's env carries
    # JAX_LOCAL_DEVICE_IDS spanning the mesh; see _sp_ify).
    tasks = run.get("tasks", n)
    timeout = run.get("timeout", args.mpi_timeout)
    cmd = _launch_prefix(tasks, args, platform)
    if args.launcher == "srun":
        # Detection-proof bootstrap: forced step task count + explicit
        # fresh-port coordinator address + per-task launch echo.
        cmd += _task_wrapper(tasks, _next_port())
    if args.launcher == "srun" and platform == "cuda":
        # Arm a per-rank Python stack dump shortly before the driver's
        # kill: a hang tail then shows the exact blocked host call on
        # every rank (faulthandler writes to the captured stderr).
        # runpy on the package == ``-m dnsjax``; sys.argv[1:] already
        # holds the dnsjax flags (argv[0] is '-c').
        cmd += [
            str(PY),
            "-c",
            (
                "import faulthandler, runpy, sys; "
                "faulthandler.dump_traceback_later("
                f"{max(30, int(timeout) - 30)}, exit=False); "
                "sys.argv[0] = 'dnsjax'; "
                "runpy.run_module('dnsjax', run_name='__main__', "
                "alter_sys=False)"
            ),
        ]
    else:
        cmd += [str(PY), "-m", "dnsjax"]
    cmd += ["--dist.platform", platform]
    cmd += ["--dist.np0", str(run["np0"]), "--dist.np1", str(run["np1"])]
    cmd += _sys_cli_flags(run["system"])
    cmd += [
        "--res.nx",
        str(run["nx"]),
        "--res.ny",
        str(run["ny"]),
        "--res.nz",
        str(run["nz"]),
        "--res.fd_order",
        "4",
        "--init.random_field",
        "True",
        "--init.random_amplitude",
        "0.1",
        "--init.random_smoothness",
        "0.4",
        "--init.random_seed",
        str(args.seed),
        "--step.dt",
        str(run["dt"]),
        "--stop.max_sim_time",
        str(run["tmax"]),
        "--stop.check_laminarization",
        "False",
        "--outs.it_stats",
        "10",
        "--outs.snapshot_save_initial",
        "False",
        "--solver.backend",
        run["backend"],
    ]
    if not run.get("snapshots", True):
        cmd += ["--outs.snapshot_save_final", "False"]
    if run.get("scheme"):
        cmd += ["--step.scheme", run["scheme"]]
    for flag, value in run.get("extra", ()):
        cmd += [flag, str(value)]

    rundir = workdir / "runs" / run["name"]
    rundir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Per-rank device assignment for the active GPU launch mode (the
    # preflight ladder locked a mode whose real micro-run works),
    # then any preflight-locked NCCL variant, then per-run overrides.
    env.update(_gpu_mode_env(args, platform))
    env.update(getattr(args, "extra_run_env", None) or {})
    env.update(run.get("env", {}))
    if args.launcher == "srun":
        # jax.distributed's own INFO progress lines land in the
        # captured stderr, so a silent init hang is diagnosable from
        # the timeout tail (the coordinator address/port comes from
        # the _task_wrapper exports).  JAX_LOG_COMPILES separates a
        # compile hang from an execution hang: "Finished tracing +
        # compilation of ..." in the tail means the program compiled
        # and the hang is in its first execution.
        env["JAX_LOGGING_LEVEL"] = "INFO"
        env["JAX_LOG_COMPILES"] = "1"
        extra = " ".join(
            x
            for x in (
                getattr(args, "xla_flags_extra", ""),
                run.get("xla_extra", ""),
            )
            if x
        )
        if extra:
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + extra).strip()
        if platform == "cuda":
            # Per-rank NCCL logs land on the (shared-FS) rundir and
            # their tails are echoed on failure below.
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("NCCL_DEBUG_FILE", str(rundir / "nccl.%h.%p.log"))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=rundir,
            env=env,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        # Keep the captured-so-far output: a hung `srun` retries "Job
        # step creation temporarily disabled" on *stderr*, which is
        # the diagnosis for a silent multi-device hang.
        stdout = _to_text(e.stdout)
        stderr = _to_text(e.stderr) + "\n[TIMEOUT]"
        rc = -1
        timed_out = True
    _log(
        workdir,
        run["name"],
        f"$ {' '.join(cmd)}\n\n{stdout}\n--- stderr ---\n{stderr}",
    )

    rec = {
        "kind": "mpi",
        "name": run["name"],
        "run": {k: v for k, v in run.items() if k not in ("extra", "env")},
        "wall_s": round(time.perf_counter() - t0, 1),
        "status": "ok" if rc == 0 else "crash",
        "rundir": str(rundir),
    }
    if rc != 0:
        rec["error"] = (
            f"timeout after {timeout:.0f}s" if timed_out else f"exit {rc}"
        )
        rec["stderr_tail"] = stderr[-1200:]
        _record(workdir, rec)
        _echo_tail(run["name"], stdout, stderr)
        _echo_nccl_logs(rundir)
        return rec

    # Success criteria (the random-smoke checklist) + benchmark line.
    if "failed to converge" in stdout:
        rec["status"] = "fail"
        rec["error"] = "corrector failed to converge"
    summary = None
    for line in stdout.splitlines():
        if ERR_PATTERN.search(line):
            summary = line
    if summary is not None:
        values = [float(v) for v in VALUE_PATTERN.findall(summary)]
        if not all(math.isfinite(v) for v in values):
            rec["status"] = "fail"
            rec["error"] = f"non-finite summary: {summary.strip()}"
        t_m = T_PATTERN.search(summary)
        if t_m is not None:
            t_final = float(t_m.group(1))
            rec["t_final"] = t_final
            if t_final < run["tmax"] - run["dt"]:
                rec["status"] = "fail"
                rec["error"] = f"ended early at t={t_final}"
    elif rec["status"] == "ok":
        rec["status"] = "fail"
        rec["error"] = "no end-of-run summary line"
    ran = RAN_PATTERN.search(stdout)
    if ran is not None:
        rec["ran_s"] = float(ran.group(1))
        rec["devices"] = int(ran.group(2))
    spt = SPT_PATTERN.search(stdout)
    if spt is not None:
        rec["s_per_t"] = float(spt.group(1))
    sprhs = SPRHS_PATTERN.search(stdout)
    if sprhs is not None:
        rec["s_per_rhs"] = float(sprhs.group(1))
    # Mean corrector iterations: a backend-roundoff-flipped iteration
    # count offsets the trajectory by O(corrector_tolerance), so this
    # is the discriminator for tolerance-scale cross-backend diffs.
    cit = CIT_PATTERN.search(stdout)
    if cit is not None:
        rec["c_per_it"] = float(cit.group(1))
    rec["pallas_lines"] = PALLAS_PATTERN.findall(stdout)
    _record(workdir, rec)
    if rec["status"] != "ok":
        _echo_tail(run["name"], stdout, stderr)
    return rec


def _final_snapshot(rundir: str) -> Path | None:
    snaps = sorted(Path(rundir).glob("state*.tar"))
    return snaps[-1] if snaps else None


def _diff_snapshots(pa: Path, pb: Path, ncomp: int) -> tuple[float, list]:
    """Max relative per-component spectral difference (JAX-free)."""
    import numpy as np

    from dnsjax.analysis.snapshot_export import read_state

    a = read_state(
        pa,
        return_physical=False,
        return_spectral=True,
        components=tuple(range(ncomp)),
    )
    b = read_state(
        pb,
        return_physical=False,
        return_spectral=True,
        components=tuple(range(ncomp)),
    )
    rel = []
    for x, y in zip(a.spectral, b.spectral, strict=True):
        denom = max(float(np.max(np.abs(x))), 1e-300)
        rel.append(float(np.max(np.abs(x - y))) / denom)
    return max(rel), [f"{r:.2e}" for r in rel]


# ── driver: sections ─────────────────────────────────────────────────


def _build_entries(args: argparse.Namespace) -> list[dict]:
    entries = []
    for system in args.systems:
        for size in args.sizes:
            nx, ny, nz = SIZES[system][size]
            entries.append(
                {
                    "name": f"{system}-{size}",
                    "system": system,
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "fd_order": 4,
                }
            )
    if "plane-couette" in args.systems and "prod" in args.sizes:
        nx, ny, nz = SIZES["plane-couette"]["prod"]
        entries.append(
            {
                "name": "plane-couette-prod-fd8",
                "system": "plane-couette",
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "fd_order": 8,
            }
        )
    if "small" in args.sizes:
        # Split-corrector stress case: total-field Dean at the
        # TimeStepping-docstring reference regime (nz = 64,
        # dt = 0.15), where the corrector cost is dominated by the
        # instantaneous mean-flow coupling (l_bf == L_mf) -- the
        # regime the split corrector targets.  The other entries
        # carry the split-vs-unsplit A/B too, but at the default
        # bench dt their correctors converge in ~1 iteration (the
        # equal-cost regime); this entry shows the FFT-refresh
        # savings.
        entries.append(
            {
                "name": "dean-split-stress",
                "system": "dean",
                "nx": 64,
                "ny": 48,
                "nz": 64,
                "fd_order": 4,
                "dt": 0.15,
            }
        )
    return entries


def _single_device_section(
    args: argparse.Namespace, workdir: Path, env_extra: dict[str, str]
) -> list[dict]:
    entries = _build_entries(args)
    results: list[dict] = []
    total = 0
    plan: list[tuple[dict, str, tuple[int, int] | None]] = []
    for entry in entries:
        for backend in BACKENDS:
            if backend == "dense":
                est = _dense_gb(
                    entry["system"], entry["nx"], entry["ny"], entry["nz"]
                )
                if est > args.dense_budget_gb:
                    results.append(
                        {
                            "kind": "single",
                            "status": "skipped",
                            "entry": f"{entry['name']}-dense",
                            "config": {
                                "system": entry["system"],
                                "backend": "dense",
                                **{
                                    k: entry[k]
                                    for k in ("nx", "ny", "nz", "fd_order")
                                },
                            },
                            "error": (
                                f"dense estimate {est:.1f} GB > budget "
                                f"{args.dense_budget_gb} GB"
                            ),
                        }
                    )
                    continue
            plan.append((entry, backend, None))
    if not args.no_tile_sweep:
        for entry in entries:
            if entry["name"] in ("pipe-prod", "plane-couette-prod"):
                for bm in TILE_SWEEP:
                    plan.append((entry, "pallas", bm))
    total = len(plan)

    for i, (entry, backend, bm) in enumerate(plan, 1):
        tag = f"{entry['name']}-{backend}"
        if bm is not None:
            tag += f"-tile{bm[0]}x{bm[1]}"
        print(f"[{i:2d}/{total}] {tag} ...", flush=True)
        rec = _spawn_child(entry, backend, args, workdir, env_extra, tag, bm)
        results.append(rec)
        if rec["status"] == "ok":
            sm = rec["step_ms"]
            uns = sm.get("icn_unsplit")
            uns_s = f"{uns:9.2f}" if uns is not None else f"{'-':>9s}"
            print(
                f"          icn {sm['icn']:9.2f} ms  unsplit {uns_s} ms  "
                f"cnab2 {sm['cnab2']:9.2f} ms  factors "
                f"{rec['factor_bytes']['total'] / 2**20:8.1f} MB",
                flush=True,
            )
        else:
            print(f"          {rec['status']}: {rec.get('error')}", flush=True)
    return results


def _sp_ify(run: dict) -> dict:
    """Rewrite one multi-device run to single-process multi-GPU: one
    task whose PJRT client addresses all ``np0 * np1`` devices (the
    offline-test topology on real GPUs) -- used when multi-process
    launches hang at this site.  The global mesh, partitioning, and
    outputs are identical; only the transport differs (no
    cross-process NCCL)."""
    n = run["np0"] * run["np1"]
    if n > 1:
        run["tasks"] = 1
        env = dict(run.get("env", {}))
        env["JAX_LOCAL_DEVICE_IDS"] = ",".join(str(i) for i in range(n))
        run["env"] = env
    return run


def _apply_locked_mode(runs: list[dict], args: argparse.Namespace) -> None:
    """Apply the preflight-locked launch topology / scheme in place."""
    if getattr(args, "locked_scheme", None):
        for run in runs:
            run.setdefault("scheme", args.locked_scheme)
    if getattr(args, "launch_mode", "mp") == "sp":
        for run in runs:
            _sp_ify(run)


def _mpi_runs(args: argparse.Namespace, platform: str) -> list[dict]:
    """The multi-device correctness matrix (padding-inducing plane)."""
    systems = [
        s
        for s in (
            "plane-couette",
            "pipe",
            "taylor-couette",
            "viscoelastic-dean",
        )
        if s in args.systems
    ]
    # Correctness runs are small (30 steps + JIT, minutes when
    # healthy): cap their timeout below --mpi-timeout so a residual
    # hang costs 10 min, not 30, before the auto-skip kicks in.
    base = {
        "nx": 34,
        "ny": 48,
        "nz": 32,
        "dt": 0.005,
        "tmax": 0.15,
        "timeout": 600.0,
    }
    runs: list[dict] = []
    for system in systems:
        for backend, np1 in (
            ("pallas", 1),
            ("pallas", 2),
            ("dense", 1),
        ):
            if np1 > args.max_gpus and platform == "cuda":
                continue
            runs.append(
                {
                    "name": f"corr-{system}-{backend}-np1x{np1}",
                    "system": system,
                    "backend": backend,
                    "np0": 1,
                    "np1": np1,
                    **base,
                }
            )
    if args.max_gpus >= 4 and platform == "cuda":
        # 2x2 mesh (wall-normal + spanwise split) on each geometry's
        # own default CGL grid -- no tanh: plane-couette (full-CGL)
        # and pipe (rigged-CGL), each diffed against its 1x1 pallas
        # reference from the main loop above.  ny=48 splits cleanly
        # over np0=2 (and the sharding layer auto-pads the y-axis
        # otherwise), so the default grid needs no divisibility
        # workaround -- the pipe np2x2/np1x4 runs already validate
        # wall-normal splitting on their default grid.
        if "plane-couette" in systems:
            runs.append(
                {
                    "name": "corr-plane-couette-pallas-np2x2",
                    "system": "plane-couette",
                    "backend": "pallas",
                    "np0": 2,
                    "np1": 2,
                    **base,
                }
            )
        if "pipe" in systems:
            runs.append(
                {
                    "name": "corr-pipe-pallas-np2x2",
                    "system": "pipe",
                    "backend": "pallas",
                    "np0": 2,
                    "np1": 2,
                    **base,
                }
            )
            runs.append(
                {
                    "name": "corr-pipe-pallas-np1x4",
                    "system": "pipe",
                    "backend": "pallas",
                    "np0": 1,
                    "np1": 4,
                    **base,
                }
            )
    _apply_locked_mode(runs, args)
    return runs


# Diff pairs: (run A, reference run B, label).
def _diff_pairs(names: set[str]) -> list[tuple[str, str, str]]:
    pairs = []
    for name in sorted(names):
        if not name.startswith("corr-"):
            continue
        stem = name[len("corr-") :]
        for system in SYS_ARGS:
            if stem.startswith(system):
                rest = stem[len(system) :]
                break
        else:
            continue
        ref = None
        label = None
        if rest == "-pallas-np1x2":
            ref, label = f"corr-{system}-pallas-np1x1", "device-count"
        elif rest == "-pallas-np1x1":
            ref, label = f"corr-{system}-dense-np1x1", "vs-dense"
        elif rest == "-pallas-np2x2":
            ref, label = f"corr-{system}-pallas-np1x1", "mesh-2x2"
        elif rest == "-pallas-np1x4":
            ref, label = f"corr-{system}-pallas-np1x1", "mesh-1x4"
        if ref in names:
            pairs.append((name, ref, label))
    return pairs


def _mpi_correctness_section(
    args: argparse.Namespace, workdir: Path, platform: str
) -> tuple[list[dict], list[dict]]:
    runs = _mpi_runs(args, platform)
    recs: dict[str, dict] = {}
    # Two consecutive multi-rank timeouts mean the launch mode is
    # systematically broken for real runs despite the preflight (each
    # timeout burns --mpi-timeout): stop feeding it and skip the
    # remaining n > 1 runs so the job still delivers the single-rank
    # results, diffs, and the echoed hang tails.
    multirank_timeouts = 0
    for i, run in enumerate(runs, 1):
        n = run["np0"] * run["np1"]
        if n > 1 and multirank_timeouts >= 2:
            recs[run["name"]] = {
                "kind": "mpi",
                "name": run["name"],
                "status": "skipped",
                "error": "skipped: multi-rank launches timing out",
            }
            _record(workdir, recs[run["name"]])
            print(
                f"[mpi {i:2d}/{len(runs)}] {run['name']} ... skipped "
                "(multi-rank launches timing out)",
                flush=True,
            )
            continue
        print(f"[mpi {i:2d}/{len(runs)}] {run['name']} ...", flush=True)
        rec = _run_mpi(run, args, workdir, platform)
        recs[run["name"]] = rec
        if n > 1 and "timeout" in rec.get("error", ""):
            multirank_timeouts += 1
        elif n > 1 and rec["status"] == "ok":
            multirank_timeouts = 0
        note = rec.get("error", "")
        if rec.get("c_per_it") is not None:
            note = f"{note}  c/it={rec['c_per_it']}".strip()
        print(f"          {rec['status']} {note}", flush=True)

    args.multirank_broken = multirank_timeouts >= 2
    diffs: list[dict] = []
    ok_names = {n for n, r in recs.items() if r["status"] == "ok"}
    for name, ref, label in _diff_pairs(ok_names):
        pa = _final_snapshot(recs[name]["rundir"])
        pb = _final_snapshot(recs[ref]["rundir"])
        drec = {
            "kind": "diff",
            "a": name,
            "b": ref,
            "label": label,
            # Differing mean corrector counts explain a
            # tolerance-scale (1e-5) trajectory divergence: a
            # roundoff-flipped convergence branch, not an error.
            "c_per_it": [
                recs[name].get("c_per_it"),
                recs[ref].get("c_per_it"),
            ],
        }
        if pa is None or pb is None:
            drec["status"] = "fail"
            drec["error"] = "missing final snapshot"
        else:
            ncomp = 9 if "viscoelastic" in name else 3
            try:
                worst, per = _diff_snapshots(pa, pb, ncomp)
                drec["max_rel_diff"] = worst
                drec["per_component"] = per
                drec["status"] = (
                    "PASS" if worst <= args.mpi_parity_tol else "FAIL"
                )
            except Exception as e:  # noqa: BLE001
                drec["status"] = "fail"
                drec["error"] = f"{type(e).__name__}: {e}"
        diffs.append(drec)
        _record(workdir, drec)
        per = (
            f"  per-component: {drec['per_component']}"
            if "per_component" in drec
            else ""
        )
        print(
            f"[diff] {label:16s} {name} vs {ref}: "
            f"{drec.get('max_rel_diff', drec.get('error'))} "
            f"-> {drec['status']}  c/it={drec['c_per_it']}{per}",
            flush=True,
        )
    return list(recs.values()), diffs


def _mpi_timing_section(args: argparse.Namespace, workdir: Path) -> list[dict]:
    runs: list[dict] = []
    schemes = (
        ("cnab2",)
        if getattr(args, "locked_scheme", None) == "cnab2"
        else ("iterative-cn", "cnab2")
    )
    for system in ("pipe", "plane-couette"):
        if system not in args.systems:
            continue
        nx, ny, nz = SIZES[system]["prod"]
        for backend in ("pallas",):
            for np1 in (1, 2):
                if np1 > args.max_gpus:
                    continue
                # Set by the correctness section when multi-rank
                # launches are systematically timing out.
                if np1 > 1 and getattr(args, "multirank_broken", False):
                    continue
                for scheme in schemes:
                    tag = "icn" if scheme == "iterative-cn" else "cnab2"
                    runs.append(
                        {
                            "name": (f"time-{system}-{backend}-np{np1}-{tag}"),
                            "system": system,
                            "backend": backend,
                            "np0": 1,
                            "np1": np1,
                            "nx": nx,
                            "ny": ny,
                            "nz": nz,
                            "dt": 0.005,
                            "tmax": 0.3,
                            "scheme": scheme,
                            "snapshots": False,
                            # Prod-size jit is minutes, the 60 steps
                            # seconds: 900 s covers both with margin.
                            "timeout": 900.0,
                        }
                    )
    _apply_locked_mode(runs, args)
    recs = []
    for i, run in enumerate(runs, 1):
        print(f"[time {i:2d}/{len(runs)}] {run['name']} ...", flush=True)
        rec = _run_mpi(run, args, workdir, "cuda")
        recs.append(rec)
        print(
            f"          {rec['status']}  s/t={rec.get('s_per_t')}  "
            f"s/rhs={rec.get('s_per_rhs')}",
            flush=True,
        )
    return recs


# ── driver: reporting ────────────────────────────────────────────────


def _mb(nbytes: int | None) -> str:
    return "-" if nbytes is None else f"{nbytes / 2**20:9.1f}"


def _single_tables(results: list[dict], parity_tol: float) -> list[str]:
    """Human tables + the pallas/dense ratio lines for the verdict."""
    by_entry: dict[str, list[dict]] = {}
    for r in results:
        if r["kind"] != "single":
            continue
        name = r["entry"]
        # strip "-{backend}" (and "-tileAxB" for the tile sweep).
        base = name.rsplit("-", 2 if "-tile" in name else 1)[0]
        by_entry.setdefault(base, []).append(r)

    verdict_lines: list[str] = []
    print("\n" + "=" * 78)
    print("SINGLE-DEVICE MATRIX")
    print("=" * 78)
    for base in sorted(by_entry):
        rows = by_entry[base]
        oks = {
            r["config"]["backend"]: r
            for r in rows
            if r["status"] == "ok" and r["config"].get("bm0") is None
        }
        ref = oks.get("dense")
        cfg = next(
            (r["config"] for r in rows if r["status"] == "ok"),
            rows[0]["config"],
        )
        print(
            f"\n{base}  (nx={cfg.get('nx')} ny={cfg.get('ny')} "
            f"nz={cfg.get('nz')})"
        )
        print(
            f"  {'backend':22s} {'icn ms':>10s} {'uns ms':>10s} "
            f"{'c s/u':>6s} {'cnab2 ms':>10s} "
            f"{'Lk ms':>8s} {'Hk ms':>8s} {'Hc ms':>8s} "
            f"{'factor MB':>10s} {'peak MB':>10s} {'parity':>9s}"
        )
        for r in rows:
            b = r["config"]["backend"]
            tile = (
                f"({r['config']['bm0']},{r['config']['bm1']})"
                if r["config"].get("bm0") is not None
                else ""
            )
            name = f"{b}{tile}"
            if r["status"] != "ok":
                print(f"  {name:22s} {r['status']}: {r.get('error')}")
                continue
            sm, sv = r["step_ms"], r["solve_ms"]
            peak = r["mem"]["peak"]
            peak_mb = (
                _mb(peak["peak_bytes_in_use"]) if peak["available"] else "-"
            )
            delta = ""
            if ref is not None and r is not ref:
                e0 = ref["parity"]["energy"]
                de = abs(r["parity"]["energy"] - e0) / max(abs(e0), 1e-300)
                delta = f"{de:.1e}"
                if de > parity_tol:
                    delta += "!"
            hc = f"{sv['Hc']:8.2f}" if sv["Hc"] is not None else f"{'-':>8s}"
            uns = sm.get("icn_unsplit")
            uns_s = f"{uns:10.2f}" if uns is not None else f"{'-':>10s}"
            csu = (
                f"{sm.get('icn_correctors', '-')}"
                f"/{sm.get('icn_unsplit_correctors', '-')}"
            )
            print(
                f"  {name:22s} {sm['icn']:10.2f} {uns_s} {csu:>6s} "
                f"{sm['cnab2']:10.2f} "
                f"{sv['Lk']:8.2f} {sv['Hk']:8.2f} {hc} "
                f"{_mb(r['factor_bytes']['total']):>10s} {peak_mb:>10s} "
                f"{delta:>9s}"
            )
        if "pallas" in oks and "dense" in oks:
            p, d = oks["pallas"], oks["dense"]
            r_icn = p["step_ms"]["icn"] / d["step_ms"]["icn"]
            r_ab = p["step_ms"]["cnab2"] / d["step_ms"]["cnab2"]
            r_fac = p["factor_bytes"]["total"] / d["factor_bytes"]["total"]
            pk_p, pk_d = p["mem"]["peak"], d["mem"]["peak"]
            r_peak = (
                pk_p["peak_bytes_in_use"] / pk_d["peak_bytes_in_use"]
                if pk_p["available"] and pk_d["available"]
                else float("nan")
            )
            verdict_lines.append(
                f"{base:28s} step icn x{r_icn:5.2f}  cnab2 x{r_ab:5.2f}  "
                f"factors x{r_fac:5.2f}  peak x{r_peak:5.2f}"
            )
    return verdict_lines


def _verdict(
    ratio_lines: list[str],
    diffs: list[dict],
    mpi_recs: list[dict],
    single: list[dict],
    timing: list[dict],
    args: argparse.Namespace | None = None,
) -> int:
    print("\n" + "=" * 78)
    print("VERDICT (pallas health checks)")
    print("=" * 78)
    code = 0

    print("\n[1] multi-GPU Pallas parity (snapshot diffs):")
    if args is not None and diffs:
        if getattr(args, "launch_mode", "mp") == "sp":
            print(
                "    NOTE measured via single-process multi-GPU (the "
                "multi-process launch hangs at this site -- a launch/"
                "transport issue, orthogonal to backend numerics; "
                "mesh, partitioning, and kernels are identical)"
            )
        if getattr(args, "locked_scheme", None):
            print(
                f"    NOTE all correctness runs used "
                f"scheme={args.locked_scheme} (iterative-cn hangs "
                "multi-process at this site)"
            )
        if getattr(args, "extra_run_env", None):
            print(f"    NOTE NCCL env locked: {args.extra_run_env}")
    if not diffs:
        print("    SKIPPED (no multi-device correctness runs)")
    for d in diffs:
        print(
            f"    {d['status']:4s} {d.get('label', ''):16s} "
            f"{d['a']} vs {d['b']}  "
            f"max_rel={d.get('max_rel_diff', d.get('error'))}"
        )
        if d["status"] != "PASS":
            code = 1

    print("\n[2] pallas vs dense-reference ratios (<1 = pallas better):")
    if not ratio_lines:
        print("    (no completed pallas+dense pairs)")
    for line in ratio_lines:
        print(f"    {line}")

    if timing:
        print("\n    production-size mpirun timing (s/t; lower = faster):")
        for r in timing:
            print(
                f"    {r['name']:40s} {r['status']:5s} "
                f"s/t={r.get('s_per_t')}  s/rhs={r.get('s_per_rhs')}"
            )
            if r["status"] not in ("ok",):
                code = 1

    print(
        "\n[3] split vs unsplit iterative-cn corrector "
        "(step.split_corrector A/B, same child; <1 = split faster):"
    )
    ab_rows = [
        r
        for r in single
        if r.get("status") == "ok"
        and r.get("step_ms", {}).get("icn_split")
        and r.get("step_ms", {}).get("icn_unsplit")
        and r["config"].get("bm0") is None
    ]
    if not ab_rows:
        print("    (no split-corrector measurements)")
    for r in ab_rows:
        sm = r["step_ms"]
        ratio = sm["icn_split"] / sm["icn_unsplit"]
        print(
            f"    {r['entry']:34s} split {sm['icn_split']:9.2f} ms  "
            f"unsplit {sm['icn_unsplit']:9.2f} ms  x{ratio:5.2f}  c "
            f"{sm.get('icn_split_correctors')}/"
            f"{sm.get('icn_unsplit_correctors')}  "
            f"dt {r['config'].get('dt')}"
        )

    print("\n[4] stability notices under backend=pallas:")
    notices = [
        f"{r.get('entry', r.get('name'))}: {line}"
        for r in single + mpi_recs
        for line in r.get("pallas_lines", [])
        if "ill-conditioned" in line
    ]
    if notices:
        # A notice is proceed-semantics, but at benchmark sizes none
        # should fire -- flag it for a look.
        code = 1
        for line in notices:
            print(f"    NOTICE {line}")
    else:
        print("    none observed -> PASS")

    crashed = [
        r for r in single + mpi_recs if r.get("status") in ("crash", "fail")
    ]
    if crashed:
        code = 1
        print("\ncrashed/failed runs:")
        for r in crashed:
            print(
                f"    {r.get('entry', r.get('name'))}: "
                f"{r.get('error', r.get('status'))}"
            )

    print(
        "\n(No-pivot stability coverage across the supported config "
        "space: scripts/pivot_stability_survey.py.)"
    )
    if not diffs:
        # No multi-device parity data was produced: an all-green code
        # here means only "nothing measured failed".
        print(
            "\nOverall: INCOMPLETE -- multi-GPU parity was not "
            "measured by this run."
        )
        return max(code, 1)
    print(
        "\nOverall: "
        + ("PASS" if code == 0 else "FAIL")
        + " on the checks measurable here."
    )
    return code


# ── driver: modes ────────────────────────────────────────────────────


def _print_banner(env: dict, workdir: Path) -> None:
    print("=" * 78)
    print("dnsjax linear-solver backend benchmark")
    print(
        f"  jax {env.get('jax')}  jaxlib {env.get('jaxlib')}  "
        f"backend {env.get('backend')}"
    )
    for d, k in zip(
        env.get("devices", []), env.get("kinds", []), strict=False
    ):
        print(f"    {d}  kind={k}")
    print(f"  workdir {workdir}  (logs + results.jsonl inside)")
    print("=" * 78)


def _plan_only(args: argparse.Namespace) -> None:
    print("\nNo GPU backend detected -> printing the planned matrix only.")
    print("Run on the cluster, or use --cpu-bench / --cpu-smoke here.\n")
    for entry in _build_entries(args):
        est = _dense_gb(entry["system"], entry["nx"], entry["ny"], entry["nz"])
        skip = (
            "  (dense would be SKIPPED)" if est > args.dense_budget_gb else ""
        )
        print(
            f"  {entry['name']:28s} nx={entry['nx']:4d} ny={entry['ny']:4d}"
            f" nz={entry['nz']:4d}  dense~{est:7.1f} GB{skip}"
        )


def _cpu_smoke(args: argparse.Namespace, workdir: Path) -> int:
    """Laptop harness self-check: exercises every driver code path."""
    print("\n--cpu-smoke: harness self-check (tiny CPU configs)")
    ok, msg = _preflight_launcher(args, "cpu", workdir)
    if not ok:
        print(f"--cpu-smoke aborted -- {msg}")
        return 1
    failures: list[str] = []
    env_extra = {"JAX_PLATFORMS": "cpu"}
    entry = {
        "name": "smoke-pc",
        "system": "plane-couette",
        "nx": 8,
        "ny": 24,
        "nz": 8,
        "fd_order": 4,
    }
    args.steps, args.parity_steps, args.reps = 3, 3, 2
    recs = {}
    for backend in ("pallas", "dense"):
        # A small (1, 4) tile for the tiny smoke plane: at nx = 8 the
        # default (2, 32) tile pads the 28-mode plane to 256 slots and
        # the padding, not the banded storage, dominates the factor
        # bytes (the documented small-plane caveat) -- the
        # pallas-smaller-than-dense sanity check needs the true plane.
        rec = _spawn_child(
            entry,
            backend,
            args,
            workdir,
            env_extra,
            f"smoke-pc-{backend}",
            bm=(1, 4) if backend == "pallas" else None,
        )
        recs[backend] = rec
        print(f"  child {backend}: {rec['status']}")
        if rec["status"] != "ok":
            failures.append(f"child {backend}: {rec.get('error')}")
    if all(r["status"] == "ok" for r in recs.values()):
        fp = recs["pallas"]["factor_bytes"]["total"]
        fd = recs["dense"]["factor_bytes"]["total"]
        print(f"  factor bytes pallas {fp} < dense {fd}: {fp < fd}")
        if not fp < fd:
            failures.append("pallas factors not smaller than dense")
        e0 = recs["dense"]["parity"]["energy"]
        de = abs(recs["pallas"]["parity"]["energy"] - e0) / abs(e0)
        print(f"  parity delta pallas vs dense: {de:.2e}")
        if de > 1e-8:
            failures.append(f"parity delta {de:.2e} > 1e-8")
        sm = recs["pallas"]["step_ms"]
        if sm.get("icn_split") is None or sm.get("icn_unsplit") is None:
            failures.append("split-corrector A/B fields missing")
        else:
            print(
                f"  split-corrector A/B: split {sm['icn_split']} ms vs "
                f"unsplit {sm['icn_unsplit']} ms (correctors "
                f"{sm['icn_split_correctors']}/"
                f"{sm['icn_unsplit_correctors']})"
            )

    # mpirun pair + snapshot diff on CPU (validates the parsers and
    # the diff path against real output).
    base = {"nx": 6, "ny": 24, "nz": 8, "dt": 0.01, "tmax": 0.05}
    mpi_recs = []
    for name, backend, np1 in (
        ("corr-plane-couette-pallas-np1x1", "pallas", 1),
        ("corr-plane-couette-pallas-np1x2", "pallas", 2),
        ("corr-plane-couette-dense-np1x1", "dense", 1),
    ):
        run = {
            "name": name,
            "system": "plane-couette",
            "backend": backend,
            "np0": 1,
            "np1": np1,
            **base,
        }
        rec = _run_mpi(run, args, workdir, "cpu")
        mpi_recs.append(rec)
        print(
            f"  mpi {name}: {rec['status']} "
            f"(t_final={rec.get('t_final')}, s/t={rec.get('s_per_t')})"
        )
        if rec["status"] != "ok":
            failures.append(f"mpi {name}: {rec.get('error')}")
    ok_names = {r["name"] for r in mpi_recs if r["status"] == "ok"}
    recs_by = {r["name"]: r for r in mpi_recs}
    diffs = []
    for name, ref, label in _diff_pairs(ok_names):
        pa = _final_snapshot(recs_by[name]["rundir"])
        pb = _final_snapshot(recs_by[ref]["rundir"])
        if pa is None or pb is None:
            failures.append(f"diff {name}: missing snapshot")
            continue
        worst, per = _diff_snapshots(pa, pb, 3)
        status = "PASS" if worst <= args.mpi_parity_tol else "FAIL"
        diffs.append((label, name, ref, worst, status))
        print(f"  diff {label} {name} vs {ref}: {worst:.2e} -> {status}")
        if status != "PASS":
            failures.append(f"diff {name} vs {ref}: {worst:.2e}")
    if not diffs:
        failures.append("no snapshot diffs ran")

    print(
        "\n--cpu-smoke: "
        + ("PASS (harness ready for the cluster)" if not failures else "FAIL:")
    )
    for f in failures:
        print(f"  {f}")
    return 1 if failures else 0


def _cpu_bench(args: argparse.Namespace, workdir: Path) -> int:
    """CPU backend timing (the CPU-production cost per backend)."""
    print("\n--cpu-bench: CPU step/solve timing per backend")
    env_extra = {"JAX_PLATFORMS": "cpu"}
    entries = []
    for system in ("plane-couette", "pipe"):
        for tag, (nx, ny, nz) in (
            ("64", (64, 64, 64)),
            ("128", (128, 96, 128)),
        ):
            entries.append(
                {
                    "name": f"cpu-{system}-{tag}",
                    "system": system,
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "fd_order": 4,
                }
            )
    results = []
    total = len(entries) * len(BACKENDS)
    i = 0
    for entry in entries:
        for backend in BACKENDS:
            i += 1
            tag = f"{entry['name']}-{backend}"
            print(f"[{i:2d}/{total}] {tag} ...", flush=True)
            rec = _spawn_child(entry, backend, args, workdir, env_extra, tag)
            results.append(rec)
            if rec["status"] == "ok":
                sm = rec["step_ms"]
                uns = sm.get("icn_unsplit")
                uns_s = f"{uns:9.1f}" if uns is not None else f"{'-':>9s}"
                print(
                    f"          icn {sm['icn']:9.1f} ms  "
                    f"unsplit {uns_s} ms  "
                    f"cnab2 {sm['cnab2']:9.1f} ms",
                    flush=True,
                )
            else:
                print(f"          {rec['status']}: {rec.get('error')}")
    ratio_lines = _single_tables(results, args.parity_tol)
    print("\nCPU pallas/dense ratios (<1 = pallas faster on CPU):")
    for line in ratio_lines:
        print(f"  {line}")
    print(
        "\nCPU split vs unsplit iterative-cn corrector "
        "(step.split_corrector A/B; <1 = split faster):"
    )
    for r in results:
        sm = r.get("step_ms", {})
        if r.get("status") != "ok" or not sm.get("icn_split"):
            continue
        print(
            f"  {r['entry']:34s} split {sm['icn_split']:9.1f} ms  unsplit "
            f"{sm['icn_unsplit']:9.1f} ms  "
            f"x{sm['icn_split'] / sm['icn_unsplit']:5.2f}  c "
            f"{sm.get('icn_split_correctors')}/"
            f"{sm.get('icn_unsplit_correctors')}"
        )
    return 0 if all(r["status"] == "ok" for r in results) else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    # child flags
    ap.add_argument("--system", choices=sorted(SYS_ARGS))
    ap.add_argument("--backend", choices=BACKENDS, default="pallas")
    # Child JAX backend, set by the driver from the section's env
    # (JAX_PLATFORMS=cpu for the CPU sections, cuda otherwise).
    ap.add_argument(
        "--platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--fd-order", dest="fd_order", type=int, default=8)
    ap.add_argument("--bm0", type=int, default=None)
    ap.add_argument("--bm1", type=int, default=None)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="timed steps per scheme (default 20 GPU / 5 CPU)",
    )
    ap.add_argument(
        "--parity-steps",
        dest="parity_steps",
        type=int,
        default=None,
        help="steps before the parity read (default 10 GPU / 5 CPU)",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=None,
        help="distinct RHS per solve timing (default 6 GPU / 4 CPU)",
    )
    # driver flags
    # Driver entries need a SIZES row; systems outside it (e.g.
    # plane-poiseuille) stay reachable via ``--child --system``.
    ap.add_argument(
        "--systems", nargs="*", default=sorted(SIZES), choices=sorted(SIZES)
    )
    ap.add_argument(
        "--sizes",
        nargs="*",
        default=["small", "prod"],
        choices=["small", "prod"],
    )
    ap.add_argument("--max-gpus", dest="max_gpus", type=int, default=4)
    ap.add_argument("--gpu-id", dest="gpu_id", default="0")
    ap.add_argument(
        "--dense-budget-gb",
        dest="dense_budget_gb",
        type=float,
        default=32.0,
    )
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument(
        "--child-timeout", dest="child_timeout", type=float, default=1800.0
    )
    ap.add_argument(
        "--mpi-timeout", dest="mpi_timeout", type=float, default=1800.0
    )
    ap.add_argument(
        "--parity-tol", dest="parity_tol", type=float, default=1e-8
    )
    ap.add_argument(
        "--mpi-parity-tol",
        dest="mpi_parity_tol",
        type=float,
        default=5e-5,
        help="PASS threshold for cross-run snapshot diffs.  Runs whose "
        "mean corrector count differs (c/it, printed per run) "
        "legitimately diverge at the corrector_tolerance scale (1e-5):"
        " backend/device-count roundoff can flip a marginal "
        "convergence branch.  Raw diffs are always printed; same-c/it "
        "pairs typically agree to ~1e-13.",
    )
    ap.add_argument("--no-tile-sweep", action="store_true")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-mpi", action="store_true")
    ap.add_argument("--skip-mpi-timing", action="store_true")
    ap.add_argument(
        "--launcher",
        default="auto",
        choices=["auto", "mpirun", "srun"],
        help="multi-process launcher for the -m dnsjax runs "
        "(auto: mpirun when on PATH, else srun job steps)",
    )
    ap.add_argument(
        "--srun-args",
        dest="srun_args",
        default="",
        help="extra flags for each srun step, e.g. "
        '"--gpus-per-task=1" (site-specific binding/gres)',
    )
    ap.add_argument(
        "--gpu-mode",
        dest="gpu_mode",
        default="auto",
        choices=["auto", *GPU_MODES],
        help="per-rank GPU assignment for srun steps (see GPU_MODES); "
        "auto probes the modes with a 2-process NCCL collective and "
        "uses the first that works",
    )
    ap.add_argument(
        "--xla-variant",
        dest="xla_variant",
        default="default",
        choices=[v[0] for v in XLA_VARIANTS],
        help="force one XLA-flag variant onto every multi-rank run "
        "(see XLA_VARIANTS; all were exonerated for the first-step "
        "hang -- manual override only)",
    )
    ap.add_argument(
        "--launch-mode",
        dest="launch_mode",
        default="auto",
        choices=["auto", "mp", "sp"],
        help="multi-device run topology: mp = one process per device "
        "(production), sp = one process addressing all devices (the "
        "offline-test topology; bypasses cross-process NCCL); auto "
        "lets the preflight bisection decide",
    )
    ap.add_argument("--oversubscribe", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--cpu-bench", action="store_true")
    a = ap.parse_args()

    cpu_mode = a.cpu_smoke or a.cpu_bench
    if a.steps is None:
        a.steps = 5 if cpu_mode else 20
    if a.parity_steps is None:
        a.parity_steps = 5 if cpu_mode else 10
    if a.reps is None:
        a.reps = 4 if cpu_mode else 6
    # Locked by the preflight bisection (or forced via flags).
    a.extra_run_env = {}
    a.locked_scheme = None
    a.xla_flags_extra = dict(XLA_VARIANTS)[a.xla_variant]

    if a.child:
        if a.system is None:
            ap.error("--child requires --system")
        run_child(a)
        return

    # Duplicate-driver guard: `srun -n N python .../solver_benchmark.py`
    # would run N full drivers in parallel (interleaved output, N x
    # children contending for the same GPU and host memory -- the
    # observed cluster failure).  Only SLURM task 0 proceeds; launch the
    # driver as a single task (or a plain process inside the
    # allocation).  Children/steps are unaffected: subprocess children
    # inherit task 0's id, and srun steps get fresh per-step ids.
    slurm_procid = os.environ.get("SLURM_PROCID")
    if slurm_procid not in (None, "0"):
        print(
            f"solver_benchmark: surplus SLURM task {slurm_procid} "
            "exiting (the driver runs as a single task; use srun -n 1)."
        )
        return
    if os.environ.get("SLURM_NTASKS") not in (None, "1"):
        print(
            "solver_benchmark: launched with SLURM_NTASKS="
            f"{os.environ['SLURM_NTASKS']}; only task 0 runs the "
            "driver -- prefer a single-task launch."
        )
    a.launcher = _resolve_launcher(a.launcher)

    # Default the workdir into the *current* directory (the sbatch
    # submission dir on a shared FS), not /tmp: a node-local /tmp
    # vanishes with the job, taking the full run logs with it.
    workdir = a.workdir or (
        Path.cwd() / f"dnsjax_solver_bench_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    env = _probe_env()
    _print_banner(env, workdir)
    print(f"  launcher {a.launcher}")

    if a.cpu_smoke:
        sys.exit(_cpu_smoke(a, workdir))
    if a.cpu_bench:
        sys.exit(_cpu_bench(a, workdir))
    if env.get("backend") != "gpu":
        _plan_only(a)
        return

    single: list[dict] = []
    if not a.skip_single:
        single = _single_device_section(
            a, workdir, {"CUDA_VISIBLE_DEVICES": a.gpu_id}
        )

    mpi_recs: list[dict] = []
    diffs: list[dict] = []
    timing: list[dict] = []
    if not a.skip_mpi and a.max_gpus >= 2:
        ok, msg = _preflight_launcher(a, "cuda", workdir)
        if not ok:
            print(f"\nMULTI-DEVICE SECTION SKIPPED -- {msg}")
        else:
            mpi_recs, diffs = _mpi_correctness_section(a, workdir, "cuda")
            if not a.skip_mpi_timing:
                timing = _mpi_timing_section(a, workdir)

    ratio_lines = _single_tables(single, a.parity_tol) if single else []
    code = _verdict(ratio_lines, diffs, mpi_recs, single, timing, a)
    print(
        f"\nDone.  Paste the full stdout back.  Raw data: "
        f"{workdir}/results.jsonl"
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
