r"""Linear-solver backend bake-off + multi-GPU Pallas validation.

Produces the data gating the SPIKE (``banded``) backend's retirement
(see the pivot-stability companion,
``scripts/pivot_stability_survey.py``):

A.  **Single-device matrix** (one subprocess per config x backend --
    the singletons capture the backend at import): per operator group
    (``Lk``/``Hk``/``Hc``) the operator class actually built (detects
    a silent SPIKE fallback), exact persistent factor bytes, device
    memory after setup and peak after stepping, isolated ``.solve``
    times, and per-step times for **both** schemes
    (``predict_and_fully_correct`` and ``step_cnab2``), plus a
    fixed-seed parity scalar (perturbation energy + ``get_stats``
    after a few steps) compared across backends.
B.  **Multi-GPU section** (``mpirun ... -m dnsjax`` production runs
    from scratch dirs): the first real multi-GPU execution of the
    Pallas Triton kernel -- correctness via JAX-free
    ``dnsjax.analysis`` snapshot diffs across device counts and
    backends (including a padding-inducing ``nx = 34`` plane, a
    ``2 x 2`` mesh, and a ``1 x 4`` case), and production-size
    timing runs parsed from the ``__main__`` benchmark summary.
C.  **CPU bench** (``--cpu-bench``): the same child measurements with
    ``JAX_PLATFORMS=cpu`` on a reduced matrix -- what CPU production
    would pay per backend after retirement (SPIKE vs the pallas
    pure-JAX sweep vs dense).

The driver is JAX-free (children own the devices); every child /
mpirun stdout is logged under ``{workdir}/logs``, results stream as
``@@RESULT`` JSON lines (also ``{workdir}/results.jsonl``), and the
run ends in summary tables plus a ``VERDICT`` section mapping the
data onto the retirement gates.

Run **on the GPU cluster** (single node, >= 1 GPU; 4 for the full
mesh cases) and **paste the full stdout back**::

    .venv/bin/python scripts/solver_benchmark.py --max-gpus 4
    .venv/bin/python scripts/solver_benchmark.py --skip-mpi-timing

On a CPU node (or the dev laptop)::

    .venv/bin/python scripts/solver_benchmark.py --cpu-bench
    .venv/bin/python scripts/solver_benchmark.py --cpu-smoke  # harness

Without a GPU and without a ``--cpu-*`` flag it prints the environment
banner and the planned matrix (with dense-size estimates) and exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
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
}

# (nx, ny, nz) per system and size class.
SIZES: dict[str, dict[str, tuple[int, int, int]]] = {
    "plane-couette": {"small": (64, 64, 64), "prod": (256, 192, 256)},
    "pipe": {"small": (64, 48, 64), "prod": (256, 192, 256)},
    "taylor-couette": {"small": (64, 48, 64), "prod": (256, 192, 256)},
    "viscoelastic-dean": {"small": (32, 48, 32), "prod": (128, 96, 128)},
}

# Dense-backend operator count: Lk + Hk components (+ 6 Hc).
N_OPS = {
    "plane-couette": 2,
    "plane-poiseuille": 2,
    "pipe": 4,
    "taylor-couette": 4,
    "dean": 4,
    "viscoelastic-dean": 10,
}

BACKENDS = ("pallas", "banded", "dense")

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
PALLAS_PATTERN = re.compile(r"^\[pallas\] .*$", re.M)


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
    from dnsjax.parameters import (
        Parameters,
        configure_jax_platform,
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
    params.solver.backend = a.backend
    if a.bm0 is not None:
        params.solver.pallas_block_m0 = a.bm0
    if a.bm1 is not None:
        params.solver.pallas_block_m1 = a.bm1
    update_parameters(Parameters())
    padded_res.set_padded_resolution(params)
    validate_parameters()

    t0 = time.perf_counter()
    m = _import_flow(a.system)
    setup_s = time.perf_counter() - t0
    flow = m.flow

    import jax.numpy as jnp

    from dnsjax.random_field import generate_random_state
    from dnsjax.sharding import sharding

    dev = jax.local_devices()[0]
    env = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", "?"),
    }

    # Operator classes + exact persistent factor bytes; a
    # PerModeBandedOperator under backend=pallas is a (silent) SPIKE
    # fallback, recorded independently of the [pallas] stdout lines.
    groups = {"Lk": flow.Lk_op, "Hk": flow.Hk_op}
    if a.system == "viscoelastic-dean":
        groups["Hc"] = flow.Hc_op
    operators: dict[str, str | None] = {}
    factor_bytes: dict[str, int] = {}
    fallbacks: list[str] = []
    for g, op in groups.items():
        if op is None:  # viscoelastic kappa == 0: no Hc group
            operators[g] = None
            factor_bytes[g] = 0
            continue
        operators[g] = type(op).__name__
        factor_bytes[g] = int(
            sum(x.nbytes for x in jax.tree_util.tree_leaves(op))
        )
        if a.backend == "pallas" and operators[g] == "PerModeBandedOperator":
            fallbacks.append(g)
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
    s = jnp.copy(state)  # the step donates its argument
    for _ in range(a.parity_steps):
        s, _err, _c = m.predict_and_fully_correct(s)
    jax.block_until_ready(s)
    parity = {
        "steps": a.parity_steps,
        "energy": float(m.get_perturbation_energy(s)),
        "stats": {k: float(v) for k, v in m.get_stats(s).items()},
    }
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
        "fallbacks": fallbacks,
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
        str(entry.get("fd_order", 4)),
        "--dt",
        str(args.dt),
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
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = "TIMEOUT"
        rc = -1
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
        result = {
            "kind": "single",
            "status": "crash",
            "config": {
                "system": entry["system"],
                "backend": backend,
                **{k: entry[k] for k in ("nx", "ny", "nz")},
            },
            "error": f"exit {rc}, no result line",
            "stderr_tail": stderr[-1200:],
        }
    result["entry"] = tag
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    _record(workdir, result)
    return result


def _sys_cli_flags(system: str) -> list[str]:
    flags = ["--phys.system", system]
    for dotted, v in SYS_ARGS[system].items():
        flags += [f"--{dotted}", str(v)]
    return flags


def _run_mpi(
    run: dict, args: argparse.Namespace, workdir: Path, platform: str
) -> dict:
    """One ``mpirun ... -m dnsjax`` run in its own scratch dir."""
    n = run["np0"] * run["np1"]
    cmd = ["mpirun"]
    if args.oversubscribe or platform == "cpu":
        cmd.append("--oversubscribe")
    cmd += ["-np", str(n), str(PY), "-m", "dnsjax"]
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
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.mpi_timeout,
            cwd=rundir,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", "TIMEOUT", -1
    _log(
        workdir,
        run["name"],
        f"$ {' '.join(cmd)}\n\n{stdout}\n--- stderr ---\n{stderr}",
    )

    rec = {
        "kind": "mpi",
        "name": run["name"],
        "run": {k: v for k, v in run.items() if k != "extra"},
        "wall_s": round(time.perf_counter() - t0, 1),
        "status": "ok" if rc == 0 else "crash",
        "rundir": str(rundir),
    }
    if rc != 0:
        rec["error"] = f"exit {rc}"
        rec["stderr_tail"] = stderr[-1200:]
        _record(workdir, rec)
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
    rec["pallas_lines"] = PALLAS_PATTERN.findall(stdout)
    _record(workdir, rec)
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
            fb = f"  FALLBACK={rec['fallbacks']}" if rec["fallbacks"] else ""
            print(
                f"          icn {sm['icn']:9.2f} ms  cnab2 "
                f"{sm['cnab2']:9.2f} ms  factors "
                f"{rec['factor_bytes']['total'] / 2**20:8.1f} MB{fb}",
                flush=True,
            )
        else:
            print(f"          {rec['status']}: {rec.get('error')}", flush=True)
    return results


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
    base = {"nx": 34, "ny": 48, "nz": 32, "dt": 0.005, "tmax": 0.15}
    runs: list[dict] = []
    for system in systems:
        for backend, np1 in (
            ("pallas", 1),
            ("pallas", 2),
            ("banded", 1),
            ("banded", 2),
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
        # 2x2 mesh: plane-couette on a tanh grid (the documented
        # clean-ny-divisibility recipe for np0 > 1) with its own 1x1
        # tanh reference; pipe keeps its default (rigged-CGL) grid.
        tanh = (("--geo.grid_type", "tanh"),)
        if "plane-couette" in systems:
            runs.append(
                {
                    "name": "corr-plane-couette-pallas-np1x1-tanh",
                    "system": "plane-couette",
                    "backend": "pallas",
                    "np0": 1,
                    "np1": 1,
                    "extra": tanh,
                    **base,
                }
            )
            runs.append(
                {
                    "name": "corr-plane-couette-pallas-np2x2-tanh",
                    "system": "plane-couette",
                    "backend": "pallas",
                    "np0": 2,
                    "np1": 2,
                    "extra": tanh,
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
        elif rest == "-banded-np1x2":
            ref, label = f"corr-{system}-banded-np1x1", "device-count"
        elif rest == "-pallas-np1x1" or rest == "-banded-np1x1":
            ref, label = f"corr-{system}-dense-np1x1", "vs-dense"
        elif rest == "-pallas-np2x2":
            ref, label = f"corr-{system}-pallas-np1x1", "mesh-2x2"
        elif rest == "-pallas-np1x4":
            ref, label = f"corr-{system}-pallas-np1x1", "mesh-1x4"
        elif rest == "-pallas-np2x2-tanh":
            ref, label = f"corr-{system}-pallas-np1x1-tanh", "mesh-2x2"
        if ref in names:
            pairs.append((name, ref, label))
        # Cross-backend at np1 = 2 (both sharded).
        if rest == "-pallas-np1x2":
            other = f"corr-{system}-banded-np1x2"
            if other in names:
                pairs.append((name, other, "pallas-vs-banded"))
    return pairs


def _mpi_correctness_section(
    args: argparse.Namespace, workdir: Path, platform: str
) -> tuple[list[dict], list[dict]]:
    runs = _mpi_runs(args, platform)
    recs: dict[str, dict] = {}
    for i, run in enumerate(runs, 1):
        print(f"[mpi {i:2d}/{len(runs)}] {run['name']} ...", flush=True)
        rec = _run_mpi(run, args, workdir, platform)
        recs[run["name"]] = rec
        note = rec.get("error", "")
        print(f"          {rec['status']} {note}", flush=True)

    diffs: list[dict] = []
    ok_names = {n for n, r in recs.items() if r["status"] == "ok"}
    for name, ref, label in _diff_pairs(ok_names):
        pa = _final_snapshot(recs[name]["rundir"])
        pb = _final_snapshot(recs[ref]["rundir"])
        drec = {"kind": "diff", "a": name, "b": ref, "label": label}
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
        print(
            f"[diff] {label:16s} {name} vs {ref}: "
            f"{drec.get('max_rel_diff', drec.get('error'))} "
            f"-> {drec['status']}",
            flush=True,
        )
    return list(recs.values()), diffs


def _mpi_timing_section(args: argparse.Namespace, workdir: Path) -> list[dict]:
    runs: list[dict] = []
    for system in ("pipe", "plane-couette"):
        if system not in args.systems:
            continue
        nx, ny, nz = SIZES[system]["prod"]
        for backend in ("pallas", "banded"):
            for np1 in (1, 2):
                if np1 > args.max_gpus:
                    continue
                for scheme in ("iterative-cn", "cnab2"):
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
                        }
                    )
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
    """Human tables + the pallas/banded ratio lines for the verdict."""
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
        ref = oks.get("dense") or oks.get("banded")
        cfg = next(
            (r["config"] for r in rows if r["status"] == "ok"),
            rows[0]["config"],
        )
        print(
            f"\n{base}  (nx={cfg.get('nx')} ny={cfg.get('ny')} "
            f"nz={cfg.get('nz')})"
        )
        print(
            f"  {'backend':22s} {'icn ms':>10s} {'cnab2 ms':>10s} "
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
            print(
                f"  {name:22s} {sm['icn']:10.2f} {sm['cnab2']:10.2f} "
                f"{sv['Lk']:8.2f} {sv['Hk']:8.2f} {hc} "
                f"{_mb(r['factor_bytes']['total']):>10s} {peak_mb:>10s} "
                f"{delta:>9s}"
            )
            if r["fallbacks"]:
                print(f"  {'':22s} ^ SPIKE FALLBACK: {r['fallbacks']}")
        if "pallas" in oks and "banded" in oks:
            p, b = oks["pallas"], oks["banded"]
            r_icn = p["step_ms"]["icn"] / b["step_ms"]["icn"]
            r_ab = p["step_ms"]["cnab2"] / b["step_ms"]["cnab2"]
            r_fac = p["factor_bytes"]["total"] / b["factor_bytes"]["total"]
            pk_p, pk_b = p["mem"]["peak"], b["mem"]["peak"]
            r_peak = (
                pk_p["peak_bytes_in_use"] / pk_b["peak_bytes_in_use"]
                if pk_p["available"] and pk_b["available"]
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
) -> int:
    print("\n" + "=" * 78)
    print("VERDICT (retirement gates; see the plan)")
    print("=" * 78)
    code = 0

    print("\nG1  multi-GPU Pallas parity (snapshot diffs):")
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

    print("\nG2/G3  pallas vs banded ratios (<1 = pallas better):")
    if not ratio_lines:
        print("    (no completed pallas+banded pairs)")
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

    print("\nG5  SPIKE fallbacks under backend=pallas:")
    fb = [
        f"{r['entry']}: {r['fallbacks']}"
        for r in single
        if r.get("status") == "ok" and r.get("fallbacks")
    ]
    for r in mpi_recs:
        for line in r.get("pallas_lines", []):
            if "SPIKE" in line:
                fb.append(f"{r['name']}: {line}")
    if fb:
        code = 1
        for line in fb:
            print(f"    FALLBACK {line}")
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
        "\n(G4 = scripts/pivot_stability_survey.py; G6 = --cpu-bench "
        "data + user judgement.)"
    )
    print(
        "\nOverall: data "
        + ("SUPPORTS" if code == 0 else "DOES NOT (yet) SUPPORT")
        + " SPIKE retirement on the gates measurable here."
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

    # mpirun pair + snapshot diff on CPU (validates the parsers and
    # the diff path against real output).
    base = {"nx": 6, "ny": 24, "nz": 8, "dt": 0.01, "tmax": 0.05}
    mpi_recs = []
    for name, backend, np1 in (
        ("corr-plane-couette-pallas-np1x1", "pallas", 1),
        ("corr-plane-couette-pallas-np1x2", "pallas", 2),
        ("corr-plane-couette-banded-np1x2", "banded", 2),
        ("corr-plane-couette-banded-np1x1", "banded", 1),
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
    """CPU backend timing (the CPU-production question, gate G6)."""
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
                print(
                    f"          icn {sm['icn']:9.1f} ms  "
                    f"cnab2 {sm['cnab2']:9.1f} ms",
                    flush=True,
                )
            else:
                print(f"          {rec['status']}: {rec.get('error')}")
    ratio_lines = _single_tables(results, args.parity_tol)
    print("\nCPU pallas/banded ratios (>1 = SPIKE faster on CPU):")
    for line in ratio_lines:
        print(f"  {line}")
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
    ap.add_argument("--fd-order", dest="fd_order", type=int, default=4)
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
    ap.add_argument("--systems", nargs="*", default=sorted(SIZES))
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
        "--mpi-parity-tol", dest="mpi_parity_tol", type=float, default=1e-6
    )
    ap.add_argument("--no-tile-sweep", action="store_true")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-mpi", action="store_true")
    ap.add_argument("--skip-mpi-timing", action="store_true")
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

    if a.child:
        if a.system is None:
            ap.error("--child requires --system")
        run_child(a)
        return

    workdir = a.workdir or Path(
        tempfile.mkdtemp(prefix="dnsjax_solver_bench_")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    env = _probe_env()
    _print_banner(env, workdir)

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
        mpi_recs, diffs = _mpi_correctness_section(a, workdir, "cuda")
        if not a.skip_mpi_timing:
            timing = _mpi_timing_section(a, workdir)

    ratio_lines = _single_tables(single, a.parity_tol) if single else []
    code = _verdict(ratio_lines, diffs, mpi_recs, single, timing)
    print(
        f"\nDone.  Paste the full stdout back.  Raw data: "
        f"{workdir}/results.jsonl"
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
