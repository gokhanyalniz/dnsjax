r"""CPU survey of the pallas backend's no-pivot stability checks.

The ``pallas`` backend factors each per-mode banded operator with a
**no-pivot** banded LU and verifies, per operator group (``Lk``
pressure Poisson, ``Hk`` velocity Helmholtz, ``Hc`` viscoelastic
conformation Helmholtz), that the factorisation is sound
(``solvers._build_pallas_operator``): genuine instability (LU element
growth above ``solvers._NO_PIVOT_GROWTH_TOL``, or a non-finite
factor/residual) is a hard ``RuntimeError`` at setup; a solve residual
above ``solver.pallas_stability_tol`` (default 1e-6) with benign
growth is mere ill-conditioning and prints a notice-and-proceed line.
This survey answers empirically whether any *real* configuration trips
the hard error, prints the notice, or even approaches the thresholds
-- the evidence backing the tolerance defaults.

It sweeps the supported configuration space (all six wall-bounded flow
systems x wall-normal grid types/stretching x ``fd_order`` x ``ny`` x
``dt`` x ``Re`` x Crank-Nicolson ``implicitness`` x near-zero-``k^2``
boxes x viscoelastic ``kappa``/``beta``/``delta``, plus seeded random
joint-corner samples) and records, for every operator group of every
configuration, the no-pivot residual and LU element growth.  The
implicit operators depend only on the grid, ``nu``,
``dt``/``implicitness``, and the mode wavenumbers -- not on the base
flow -- so tiny ``nx = 4``/``nz = 8`` mode planes with *large* ``lx``
(smallest nonzero ``k^2 ~ (2 pi / lx)^2``, the near-singular Poisson
worst case) cover the conditioning space cheaply.

Everything runs on CPU (``JAX_PLATFORMS=cpu``; no mpirun, no time
stepping): importing a flow module builds the operators and prints the
``[pallas] {group}: ...`` check lines, which the child captures and
parses; a setup ``RuntimeError`` is recorded as a ``stability-error``
result.  One subprocess per configuration (the geometry modules
capture the global singletons at import time).

Usage (from the repo root; ~150 configs, ~10-25 min)::

    uv run python scripts/pivot_stability_survey.py
    uv run python scripts/pivot_stability_survey.py --quick   # ~12
    uv run python scripts/pivot_stability_survey.py --jobs 8

Results stream as ``@@RESULT`` JSON lines (also written to ``--out``,
default ``pivot_survey_results.jsonl``) and end in an aggregation +
``VERDICT`` section.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
SELF = Path(__file__).resolve()
RESULT_TAG = "@@RESULT "

# Mirrors the ``solver.pallas_stability_tol`` default in
# ``parameters.py`` (the children run with the model default).
STABILITY_TOL = 1e-6
# Mirrors ``solvers._NO_PIVOT_GROWTH_TOL`` (the hard-error bound).
GROWTH_TOL = 1e3
# Residuals above tol/MARGIN (growths above GROWTH_TOL/MARGIN) are
# flagged as "near-tol" offenders even when no notice/error fired.
MARGIN = 10.0

SYSTEMS = [
    "plane-couette",
    "plane-poiseuille",
    "pipe",
    "taylor-couette",
    "dean",
    "viscoelastic-dean",
]

# Per-system baseline physics (the test-suite-standard values from
# tests/test_random_smoke.py; the base flow does not enter the banded
# operators, so only nu = f(Re) matters here).  taylor-couette derives
# ``re := re1``; viscoelastic-dean derives ``re := wi/el`` (beta /
# epsilon / kappa / delta get their update_parameters defaults 0.8 /
# 1e-3 / 5e-5 / 11 unless overridden).
BASELINES: dict[str, dict] = {
    "plane-couette": {"re": 330.0, "lx": 5.0, "lz": 5.0},
    "plane-poiseuille": {"re": 660.0, "lx": 5.0, "lz": 5.0},
    "pipe": {"re": 1800.0, "lx": 5.0},
    "taylor-couette": {"re1": 400.0, "re2": -400.0, "eta": 0.5, "lx": 5.0},
    "dean": {"re": 1000.0, "eta": 0.5, "lx": 5.0},
    "viscoelastic-dean": {"wi": 20.0, "el": 20.0, "lx": 5.0},
}

# Common numerics baseline.  nx = 4 / nz = 8 keep the mode plane tiny
# (per-mode conditioning depends on k^2, which lx/lz control; nz = 6
# would trip the 3/2-rule "difference cannot be odd" check).
COMMON = {
    "fd_order": 4,
    "ny": 64,
    "nx": 4,
    "nz": 8,
    "dt": 0.01,
    "implicitness": 0.5,
}

# Wall-normal grid pool per system: (grid_type, grid_stretch) with
# None = the resolved scheme default (full CGL for Cartesian/annular;
# for the pipe, half-CGL under the survey's fixed iterative-cn
# scheme), and an explicit "cgl" pipe entry for the rigged-CGL grid
# (the cnab2 default) so both radial grids stay surveyed.
GRID_POOL: dict[str, list[tuple[str | None, float | None]]] = {
    "plane-couette": [
        (None, None),
        ("tanh", 1.0),
        ("tanh", 1.5),
        ("tanh", 3.0),
    ],
    "plane-poiseuille": [
        (None, None),
        ("tanh", 1.0),
        ("tanh", 1.5),
        ("tanh", 3.0),
    ],
    "pipe": [(None, None), ("cgl", None), ("tanh", 1.5)],
    "taylor-couette": [(None, None), ("tanh", 1.5)],
    "dean": [(None, None), ("tanh", 1.5)],
    "viscoelastic-dean": [(None, None), ("tanh", 1.5)],
}

FD_ORDERS = [2, 4, 6, 8]
NYS = [24, 64, 128, 192]
DTS = [1e-4, 1e-3, 1e-2, 1e-1]
RES = [1e2, 1e3, 1e4, 1e5]
IMPLICITNESSES = [0.5, 1.0]
LXS = [5.0, 100.0, 1000.0]

# The two ``_build_pallas_operator`` check-line variants (solvers.py):
# healthy no-pivot LU, and the above-tolerance-residual
# ill-conditioning notice.  Both carry the residual and the LU element
# growth.
_PALLAS_LINE = re.compile(
    r"^\[pallas\] (?P<g>\w+): (?:"
    r"no-pivot banded LU \(residual (?P<ok>\S+), growth (?P<gok>\S+)\)"
    r"|residual (?P<bad>\S+) > tol \S+, growth (?P<gbad>\S+) benign"
    r")",
    re.M,
)


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


def _child_config(a: argparse.Namespace) -> dict:
    """The child's own view of its configuration (echoed in the
    result so a manually-run child is self-describing)."""
    keys = [
        "system",
        "grid_type",
        "grid_stretch",
        "fd_order",
        "ny",
        "nx",
        "nz",
        "lx",
        "lz",
        "dt",
        "re",
        "re1",
        "re2",
        "eta",
        "wi",
        "el",
        "beta",
        "epsilon",
        "kappa",
        "delta",
        "implicitness",
    ]
    return {k: getattr(a, k) for k in keys if getattr(a, k) is not None}


def run_child(a: argparse.Namespace) -> None:
    """Configure the singletons, import the flow, report the
    per-group no-pivot residuals and LU element growths."""
    import contextlib
    import io

    from dnsjax.parameters import (
        Parameters,
        configure_jax_platform,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    # CPU-only survey (no-pivot-LU conditioning; no time stepping, no
    # GPU kernels).  The parent also pins JAX_PLATFORMS=cpu in the child
    # env; this records it on params and sets x64 before any array.
    configure_jax_platform("cpu")

    cfg = _child_config(a)

    params.phys.system = a.system
    for field in ("re", "re1", "re2", "wi", "el", "beta", "epsilon", "kappa"):
        v = getattr(a, field)
        if v is not None:
            setattr(params.phys, field, v)
    for field in ("eta", "delta", "grid_stretch"):
        v = getattr(a, field)
        if v is not None:
            setattr(params.geo, field, v)
    params.geo.lx = a.lx
    if a.lz is not None:
        params.geo.lz = a.lz
    params.res.nx = a.nx
    params.res.ny = a.ny
    params.res.nz = a.nz
    params.res.fd_order = a.fd_order
    params.res.double_precision = True
    params.step.dt = a.dt
    params.step.implicitness = a.implicitness
    params.step.scheme = "iterative-cn"

    # grid_type and the backend must go through the layering call, not
    # direct ``params.*`` assignments: ``update_parameters`` re-resolves
    # both per-family defaults for any field not recorded in
    # ``_user_set_fields``, so a direct assignment is overwritten (a
    # "tanh" pool entry would silently resolve back to the
    # scheme-default CGL grid).
    explicit: dict = {"solver": {"backend": "pallas"}}
    if a.grid_type is not None:
        explicit["geo"] = {"grid_type": a.grid_type}

    try:
        update_parameters(Parameters(**explicit))
        padded_res.set_padded_resolution(params)
        validate_parameters()
    except (ValueError, RuntimeError) as e:
        print(
            RESULT_TAG
            + json.dumps(
                {"status": "invalid", "config": cfg, "error": str(e)}
            ),
            flush=True,
        )
        return

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            m = _import_flow(a.system)
    except RuntimeError as e:
        # The setup stability check hard-errored (genuine no-pivot
        # instability) -- the survey's primary offender signal.
        print(buf.getvalue(), end="")
        print(
            RESULT_TAG
            + json.dumps(
                {
                    "status": "stability-error",
                    "config": cfg,
                    "error": str(e),
                }
            ),
            flush=True,
        )
        return
    setup_text = buf.getvalue()
    if a.verbose:
        print(setup_text, end="")

    # Parse the per-group check lines printed at operator setup.
    groups: dict[str, dict | None] = {}
    for match in _PALLAS_LINE.finditer(setup_text):
        g = match.group("g")
        if match.group("ok") is not None:
            groups[g] = {
                "residual": float(match.group("ok")),
                "growth": float(match.group("gok")),
                "notice": False,
            }
        else:  # above-tol residual: ill-conditioning notice
            groups[g] = {
                "residual": float(match.group("bad")),
                "growth": float(match.group("gbad")),
                "notice": True,
            }

    flow = m.flow
    warn: list[str] = []
    for g, op in (
        ("Lk", flow.Lk_op),
        ("Hk", flow.Hk_op),
        ("Hc", getattr(flow, "Hc_op", None)),
    ):
        if op is None:
            if g == "Hc":
                groups.setdefault("Hc", None)  # kappa == 0: no group
            continue
        rec = groups.get(g)
        if rec is None:
            groups[g] = rec = {
                "residual": None,
                "growth": None,
                "notice": None,
            }
            warn.append(f"{g}: no [pallas] line captured")
        rec["op"] = type(op).__name__
    if a.system != "viscoelastic-dean":
        groups.pop("Hc", None)

    out = {"status": "ok", "config": cfg, "groups": groups}
    if warn:
        out["warn"] = warn
    print(RESULT_TAG + json.dumps(out), flush=True)


# ── sweep construction ───────────────────────────────────────────────


def _base(system: str) -> dict:
    cfg = dict(COMMON)
    cfg.update(BASELINES[system])
    cfg["system"] = system
    return cfg


def _set_re(cfg: dict, re_val: float) -> None:
    """Set the Reynolds number through each system's own knobs."""
    if cfg["system"] == "taylor-couette":
        cfg["re1"], cfg["re2"] = re_val, -re_val
    elif cfg["system"] == "viscoelastic-dean":
        cfg["wi"] = re_val * cfg["el"]  # re := wi / el
    else:
        cfg["re"] = re_val


def _set_grid(cfg: dict, grid: tuple[str | None, float | None]) -> None:
    gt, stretch = grid
    if gt is not None:
        cfg["grid_type"] = gt
    if stretch is not None:
        cfg["grid_stretch"] = stretch


def build_configs(quick: bool, corners: int) -> list[dict]:
    """The survey sweep: per-system baselines, one-factor sweeps, and
    seeded random joint-corner samples; deduplicated."""
    cfgs: list[dict] = []

    for s in SYSTEMS:
        cfgs.append(_base(s))
    if not quick:
        for s in SYSTEMS:
            for fd in FD_ORDERS:
                cfgs.append({**_base(s), "fd_order": fd})
            for ny in NYS:
                cfgs.append({**_base(s), "ny": ny})
            for dt in DTS:
                cfgs.append({**_base(s), "dt": dt})
            for re_val in RES:
                cfg = _base(s)
                _set_re(cfg, re_val)
                cfgs.append(cfg)
            for impl in IMPLICITNESSES:
                cfgs.append({**_base(s), "implicitness": impl})
            # Near-zero k^2: large box with the tiny nx = 4 plane, so
            # the smallest nonzero k probes the near-singular Poisson.
            for lx in LXS:
                cfgs.append({**_base(s), "lx": lx})
            for grid in GRID_POOL[s]:
                cfg = _base(s)
                _set_grid(cfg, grid)
                cfgs.append(cfg)
        for s in ("plane-couette", "plane-poiseuille"):
            cfgs.append({**_base(s), "lz": 1000.0})
        for s in ("taylor-couette", "dean"):
            for eta in (0.3, 0.9):
                cfgs.append({**_base(s), "eta": eta})
        for kappa in (0.0, 1e-3):
            cfgs.append({**_base("viscoelastic-dean"), "kappa": kappa})
        cfgs.append({**_base("viscoelastic-dean"), "beta": 0.2})
        cfgs.append({**_base("viscoelastic-dean"), "delta": 1.0})

    # Joint corners: deterministic random combinations of the extremes
    # (adverse stacks like dt = 0.1 x Re = 100 x implicitness = 1.0 x
    # fd_order = 8 x tanh@3.0 x ny = 192).
    rng = random.Random(0)
    n_corners = 4 if quick else corners
    for _ in range(n_corners):
        s = rng.choice(SYSTEMS)
        cfg = _base(s)
        cfg["fd_order"] = rng.choice(FD_ORDERS)
        cfg["ny"] = rng.choice(NYS)
        cfg["dt"] = rng.choice(DTS)
        _set_re(cfg, rng.choice(RES))
        cfg["implicitness"] = rng.choice(IMPLICITNESSES)
        cfg["lx"] = rng.choice(LXS)
        _set_grid(cfg, rng.choice(GRID_POOL[s]))
        if cfg.get("grid_type") == "tanh":
            cfg["grid_stretch"] = rng.choice([1.0, 1.5, 3.0])
        if s in ("taylor-couette", "dean"):
            cfg["eta"] = rng.choice([0.3, 0.5, 0.9])
        if s == "viscoelastic-dean":
            cfg["kappa"] = rng.choice([0.0, 5e-5, 1e-3])
            cfg["beta"] = rng.choice([0.2, 0.8])
            cfg["delta"] = rng.choice([1.0, 11.0])
        cfgs.append(cfg)

    seen: set[str] = set()
    unique: list[dict] = []
    for cfg in cfgs:
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


# ── driver mode ──────────────────────────────────────────────────────


def _cfg_label(cfg: dict) -> str:
    """Compact one-line label of a config's non-baseline knobs."""
    parts = [cfg["system"]]
    base = _base(cfg["system"])
    for k in sorted(cfg):
        if k == "system":
            continue
        if cfg[k] != base.get(k):
            parts.append(f"{k}={cfg[k]}")
    return " ".join(parts) if len(parts) > 1 else parts[0] + " (baseline)"


def _spawn(cfg: dict, timeout: float) -> dict:
    """Run one child subprocess; return the parsed result record."""
    cmd = [str(PY), str(SELF), "--child"]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        return {"status": "crash", "config": cfg, "error": "timeout"}
    elapsed = time.perf_counter() - t0
    result: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            result = json.loads(line[len(RESULT_TAG) :])
    if result is None:
        return {
            "status": "crash",
            "config": cfg,
            "error": f"exit {proc.returncode}, no result line",
            "stderr_tail": proc.stderr[-1500:],
        }
    result["config"] = cfg  # driver's canonical view
    result["elapsed_s"] = round(elapsed, 2)
    return result


def _fmt_res(r: float | None) -> str:
    return "n/a" if r is None else f"{r:.2e}"


def _aggregate(results: list[dict], out_path: Path) -> int:
    """Print the aggregation + VERDICT; return the exit code."""
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    oks = [r for r in results if r["status"] == "ok"]
    invalid = [r for r in results if r["status"] == "invalid"]
    crashes = [r for r in results if r["status"] == "crash"]
    stab_errors = [r for r in results if r["status"] == "stability-error"]

    # (residual, growth, group, label, config) per group record.
    rows: list[tuple[float, float, str, str, dict]] = []
    offenders: list[str] = []
    warns: list[str] = []
    for r in stab_errors:
        offenders.append(
            f"STABILITY-ERROR  {_cfg_label(r['config'])}: {r['error']}"
        )
    for r in oks:
        label = _cfg_label(r["config"])
        for g, rec in r["groups"].items():
            if rec is None:  # viscoelastic kappa = 0: no Hc group
                continue
            resid = rec.get("residual")
            growth = rec.get("growth")
            if rec.get("notice") or resid is None or not math.isfinite(resid):
                offenders.append(
                    f"NOTICE    {g:2s} residual={_fmt_res(resid)} "
                    f"growth={_fmt_res(growth)}  {label}"
                )
            elif (
                resid > STABILITY_TOL / MARGIN
                or (growth or 0.0) > GROWTH_TOL / MARGIN
            ):
                offenders.append(
                    f"NEAR-TOL  {g:2s} residual={resid:.2e} "
                    f"growth={_fmt_res(growth)}  {label}"
                )
            if resid is not None and math.isfinite(resid):
                rows.append((resid, growth or 0.0, g, label, r["config"]))
        for w in r.get("warn", []):
            warns.append(f"{label}: {w}")

    print("\n" + "=" * 72)
    print("AGGREGATION")
    print("=" * 72)
    print(
        f"configs: {len(results)} total = {len(oks)} ok + "
        f"{len(invalid)} invalid + {len(stab_errors)} stability-error "
        f"+ {len(crashes)} crashed; group records: {len(rows)}"
    )

    if rows:
        print("\nmax no-pivot residual per (system, group):")
        per_sys: dict[tuple[str, str], tuple[float, float, str]] = {}
        for resid, growth, g, label, cfg in rows:
            key = (cfg["system"], g)
            if key not in per_sys or resid > per_sys[key][0]:
                per_sys[key] = (resid, growth, label)
        for (system, g), (resid, growth, label) in sorted(per_sys.items()):
            print(
                f"  {system:18s} {g:2s}  {resid:.2e} "
                f"(growth {growth:.1e})  at: {label}"
            )

        print("\ntop 10 residuals overall:")
        for resid, growth, g, label, _cfg in sorted(rows, reverse=True)[:10]:
            print(f"  {resid:.2e}  growth {growth:.1e}  {g:2s}  {label}")

        g_max, g_grp, g_label = max(
            (growth, g, label) for _r, growth, g, label, _c in rows
        )
        print(
            f"\nmax LU element growth: {g_max:.1e} ({g_grp}, {g_label}); "
            f"hard-error bound {GROWTH_TOL:.0e}"
        )

    if invalid:
        print("\ninvalid configs (rejected by parameter validation):")
        for r in invalid:
            print(f"  {_cfg_label(r['config'])}: {r['error']}")
    if crashes:
        print("\ncrashed configs:")
        for r in crashes:
            print(f"  {_cfg_label(r['config'])}: {r['error']}")
    if warns:
        print("\nwarnings:")
        for w in warns:
            print(f"  {w}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    code = 0
    if offenders:
        print(
            f"{len(offenders)} group record(s) at/near the stability "
            f"thresholds (residual tol {STABILITY_TOL:.0e}, growth "
            f"bound {GROWTH_TOL:.0e}, margin {MARGIN:g}x):"
        )
        for line in offenders:
            print(f"  {line}")
        code = 1
    elif rows:
        worst, growth, g, label, _cfg = max(rows)
        g_max = max(gr for _r, gr, _g, _l, _c in rows)
        print(
            "NO STABILITY ERROR OR NOTICE TRIGGERED across "
            f"{len(oks)} configurations.\n"
            f"Max no-pivot residual {worst:.2e} ({g}, {label}); "
            f"margin to the {STABILITY_TOL:.0e} notice threshold: "
            f"{STABILITY_TOL / worst:.1e}x.\n"
            f"Max LU element growth {g_max:.1e}; margin to the "
            f"{GROWTH_TOL:.0e} hard-error bound: {GROWTH_TOL / g_max:.1e}x."
        )
    else:
        print("No successful configurations -- survey inconclusive.")
        code = 1
    if crashes:
        print(f"({len(crashes)} crashed config(s) -- see above.)")
        code = 1
    print(f"\nresults written to {out_path}")
    return code


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    # child-mode config flags
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--grid-type", dest="grid_type", default=None)
    ap.add_argument("--grid-stretch", dest="grid_stretch", type=float)
    ap.add_argument("--fd-order", dest="fd_order", type=int, default=4)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--lx", type=float, default=5.0)
    ap.add_argument("--lz", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--re", type=float, default=None)
    ap.add_argument("--re1", type=float, default=None)
    ap.add_argument("--re2", type=float, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--wi", type=float, default=None)
    ap.add_argument("--el", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument(
        "--implicitness",
        type=float,
        default=0.5,
        help="Crank-Nicolson weight c (enters the Hk/Hc diagonal)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="child: re-emit the captured setup text",
    )
    # driver-mode flags
    ap.add_argument(
        "--quick",
        action="store_true",
        help="baselines + a few corners only (~12 configs)",
    )
    ap.add_argument(
        "--corners",
        type=int,
        default=30,
        help="number of random joint-corner samples",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel child subprocesses",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-child timeout in seconds",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("pivot_survey_results.jsonl"),
        help="JSONL results file",
    )
    a = ap.parse_args()

    if a.child:
        if a.system is None:
            ap.error("--child requires --system")
        run_child(a)
        return

    # Duplicate-driver guard (as in solver_benchmark.py): under
    # `srun -n N python .../pivot_stability_survey.py` every task would
    # run the full survey; only SLURM task 0 proceeds.
    slurm_procid = os.environ.get("SLURM_PROCID")
    if slurm_procid not in (None, "0"):
        print(
            f"pivot_stability_survey: surplus SLURM task {slurm_procid} "
            "exiting (the driver runs as a single task; use srun -n 1)."
        )
        return

    configs = build_configs(a.quick, a.corners)
    print("=" * 72)
    print("Pivot-stability survey (no-pivot banded LU checks)")
    print(
        f"  {len(configs)} configs x 1 subprocess each, CPU only, "
        f"{a.jobs} parallel jobs"
    )
    print(
        f"  notice threshold: pallas_stability_tol = "
        f"{STABILITY_TOL:.0e} (model default); hard-error growth "
        f"bound: {GROWTH_TOL:.0e}"
    )
    print("=" * 72)

    t0 = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(_spawn, cfg, a.timeout) for cfg in configs]
        for i, fut in enumerate(futures, 1):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                bits = []
                for g, rec in r["groups"].items():
                    if rec is None:
                        continue
                    mark = "NOTICE" if rec.get("notice") else ""
                    bits.append(f"{g}={_fmt_res(rec.get('residual'))}{mark}")
                detail = "  ".join(bits)
            else:
                detail = r["status"].upper()
            print(
                f"[{i:3d}/{len(configs)}] {detail:44s} "
                f"{_cfg_label(r['config'])}",
                flush=True,
            )
    print(f"\nsurvey wall time: {time.perf_counter() - t0:.1f}s")

    sys.exit(_aggregate(results, a.out))


if __name__ == "__main__":
    main()
