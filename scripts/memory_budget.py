#!/usr/bin/env python3
r"""Per-rank memory of one production run on candidate layouts, offline.

Before a run is submitted, the questions are how much memory each rank
needs on a given ``(np0, np1)`` grid, and whether a node full of such
ranks fits.  This script answers the data part with XLA's own buffer
assignment, the instrument the twin "Memory" notes use: the run's time
step is lowered **as the run calls it** (the ``fourier`` and ``flow``
singletons passed as arguments; a ``jit`` of the bound step would bake
them in as constants instead and measure a different program),
compiled, and read through ``memory_analysis()``.  Nothing executes,
so the figures are deterministic: no hardware, no timing noise.

A production problem does not fit on the machine that sizes it, so each
layout is compiled at a **reduced** problem, on that layout's own grid
of forced host devices, and extrapolated.  Every per-device array has
exactly one streamwise extent -- the stored ``kx`` modes in spectral
space, the oversampled ``x`` points in physical space -- and both are
proportional to ``nx``, so a per-device byte count is
``c + k * nx`` along ``nx``, with ``c`` the arrays that have no
streamwise axis (the dense wall-normal matrices, the wavenumber
vectors).  Each row therefore compiles **two** reduced problems, at one
and two ``kx`` modes per device (``nx = 2 np1`` and ``4 np1``; 16 and
32 when ``np1 = 1``), fits ``c`` and ``k``, and evaluates the fit at the
production ``nx``.  Scaling a single reduced problem instead would
scale ``c`` too, and overstate a layout with many ``kx`` modes per
device several-fold.

Every axis the grid splits keeps its production size, so every
divisibility pad and every FFT-friendly rounding is the production one
-- with one exception.  The reduced problems stay inside a memory
budget (``--child-field-mib``, the size of one reduced field; a few
GiB of peak per child at the default), and a layout with many ``kx``
modes per device exceeds it at the production wall-normal size.  Such a
layout is also reduced in ``ny`` (wall-bounded flows; to a multiple of
``np0``, so the reduced problem carries no wall-normal padding), and
scaled back by the larger of the two per-device ratios: physical rows,
padding included, and spectral rows.  The printed ``y`` column shows
the reduced size, so such rows can be told apart.

What is measured is the step program -- ``predict_and_fully_correct``,
or ``step_cnab2`` under ``step.scheme = "cnab2"`` -- per device:
arguments (the state and every operator, resident for the whole run),
temporaries, and outputs not aliased to an input.  What is *not*: each
process's own runtime (Python, jaxlib, the compiled programs, MPI
buffers).  The closing ``Peak host memory`` line of a small CPU run
measures that; pass it as ``--overhead-gib`` to compare a node's total
against ``--node-gib``.

Usage (from the repository root)::

    # the configuration of a parameters.toml, three layouts
    .venv/bin/python scripts/memory_budget.py --toml parameters.toml \
        --layouts 128x1 1x128 4x32 --chunks 1 2 \
        --ranks-per-node 128 --overhead-gib 0.6 --node-gib 230
    # command-line parameters instead of a TOML
    .venv/bin/python scripts/memory_budget.py --solver-args \
        "--phys.system plane-poiseuille --res.nx 1280 --res.ny 385 \
         --res.nz 320" --layouts 1x128

One subprocess per reduced problem (the singletons are built at
import, so a problem needs a process of its own), run one at a time;
each forces ``np0 * np1`` CPU devices.  Expect a minute or two per row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GIB = 2**30

#: The fewest wall-normal points a reduced problem keeps.
NY_MIN = 33

#: The reported quantities, as ``memory_analysis`` fields.
FIELDS = ("argument", "temp", "output", "alias")


def _layout(text: str) -> tuple[int, int]:
    a, b = text.lower().split("x")
    return int(a), int(b)


def _reduction(
    np0: int, np1: int, ny: int, nz: int, wall_bounded: bool, budget: int
) -> tuple[tuple[int, int], int]:
    """``((nx_lo, nx_hi), ny_red)`` for one layout.

    *budget* bounds one reduced field, ``nx_hi * ny_red * nz`` doubles,
    in bytes.  ``ny`` shrinks only when the production size would
    exceed it, to a multiple of ``np0`` and no fewer than
    :data:`NY_MIN` points.
    """
    nx_lo = 2 * np1 if np1 > 1 else 16
    nx_hi = 2 * nx_lo
    ny_red = ny
    if wall_bounded and nx_hi * ny * nz * 8 > budget:
        rows = max(budget // (nx_hi * nz * 8), NY_MIN)
        ny_red = min(
            ny, max(np0 * (rows // np0), np0 * math.ceil(NY_MIN / np0))
        )
    return (nx_lo, nx_hi), ny_red


def _production_nx(p, np0: int, np1: int) -> float:
    """The ``nx`` at which the fit reproduces the production device.

    The stored ``kx`` modes per device and the oversampled ``x`` points
    are both proportional to ``nx`` before divisibility padding and
    FFT-friendly rounding; whichever of the two the production layout
    inflates more decides (the conservative reading).
    """
    from dnsjax.parameters import PaddedResolution

    q = p.model_copy(deep=True)
    q.dist.np0, q.dist.np1 = np0, np1
    pr = PaddedResolution()
    pr.set_padded_resolution(q)
    kx_dev = math.ceil((p.res.nx // 2) / np1)
    return max(2 * np1 * kx_dev, 2 * pr.nx_padded / 3, p.res.nx)


def _production_padding(p, np0: int, np1: int, periodic: bool) -> str:
    """The production layout's pads, as its start-up lines report them."""
    from dnsjax.parameters import PaddedResolution, round_up_padded

    q = p.model_copy(deep=True)
    q.dist.np0, q.dist.np1 = np0, np1
    pr = PaddedResolution()
    pr.set_padded_resolution(q)
    y = "" if periodic else f"y +{round_up_padded(p.res.ny, np0) - p.res.ny}, "
    kz = round_up_padded(p.res.nz - 1, np0)
    kx = round_up_padded(p.res.nx // 2, np1)
    return f"  [{y}kz {kz}, kx {kx}, z_pad {pr.nz_padded}]"


def _y_scale(ny: int, ny_red: int, np0: int) -> float:
    """Production / reduced per-device wall-normal rows (the larger of
    the physical rows, padding included, and the spectral ones)."""
    if ny_red == ny:
        return 1.0
    phys = math.ceil(ny / np0) / math.ceil(ny_red / np0)
    return max(phys, ny / ny_red)


def _child(args: argparse.Namespace) -> int:
    """Compile the step at one reduced problem; print one JSON line."""
    from dnsjax.bootstrap import (
        configure_jax_platform,
        platform_from_argv,
        resolve_parameters,
    )
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    cli = shlex.split(args.solver_args) + [
        "--dist.np0",
        str(args.np0),
        "--dist.np1",
        str(args.np1),
        "--solver.rhs_transform_chunks",
        str(args.chunk),
    ]
    resolve_parameters(cli, toml_path=Path(args.toml) if args.toml else False)
    update_parameters(Parameters(res={"nx": args.nx_red, "ny": args.ny_red}))
    padded_res.set_padded_resolution(params)
    validate_parameters()
    configure_jax_platform(
        platform_from_argv(), double_precision=params.res.double_precision
    )

    import importlib

    import jax.numpy as jnp

    from dnsjax.flows.registry import spec_for

    spec = spec_for(params.phys.system)
    mod = importlib.import_module(spec.flow_module)
    state = mod.init_state()
    to_solver = getattr(mod, "to_solver_basis", None)
    if to_solver is not None:
        state = to_solver(state)

    def cells(fn):
        return dict(
            zip(
                fn.__code__.co_freevars,
                (c.cell_contents for c in fn.__closure__),
                strict=True,
            )
        )

    if params.step.scheme == "cnab2":
        c = cells(mod.step_cnab2)
        carry = jnp.zeros_like(state[: spec.n_components])
        lowered = c["_step_cnab2_jit"].lower(
            state, carry, c["fourier"], c["flow"]
        )
    else:
        c = cells(mod.predict_and_fully_correct)
        lowered = c["_predict_and_fully_correct_jit"].lower(
            state, c["fourier"], c["flow"]
        )
    m = lowered.compile().memory_analysis()
    print(
        "MEMORY_BUDGET "
        + json.dumps(
            {
                "argument": m.argument_size_in_bytes,
                "output": m.output_size_in_bytes,
                "alias": m.alias_size_in_bytes,
                "temp": m.temp_size_in_bytes,
            }
        ),
        flush=True,
    )
    return 0


def _run_child(
    args: argparse.Namespace,
    np0: int,
    np1: int,
    chunk: int,
    nx_red: int,
    ny_red: int,
) -> dict | None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--np0",
        str(np0),
        "--np1",
        str(np1),
        "--chunk",
        str(chunk),
        "--nx-red",
        str(nx_red),
        "--ny-red",
        str(ny_red),
        "--solver-args",
        args.solver_args,
    ]
    if args.toml:
        cmd += ["--toml", str(Path(args.toml).resolve())]
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={np0 * np1} "
        + env.get("XLA_FLAGS", "")
    ).strip()
    env["DNSJAX_QUIET_STARTUP"] = "1"
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=REPO
    )
    for line in proc.stdout.splitlines():
        if line.startswith("MEMORY_BUDGET "):
            return json.loads(line.split(" ", 1)[1])
    print(
        f"  {np0}x{np1} chunks {chunk} at nx {nx_red}, ny {ny_red} failed, "
        f"exit {proc.returncode}:"
    )
    for ln in (proc.stdout + proc.stderr).splitlines()[-12:]:
        print(f"    | {ln}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--toml", default=None, help="parameters.toml to read")
    ap.add_argument("--solver-args", default="", help="solver CLI flags")
    ap.add_argument("--layouts", nargs="+", default=["1x1"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[1])
    ap.add_argument("--ranks-per-node", type=int, default=None)
    ap.add_argument("--overhead-gib", type=float, default=0.0)
    ap.add_argument("--node-gib", type=float, default=None)
    ap.add_argument(
        "--child-field-mib",
        type=float,
        default=48.0,
        help="largest reduced field (MiB); bounds each child's memory",
    )
    # Child-process plumbing.
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--np0", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--np1", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--chunk", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--nx-red", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--ny-red", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(args)

    from dnsjax.bootstrap import resolve_parameters
    from dnsjax.flows.registry import periodic_systems, walled_systems
    from dnsjax.parameters import params

    resolve_parameters(
        shlex.split(args.solver_args),
        toml_path=Path(args.toml) if args.toml else False,
    )
    prod = params.model_copy(deep=True)
    wall = prod.phys.system in walled_systems
    periodic = prod.phys.system in periodic_systems
    ny, nz = prod.res.ny, prod.res.nz
    budget = int(args.child_field_mib * 2**20)
    print(
        f"{prod.phys.system}, {prod.res.nx} x {ny} x {nz}, fd_order "
        f"{prod.res.fd_order}, scheme {prod.step.scheme}, "
        f"{'double' if prod.res.double_precision else 'single'} precision"
    )
    print(
        "\nGiB per rank at production size: args = state + operators "
        "(resident),\ntemp = the step's peak transient, out = outputs "
        "not aliased to inputs,\nfixed = the part independent of nx "
        "(already in the others); y = the\nreduced wall-normal size when "
        "it is not the production one.\n"
    )
    head = (
        f"{'layout':>8} {'chunks':>6} {'args':>7} {'temp':>7} {'out':>6} "
        f"{'rank':>7} {'fixed':>6} {'node':>7}  {'y':>4}"
    )
    print(head)
    worst = 0
    for text in args.layouts:
        np0, np1 = _layout(text)
        ranks = np0 * np1
        per_node = min(args.ranks_per_node or ranks, ranks)
        (nx_lo, nx_hi), ny_red = _reduction(np0, np1, ny, nz, wall, budget)
        nx_eval = _production_nx(prod, np0, np1)
        ys = _y_scale(ny, ny_red, np0)
        pads = _production_padding(prod, np0, np1, periodic)
        for chunk in args.chunks:
            lo = _run_child(args, np0, np1, chunk, nx_lo, ny_red)
            hi = (
                _run_child(args, np0, np1, chunk, nx_hi, ny_red)
                if lo
                else None
            )
            if lo is None or hi is None:
                worst = 1
                continue
            pred = {}
            fixed = 0.0
            for key in FIELDS:
                k = (hi[key] - lo[key]) / (nx_hi - nx_lo)
                c = lo[key] - k * nx_lo
                pred[key] = (c + k * nx_eval) * ys / GIB
                if key in ("argument", "temp"):
                    fixed += c * ys / GIB
            out = pred["output"] - pred["alias"]
            rank = pred["argument"] + pred["temp"] + out
            node = per_node * (rank + args.overhead_gib)
            line = (
                f"{text:>8} {chunk:>6} {pred['argument']:7.3f} "
                f"{pred['temp']:7.3f} {out:6.3f} {rank:7.3f} {fixed:6.3f} "
                f"{node:7.1f}  {ny_red if ny_red != ny else '':>4}"
            )
            if args.node_gib is not None:
                line += "  fits" if node <= args.node_gib else "  OVER"
            print(line + pads, flush=True)
    note = (
        f"\nnode = {args.ranks_per_node or 'all'} ranks x (rank + "
        f"{args.overhead_gib:.2f} GiB per-process overhead)"
    )
    if args.node_gib is not None:
        note += f", against {args.node_gib:.0f} GiB"
    print(note + ".")
    return worst


if __name__ == "__main__":
    sys.exit(main())
