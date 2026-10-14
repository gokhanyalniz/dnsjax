#!/usr/bin/env python3
r"""Which `$t^n$` work one iterative-CN step shares between its passes.

One iterative-CN step runs the predictor pass and the first correction
pass, and more correction passes inside a ``while_loop`` when the
corrector iterates.  Every pass builds the right-hand side
`$u^n/\Delta t + (1 - c)\,\nu L u^n + \ldots$`, whose `$u^n$` half does
not change within the step.  Outside the loop XLA shares that half
between the two passes only through common-subexpression elimination,
which fails wherever a step-invariant column is batched with a varying
one in a single GEMM; inside the loop, the same batching keeps
loop-invariant code motion from hoisting it.

This script counts the dense wall-normal GEMMs (``dot`` instructions of
the optimized CPU HLO, on the ``solver.wall_normal_matvec = "dense"``
path) of four programs on one flow -- the RHS alone, RHS + predictor,
two RHS + one correction, and the whole ``K = 1`` step -- and reports
how many GEMMs the step shares between its two passes.  A count says
where work is shared, not what it costs; time a change with an
interleaved A/B (``node_benchmark.py`` with two ``--exe``).

Usage (from the repository root)::

    uv run python lab/scripts/performance/step_sharing_census.py \
        --solver-args "--phys.system taylor-couette --geo.eta 0.5 \
        --phys.re1 400 --phys.re2 -400"

``--solver-args`` takes the solver's own flags (the flow's public names);
the default resolution is ``--res.nx 16 --res.nz 16 --res.ny 33`` in
internal names.  ``--stepping MODULE`` names the module whose
``_get_rhs`` / ``_predict`` / ``_correct`` to count instead of the
core geometry's (a lab flow's, e.g.
``dnsjax_lab.geometries.wall_bounded._viscoelastic_stepping``).

Measured at that default resolution: plane Couette shares 2 of the 6
GEMMs of a pass, the pipe 7 of 17, and Taylor-Couette none of 6 --
stage 1 of ``annular._imm_iteration_vw`` batches `$u_r^n$` with the
iterate's `$u_\theta$` in one ``A_base`` GEMM, which makes `$\varphi^n$`
and the whole stage-4 ``A_pair`` GEMM dataflow-varying.  Splitting that
batch in two shares 2 GEMMs (3 field-GEMMs) per step.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import sys

#: Each wall-bounded geometry's stepping module.
_STEPPING = {
    "cartesian": "dnsjax.geometries.wall_bounded.cartesian",
    "annular": "dnsjax.geometries.wall_bounded.annular",
    "cylindrical": "dnsjax.geometries.wall_bounded._cylindrical_stepping",
}


def _count_dots(hlo: str) -> int:
    """``dot`` instructions reachable from the entry computation.

    Fusions, loop bodies and called computations included: each
    computation's own ``dot`` lines plus its callees' (``calls=``,
    ``body=``, ``condition=``, ``to_apply=``).
    """
    comps: dict[str, dict] = {}
    cur = None
    for line in hlo.splitlines():
        head = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if head:
            cur = head.group(2)
            comps[cur] = {"dots": 0, "calls": [], "entry": bool(head.group(1))}
            continue
        if cur is None:
            continue
        if re.search(r"\bdot\(", line):
            comps[cur]["dots"] += 1
        for kw in ("calls=", "body=", "condition=", "to_apply="):
            comps[cur]["calls"] += re.findall(kw + r"%?([\w.\-]+)", line)

    def total(name: str) -> int:
        c = comps[name]
        return c["dots"] + sum(total(k) for k in c["calls"] if k in comps)

    return total(next(k for k, v in comps.items() if v["entry"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--solver-args",
        required=True,
        help="the solver's flags: the flow and its parameters",
    )
    ap.add_argument(
        "--stepping",
        default=None,
        help="module whose _get_rhs / _predict / _correct to count "
        "(default: the flow's core geometry)",
    )
    args = ap.parse_args(argv)

    from dnsjax.bootstrap import configure_jax_platform, resolve_parameters
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    cli = shlex.split(args.solver_args)
    resolve_parameters(cli, toml_path=False)
    res = {"double_precision": True}
    if not any(a.startswith("--res.") for a in cli):
        res |= {"nx": 16, "ny": 33, "nz": 16}
    update_parameters(
        Parameters(res=res, solver={"wall_normal_matvec": "dense"})
    )
    padded_res.set_padded_resolution(params)
    configure_jax_platform("cpu")

    import jax

    from dnsjax.flows.registry import import_flow_module, spec_for

    system = params.phys.system
    geometry = spec_for(system).geometry
    if args.stepping is None and geometry not in _STEPPING:
        raise SystemExit(
            f"{system!r} is not a wall-bounded flow of a core geometry; "
            "pass --stepping"
        )
    g = importlib.import_module(args.stepping or _STEPPING[geometry])
    m = import_flow_module(system)
    fourier, flow = m.fourier, m.flow
    s0 = m.init_state()
    if hasattr(m, "to_solver_basis"):
        s0 = m.to_solver_basis(s0)
    p0 = s0 * 1.0001

    def rhs(s, f, fl):
        return g._get_rhs(s, f, fl)

    def predict(s, f, fl):
        return g._predict(s, g._get_rhs(s, f, fl), f, fl)

    def correct(s, p, f, fl):
        r = g._get_rhs(s, f, fl)
        return g._correct(s, p, r, g._get_rhs(p, f, fl), f, fl)

    def step(s, f, fl):
        r = g._get_rhs(s, f, fl)
        p = g._predict(s, r, f, fl)
        return g._correct(s, p, r, g._get_rhs(p, f, fl), f, fl)

    def dots(fn, *a):
        return _count_dots(jax.jit(fn).lower(*a).compile().as_text())

    n_rhs = dots(rhs, s0, fourier, flow)
    n_pred = dots(predict, s0, fourier, flow) - n_rhs
    n_corr = dots(correct, s0, p0, fourier, flow) - 2 * n_rhs
    n_step = dots(step, s0, fourier, flow)
    unshared = 2 * n_rhs + n_pred + n_corr
    print(
        f"{system}: RHS {n_rhs} GEMMs; predictor pass {n_pred}, "
        f"correction pass {n_corr}; a K = 1 step {n_step} of "
        f"{unshared} unshared -> {unshared - n_step} shared"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
