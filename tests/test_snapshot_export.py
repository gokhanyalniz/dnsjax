#!/usr/bin/env python3
r"""Tests for the JAX-free snapshot analysis API (``dnsjax.analysis``).

Validates :func:`dnsjax.analysis.read_state` (spectral<->physical
transforms, the snapshot-native layout / component basis, the
coordinate tuples, component and wall-normal subsetting, and
object-like params/stats access) plus the field operators: the
public ``derivative``/``gradient`` wrappers (axis wiring, ``ik``
scaling, plain ``D1``, the pipe parity requirement), ``curl`` and
``divergence`` against the solver's own operators node-for-node, and
``integrate`` -- all **without importing JAX in the test process**, so
the JAX-free guarantee is asserted directly, at import and again after
the reads (a flow's ``analysis_metric`` module is imported on its first
read).  The machinery: ``tests/_snapshot_export.py``; the lab's flows
have cases of their own, ``lab/tests/test_snapshot_export.py``.

Fixtures are generated in JAX subprocesses (forced CPU devices, one per
flow system, no MPI), mirroring ``tests/test_localized_rolls.py``: each
writes ``state.tar`` (a random divergence-free IC) and, for the
wall-bounded families, ``omega.tar`` -- dnsjax's own ``_curl_fn`` of
that state, saved raw in the ``(.,r,θ)`` / ``(x,y,z)`` basis -- so the
analysis ``curl`` can be checked against the solver node-for-node.

The pipe is generated twice, at azimuthal wedges ``m0 = 1`` and
``m0 = 2``.  The radial parity class is set by the **physical**
azimuthal wavenumber ``m = m0 * h``, which for every *odd* ``m0``
(the full circle included) agrees with the harmonic index ``h`` -- so
only an even wedge can catch a selector keyed on ``h``.

``div_true.npy`` does the same for ``divergence``: the ground truth is
taken on ``DIV_PROBE``, the IC rescaled per component to carry O(1)
divergence (why: its comment in ``tests/_snapshot_export.py``), in the
solver's decoupled ``u_pm`` basis (``solver_divergence`` there; why:
its docstring), so the check also pins the physical <-> u_pm
equivalence the stored basis relies on.

Run directly::

    uv run python tests/test_snapshot_export.py            # all systems
    uv run python tests/test_snapshot_export.py --system pipe
"""

from __future__ import annotations

import sys

from _snapshot_export import main, solver_divergence


def _cartesian(state, probe, outdir):
    from dnsjax.flows.wall_bounded.plane_couette import flow, get_stats
    from dnsjax.geometries.wall_bounded.cartesian import _curl_fn, fourier

    omega = _curl_fn(state, fourier, flow)
    return get_stats, omega, solver_divergence(probe, flow, fourier)


def _pipe(state, probe, outdir):
    from dnsjax.flows.wall_bounded.pipe import flow, get_stats, to_solver_basis
    from dnsjax.geometries.wall_bounded.cylindrical import _curl_fn, fourier

    omega = _curl_fn(state, fourier, flow)
    div = solver_divergence(to_solver_basis(probe), flow, fourier)
    return get_stats, omega, div


def _annular(state, probe, outdir):
    from dnsjax.flows.wall_bounded.taylor_couette import (
        flow,
        get_stats,
        to_solver_basis,
    )
    from dnsjax.geometries.wall_bounded.annular import _curl_fn, fourier

    omega = _curl_fn(state, fourier, flow)
    div = solver_divergence(to_solver_basis(probe), flow, fourier)
    return get_stats, omega, div


def _periodic(state, probe, outdir):
    # Curl is pure Fourier: no ground-truth dump.
    from dnsjax.flows.triply_periodic.monochromatic import get_stats

    return get_stats, None, None


# Both basis paths and the pipe's parity, the pipe at m0 = 1 and 2 (why
# both: the module docstring).
CASES: list[dict] = [
    {"system": "plane-couette", "fixtures": _cartesian},
    {"system": "pipe", "fixtures": _pipe},
    {"system": "pipe", "m0": 2, "fixtures": _pipe},
    {
        "system": "taylor-couette",
        "phys": {"re1": 100.0, "re2": -100.0},  # re derives from re1
        "geo": {"eta": 0.5},
        "fixtures": _annular,
    },
    {"system": "kolmogorov", "fixtures": _periodic},
]


if __name__ == "__main__":
    sys.exit(main(__file__, CASES))
