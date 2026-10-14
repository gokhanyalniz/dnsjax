r"""No host array goes through JAX's cross-process equality check.

``jax.device_put`` of a host (NumPy) value onto a sharding that spans
several processes first checks that every process passed the same
value, by gathering all ``W`` copies onto every process
(``jax.experimental.multihost_utils.assert_equal``): each process
briefly holds about twice ``W`` times the array, so the cost grows
with the job, not with the problem.  On ARCHER2 that check, on the two
dense wall-normal derivative matrices alone, was the per-rank memory
peak of every 4-node layout (``sharding.Sharding.distribute`` has the
numbers).  Host arrays reach the mesh through ``distribute`` instead,
which builds each process's shards from its own copy.

Each case runs the solver on two ranks under a recorder of that check,
with a positive control (the machinery: ``tests/_host_placement.py``;
the lab's cases: ``lab/tests/test_host_placement.py``).

The cases build every core flow family and step it a few times:
Cartesian, resumed onto a changed ``ny`` (the regrid matrix), with
adaptive ``dt`` and ``[lowres]``, and in single precision; Kolmogorov
from localized rolls; the pipe, and with cnab2; Taylor-Couette, and on
its ``m0 = 2`` wedge.  In single precision the check cannot pass at
all: it compares the gathered copies, canonicalized to float32, with
the float64 host originals, so no float64 array survives it.

Run as a script (``--only`` takes case names; a ``-regrid`` case
brings the case it reads from along)::

    uv run python tests/test_host_placement.py
"""

from __future__ import annotations

import sys

from _host_placement import (
    CART,
    COMMON,
    CYL,
    LOWRES,
    PIPE,
    POISEUILLE,
    RUN,
    SEED,
    from_snapshot,
    main,
)

_TC = [
    "--phys.system",
    "taylor-couette",
    "--phys.re1",
    "400",
    "--phys.re2",
    "-400",
    "--geo.eta",
    "0.5",
    "--geo.lz",
    "5",
]

#: ``(name, entry point, arguments, source case or None, prepare or
#: None)``: the ``-regrid`` case starts from its source's initial
#: snapshot.
CASES: list[tuple] = [
    ("cartesian", "dnsjax", [*COMMON, *POISEUILLE, *CART], None, None),
    (
        "cartesian-regrid",
        "dnsjax",
        [
            *COMMON,
            *POISEUILLE,
            "--res.nx",
            "16",
            "--res.ny",
            "19",
            "--res.nz",
            "16",
        ],
        "cartesian",
        from_snapshot,
    ),
    (
        "cartesian-adaptive-lowres",
        "dnsjax",
        [
            *RUN,
            *SEED,
            "--step.adaptive",
            "True",
            "--step.dt",
            "0.005",
            "--step.dt_max",
            "0.01",
            *POISEUILLE,
            *CART,
            *LOWRES,
            "--lowres.pressure",
            "True",
        ],
        None,
        None,
    ),
    (
        "cartesian-single",
        "dnsjax",
        [*COMMON, *POISEUILLE, *CART, "--res.double_precision", "False"],
        None,
        None,
    ),
    (
        "kolmogorov-rolls",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "100",
            "--init.localized_rolls",
            "True",
            "--res.nx",
            "16",
            "--res.ny",
            "16",
            "--res.nz",
            "16",
        ],
        None,
        None,
    ),
    ("pipe", "dnsjax", [*COMMON, *PIPE, *CYL], None, None),
    (
        "pipe-cnab2",
        "dnsjax",
        [*COMMON, *PIPE, *CYL, "--step.scheme", "cnab2"],
        None,
        None,
    ),
    ("taylor-couette", "dnsjax", [*COMMON, *_TC, *CYL], None, None),
    (
        "taylor-couette-wedge",
        "dnsjax",
        [*COMMON, *_TC, "--geo.m0", "2", *CYL],
        None,
        None,
    ),
]

#: The module behind each entry point (``pyproject.toml``).
ENTRIES = {"dnsjax": "dnsjax.__main__"}


if __name__ == "__main__":
    sys.exit(main(__file__, CASES, ENTRIES, __doc__.splitlines()[0]))
