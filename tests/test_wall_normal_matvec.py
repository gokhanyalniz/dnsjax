r"""Step parity across ``solver.wall_normal_matvec``: GEMM vs stencil.

The as-run predictor-corrector step under the GEMM and the stencil must
land on the same state (the runner and the bound:
``tests/_wall_normal_matvec.py``).  One worker per case, covering every
core geometry's derivative call sites: plane Couette (Cartesian), the
pipe and Taylor-Couette (annular); then the call sites those three
cannot reach, the legacy primitive pass of each geometry
(``res.consistent_imm = False``: the ``_*_primitive_imm`` modules' own
``D1``/``D2``/``A_base`` stacks).

Run as a script::

    uv run python tests/test_wall_normal_matvec.py
"""

from __future__ import annotations

from _wall_normal_matvec import cli

CASES: dict[str, dict[str, object]] = {
    "plane-couette": {"phys": {"re": 400.0}, "geo": {"lx": 6.0, "lz": 3.0}},
    "pipe": {"phys": {"re": 1800.0}, "geo": {"lx": 5.0}},
    "taylor-couette": {
        "phys": {"re1": 100.0, "re2": 0.0},
        "geo": {"eta": 0.5},
    },
}

_LEGACY = {"consistent_imm": False}

# The call sites the three above do not reach.
CASES |= {
    "plane-couette-legacy": {
        **CASES["plane-couette"],
        "system": "plane-couette",
        "res": _LEGACY,
    },
    "pipe-legacy": {**CASES["pipe"], "system": "pipe", "res": _LEGACY},
    "taylor-couette-legacy": {
        **CASES["taylor-couette"],
        "system": "taylor-couette",
        "res": _LEGACY,
    },
}


if __name__ == "__main__":
    cli(CASES, __file__)
