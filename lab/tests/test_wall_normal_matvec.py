r"""Step parity across ``solver.wall_normal_matvec``, for the lab flows.

The lab's cases of ``tests/test_wall_normal_matvec.py`` (the runner and
the bound: ``tests/_wall_normal_matvec.py``): the curved pipe (its
continuity rows rebuild a parity-reduced ``D1`` from ``D1_pos`` and the
trimmed ghost), viscoelastic Dean (the component-leading 9-, 6- and
3-field tensor stacks) and the viscoelastic pipe (its parity-reduced
tensor stacks).

Run as a script::

    uv run python lab/tests/test_wall_normal_matvec.py
"""

from __future__ import annotations

import _repo  # noqa: F401 -- the core suite's helpers
from _wall_normal_matvec import cli

CASES: dict[str, dict[str, object]] = {
    "curved-pipe": {
        "phys": {"re": 1800.0},
        "geo": {"lx": 5.0, "curvature": 0.3},
    },
    "viscoelastic-dean": {
        "phys": {"wi": 20.0, "el": 20.0},
        "geo": {"lx": 5.0},
        "init": {"random_conformation_amplitude": 10.0},
    },
    "viscoelastic-pipe": {
        "phys": {"wi": 20.0, "el": 0.02},
        "geo": {"lx": 5.0},
        "init": {"random_conformation_amplitude": 10.0},
    },
}


if __name__ == "__main__":
    cli(CASES, __file__)
