r"""No host array goes through JAX's cross-process check, for the lab.

The lab's cases of ``tests/test_host_placement.py`` (the machinery and
the contract: ``tests/_host_placement.py``), each on two ranks:
Cartesian with ``[probes]``, with ``[force]`` (its per-kick columns),
and as the parent of a twin member (``dnsjax-twin``, its streams and
reduced differences on); the pipe with ``[force]`` (its mode columns);
the curved pipe; Dean and the quasi-Keplerian wedge; and both
viscoelastic flows.

Run as a script (``--only`` takes case names; a ``-force`` or
``-twin`` case brings the case it reads from along)::

    uv run python lab/tests/test_host_placement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
import numpy as np
from _host_placement import (
    CART,
    COMMON,
    CYL,
    FIXED_DT,
    LOWRES,
    PIPE,
    POISEUILLE,
    RUN,
    from_snapshot,
    main,
)

_PROBES = ["--probes.modes", "1,1", "--probes.it_probes", "1"]
_FORCE = [
    "--force.modes",
    "1,1",
    "--force.amplitude",
    "1e-4",
    "--force.it_force",
    "2",
    "--force.seed",
    "1",
    *_PROBES,
]
_TWIN = [
    "--twin.e0",
    "1e-6",
    "--twin.seed",
    "3",
    "--twin.it_spectra",
    "1",
    "--twin.it_yspectra",
    "1",
    "--twin.it_ybudget",
    "1",
    "--twin.it_spectra3d",
    "1",
    "--twin.it_budget3d",
    "1",
    "--twin.it_lowres_delta",
    "1",
]


def _with_profiles(cwd: Path, args: list[str], source: Path, grid):
    """A ``-force`` case's profiles, on its source case's wall grid.

    Two channels of random 3-component profiles for mode ``(1, 1)``, on
    the grid ``dnsjax_lab.extensions.forcing`` requires.
    """
    if not grid:
        return f"no grid from case {source.name}"
    system = args[args.index("--phys.system") + 1]
    rng = np.random.default_rng(7)
    shape = (2, 3, len(grid))
    profiles = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    path = cwd / "profiles.npz"
    np.savez(
        path, system=system, code_grid=np.asarray(grid), profiles_1_1=profiles
    )
    return [*args, "--force.profiles", str(path)]


#: ``(name, entry point, arguments, source case or None, prepare or
#: None)``: a ``-force`` case writes its profiles on its source case's
#: wall-normal grid; the ``-twin`` case starts from its initial
#: snapshot.
CASES: list[tuple] = [
    (
        "cartesian",
        "dnsjax",
        [*COMMON, *POISEUILLE, *CART, *_PROBES],
        None,
        None,
    ),
    (
        "cartesian-force",
        "dnsjax",
        [*COMMON, *POISEUILLE, *CART, *_FORCE],
        "cartesian",
        _with_profiles,
    ),
    (
        "cartesian-twin",
        "dnsjax-twin",
        [*RUN, *FIXED_DT, *_TWIN, *LOWRES],
        "cartesian",
        from_snapshot,
    ),
    ("pipe", "dnsjax", [*COMMON, *PIPE, *CYL], None, None),
    (
        "pipe-force",
        "dnsjax",
        [*COMMON, *PIPE, *CYL, *_FORCE],
        "pipe",
        _with_profiles,
    ),
    (
        "curved-pipe",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "curved-pipe",
            "--phys.re",
            "1800",
            "--geo.curvature",
            "0.3",
            "--geo.lz",
            "5",
            *CYL,
        ],
        None,
        None,
    ),
    (
        "dean",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            *CYL,
        ],
        None,
        None,
    ),
    (
        "quasi-keplerian-wedge",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "100",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
            "--geo.m0",
            "2",
            *CYL,
        ],
        None,
        None,
    ),
    (
        "viscoelastic-pipe",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "viscoelastic-pipe",
            "--phys.wi",
            "20",
            "--phys.el",
            "0.02",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
            *CYL,
        ],
        None,
        None,
    ),
    (
        "viscoelastic-dean",
        "dnsjax",
        [
            *COMMON,
            "--phys.system",
            "viscoelastic-dean",
            "--phys.wi",
            "20",
            "--phys.el",
            "20",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
            *CYL,
        ],
        None,
        None,
    ),
]

#: The module behind each entry point (``pyproject.toml``).
ENTRIES = {
    "dnsjax": "dnsjax.__main__",
    "dnsjax-twin": "dnsjax_lab.twin.driver",
}


if __name__ == "__main__":
    sys.exit(main(__file__, CASES, ENTRIES, __doc__.splitlines()[0]))
