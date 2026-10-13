r"""Pallas-backend validation and benchmark, with the lab's flows.

Runs ``scripts/solver_benchmark.py`` (its docstring is the manual) with
the lab's rows added to its tables: Dean and the two viscoelastic flows
(their physics, sizes, dense-size estimates and stepper factories), the
conformation solve ``Hc`` (6 components) as a further operator group,
viscoelastic Dean in the multi-device correctness matrix, and the
split-corrector stress entry ``dean-split-stress``.  The children
re-enter this script, so they see the same tables::

    .venv/bin/python lab/scripts/solver_benchmark.py --max-gpus 4
    .venv/bin/python lab/scripts/solver_benchmark.py --cpu-smoke

``dean-split-stress`` is total-field Dean at ``nz = 64``,
``dt = 0.15``, the regime of the :mod:`dnsjax.parameters` Design notes
("The split corrector stays opt-in"): there the corrector cost is
dominated by the instantaneous mean-flow coupling (``l_bf == L_mf``),
the unsplit corrector reaches its cap, and the split's FFT-refresh
savings show.  Every other entry runs the same split-vs-unsplit A/B at
the bench ``dt``, where both converge in about one iteration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "scripts" / "solver_benchmark.py"
_spec = importlib.util.spec_from_file_location("solver_benchmark", _CORE)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

bench.SELF = Path(__file__).resolve()
bench.SYS_ARGS.update(
    {
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
)
bench.SIZES.update(
    {
        "viscoelastic-dean": {"small": (32, 48, 32), "prod": (128, 96, 128)},
        "viscoelastic-pipe": {"small": (32, 48, 32), "prod": (128, 96, 128)},
    }
)
bench.EXTRA_ENTRIES.append(
    (
        "small",
        {
            "name": "dean-split-stress",
            "system": "dean",
            "nx": 64,
            "ny": 48,
            "nz": 64,
            "fd_order": 4,
            "dt": 0.15,
        },
    )
)
# Lk + the Hk components + the 6 Hc of the sPTT flows (the core table's
# comment says which Hk families each formulation carries).
bench.N_OPS.update({"dean": 3, "viscoelastic-dean": 9, "viscoelastic-pipe": 9})
bench.N_OPS_LEGACY.update(
    {"dean": 4, "viscoelastic-dean": 10, "viscoelastic-pipe": 10}
)
bench.STEPPERS.update(
    {
        "dean": "dnsjax.geometries.wall_bounded.annular:build_annular_stepper",
        "viscoelastic-dean": (
            "dnsjax_lab.geometries.wall_bounded.annular_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
        "viscoelastic-pipe": (
            "dnsjax_lab.geometries.wall_bounded.cylindrical_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
    }
)
bench.MPI_SYSTEMS.append("viscoelastic-dean")
# The conformation solve; unbuilt (``None``) at ``kappa = 0`` and in
# every Newtonian flow.
bench.OP_GROUPS["Hc"] = ("Hc_op", 6)

if __name__ == "__main__":
    bench.main()
