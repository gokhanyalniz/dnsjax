r"""Random-IC smoke tests for the lab's flows.

The lab's table of ``tests/test_random_smoke.py`` (the runner and the
success criteria: ``tests/_random_smoke.py``): every lab stepping
machinery -- the curved pipe, viscoelastic-pipe, dean,
viscoelastic-dean -- from a random divergence-free perturbation, at a
Reynolds number above the onset of transition, on a small domain at
low resolution, integrated a short time with no error, NaN, or
blow-up.  ``quasi-keplerian`` needs no entry of its own: it binds the
Taylor-Couette machinery the core table covers, and its parameter
derivation is ``test_quasi_keplerian.py``'s.

Three entries (``dean-cnab2``, ``viscoelastic-pipe-cnab2``,
``viscoelastic-dean-cnab2``) pass ``--step.scheme cnab2``:
``dean-cnab2`` covers the total-field ``_l_bf == 0`` corrector path;
the two viscoelastic entries the 9-component tensor coupling -- the
pipe's also with a nonzero moving frame, which the annular
``u_grid = 0`` default never exercises.

``dean-split-corrector`` covers the opt-in split corrector on the
force-driven Dean flow, where the coupling is the pure instantaneous
mean-flow term (``l_bf == L_mf``) -- the regime where the two corrector
structures most differ.

Two ``*-legacy-imm`` entries drive the retired primitive `$(v, p)$`
path through both 9-component viscoelastic compositions
(viscoelastic-dean inheriting the annular pass, viscoelastic-pipe the
cylindrical one, each exercised nowhere else) under iterative-cn.

Two ``*-pallas-mpi-pad`` entries repeat the core table's sharded,
padded shape for the banded operators its entries do not reach:
``viscoelastic-dean`` (the annular stacked ``Hk`` plus the conformation
Helmholtz operator ``Hc``, `$\kappa > 0$`) and ``viscoelastic-pipe``
(both of those at once -- every ``Hc`` band and every per-slot ghost
sign is a mode-sharded stacked array there).  A ``curved-pipe-mpi-pad``
entry adds the toroidal metric on that shape: its metric diagnostics
are jitted programs that a closed-over global array would fail on two
processes.

Usage::

    uv run python lab/tests/test_random_smoke.py
    uv run python lab/tests/test_random_smoke.py --np 2
"""

from __future__ import annotations

import _repo  # noqa: F401 -- the core suite's helpers
from _random_smoke import main

# ── configuration ────────────────────────────────────────────────────

# Reynolds numbers above the onset of (subcritical) transition: Dean
# force-driven at re = 1000, eta = 0.5; the azimuthal extent is forced
# to 2*pi by the code and the radius / gap is fixed by geometry, so
# only the axial lx is set.
SYSTEMS: list[dict] = [
    {
        # The toroidal pipe: the same cylindrical stepping with a
        # metric, driving the full curvature path -- the pointwise
        # metric factors in the RHS and the O(kappa) defect in the
        # continuity rows -- through the CLI on a nonlinear field.  A
        # *total*-field flow, so its ``E'`` is the k_s != 0 energy and
        # the run starts from the straight-pipe profile plus the random
        # perturbation.  Curvature 0.3 rather than the shipped 0.037:
        # the terms under test scale with it, and the corrector's
        # contraction (~kappa) is worth exercising away from zero.
        # Correctness lives in test_curved_pipe.py; this is the
        # integration gate.
        "name": "curved-pipe",
        "args": [
            "--phys.system",
            "curved-pipe",
            "--phys.re",
            "1800",
            "--geo.curvature",
            "0.3",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "dean",
        "args": [
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # Viscoelastic (sPTT) pipe flow: the axially driven 9-component
        # twin of viscoelastic-dean, and the only flow that drives the
        # coupled tensor path through the pipe's **parity-reduced**
        # radial operators -- so this is where a wrong per-slot
        # conformation parity sign would show up nonlinearly (the
        # laminar smoke's axisymmetric state cannot see it: the
        # near-axis ghost correction only bites on m != 0 content).
        # Total-field IC = analytical laminar pair + windowed,
        # axis-enveloped tensor noise.  Re = wi/el is derived; the
        # kappa > 0 default builds the conformation Helmholtz operator.
        # Wi = 20, El = 0.02 => Re = 1000 -- the flow's own default
        # regime, and close to the Newtonian ``pipe`` entry's Re = 1800,
        # so the two are comparable.  (The annular entries below run at
        # Wi = El = 20, Re = 1: that flow's subject is the inertialess
        # limit, not this one's.)
        "name": "viscoelastic-pipe",
        "args": [
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
        ],
    },
    {
        # The pipe spin quad carrying the 9-component total field,
        # with ``res.consistent_imm`` off, at the flow's default
        # Re = 1000 (the Newtonian ``pipe-legacy-imm`` entry runs
        # 1800).  A resolution-pinned smoke entry cannot detect
        # defects that appear only under wall-normal refinement at
        # fixed Re; those need a default-vs-legacy comparison at the
        # intended (Re, nr).
        "name": "viscoelastic-pipe-legacy-imm",
        "args": [
            "--phys.system",
            "viscoelastic-pipe",
            "--res.consistent_imm",
            "False",
            "--phys.wi",
            "20",
            "--phys.el",
            "0.02",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
        ],
    },
    {
        # Viscoelastic (sPTT) Dean flow: the 9-component state (3 velocity
        # + 6 conformation-tensor) driven through the full coupled
        # nonlinear path (advection Christoffels, stretching, relaxation,
        # polymer-stress divergence).  Total-field IC = analytical laminar
        # pair + windowed tensor noise (``random_conformation_amplitude``).
        # Re = wi/el is derived (no --phys.re).  The kappa > 0 default
        # also builds the conformation Helmholtz operator (``Hc_op``).
        # wi = el = 20 (Re = 1) and a reduced conformation-noise amplitude
        # (10, against the default 700) keep the coupled system stable at
        # the coarse smoke resolution -- the default wi = 105 with the
        # full noise is a genuinely stiff elastic instability on a 32^3
        # grid, not a solver defect.
        "name": "viscoelastic-dean",
        "args": [
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
        ],
    },
    {
        "name": "viscoelastic-dean-legacy-imm",
        "args": [
            "--phys.system",
            "viscoelastic-dean",
            "--res.consistent_imm",
            "False",
            "--phys.wi",
            "20",
            "--phys.el",
            "20",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
        ],
    },
    {
        # No dedicated cartesian / plain-annular pallas-pad entries:
        # solver.backend already *defaults* to pallas, so
        # plane-couette-mpi-pad above runs the cartesian banded
        # build/solve on the sharded padded axis at 32^3, and the
        # viscoelastic entry below covers the annular stacked-Hk (+
        # Hc) sharded build; single-device band-vs-dense parity is
        # unit-covered per geometry (test_cartesian / test_annular)
        # and the shard_map-local solve by test_banded_solver_sharded.
        # Pallas banded backend on the viscoelastic annular
        # (viscoelastic-dean) geometry, same sharded/padded mode-axis
        # guard (np1 = 2, nx // 2 = 3 padded to 4): builds the annular Lk
        # / 2x2-IMM Hk operators **and** the 6-component stacked
        # conformation Helmholtz operator ``Hc_op`` (kappa > 0) in banded
        # storage on the sharded axis, and solves the 9-component coupled
        # nonlinear path.  CPU runs the pure-JAX banded sweep (the Triton
        # kernel is GPU-only).  Same stable reduced parameters as the
        # single-device ``viscoelastic-dean`` entry.
        "name": "viscoelastic-dean-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "check_laminarization": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
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
            "--solver.backend",
            "pallas",
        ],
    },
    {
        # The same multi-device 9-component guard for the *pipe*, where
        # the per-slot parity masks additionally ride the sharded
        # azimuthal axis: every ``m_is_even``-selected band and every
        # per-slot ghost sign is a stacked, mode-sharded array here, so
        # this is the entry that would catch a cross-spec stack in the
        # conformation operators or the fused radial GEMM.
        "name": "viscoelastic-pipe-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "check_laminarization": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
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
            "--solver.backend",
            "pallas",
        ],
    },
    {
        # The curved pipe on the same sharded, padded shape: its basis
        # maps and diagnostics carry the metric `h`, a global array
        # that a jitted function must take as an argument -- closed
        # over, it traces on one process and is refused on two.  Under
        # constant bulk velocity, so the t = t0 driving inference runs
        # too.
        "name": "curved-pipe-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "check_laminarization": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
            "--phys.system",
            "curved-pipe",
            "--phys.re",
            "1800",
            "--geo.curvature",
            "0.3",
            "--phys.driving",
            "constant_bulk_velocity",
            "--geo.lz",
            "5",
        ],
    },
    {
        # CN/AB2 on Dean flow (annular, force-driven): the one
        # **total-field** flow (``base_flow = curl_base_flow = 0``, driven
        # by the azimuthal body force), so its ``_l_bf`` is identically
        # zero and the base-flow-coupling corrector is trivial -- a
        # distinct cnab2 code path exercised by no other entry.  Both
        # walls are stationary (``U -> 0`` at each), so the total-field
        # advection is not wall-stiff and cnab2 is bounded only by the
        # ordinary self-advection CFL (fine at ny=32, this Re).
        "name": "dean-cnab2",
        "res": {"nx": 32, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # CN/AB2 on the viscoelastic (sPTT) Dean flow: the 9-component
        # total-field tensor path.  Its ``_l_bf`` is the viscoelastic
        # coupling (velocity mean-flow + polymer-stress divergence;
        # conformation mean advection / stretching + linear relaxation),
        # all FFT-free, so the explicit AB2 remainder is the pure
        # fluctuation-fluctuation nonlinearity -- a distinct cnab2 code
        # path (9 components, elastic coupling) exercised by no other
        # entry.  Same stable reduced parameters as the iterative-cn
        # ``viscoelastic-dean`` entry; cnab2 matches iterative-cn to
        # O(dt^2) at ~1 FFT/step vs its ~4.
        # CN/AB2 on the viscoelastic pipe: the FFT-free ``_l_bf``
        # here carries the pipe's mean-flow coupling, the polymer
        # divergence through the parity-reduced radial operators, and
        # the moving-frame term (u_grid = 1/2 by default, unlike the
        # annular twin) -- and the AB2 remainder must stay the pure
        # fluctuation-fluctuation nonlinearity.
        "name": "viscoelastic-pipe-cnab2",
        "res": {"nx": 32, "ny": 32, "nz": 32},
        "args": [
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
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        "name": "viscoelastic-dean-cnab2",
        "res": {"nx": 32, "ny": 32, "nz": 32},
        "args": [
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
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # Split iterative-cn corrector (step.split_corrector=True):
        # every other iterative-cn entry runs the unsplit corrector
        # (the default), so this opt-in override keeps the split
        # corrector integration-covered.  Dean, because its coupling is
        # the pure instantaneous mean-flow term (l_bf == L_mf) -- the
        # regime where the two corrector structures most differ.  Same
        # config as the iterative-cn ``dean`` entry.
        "name": "dean-split-corrector",
        "res": {"nx": 32, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            "--step.split_corrector",
            "True",
        ],
    },
]


if __name__ == "__main__":
    main(SYSTEMS)
