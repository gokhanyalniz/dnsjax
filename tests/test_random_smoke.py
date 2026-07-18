"""Random-IC smoke tests: exercise time integration for all flows.

Starts every implemented flow from a random divergence-free perturbation
of its base flow (the in-process ``init.random_field`` start mode, no
snapshot file), at a Reynolds number above the onset of transition, on a
small domain at low resolution, and integrates a short time -- verifying
the run completes with no error, NaN, or blow-up.

Unlike ``tests/test_laminar_smoke.py`` (which starts from ``u' = 0``, so
all `$\\omega'$`/`$u'$`-proportional terms -- including the rotational
nonlinear term -- vanish), this test feeds a non-trivial field through
the full nonlinear path, catching advection / ``rhs.py`` regressions a
laminar run reports as ``err = 0``.  It is also the triply-periodic
family's first stepping test (via Kolmogorov).

One entry (``plane-couette-default-ic``) omits ``--init.random_field``
to verify random is the **default** start mode -- with no snapshot and
no explicit mode selected, the run must start from a random field (not
the laminar state), guarding the snapshot-first / random-default
precedence in ``__main__.py``.  Six entries (``plane-couette-cnab2``,
``pipe-cnab2``, ``taylor-couette-cnab2``, ``dean-cnab2``,
``viscoelastic-dean-cnab2``, ``kolmogorov-cnab2``) pass
``--step.scheme cnab2`` to drive the alternative CN/AB2 time-stepping
scheme -- Crank-Nicolson viscous + explicit 2nd-order Adams-Bashforth
nonlinear, one FFT eval per step -- across all four geometry families
(``dean-cnab2`` additionally covers the total-field ``_l_bf == 0``
corrector path; ``viscoelastic-dean-cnab2`` the 9-component tensor
coupling).
For the wall-bounded families this exercises the fix that makes the
stiff base-flow coupling implicit (an FFT-free corrector; see ``_l_bf``
/ ``step_cnab2``) plus the iterative-CN self-start, so cnab2 stays
stable on the wall-clustered CGL grid where a naive explicit-AB2 of the
coupling blows up at CFL << 1; triply-periodic (Kolmogorov) drives the
plain no-corrector path (uniform Fourier grid, no coupling stiffness).
The two moving-wall entries use ``ny = 48`` on purpose (the coupling
stiffness the fix targets scales with the near-wall clustering); the
stationary-wall pipe uses ``ny = 32`` (its coupling is mild -- ``U -> 0``
at the wall -- so cnab2 there is bounded only by the ordinary explicit
self-advection CFL, which ``ny = 48`` on the fine near-wall CGL grid
would violate at this ``Re``, an inherent explicit-scheme limit, not the
coupling bug).

The wall-bounded iterative-cn entries run the **unsplit** corrector
(``step.split_corrector`` defaults off).  A trailing
``dean-split-corrector`` entry passes ``--step.split_corrector True``
to cover the opt-in split corrector (linear coupling iterated FFT-free
between full-RHS refreshes; see ``_split_core`` in ``timestep.py``) on
the force-driven Dean flow, where the coupling is the pure
instantaneous mean-flow term (``l_bf == L_mf``) -- the regime where the
two corrector structures most differ.

A trailing ``plane-couette-chunked`` entry passes
``--solver.rhs_transform_chunks 2`` to drive the chunked RHS inverse
transform of ``rhs.get_nonlin`` (``fft.chunked_transform``, the
memory/throughput knob; every other entry runs the default fused
6-field batch) end to end through the jitted stepper.  The
viscoelastic analogue is unit-covered by
``test_viscoelastic.py::test_rhs_transform_chunks_parity``.

A trailing forced multi-device, padding-inducing entry (``mpi-pad``:
``mpirun --oversubscribe -np 2``, ``np1 = 2``, ``nx // 2`` not
divisible by ``np1``) is the regression guard for the per-device
(no-replication) random-IC build with spectral padding.  Four
``*-pallas-mpi-pad`` entries repeat that shape with
``--solver.backend pallas`` to guard the sharded per-mode banded
operator build/solve (the CPU pure-JAX banded path; the Triton kernel
itself is GPU-only): ``pipe`` (parity-selected base band),
``plane-couette`` (single shared ``Hk``), ``taylor-couette`` (three
stacked ``Hk``), and ``viscoelastic-dean`` (additionally the
conformation Helmholtz operator ``Hc``, `$\\kappa > 0$`).

A trailing ``kolmogorov-nan-guard`` entry inverts the success
criteria: it forces a genuine blow-up (cnab2 at a dt far past the
advective limit) and requires the solver's non-finite guard to abort
the run itself -- exit code 3 and one ``FATAL: non-finite ...``
diagnostic line (see the ``__main__`` module docstring) -- instead of
completing or dying uncontrolled.  A small ``nbuffer`` makes the
in-loop buffered-stats scan see the blow-up promptly.

Transition to turbulence is **not** expected to develop by the default
``t = 1`` at this resolution/box; the success metric is purely that
integration completes cleanly, not that the flow becomes turbulent.

Each system steps at ``--dt`` (default 0.01), capped per-system where the
corrector needs a smaller step (Kolmogorov: 0.005 -- a corrector-rate,
not advective-CFL, limit; see ``SYSTEMS``).

Each system runs in a separate subprocess (the geometry modules capture
global singletons at import time) in its own temporary directory, so
``parameters.toml`` is not loaded (model defaults + CLI args only) and
the per-system ``stats.dat`` does not collide.

Success criteria per system:

1. subprocess exit code 0 (catches hard crashes / exceptions);
2. the run reached the end (final ``t`` `$\\geq$`
   ``max_sim_time - dt``): it was not cut short by corrector
   divergence (the main loop stops once ``err`` reaches
   ``corrector_tolerance``);
3. ``"Corrector failed to converge"`` absent from stdout;
4. the final corrector error is finite and below ``corrector_tolerance``
   (catches a late divergence in the last ``it_error_check`` steps);
5. every numeric value on the final summary line is finite (NaN/Inf
   print as ``nan``/``inf``).

Usage (single device)::

    uv run python tests/test_random_smoke.py

Faster subset (one system, coarse / short)::

    uv run python tests/test_random_smoke.py \
        --systems plane-couette --res 16 --max-sim-time 0.2

Two devices via MPI::

    uv run python tests/test_random_smoke.py --np 2
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile

from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

# ── configuration ────────────────────────────────────────────────────

# Reynolds numbers above the onset of (subcritical) transition.  TC is
# counter-rotating (re1 = 400 > 0 so update_parameters sets Re_ref = re1);
# Dean is force-driven at re = 1000.  eta = 0.5 for the annular family.
# For pipe / TC / Dean the azimuthal extent is forced to 2*pi by the
# code and the radius / gap is fixed by geometry, so only the axial lx
# is set; Cartesian / periodic set both lx and lz (periodic ly is
# hardwired to 4 in update_parameters).
SYSTEMS: list[dict] = [
    {
        "name": "kolmogorov",
        # The iterative Crank-Nicolson corrector contracts to the 1e-5
        # tolerance within max_corrector_iterations only for dt <~ 0.005
        # here (at dt = 0.01 it stalls at ~1.4e-5 after 10 iterations);
        # cap dt so the global default does not exceed it.  The CFL is
        # tiny (~0.08), so this is a corrector-rate limit, not advective.
        "max_dt": 0.005,
        "args": [
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "620",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "plane-couette",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "plane-poiseuille",
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--phys.re",
            "660",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "pipe",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "taylor-couette",
        "args": [
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
        ],
    },
    {
        # Quasi-Keplerian (co-rotating, linearly stable): the random
        # perturbation decays, but the run still drives the full
        # rotational-nonlinear path for the new system.
        "name": "quasi-keplerian",
        "args": [
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "400",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
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
        # Viscoelastic (sPTT) Dean flow: the 9-component state (3 velocity
        # + 6 conformation-tensor) driven through the full coupled
        # nonlinear path (advection Christoffels, stretching, relaxation,
        # polymer-stress divergence).  Total-field IC = analytical laminar
        # pair + windowed tensor noise (``random_conformation_amplitude``).
        # Re = wi/el is derived (no --phys.re).  The kappa > 0 default
        # also builds the conformation Helmholtz operator (``Hc_op``).
        # wi = el = 20 (Re = 1) and a reduced conformation-noise amplitude
        # (10, vs the reference restart default 700) keep the coupled
        # system robust at the coarse smoke resolution -- the reference
        # wi = 105 with the full noise is a genuinely stiff elastic
        # instability on a 32^3 grid, not a solver defect.
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
        # Regression guard for the per-device (no-replication) random
        # build with spectral padding: always multi-device on the kx axis
        # (np1 = 2), with nx // 2 = 17 not divisible by np1 (padded to
        # 18).  Exercises the fixed multi-device spectral-padding path.
        "name": "plane-couette-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 34, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # Regression guard for the Pallas banded backend
        # (solver.backend=pallas) on a sharded, padded mode axis:
        # multi-device on the kx/axial axis (np1 = 2) with nx // 2 = 3
        # not divisible by np1 (padded to 4).  The per-mode banded
        # operators are built and the nonlinear path solved on the
        # sharded axis -- catching sharding bugs the single-device run
        # misses.  On CPU the backend runs the pure-JAX banded sweep
        # (the Triton kernel is GPU-only; see the GPU-validation plan).
        "name": "pipe-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--solver.backend",
            "pallas",
        ],
    },
    {
        # Pallas banded backend on the Cartesian (plane-couette)
        # geometry, same sharded/padded mode-axis guard as
        # pipe-pallas-mpi-pad (np1 = 2, nx // 2 = 3 padded to 4): builds
        # the single shared Hk and the Lk pressure operator in banded
        # storage and solves the nonlinear path on the sharded axis.  On
        # CPU the pure-JAX banded sweep runs (Triton kernel is GPU-only).
        "name": "plane-couette-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--solver.backend",
            "pallas",
        ],
    },
    {
        # Pallas banded backend on the annular (taylor-couette)
        # geometry, same sharded/padded guard: builds the Lk and the
        # three stacked Hk (m+1, m-1, m) operators in banded storage and
        # solves on the sharded axis.  CPU runs the pure-JAX banded sweep.
        "name": "taylor-couette-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        "args": [
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
            "--solver.backend",
            "pallas",
        ],
    },
    {
        # Pallas banded backend on the viscoelastic annular
        # (viscoelastic-dean) geometry, same sharded/padded mode-axis
        # guard (np1 = 2, nx // 2 = 3 padded to 4): builds the annular Lk
        # / 2x2-IMM Hk operators **and** the 6-component stacked
        # conformation Helmholtz operator ``Hc_op`` (kappa > 0) in banded
        # storage on the sharded axis, and solves the 9-component coupled
        # nonlinear path.  CPU runs the pure-JAX banded sweep (the Triton
        # kernel is GPU-only).  Same robust reduced parameters as the
        # single-device ``viscoelastic-dean`` entry.
        "name": "viscoelastic-dean-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
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
        # Default start mode: no --init.random_field flag is passed, so
        # the run must fall through to the random-IC default
        # (start_from_laminar defaults off).  Guards the snapshot-first /
        # random-default precedence in __main__.py: with no snapshot and
        # no explicit mode, the IC is a random field, not the laminar
        # state.  run_smoke_test additionally asserts the random-IC
        # startup line is present.
        "name": "plane-couette-default-ic",
        "omit_random_flag": True,
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # CN/AB2 scheme (step.scheme=cnab2) on the Cartesian
        # (plane-Couette) geometry, driven through the full nonlinear
        # path.  Run at a **higher wall-normal resolution** (ny=48) than
        # the other Cartesian entries on purpose: plane-Couette is a
        # *moving-wall* flow (``U = y``, so ``U ~ O(1)`` at the walls),
        # so on the wall-clustered CGL grid the rotational base-flow
        # coupling ``U d(u')/dy`` is a stiff explicit term and a naive
        # explicit-AB2 nonlinear blows up at dt=0.01 once ny >~ 40
        # (CFL << 1).  This guards the fix that makes the base-flow
        # coupling implicit (FFT-free corrector; see ``_l_bf`` /
        # ``step_cnab2``), so cnab2 reports a *real* corrector
        # count/error and criterion 4 (err < tol) is a genuine check,
        # integrating cleanly like the iterative-cn entries.
        "name": "plane-couette-cnab2",
        "res": {"nx": 32, "ny": 48, "nz": 32},
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # CN/AB2 on the cylindrical (pipe) geometry, guarding the
        # cylindrical ``_l_bf`` build + step path.  Pipe is a
        # *stationary-wall* flow (``U_z = 1 - r^2``, ``U -> 0`` at the
        # wall), so its base-flow coupling is mild (the ``U`` weight
        # kills the near-wall stiffness) and cnab2 is bounded only by
        # the ordinary explicit self-advection CFL.  For the pipe the
        # binding term is the **near-axis azimuthal** advection: the
        # innermost radial node sits at ``r_0 ~ pi/(2 ny)`` (the
        # rigged-CGL grid, the cnab2 default this entry resolves to)
        # where the azimuthal arc length
        # ``r_0 dtheta`` is tiny, so
        # ``CFL_th = dt |u_th(r_0)| nz / (2 pi r_0)`` (the dominant
        # ``steps.dat`` column) scales linearly with nz and with the
        # perturbation amplitude -- *not* the near-wall ``1/N^2``
        # radial-spacing story (ny=72 integrates configs ny=48 fails).
        # The instability is the weak AB2 imaginary-axis one: it needs
        # *sustained* CFL_th >~ 0.5, so pass/fail is seed-/trajectory-
        # marginal at ny>=48 for this Re/amplitude.  It is fluctuation
        # (m = +-1) driven, so neither the implicit ``L_bf`` corrector
        # (which converges cleanly throughout) nor
        # ``step.implicit_mean_coupling`` can remove it -- but the
        # **rigged-CGL** radial grid (the cnab2 default) doubles
        # ``r_0`` vs the half-CGL grid (the iterative-cn default),
        # doubling the admissible ``dt``
        # (measured dt* ~ 0.0125 -> 0.0175 at 32^3/Re=1800); the
        # tighter half-CGL grid (``geo.grid_type = "half-cgl"``)
        # destabilises cnab2 and is iterative-cn-only.  ny=32 here
        # like the other pipe entries.
        "name": "pipe-cnab2",
        "res": {"nx": 32, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # CN/AB2 on the annular (Taylor-Couette) geometry at ny=48,
        # guarding the annular ``_l_bf`` (three stacked Hk) build +
        # implicit-coupling corrector + iterative-CN self-start.  Uses a
        # standard (inner-rotating, stationary-outer) config, which is a
        # *moving-wall* flow (``U_theta ~ O(1)`` at the inner wall): its
        # ``U_theta d(u')/dr`` coupling is stiff on the wall-clustered
        # grid, so ny=48 stresses exactly the stiffness the fix removes.
        # NOTE: *strongly counter-rotating* Taylor-Couette
        # (re1=400/re2=-400) is deliberately **not** used here -- its
        # base flow is so non-normal that the explicit self-advection is
        # amplified into a (delayed) blow-up needing ~8x smaller dt, an
        # inherent explicit-nonlinear limit the coupling corrector cannot
        # remove (it converges cleanly); such flows want ``iterative-cn``
        # or a much smaller dt.  See the ``TimeStepping`` docstring.
        "name": "taylor-couette-cnab2",
        "res": {"nx": 32, "ny": 48, "nz": 32},
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
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
        # entry.  Same robust reduced parameters as the iterative-cn
        # ``viscoelastic-dean`` entry; cnab2 matches iterative-cn to
        # O(dt^2) at ~1 FFT/step vs its ~4.
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
        # CN/AB2 on the triply-periodic (Kolmogorov) geometry: the plain
        # no-corrector explicit-AB2 path (l_bf_fn is None -- the uniform
        # Fourier grid has no wall clustering and no base-flow-coupling
        # stiffness, so there is nothing to make implicit).  ``err`` is
        # reported as 0 (no corrector).  Guards the triply-periodic
        # ``step_cnab2`` branch and its self-start seeding.
        "name": "kolmogorov-cnab2",
        "args": [
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "620",
            "--geo.lx",
            "5",
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
    {
        # Chunked RHS transforms (solver.rhs_transform_chunks=2) on a
        # Newtonian flow: every other entry runs the default fused
        # 6-field batch of ``get_nonlin``, so this keeps the
        # ``fft.chunked_transform`` path integration-covered (k = 2
        # splits velocity / vorticity into separate inverse
        # transforms; results are identical, so the standard criteria
        # apply unchanged).  Same config as the iterative-cn
        # ``plane-couette`` entry.
        "name": "plane-couette-chunked",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--solver.rhs_transform_chunks",
            "2",
        ],
    },
    {
        # Adaptive CFL dt (iterative-cn, Cartesian): a deliberately
        # small initial dt keeps the measured CFL far below
        # cfl_target, so the controller must grow dt (1.2x per
        # accepted evaluation) up to dt_max during the run -- the
        # asserted "[adaptive]" log line.  Guards the on-device
        # Hk/IMM leaf rebuild (set_dt) inside a full solver run --
        # on 2 devices (np1 = 2), so the sharded rebuild (shard_map
        # tile pad, sharded IMM solves) is integration-covered.
        "name": "plane-couette-adaptive",
        "force_np": 2,
        "oversubscribe": True,
        "force_dt": 0.002,
        "slack_dt": 0.01,
        "expect_pattern": r"\[adaptive\] .*dt .*->",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--step.adaptive",
            "True",
            "--step.dt_max",
            "0.01",
            "--step.cfl_target",
            "0.5",
            "--step.cfl_cadence",
            "2",
        ],
    },
    {
        # Adaptive CFL dt under CN/AB2 on the cylindrical geometry:
        # exercises the kappa-weighted AB2 step and the stacked
        # (+,-,z) Hk rebuild.  dt_max stays below the rigged-grid
        # cnab2 stability step (~0.0125 at 32^3; see pipe-cnab2) and
        # cfl_target below the azimuthal AB2 limit.
        "name": "pipe-adaptive-cnab2",
        "force_dt": 0.002,
        "slack_dt": 0.008,
        "expect_pattern": r"\[adaptive\] .*dt .*->",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
            "--step.adaptive",
            "True",
            "--step.dt_max",
            "0.008",
            "--step.cfl_target",
            "0.3",
            "--step.cfl_cadence",
            "2",
        ],
    },
    {
        # Adaptive CFL dt on the triply-periodic path: the
        # ldt_1/ildt_2 recompute.  dt_max respects the Kolmogorov
        # corrector-contraction cap (~0.005; see the kolmogorov
        # entry).
        "name": "kolmogorov-adaptive",
        "force_dt": 0.002,
        "slack_dt": 0.004,
        "expect_pattern": r"\[adaptive\] .*dt .*->",
        "args": [
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "620",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--step.adaptive",
            "True",
            "--step.dt_max",
            "0.004",
            "--step.cfl_target",
            "0.5",
            "--step.cfl_cadence",
            "2",
        ],
    },
    {
        # Non-finite guard (inverted criteria; see run_smoke_test):
        # cnab2 at dt = 1.0 (200x Kolmogorov's iterative-cn limit,
        # CFL >> 1) blows up within tens of steps; the run must abort
        # itself with exit code 3 and a "FATAL: non-finite ..." line
        # (whichever guard layer sees it first).  it_stats = 1 with
        # nbuffer = 10 makes the in-loop buffered-stats scan the
        # likely detector; max_sim_time is only the safety ceiling.
        # force_np = 1 keeps the mpirun exit-code propagation exact.
        "name": "kolmogorov-nan-guard",
        "expect_nonfinite_abort": True,
        "force_np": 1,
        "force_dt": 1.0,
        "max_sim_time": 1000.0,
        "it_stats": 1,
        "args": [
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "620",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--step.scheme",
            "cnab2",
            "--outs.nbuffer",
            "10",
        ],
    },
]

# Default time-stepping ``corrector_tolerance`` (TimeStepping model
# default; the subprocesses run in temp dirs, so no parameters.toml).
CORRECTOR_TOLERANCE = 1e-5

# A signed float, or ``nan`` / ``inf`` (how non-finite values print).
_NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
# ``\bt`` avoids matching the ``t`` in ``c/it =`` ('i' is a word char, so
# there is no word boundary before that ``t``).
T_PATTERN = re.compile(rf"\bt\s*=\s*({_NUM})")
ERR_PATTERN = re.compile(rf"\berr\s*=\s*({_NUM})")
VALUE_PATTERN = re.compile(rf"=\s*({_NUM})")

# ── helpers ──────────────────────────────────────────────────────────


def _build_command(
    system: dict, args: argparse.Namespace, dt: float
) -> list[str]:
    """Build the ``mpirun ... -m dnsjax`` command for one system.

    Per-system ``force_np`` / ``force_np0`` / ``res`` /
    ``oversubscribe`` / ``max_sim_time`` / ``it_stats`` override the
    suite defaults (used by the multi-device padded entry and the
    nan-guard entry).
    """
    np_count = system.get("force_np", args.np)
    np0 = system.get("force_np0", args.np0)
    res = system.get("res", {})
    nx = str(res.get("nx", args.res))
    ny = str(res.get("ny", args.res))
    nz = str(res.get("nz", args.res))
    # Cylindrical/annular flows take the aliased public resolution
    # names (axial nz, radial nr, azimuthal ntheta); the internal
    # meaning (nx = axial, ny = radial, nz = azimuthal) is unchanged.
    cyl_annular = any(
        s in system["args"]
        for s in (
            "pipe",
            "taylor-couette",
            "quasi-keplerian",
            "dean",
            "viscoelastic-dean",
        )
    )
    res_flags = (
        ["--res.nz", nx, "--res.nr", ny, "--res.ntheta", nz]
        if cyl_annular
        else ["--res.nx", nx, "--res.ny", ny, "--res.nz", nz]
    )
    max_sim_time = system.get("max_sim_time", args.max_sim_time)
    it_stats = system.get("it_stats", args.it_stats)

    base = ["mpirun"]
    if system.get("oversubscribe"):
        base.append("--oversubscribe")
    base += [
        "-np",
        str(np_count),
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.platform",
        args.platform,
        "--dist.np0",
        str(np0),
        "--dist.np1",
        str(np_count // np0),
    ]
    # In-process random divergence-free IC (no snapshot file).  The
    # default-IC entry omits this flag to verify that random is the
    # default start mode when nothing else is selected.
    if not system.get("omit_random_flag"):
        base += ["--init.random_field", "True"]
    base += [
        "--init.random_amplitude",
        str(args.amplitude),
        "--init.random_smoothness",
        str(args.smoothness),
        "--init.random_seed",
        str(args.seed),
        *res_flags,
        "--step.dt",
        str(dt),
        "--stop.max_sim_time",
        str(max_sim_time),
        "--outs.it_stats",
        str(it_stats),
        # Laminarization check off: a decaying transient must not cut
        # the run short before max_sim_time.
        "--stop.check_laminarization",
        "False",
    ]
    return base + system["args"]


def _final_summary_line(stdout: str) -> str | None:
    """Return the final ``t = ... err = ...`` summary line, if any.

    The per-step error appears only on the end-of-run summary line, so
    the last line carrying ``err =`` is that summary (the initial
    warm-up ``t = 0.00 ...`` print has no ``err``).
    """
    summary: str | None = None
    for line in stdout.splitlines():
        if ERR_PATTERN.search(line):
            summary = line
    return summary


def _check_run(stdout: str, name: str, max_sim_time: float, dt: float) -> str:
    """Validate one completed run; return the summary line for the PASS.

    Raises ``AssertionError`` on any failed criterion.
    """
    if "failed to converge" in stdout:
        raise AssertionError(f"{name}: corrector failed to converge")

    summary = _final_summary_line(stdout)
    if summary is None:
        raise AssertionError(
            f"{name}: no end-of-run summary line found\n{stdout[-2000:]}"
        )

    t_match = T_PATTERN.search(summary)
    err_match = ERR_PATTERN.search(summary)
    if t_match is None or err_match is None:
        raise AssertionError(f"{name}: cannot parse summary line: {summary!r}")

    t_final = float(t_match.group(1))
    err = float(err_match.group(1))

    # Reached the end (not cut short by corrector divergence).  The last
    # step lands t on (or just past) max_sim_time; allow one dt of slack.
    if not (t_final >= max_sim_time - dt):
        raise AssertionError(
            f"{name}: run ended early at t={t_final} (< {max_sim_time}); "
            "integration did not complete"
        )

    # Final corrector error finite and converged (catches a divergence
    # in the last it_error_check steps that the loop did not stop on).
    if not (math.isfinite(err) and err < CORRECTOR_TOLERANCE):
        raise AssertionError(
            f"{name}: final corrector error {err:.3e} not in "
            f"[0, {CORRECTOR_TOLERANCE:.0e})"
        )

    # No NaN / Inf anywhere on the summary line (every diagnostic finite).
    values = [float(v) for v in VALUE_PATTERN.findall(summary)]
    if not all(math.isfinite(v) for v in values):
        raise AssertionError(
            f"{name}: non-finite value on summary line: {summary!r}"
        )

    return summary


# ── test runner ──────────────────────────────────────────────────────


def run_smoke_test(system: dict, args: argparse.Namespace) -> None:
    """Run a single random-IC smoke test (in a fresh directory).

    Further per-entry keys: ``expect_pattern`` (regex that must match
    the run's stdout, e.g. the adaptive-dt "[adaptive]" change line)
    and ``slack_dt`` (the end-of-run ``t`` slack for runs whose step
    grows -- the adaptive entries pass their ``dt_max``).
    """
    name = system["name"]
    # Per-system dt cap: some systems need a smaller step than the
    # global default for the corrector to converge (see SYSTEMS).
    # ``force_dt`` overrides outright (the nan-guard entry needs a dt
    # *larger* than any sane default).
    dt = system.get("force_dt", min(args.dt, system.get("max_dt", math.inf)))
    cmd = _build_command(system, args, dt)

    with tempfile.TemporaryDirectory(prefix=f"rand_{name}_") as workdir:
        result = run_live(cmd, timeout=args.timeout, cwd=workdir)

        if system.get("expect_nonfinite_abort"):
            # Inverted criteria: the run must have aborted itself via
            # the non-finite guard -- exit code 3 (mpirun -np 1
            # propagates the child's code) and the FATAL diagnostic.
            fatal_lines = [
                line
                for line in result.stdout.splitlines()
                if "FATAL: non-finite" in line
            ]
            if result.returncode != 3 or not fatal_lines:
                print(f"  FAIL  {name}: exit code {result.returncode}")
                print(
                    result.stdout[-2000:] if result.stdout else "(no stdout)"
                )
                print(
                    result.stderr[-2000:] if result.stderr else "(no stderr)"
                )
                raise AssertionError(
                    f"{name}: expected the non-finite guard abort (exit 3 "
                    f"+ 'FATAL: non-finite ...'), got exit "
                    f"{result.returncode} with {len(fatal_lines)} FATAL "
                    "line(s)"
                )
            print(f"  PASS  {name}  ({fatal_lines[0].strip()})")
            return

        if result.returncode != 0:
            print(f"  FAIL  {name}: exit code {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise AssertionError(
                f"{name} exited with code {result.returncode}"
            )

        # Default-IC entry: confirm the run actually took the random
        # branch (no snapshot, no explicit mode => random default).
        if system.get("omit_random_flag") and (
            "Started from an in-process random IC" not in result.stdout
        ):
            raise AssertionError(
                f"{name}: default start mode did not select the random IC "
                "('Started from an in-process random IC' missing from stdout)"
            )

        summary = _check_run(
            result.stdout,
            name,
            args.max_sim_time,
            system.get("slack_dt", dt),
        )

        pattern = system.get("expect_pattern")
        if pattern and re.search(pattern, result.stdout) is None:
            raise AssertionError(
                f"{name}: expected pattern {pattern!r} not found in stdout"
            )

    print(f"  PASS  {name}  ({summary.strip()})")


# ── main ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-IC smoke tests for all flows",
    )
    parser.add_argument(
        "--np", type=int, default=1, help="Number of devices (mpirun -np)"
    )
    parser.add_argument(
        "--np0",
        type=int,
        default=1,
        help="np0 mesh axis (wall-normal / kz split)",
    )
    parser.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend forwarded to each `python -m dnsjax` child "
        "(default cpu).  Use cuda to run the suite on GPU(s).",
    )
    parser.add_argument(
        "--res", type=int, default=32, help="Cubic resolution nx=ny=nz"
    )
    parser.add_argument("--max-sim-time", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--smoothness", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--it-stats", type=int, default=10)
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Subset of system names to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-system subprocess timeout in seconds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    systems = SYSTEMS
    if args.systems:
        wanted = set(args.systems)
        systems = [s for s in SYSTEMS if s["name"] in wanted]
        unknown = wanted - {s["name"] for s in SYSTEMS}
        if unknown:
            print(f"Unknown system(s): {sorted(unknown)}")
            sys.exit(2)

    print(
        f"Random-IC smoke tests on platform '{args.platform}' via "
        f"mpirun (default -np {args.np}, np0={args.np0}; some entries "
        "force their own device count).  Each child prints its own "
        "device banner.",
        flush=True,
    )

    passed = 0
    failed = 0
    for system in systems:
        try:
            run_smoke_test(system, args)
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {system['name']}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
