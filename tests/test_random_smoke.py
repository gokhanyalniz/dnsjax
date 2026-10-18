r"""Random-IC smoke tests: exercise time integration for the core flows.

Starts every distinct stepping machinery of the core flows
(kolmogorov, plane-couette, pipe, taylor-couette) from a random
divergence-free perturbation of its base flow (the in-process
``init.random_field`` start mode, no snapshot file), at a Reynolds
number above the onset of transition, on a small domain at low
resolution, and integrates a short time -- verifying the run
completes with no error, NaN, or blow-up.  ``plane-poiseuille`` needs
no entries of its own: it binds the plane-couette machinery, with its
flow-specific pieces covered by ``test_laminar_smoke.py`` and the
published transient-growth values.  The runner and the per-row keys:
``tests/_random_smoke.py``; the lab's flows have a table of their own,
``lab/tests/test_random_smoke.py``.

Unlike ``tests/test_laminar_smoke.py`` (which starts from ``u' = 0``, so
all `$\omega'$`/`$u'$`-proportional terms -- including the rotational
nonlinear term -- vanish), this test feeds a non-trivial field through
the full nonlinear path, catching advection / ``rhs.py`` regressions a
laminar run reports as ``err = 0``.  It is also the triply-periodic
family's first stepping test (via Kolmogorov).

What the entries pin beyond a clean run (each entry's comment says how
it is built and why):

- ``plane-couette``: random is the **default** start mode (it omits
  ``--init.random_field``; the snapshot-first / random-default
  precedence in ``__main__.py``), and the chunked RHS inverse transform
  of ``rhs.get_nonlin`` steps through the jitted stepper
  (``--solver.rhs_transform_chunks 2``).
- ``plane-couette-gemm``, ``pipe-gemm``, ``taylor-couette-gemm``: the
  dense wall-normal GEMM, the GPU path and the reference, steps
  (``--solver.wall_normal_matvec dense``; CPU defaults to the stencil).
- ``pipe-half-tanh``: the pipe's one named non-CGL radial grid, whose
  weights take the positive-order composite rule
  (``fd.positive_quadrature``) no CGL entry reaches.
- ``plane-couette-cnab2``, ``pipe-cnab2``, ``taylor-couette-cnab2``,
  ``kolmogorov-cnab2``: the CN/AB2 scheme (``--step.scheme cnab2``;
  Crank-Nicolson viscous, explicit second-order Adams-Bashforth
  nonlinear, one FFT evaluation per step) on all four geometry families
  -- on the wall-bounded ones the implicit base-flow coupling and the
  iterative-CN self-start, which keep it stable on the wall-clustered
  CGL grid, on Kolmogorov the plain no-corrector path.
- ``taylor-couette-split-corrector``: the opt-in split corrector
  (``--step.split_corrector True``) on the annular machinery; every
  other wall-bounded iterative-cn entry runs the unsplit default.
- Every entry but the ``*-legacy-imm`` ones runs the **default**
  ``res.consistent_imm`` reconstruction, so the nonlinear stability gate
  the one-step ``test_imm_continuity.py`` guard cannot provide covers the
  whole table.  The five ``*-legacy-imm`` entries (plane-couette and
  taylor-couette under both schemes, the pipe under iterative-cn) drive
  the legacy primitive `$(v, p)$` path (``--res.consistent_imm False``)
  through the same nonlinear runs.  No pipe entry reaches the
  cylindrical default scheme's low-``Re`` limit: they run decades above
  its threshold, deliberately (the ``_cylindrical_stepping`` Design
  notes).
- ``plane-couette-mpi-pad``, ``pipe-pallas-mpi-pad``
  (``mpirun --oversubscribe -np 2``, ``np1 = 2``, ``nx // 2`` not
  divisible by ``np1``): the per-device (no-replication) random-IC build with
  spectral padding and the banded operators on that sharded padded
  axis, and ``[lowres]`` snapshots written and read back
  (``_check_lowres``).  The multi-process entries alone keep
  ``stop.check_laminarization`` on (their `$E'$` stays decades above its
  threshold): a flow's global arrays must reach every jitted program as
  arguments, a closed-over one traces on one process and is refused on
  two, and the laminarization read and the basis crossing are such
  programs.
- ``plane-couette-adaptive``, ``pipe-adaptive-cnab2``,
  ``kolmogorov-adaptive``: the adaptive CFL step grows ``dt`` (the
  asserted ``[adaptive]`` line) through the on-device ``set_dt`` rebuild.
- ``plane-couette-error-max``: the corrector-error check judges every
  step since the previous host sync, not only the one it lands on (its
  closing ``err =`` against its own ``corrector.dat``).
- ``kolmogorov-nan-guard``: inverted criteria -- a genuine blow-up must
  end in the solver's own non-finite guard, exit code 3 and one
  ``FATAL: non-finite ...`` line (the ``__main__`` module docstring),
  not a completed or uncontrolled run.

Transition to turbulence is **not** expected by the default ``t = 1``
at this resolution and box: success is a clean integration, not
turbulence.  Each system steps at ``--dt`` (default 0.01) unless its
entry forces its own (``force_dt``: the adaptive and non-finite-guard
entries), in a separate subprocess (the
geometry modules capture global singletons at import time) and its own
temporary directory, so ``parameters.toml`` is not loaded (model
defaults + CLI args only) and the per-system ``stats.dat`` does not
collide.  Success criteria per system: exit code 0, the shared criteria
of ``tests/_smoke.py``, and on CPU the closing memory lines over every
rank (``tests/_random_smoke.py``).

Usage (single device)::

    uv run python tests/test_random_smoke.py

Faster subset (one system, coarse / short)::

    uv run python tests/test_random_smoke.py \
        --systems plane-couette --res 16 --max-sim-time 0.2

Two devices via MPI::

    uv run python tests/test_random_smoke.py --np 2
"""

from __future__ import annotations

from _random_smoke import main

# ── configuration ────────────────────────────────────────────────────

# Reynolds numbers above the onset of (subcritical) transition.  TC is
# counter-rotating (re1 = 400 > 0 so update_parameters sets Re_ref = re1),
# eta = 0.5.  For pipe / TC the azimuthal extent is forced to 2*pi by
# the code and the radius / gap is fixed by geometry, so only the axial
# lx is set; Cartesian / periodic set both lx and lz (periodic ly is
# hardwired to 4 in update_parameters).
SYSTEMS: list[dict] = [
    {
        "name": "kolmogorov",
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
        # No snapshot and no explicit mode (start_from_laminar defaults
        # off): run_smoke_test asserts the random-IC startup line.  The
        # chunked transform is fft.chunked_transform; every other entry
        # runs the default fused 6-field batch, and chunk-vs-fused
        # bit-parity is unit-guarded (test_padding.py).  The
        # plane-poiseuille specifics: the laminar smoke (pp + pp-cbv)
        # and the transient-growth PP case.
        "name": "plane-couette",
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
            "--solver.rhs_transform_chunks",
            "2",
        ],
    },
    {
        # The default spin-quad formulation on a *random* IC: the guard
        # that it carries no resolved-IC restriction (a composed,
        # non-grid-scale-dissipative `$D_2$` would blow up on a
        # grid-white draw near the axis; the reformulation composes
        # nothing).  The half-CGL grid (the iterative-cn default) puts
        # the innermost node closest to the axis, the hardest case for
        # every 1/r term.
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
    # The pipe's wall-clustered one-sided tanh grid.
    {
        "name": "pipe-half-tanh",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--geo.grid_type",
            "half-tanh",
        ],
    },
    {
        # The default u_r-omega_r pair, whose spin coupling is the one
        # linear term still lagged to the corrector iterate (measured
        # contraction <= 0.02, but on *linear* data): the baseline
        # entry where that loop meets a real nonlinear source.
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
    # The GEMM rows, one per geometry; the two paths' agreement is
    # pinned by test_wall_normal_matvec.py.
    {
        "name": "plane-couette-gemm",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--solver.wall_normal_matvec",
            "dense",
        ],
    },
    {
        "name": "pipe-gemm",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--solver.wall_normal_matvec",
            "dense",
        ],
    },
    {
        "name": "taylor-couette-gemm",
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
            "--solver.wall_normal_matvec",
            "dense",
        ],
    },
    {
        # nx // 2 = 17 on np1 = 2 (padded to 18); solver.backend
        # defaults to pallas, so this also builds and solves the
        # cartesian banded operators on the sharded padded axis.
        "name": "plane-couette-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "check_laminarization": True,
        "res": {"nx": 34, "ny": 32, "nz": 32},
        # Reduced snapshots with the static pressure: kx = 9 does not
        # divide np1 either.
        "lowres": {
            "it_lowres": 50,
            "res": {"nx": 18, "ny": 17, "nz": 12},
            "shape": (3, 17, 11, 9),
            "pressure": True,
        },
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
        # The banded backend on a sharded, padded mode axis: nx // 2 = 3
        # on np1 = 2 (padded to 4), the pipe's parity-selected base
        # band.  On CPU the backend runs the pure-JAX banded sweep (the
        # Triton kernel is GPU-only).  Single-device band-vs-dense
        # parity per geometry: test_cartesian.py / test_cylindrical.py /
        # test_annular.py; the shard_map-local solve:
        # test_banded_solver_sharded.py.
        "name": "pipe-pallas-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "check_laminarization": True,
        "res": {"nx": 6, "ny": 24, "nz": 8},
        # Reduced snapshots through the radial parity regrid, under
        # the public names (no pressure for the pipe).
        "lowres": {
            "it_lowres": 50,
            "res": {"nz": 4, "nr": 13, "ntheta": 6},
            "shape": (3, 13, 5, 2),
            "pressure": False,
        },
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
        # ny = 48, above the other Cartesian entries on purpose:
        # plane-Couette is a *moving-wall* flow (``U = y``, ``O(1)`` at
        # the walls), so on the wall-clustered CGL grid the coupling
        # ``U d(u')/dy`` is stiff and a naive explicit AB2 blows up at
        # dt = 0.01 once ny >~ 40 (CFL << 1).  The implicit coupling
        # (FFT-free corrector: ``_l_bf`` / ``step_cnab2``) reports a
        # *real* corrector count and error, so ``tests/_smoke.py``'s
        # criterion 3 (err < tol) is a genuine check.  Under cnab2 the
        # default scheme's v-omega_y source projections act on the AB2/CN
        # combination assembled from ``_l_bf`` and the carried nonlinear
        # history.
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
        # The laminar smoke cannot separate the two formulations (every
        # term is linear in u', so all vanish at u' = 0) and
        # ``tests/test_imm_continuity.py`` takes one step, so only
        # entries of this shape integrate either through a real
        # nonlinear run.  That matters most here: a state-side
        # projection passes every *linear* gate and still goes
        # non-finite within ~6 steps of exactly this configuration (the
        # ``cartesian._imm_iteration`` docs).
        "name": "plane-couette-legacy-imm",
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
            "--res.consistent_imm",
            "False",
        ],
    },
    {
        # Under cnab2 the AB2/CN split hands ``_imm_iteration`` a
        # combination assembled from ``_l_bf`` and the carried
        # nonlinear history, which no other entry integrates
        # nonlinearly.
        "name": "plane-couette-legacy-imm-cnab2",
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
            "--res.consistent_imm",
            "False",
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # The annular half of the same guard: the legacy primitive
        # path on the annular machinery.
        "name": "taylor-couette-legacy-imm",
        "res": {"nx": 24, "ny": 40, "nz": 24},
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "500",
            "--phys.re2",
            "-200",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "3",
            "--res.consistent_imm",
            "False",
        ],
    },
    {
        # The annular legacy path under cnab2 (the combination as
        # above).
        "name": "taylor-couette-legacy-imm-cnab2",
        "res": {"nx": 24, "ny": 40, "nz": 24},
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "500",
            "--phys.re2",
            "-200",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "3",
            "--res.consistent_imm",
            "False",
            "--step.scheme",
            "cnab2",
        ],
    },
    {
        # The pipe half: the legacy primitive path, on the half-CGL grid
        # (the iterative-cn default), whose innermost node sits closest
        # to the axis.
        "name": "pipe-legacy-imm",
        "res": {"nx": 24, "ny": 40, "nz": 24},
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
            "--res.consistent_imm",
            "False",
        ],
    },
    {
        # The cylindrical ``_l_bf`` build and step.  Pipe is a
        # *stationary-wall* flow (``U -> 0`` at the wall), so its
        # coupling is mild and cnab2 is bounded only by the ordinary
        # explicit self-advection CFL, whose binding term is the
        # **near-axis azimuthal** advection: the
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
        # destabilizes cnab2 and is iterative-cn-only.  ny=32 here
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
        # The annular ``_l_bf`` (three stacked Hk) build at ny = 48: the
        # inner-rotating, stationary-outer configuration is a
        # *moving-wall* flow (``U_theta ~ O(1)`` at the inner wall), so
        # ``U_theta d(u')/dr`` is stiff on the wall-clustered grid,
        # exactly the stiffness the implicit coupling removes.  Under
        # cnab2 the default scheme's lagged spin coupling reads a
        # *carry*-derived iterate rather than a corrector one.
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
        # l_bf_fn is None: the uniform Fourier grid has no wall
        # clustering and no coupling stiffness to make implicit, so
        # ``err`` is reported as 0 (no corrector).  Also the
        # triply-periodic ``step_cnab2`` self-start seeding.
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
        # The split corrector iterates the linear coupling FFT-free
        # between full-RHS refreshes (``_split_core`` in
        # ``timestep.py``).  Same config as the iterative-cn
        # ``taylor-couette`` entry.
        "name": "taylor-couette-split-corrector",
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
            "--step.split_corrector",
            "True",
        ],
    },
    {
        # A small initial dt keeps the CFL far below cfl_target, so the
        # controller must grow dt (1.2x per accepted evaluation) up to
        # dt_max.  On 2 devices (np1 = 2), so the sharded Hk/IMM leaf
        # rebuild (shard_map tile pad, sharded IMM solves) runs too.
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
        # The kappa-weighted AB2 step and the stacked (+,-,z) Hk
        # rebuild.  dt_max stays below the rigged-grid cnab2 stability
        # step (~0.0125 at 32^3; see pipe-cnab2) and cfl_target below
        # the azimuthal AB2 limit.
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
        # The ldt_1/ildt_2 recompute.  The CFL is tiny here, so the
        # controller grows dt until dt_max stops it: any dt_max above
        # force_dt exercises the rebuild.
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
        # At it_error_check = it_corrector = 10 over 10 steps, the single
        # corrector.dat row is step 1's error (the largest of the
        # decaying post-IC sequence) and the closing "err =" -- the
        # maximum since the last check -- must not fall below it; a
        # loop reading only the check step's own error reports step
        # 10's smaller one (``check_error_max``).
        "name": "plane-couette-error-max",
        "max_sim_time": 0.1,
        "check_error_max": True,
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
            "--outs.it_error_check",
            "10",
            "--outs.it_corrector",
            "10",
        ],
    },
    {
        # cnab2 at dt = 1.0 (100x the default dt, CFL >> 1) blows up
        # within tens of steps; whichever guard layer sees it first
        # aborts (run_smoke_test inverts the criteria).  it_stats = 1 with
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


if __name__ == "__main__":
    main(SYSTEMS)
