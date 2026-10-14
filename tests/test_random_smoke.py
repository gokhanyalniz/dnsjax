r"""Random-IC smoke tests: exercise time integration for the core flows.

Starts every distinct stepping machinery of the core flows
(kolmogorov, plane-couette, pipe, taylor-couette) from a random
divergence-free perturbation of its base flow (the in-process
``init.random_field`` start mode, no snapshot file), at a Reynolds
number above the onset of transition, on a small domain at low
resolution, and integrates a short time -- verifying the run
completes with no error, NaN, or blow-up.  ``plane-poiseuille`` needs
no entries of its own: it binds the plane-couette machinery (see the
SYSTEMS comments), with its flow-specific pieces covered by
``test_laminar_smoke.py`` and the published transient-growth values.
The runner and the per-row keys: ``tests/_random_smoke.py``; the lab's
flows have a table of their own, ``lab/tests/test_random_smoke.py``.

Unlike ``tests/test_laminar_smoke.py`` (which starts from ``u' = 0``, so
all `$\omega'$`/`$u'$`-proportional terms -- including the rotational
nonlinear term -- vanish), this test feeds a non-trivial field through
the full nonlinear path, catching advection / ``rhs.py`` regressions a
laminar run reports as ``err = 0``.  It is also the triply-periodic
family's first stepping test (via Kolmogorov).

The base ``plane-couette`` entry doubles as two extra guards: it
omits ``--init.random_field`` to verify random is the **default**
start mode (with no snapshot and no explicit mode selected, the run
must start from a random field, not the laminar state -- the
snapshot-first / random-default precedence in ``__main__.py``), and
it passes ``--solver.rhs_transform_chunks 2`` to drive the chunked
RHS inverse transform of ``rhs.get_nonlin`` (``fft.chunked_transform``)
through the jitted stepper -- every other entry runs the default
fused 6-field batch, and the chunk-vs-fused bit-parity is
unit-covered by ``test_padding.py``.
Three ``*-gemm`` entries (``plane-couette-gemm``, ``pipe-gemm``,
``taylor-couette-gemm``) pass ``--solver.wall_normal_matvec dense``:
on CPU the wall-normal derivatives default to the stencil, so these
keep the dense GEMM -- the GPU path, and the reference -- stepping
here, one per geometry (the two paths' agreement:
``test_wall_normal_matvec.py``).
A ``pipe-half-tanh`` entry runs the pipe on its one named non-CGL
radial grid, whose weights take the positive-order composite rule
(``fd.positive_quadrature``) that no CGL entry reaches.
Four entries (``plane-couette-cnab2``,
``pipe-cnab2``, ``taylor-couette-cnab2``, ``kolmogorov-cnab2``) pass
``--step.scheme cnab2`` to drive the alternative CN/AB2 time-stepping
scheme -- Crank-Nicolson viscous + explicit 2nd-order Adams-Bashforth
nonlinear, one FFT eval per step -- across all four geometry families.
For the wall-bounded families this exercises the implicit treatment of
the stiff base-flow coupling (an FFT-free corrector; see ``_l_bf``
/ ``step_cnab2``) plus the iterative-CN self-start, so cnab2 stays
stable on the wall-clustered CGL grid where a naive explicit-AB2 of the
coupling blows up at CFL << 1; triply-periodic (Kolmogorov) drives the
plain no-corrector path (uniform Fourier grid, no coupling stiffness).
The two moving-wall entries use ``ny = 48`` on purpose (the coupling
stiffness that treatment targets scales with the near-wall clustering);
the stationary-wall pipe uses ``ny = 32`` (its coupling is mild --
``U -> 0`` at the wall -- so cnab2 there is bounded only by the
ordinary explicit
self-advection CFL, which ``ny = 48`` on the fine near-wall CGL grid
would violate at this ``Re``, an inherent explicit-scheme limit, not a
coupling instability).

The wall-bounded iterative-cn entries run the **unsplit** corrector
(``step.split_corrector`` defaults off).  A trailing
``taylor-couette-split-corrector`` entry passes
``--step.split_corrector True`` to cover the opt-in split corrector
(linear coupling iterated FFT-free between full-RHS refreshes; see
``_split_core`` in ``timestep.py``) on the annular machinery.

Every entry above runs the **default** ``res.consistent_imm``
reconstruction scheme, so the nonlinear stability gate the one-step
``test_imm_continuity.py`` guard cannot provide is covered for the
whole table (a state-side projection passes every linear gate yet
goes non-finite within ~6 steps of exactly the plane-couette
configuration here).  Five ``*-legacy-imm`` entries then drive the
legacy primitive `$(v, p)$` path (``--res.consistent_imm False``)
through the same nonlinear runs, so the non-default formulation keeps
its own coverage:
plane-couette and taylor-couette under both schemes, plus the pipe
under iterative-cn.  None of the pipe
entries can see the cylindrical default scheme's low-``Re`` limit --
they run decades above its threshold, deliberately (that limit and its
measurements: the ``_cylindrical_stepping`` Design notes).

A trailing forced multi-device, padding-inducing entry (``mpi-pad``:
``mpirun --oversubscribe -np 2``, ``np1 = 2``, ``nx // 2`` not
divisible by ``np1``) is the regression guard for the per-device
(no-replication) random-IC build with spectral padding -- and, since
``solver.backend`` defaults to pallas, it also runs the *cartesian*
banded operator build/solve on that sharded padded axis.  A
``pipe-pallas-mpi-pad`` entry repeats the sharded/padded shape for the
banded-operator variant that entry does not reach (the CPU pure-JAX
banded path; the Triton kernel itself is GPU-only): the pipe's
parity-selected base band; single-device band-vs-dense parity per
geometry is unit-covered by ``test_cartesian.py`` /
``test_cylindrical.py`` / ``test_annular.py``, the shard_map-local
solve by ``test_banded_solver_sharded.py``.  ``plane-couette-mpi-pad`` and
``pipe-pallas-mpi-pad`` also write ``[lowres]`` snapshots (the static
pressure in the first, the radial parity regrid in the second) and
read them back (``_check_lowres``).  The multi-process entries are
also the only ones that keep ``stop.check_laminarization`` on (their
`$E'$` stays decades above its threshold): a flow's global arrays must
reach every jitted program as arguments, a closed-over one traces on
one process and is refused on two, and the laminarization read and
the basis crossing are such programs.

A ``plane-couette-error-max`` entry pins that the corrector-error
check judges every step since the previous host sync, not only the one
it lands on (its closing ``err =`` against its own ``corrector.dat``;
see the entry).

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

Each system steps at ``--dt`` (default 0.01) unless its entry forces
its own (``force_dt``: the adaptive and non-finite-guard entries).

Each system runs in a separate subprocess (the geometry modules capture
global singletons at import time) in its own temporary directory, so
``parameters.toml`` is not loaded (model defaults + CLI args only) and
the per-system ``stats.dat`` does not collide.

Success criteria per system: exit code 0, the shared criteria of
``tests/_smoke.py`` (reached the end, no corrector failure, a finite
converged final error, a finite summary line), and on CPU the closing
memory lines over every rank (``tests/_random_smoke.py``).

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
        # The base Cartesian entry doubles as two extra guards.  It
        # OMITS --init.random_field: with no snapshot and no explicit
        # mode the run must fall through to the random-IC default
        # (start_from_laminar defaults off) -- the snapshot-first /
        # random-default precedence in __main__.py; run_smoke_test
        # asserts the random-IC startup line.  And it passes
        # --solver.rhs_transform_chunks 2 to drive the chunked RHS
        # inverse transform of rhs.get_nonlin (fft.chunked_transform)
        # end to end through the jitted stepper -- every other entry
        # runs the default fused 6-field batch, and chunk-vs-fused
        # bit-parity is unit-guarded (test_padding.py).
        # plane-poiseuille needs no entry of its own: it shares this
        # entry's Cartesian nonlinear machinery, and its
        # driving/base-flow specifics are covered by the laminar
        # smoke (pp + pp-cbv) and the transient-growth PP case.
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
    # The pipe's wall-clustered one-sided tanh grid, the one named
    # non-CGL radial grid: it steps on the positive-order composite
    # quadrature (``fd.positive_quadrature``), which no CGL row reaches.
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
    # The GEMM path of the wall-normal derivatives, one row per
    # geometry: ``solver.wall_normal_matvec`` defaults to the stencil on
    # CPU, so without these the GEMM -- the GPU path, and the reference
    # -- would never step here.  Agreement of the two paths is pinned by
    # test_wall_normal_matvec.py; these drive the GEMM end to end.
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
        # Regression guard for the per-device (no-replication) random
        # build with spectral padding: always multi-device on the kx axis
        # (np1 = 2), with nx // 2 = 17 not divisible by np1 (padded to
        # 18).  Exercises the multi-device spectral-padding path.
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
        # Regression guard for the Pallas banded backend
        # (solver.backend=pallas) on a sharded, padded mode axis:
        # multi-device on the kx/axial axis (np1 = 2) with nx // 2 = 3
        # not divisible by np1 (padded to 4).  The per-mode banded
        # operators are built and the nonlinear path solved on the
        # sharded axis -- catching sharding bugs the single-device run
        # misses.  On CPU the backend runs the pure-JAX banded sweep
        # (the Triton kernel is GPU-only).
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
        # CN/AB2 scheme (step.scheme=cnab2) on the Cartesian
        # (plane-Couette) geometry, driven through the full nonlinear
        # path.  Run at a **higher wall-normal resolution** (ny=48) than
        # the other Cartesian entries on purpose: plane-Couette is a
        # *moving-wall* flow (``U = y``, so ``U ~ O(1)`` at the walls),
        # so on the wall-clustered CGL grid the rotational base-flow
        # coupling ``U d(u')/dy`` is a stiff explicit term and a naive
        # explicit-AB2 nonlinear blows up at dt=0.01 once ny >~ 40
        # (CFL << 1).  This guards the implicit treatment of the
        # base-flow coupling (FFT-free corrector; see ``_l_bf`` /
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
        # The legacy primitive (v, p) path, driven nonlinearly: the
        # plane-couette entry above already covers the default
        # v-omega_y scheme, and this one keeps the legacy formulation
        # under the same nonlinear load.  The laminar smoke cannot
        # separate them (every term is linear in u', so all vanish at
        # u' = 0) and the discrete-continuity guard
        # (``tests/test_imm_continuity.py``) takes one step, so only
        # entries of this shape integrate either formulation through a
        # real nonlinear run.  That matters more here than anywhere
        # else in this file: a state-side projection passes every
        # *linear* gate and still goes non-finite within ~6 steps of
        # exactly this configuration (the
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
        # The same scheme under cnab2, which is a genuinely different
        # path into it: the AB2/CN split hands ``_imm_iteration`` a
        # combination assembled from ``_l_bf`` and the carried
        # nonlinear history, and that combination is what the
        # v-omega_y source projections act on, and no other entry
        # integrates this combination nonlinearly.
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
        # The annular half of the same guard: the u_r-omega_r pair,
        # whose spin coupling is the one linear term still lagged to
        # the corrector iterate (measured contraction <= 0.02, but
        # measured on *linear* data -- this entry is where the loop
        # meets a real nonlinear source).
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
        # The annular flag-on scheme under cnab2 -- a genuinely
        # different path in: the AB2/CN split hands ``_imm_iteration``
        # a combination assembled from ``_l_bf`` and the carried
        # nonlinear history, and the lagged spin coupling then reads a
        # *carry*-derived iterate rather than a corrector one.
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
        # The pipe half: the spin-quad formulation, on a *random* IC
        # -- the guard that the flag carries no resolved-IC
        # restriction (a composed, non-grid-scale-dissipative `$D_2$`
        # would blow up on a grid-white draw near the axis; the
        # reformulation composes nothing).  Half-CGL grid
        # (the iterative-cn default), i.e. the innermost node sits
        # closest to the axis -- the hardest case for every 1/r term
        # in the scheme.
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
        # CN/AB2 on the annular (Taylor-Couette) geometry at ny=48,
        # guarding the annular ``_l_bf`` (three stacked Hk) build +
        # implicit-coupling corrector + iterative-CN self-start.  Uses a
        # standard (inner-rotating, stationary-outer) config, which is a
        # *moving-wall* flow (``U_theta ~ O(1)`` at the inner wall): its
        # ``U_theta d(u')/dr`` coupling is stiff on the wall-clustered
        # grid, so ny=48 stresses exactly the stiffness the implicit
        # coupling removes.
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
        # corrector integration-covered, on the annular machinery.
        # Same config as the iterative-cn ``taylor-couette`` entry.
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
        # ldt_1/ildt_2 recompute.  The CFL is tiny here, so the
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
        # Corrector-error guard between host checks: the loop syncs the
        # error every it_error_check steps and must judge every step
        # since the previous check, not only the one it lands on.  At
        # it_error_check = it_corrector = 10 over 10 steps, the single
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
        # Non-finite guard (inverted criteria; see run_smoke_test):
        # cnab2 at dt = 1.0 (100x the default dt, CFL >> 1) blows up
        # within tens of steps; the run must abort
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


if __name__ == "__main__":
    main(SYSTEMS)
