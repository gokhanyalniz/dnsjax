"""Simulation parameter management via Pydantic models and TOML files.

Configuration is layered, lowest priority first: hard-coded defaults ->
parameters embedded in a resumed snapshot (:func:`read_snapshot_params`)
-> ``parameters.toml`` (if present) -> command-line arguments.  The
snapshot layer is skipped for the parameters that must be known to
configure JAX *before* the snapshot is read (``dist.np0``, ``dist.np1``,
``dist.platform``, ``res.double_precision``); those come only from
defaults / TOML / CLI.  The global singletons ``params``,
``derived_params``, and ``padded_res`` are mutated in-place by
:func:`update_parameters` so that every module sees the same state.
"""

import tomllib
from dataclasses import dataclass
from datetime import timedelta
from math import cos, pi, sin
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

monochromatic_systems: list[str] = ["kolmogorov", "waleffe"]
periodic_systems: list[str] = ["decaying-box", *monochromatic_systems]

cartesian_systems: list[str] = ["plane-couette", "plane-poiseuille"]
cylindrical_systems: list[str] = ["pipe"]
annular_systems: list[str] = ["taylor-couette", "dean"]
walled_systems: list[str] = [
    *cartesian_systems,
    *cylindrical_systems,
    *annular_systems,
]

ns_to_s: float = 10 ** (-9)  # nanoseconds to seconds


class Distribution(BaseModel):
    r"""Device distribution and backend platform.

    The total device count is ``np0 * np1``.  Both default
    to 1 (single device).

    Double parallelisation
    ----------------------
    ``np0`` splits the wall-normal axis (`$y$` / `$r$`) in
    physical space and the spanwise-wavenumber axis
    (`$k_z$` / `$m$`) in spectral space.  ``np1`` splits
    the spanwise axis (`$z$`) in physical space and the
    streamwise-wavenumber axis (`$k_x$`) in spectral space.
    Each device holds the full wall-normal extent in spectral
    space, so FD / SPIKE solves are unchanged.

    When ``np0 == 1`` (default), the decomposition collapses
    to the original 1D scheme (only `$k_x$` / `$z$`
    distributed).

    Divisibility
    ------------
    ``np1`` requires ``nz_padded % np1 == 0`` (where
    ``nz_padded = oversampling_factor * nz // 2``).
    ``np0`` auto-pads when needed: the spectral `$k_z$`
    axis is zero-padded to the next multiple of ``np0``;
    for wall-bounded flows the physical `$y$` axis is
    likewise zero-padded (stripped after the
    `$y \leftrightarrow k_z$` reshard); for periodic flows
    ``ny_padded`` is bumped to the next multiple of ``np0``
    (marginally more oversampling, physically neutral).
    Power-of-2 ``ny`` avoids the padding overhead.  Note
    that CGL grids traditionally use ``ny = 2^k + 1``
    (``N + 1`` collocation points for ``N`` Chebyshev
    polynomials), but any ``ny >= 2`` is valid: the code
    uses finite differences, not spectral Chebyshev
    transforms, and the Clenshaw--Curtis quadrature handles
    both even and odd ``ny``.
    """

    np0: int = Field(ge=1, default=1)
    np1: int = Field(ge=1, default=1)
    platform: Literal["cpu", "cuda", "rocm", "tpu"] = "cpu"


class Physics(BaseModel):
    """Physical parameters: Reynolds number, flow system, dealiasing."""

    re: float = Field(gt=0, default=1000)  # Reynolds number
    # Taylor-Couette control parameters (system == "taylor-couette").
    # Inner / outer cylinder Reynolds numbers, with gap d = r2 - r1:
    #   re1 = Omega1 * r1_dim * d / nu,  re2 = Omega2 * r2_dim * d / nu.
    # Sign convention: re1 >= 0; re2 may be negative (counter-rotation).
    # Case 1 (inner-driven): re1 > 0  -> Re_ref = re1.
    # Case 2 (outer-driven): re1 == 0, re2 > 0 -> Re_ref = re2.
    # ``update_parameters`` validates these, derives the circular-Couette
    # coefficients A0, B0 and radii (onto ``derived_params``), and sets
    # ``re = Re_ref`` so every downstream 1/re viscous/IMM/stats path is
    # reused unchanged.  See the annular branch of ``update_parameters``
    # and ``flows.wall_bounded.taylor_couette``.
    re1: float | None = None
    re2: float | None = None
    # Default "plane-couette": a wall-bounded flow that integrates
    # cleanly from the default random IC at the default dt (Kolmogorov +
    # random needs a smaller dt; see the random-IC corrector note in the
    # root CLAUDE.md).  Kolmogorov: sine forcing.  Waleffe: cosine
    # forcing + Ry symmetry (not yet implemented).
    system: Literal[*periodic_systems, *walled_systems] = "plane-couette"
    # (n + 1) / 2 oversampling in each direction
    # to dealias the n'th order nonlinearity
    # oversampling_factor = n + 1
    oversampling_factor: int = Field(ge=2, default=3)
    oversample_y: bool = True
    driving: Literal[
        "constant_pressure_gradient", "constant_bulk_velocity"
    ] = "constant_pressure_gradient"
    # Zero the mean velocity in the undriven homogeneous direction.
    # Cartesian: the spanwise (z) direction.  Taylor-Couette: the axial
    # (z) direction (no axial bulk velocity); the azimuthal mean evolves
    # freely.  Independent of ``driving``.
    block_mean_spanwise_velocity: bool = False
    # Speed U_grid of the moving frame of reference, translating along
    # the homogeneous "grid" direction: streamwise x (Cartesian) or
    # axial z (cylindrical / annular).  The time derivative becomes
    # d/dt - U_grid d/dx_0, i.e. the *convective-form* frame term
    # +U_grid d/dx_0 u' = i k_0 U_grid u' is added to the RHS -- a
    # mode-diagonal, non-stiff, divergence-free (projection-neutral)
    # term, integrated implicitly (inside the iterative-CN corrector;
    # via ``_l_bf`` for CN/AB2).  It de-advects snapshots, improves
    # temporal accuracy, and relaxes the corrector-contraction dt limit
    # (the advecting velocity drops to ``U - U_grid``).  NOT the
    # rotational-form splitting `omega' x c + grad(c . u')` of the
    # removed first implementation, whose explicit `c d/dy u'` piece
    # was wall-stiff and blew up.  When ``None`` (default) it resolves
    # to the laminar bulk velocity in the grid direction (1/2 pipe,
    # 2/3 plane-Poiseuille, 0 otherwise); see ``update_parameters`` and
    # ``derived_params.u_grid``.  Only meaningful for wall-bounded
    # systems (periodic flows reject it).  A changed ``u_grid`` on
    # resume is trajectory-defining (the stored fields drift between
    # frames); pre-feature snapshots resume into the new default.
    u_grid: float | None = None


class Geometry(BaseModel):
    r"""Domain size and optional tilt angle for the forcing direction.

    Wall-normal grid selection (precedence order):

    1. ``wall_grid`` (file path): load a custom grid from file.
       A custom grid always overrides dnsjax's grid generation.
    2. ``grid_type``: generate a named grid at startup.
    3. Default: CGL (Cartesian) or radial CGL (cylindrical; shaped
       by ``axis_gap``).

    Setting both ``wall_grid`` and ``grid_type`` is an error.

    ``axis_gap`` (cylindrical geometry only) sets how far the
    innermost *generated* radial point sits from the axis: the
    radial grid keeps the ``ny`` outermost positive points of a
    `$(2 n_y + g)$`-point CGL grid on `$[-1, 1]$` (``g =
    axis_gap``), so with the near-axis spacing `$\Delta r \approx
    \pi/(2 n_y)$` the innermost point is `$r_0 \approx
    (g{+}1)\,\Delta r/2$` (exactly `$r_0 = \sin(\pi (g{+}1) /
    (2\,(2 n_y + g - 1)))$`).  ``g = 0`` is the legacy half-CGL
    grid (even auxiliary total, staggered `$r_0 = \Delta r/2$`);
    the default ``g = 1`` uses an odd total whose centre point
    falls exactly on the coordinate-singular axis and drops it,
    doubling `$r_0$`.  Rationale: the near-axis *azimuthal*
    advection CFL `$\propto 1/r_0$` is a stability artifact of
    explicit stepping evaluated at grid points only -- no degree
    of freedom lives in `$[0, r_0)$` (parity ghosts close the FD
    stencils across the axis; the quadrature covers the segment
    with an even-parity gap rule) -- so that CFL relaxes
    `$\propto (g{+}1)$` at a truncation-level accuracy cost (the
    parity-mirrored gap across the axis widens to
    `$(g{+}1)\,\Delta r$`).  The default ``g = 1`` realises the
    full 2x on the admissible cnab2 ``dt`` (measured); ``g >= 2``
    keeps relaxing the azimuthal CFL but its wider mirrored axis
    hole introduces its own explicit near-axis instability
    (radially growing, coupling-corrector-stressing) that
    *lowers* cnab2's admissible ``dt`` below the ``g = 1`` value
    -- ``iterative-cn`` integrates ``g >= 2`` cleanly, so larger
    gaps are for that scheme.  Ignored by ``wall_grid`` /
    ``grid_type`` grids; a non-default value is rejected for
    non-cylindrical systems.
    """

    lx: float = Field(gt=0, default=4.0)
    lz: float = Field(gt=0, default=4.0)
    tilt_degree: float = Field(gt=-180, le=180, default=0)
    # Radius ratio eta = r1/r2 for the annular (Taylor-Couette)
    # geometry.  Non-dim radii r1 = eta/(1-eta), r2 = 1/(1-eta) on the
    # gap d = r2 - r1 = 1.  Required for system == "taylor-couette".
    eta: float | None = Field(default=None, gt=0, lt=1)
    wall_grid: Path | None = None
    grid_type: Literal["cgl", "tanh"] | None = None
    grid_stretch: float = Field(gt=0, default=1.5)
    axis_gap: int = Field(ge=0, default=1)


class Resolution(BaseModel):
    """Grid resolution (number of Fourier modes before dealiasing)."""

    # Number of grid points = (before oversampling) # of Fourier modes
    nx: int = Field(ge=1, default=128)
    ny: int = Field(ge=1, default=128)
    nz: int = Field(ge=1, default=128)
    fd_order: int = Field(ge=2, default=4)
    double_precision: bool = True  # use double-precision floating point


class Initiation(BaseModel):
    """Initial condition: from a snapshot, a random field (default), or
    laminar.

    Start-mode precedence (resolved in ``__main__.py``): a provided
    ``snapshot`` file (a single-file tar snapshot, or a legacy ``.npz``;
    see :mod:`dnsjax.snapshot`) wins over every in-process mode;
    otherwise ``start_from_laminar`` (the laminar / closed-form base
    state); otherwise ``localized_rolls`` (an in-process deterministic
    streamwise-localized-rolls perturbation, wall-bounded only);
    otherwise ``random_field`` -- an in-process random divergence-free
    perturbation, which is **the default**: a run with no snapshot and
    no explicit mode selected starts from a random IC.  The ``random_*``
    knobs feed :func:`dnsjax.random_field.generate_random_state`; the
    ``localized_rolls_*`` knobs feed
    :func:`dnsjax.localized_rolls.generate_localized_rolls`.

    Resume policy: when ``snapshot`` is a dnsjax snapshot, ``it``/``t``/
    ``isnap`` are inherited only when none of the Physics/Geometry/
    Resolution parameters were overridden to a value different from the
    snapshot's (a *continuation*).  Any such change starts a NEW
    trajectory by default (``it = t = isnap = 0``); ``force_resume``
    keeps the run continuous instead.  See
    :func:`trajectory_defining_changes`.
    """

    # Start from the laminar / closed-form base state (zero
    # perturbation; for Dean the analytical laminar profile).  Defaults
    # off -- it must be set explicitly; when set it takes precedence over
    # ``localized_rolls`` and the default ``random_field`` (but not a
    # provided snapshot).
    start_from_laminar: bool = False
    snapshot: Path | None = None
    t0: float = 0  # Initial value of time
    it0: int = 0  # Initial value of number of time steps taken
    # Initial value of the snapshot counter ``isnap`` (snapshots are
    # named ``state{isnap}.tar``).  Mirrors ``it0``/``t0``: this is the
    # fresh-start value, and on a *continuation* resume it is inherited
    # from the snapshot (the resumed file's index + 1) instead.
    isnap0: int = Field(ge=0, default=0)
    # Continue the resumed trajectory (inherit ``it``/``t``/``isnap``)
    # even when Physics/Geometry/Resolution parameters differ from the
    # snapshot, instead of starting a new trajectory.  Does not bypass
    # the hard nx/nz/system/precision mismatches that
    # ``snapshot.validate_snapshot_params`` rejects.
    force_resume: bool = False
    # Generate an in-process random divergence-free perturbation of the
    # base flow as the initial condition.  This is the **default** start
    # mode (lowest precedence among the non-snapshot modes): it is used
    # when no snapshot is given and neither ``start_from_laminar`` nor
    # ``localized_rolls`` is set.  Ignored when a snapshot is given.  For
    # Dean the analytical laminar profile is added (total-field IC).
    random_field: bool = True
    random_amplitude: float = 0.1  # target L2 norm of the perturbation
    random_smoothness: float = Field(
        gt=0, lt=1, default=0.4
    )  # spectral decay rate (0 < s < 1)
    random_seed: int = 1  # NumPy PRNG seed (device-count independent)
    random_mean_flow: bool = False  # also perturb the mean (kx=kz=0) mode
    # Generate an in-process deterministic localized-rolls ("turbulent
    # spot") perturbation (wall-bounded only; higher precedence than the
    # default ``random_field``, lower than ``start_from_laminar``).  A
    # compact fixed-physical structure normalized so
    # peak |u'| = amplitude, localized in every homogeneous direction
    # (growing a box length adds laminar around the spot).  ``width`` is
    # the physical localization half-width (flow units); ``wavelength`` is
    # the cross-roll spanwise wavelength (flow units; ignored by the pipe,
    # whose cross-section is the fixed m = +-1 mode).  For Dean the
    # analytical laminar profile is added (total-field IC).
    localized_rolls: bool = False
    localized_rolls_amplitude: float = 0.1  # peak perturbation velocity
    localized_rolls_width: float = 2.0  # physical localization width
    localized_rolls_wavelength: float = 4.0  # cross-roll wavelength


class Outputs(BaseModel):
    """Output frequency controls (in time-step counts)."""

    # All outputs are with respect to the number of time steps taken
    it_stats: int | None = None  # How often to compute stats
    # How often (in steps) to record the time-step (CFL) diagnostic
    # into ``steps.dat``.  The measurement is taken from the current
    # state `$u^n$` at the step's first nonlinear-term evaluation
    # (no extra Fourier transforms).  ``None`` disables it.
    it_steps: int | None = None
    it_snapshot: int | None = None  # How often to save snapshots
    # How often (in steps) to record the corrector diagnostic (the
    # corrector iteration count ``c`` and the final corrector error)
    # into ``corrector.dat``, with the same on-device buffering and file
    # format as ``stats.dat``.  ``None`` disables it.  When set,
    # ``it_error_check`` must be <= ``it_corrector`` (see
    # ``validate_parameters``) so the convergence check is at least as
    # frequent as the logging.
    it_corrector: int | None = Field(default=None, ge=1)
    # How often (in steps) to sync the corrector error to the host
    # for the convergence check.  Between checks the host enqueues
    # steps ahead of the device (JAX async dispatch); corrector
    # divergence is therefore detected up to ``it_error_check``
    # steps late, each late step bounded by
    # ``max_corrector_iterations``.  1 restores a per-step check
    # (and a per-step host-device sync).  Must be <= ``it_corrector``
    # when the corrector diagnostic is enabled.
    it_error_check: int = Field(ge=1, default=10)
    # Rows buffered on device before flushing ``stats.dat`` /
    # ``steps.dat`` / ``corrector.dat`` to disk.
    nbuffer: int = Field(ge=1, default=100)
    stats_precision: int = Field(ge=1, le=17, default=9)
    # Snapshot filenames are ``state{isnap:0Nd}.tar`` with N =
    # ``snapshot_pad_width`` (a *minimum* width; a larger ``isnap`` is
    # not truncated).  ``isnap`` starts at ``init.isnap0`` and is
    # incremented on every snapshot written.
    snapshot_pad_width: int = Field(ge=1, default=5)
    # Embed the state's stats (``get_stats``) into every snapshot as a
    # ``_dnsjax_stats.json`` member.  The periodic-snapshot path reuses
    # the ``it_stats`` computation when the iterations coincide, else it
    # computes the stats once for the snapshot.
    snapshot_embed_stats: bool = True
    # Save the initial condition as ``state00000.tar`` when the run does
    # not *continue* a dnsjax snapshot trajectory (random / localized-
    # rolls / legacy-``.npz`` / laminar start, or a resume that changed
    # the Physics/Geometry/Resolution).  Independent of ``it_snapshot``.
    snapshot_save_initial: bool = True
    # Save the final state as a snapshot when the simulation terminates.
    # Independent of ``it_snapshot``; skipped when the final state was
    # just written (e.g. it coincided with a periodic snapshot).
    snapshot_save_final: bool = True
    # How processes write the snapshot's shared tar file:
    #   "concurrent": all processes write their disjoint byte ranges
    #                 at once (fast; POSIX/parallel filesystems).
    #   "serial":     rank-ordered (token-passing) writes, one
    #                 process at a time -- safe on filesystems such
    #                 as NFS where concurrent writes can corrupt
    #                 data.  No effect for single-process runs.
    snapshot_write_mode: Literal["concurrent", "serial"] = "concurrent"


class TimeStepping(BaseModel):
    r"""Time integration parameters.

    Two schemes (``scheme``), both semi-implicit (implicit viscous,
    IMM pressure) and both second-order:

    - ``"iterative-cn"`` (default): Euler predictor + iterative
      Crank-Nicolson corrector (Willis 2017).  Implicitness *c* is
      applied to **both** the viscous *and* the nonlinear term; the
      nonlinear ``c N^{n+1}`` is resolved by the corrector fixed-point
      iteration (the RHS is re-evaluated until the correction converges).
      Stable well past the advective CFL; costs ``2 + num_corrector``
      RHS/FFT evaluations per step.
    - ``"cnab2"``: Crank-Nicolson viscous (implicitness *c*) + 2nd-order
      Adams-Bashforth nonlinear (explicit ``1.5 N^n - 0.5 N^{n-1}``).
      **One** expensive nonlinear/FFT evaluation per step; the previous
      nonlinear RHS is carried by the main loop, seeded by an
      ``iterative-cn`` self-start on the very first step.  Explicit
      *self-advection*, so ``dt`` is advective-CFL-limited -- a net win
      (~3x fewer FFTs) on CFL-limited (turbulent) runs.

      Wall-bounded caveat and coupling corrector.  In the rotational
      perturbation form the nonlinear term includes the *linear*
      base-flow coupling ``L_bf = u' x curl(U) + U x omega'``, whose
      ``U d(u')/dy`` piece is a wall-normal derivative.  On the
      wall-clustered CGL grid (``dy ~ 1/N^2`` near the wall) treating
      ``L_bf`` explicitly gives it a Chebyshev-type CFL ``dt <~ 1/N^2``
      -- far below the advective limit and *amplitude-independent* -- so
      a naive explicit-AB2 cnab2 blows up at CFL << 1 for moving-wall
      flows (plane-Couette, Taylor-Couette, where ``U ~ O(1)`` at the
      wall).  To restore the advective limit, wall-bounded cnab2 makes
      only the *self-advection* ``u' x omega'`` explicit (AB2) and treats
      ``L_bf`` implicitly (Crank-Nicolson) via an **FFT-free** fixed-point
      corrector (it re-evaluates only the matrix-free ``L_bf``, no FFT),
      so ``corrector_tolerance`` / ``max_corrector_iterations`` **do**
      apply here.  The first step self-starts with ``iterative-cn``
      (also seeding ``N^{n-1}``), and if the coupling corrector fails to
      converge on a step (its Picard rate reaches 1 -- only at ``dt`` well
      past the advective limit) that step automatically falls back to a
      full ``iterative-cn`` step (a stdout diagnostic is printed).  The
      residual ``dt`` bound is then the ordinary explicit self-advection
      CFL of the *fluctuations* on the clustered grid (stationary-wall
      flows -- Poiseuille, pipe, Dean -- are bounded only by this, their
      ``L_bf`` being mild as ``U -> 0`` at the wall).  Where it binds is
      geometry-specific: for the **pipe** it is the *near-axis
      azimuthal* advection (the innermost radial node
      ``r_0 ~ (geo.axis_gap + 1) pi/(4 ny)`` makes
      ``CFL_th = dt |u_th(r_0)| nz/(2 pi r_0)`` the dominant
      column -- linear in ``nz`` and in the fluctuation amplitude, and
      a *weak* AB2 imaginary-axis instability, so it needs sustained
      ``CFL_th >~ 0.5`` and pass/fail is trajectory-marginal near the
      boundary; the default ``geo.axis_gap = 1`` doubles ``r_0`` and
      -- measured -- the admissible ``dt``, but ``axis_gap >= 2``
      trades it for a different explicit near-axis instability and
      suits only ``iterative-cn``); Cartesian flows feel the
      near-wall ``dy ~ 1/N^2`` spacing instead.  A strongly
      non-normal base flow
      (counter-rotating Taylor-Couette) amplifies the explicit
      self-advection error further into a delayed blow-up needing a
      much smaller ``dt``.  These are inherent explicit-nonlinear
      limits, not the coupling bug: such regimes want ``iterative-cn``
      or a smaller ``dt``.  Triply-periodic cnab2 has none of this (uniform
      Fourier grid, no coupling stiffness): it is the plain one-FFT
      no-corrector explicit-AB2 step.

      ``implicit_mean_coupling`` (wall-bounded cnab2 only, default on)
      additionally folds the coupling with the *instantaneous mean
      flow* -- ``L_mf = u' x curl(mean u') + (mean u') x omega'``, the
      same cross-product structure as ``L_bf`` with the time-varying
      mean profile ``extract_mean_mode(u')`` in place of ``U`` -- into
      the implicit coupling term, still FFT-free (the mean mode is a
      ``psum``; each geometry ``_l_bf`` adds the mean profiles onto the
      base-flow profiles, the coupling being linear in the profile
      pair).  The explicit AB2 remainder is then the pure
      fluctuation-fluctuation advection: the mean-flow *distortion*
      (streaks; for total-field Dean the entire evolving mean profile,
      whose ``L_bf`` is otherwise zero) no longer rides the explicit
      term, removing its advective-CFL contribution.  The double-counted
      mean-mean product is a purely wall-normal (radial) profile at the
      mean mode, absorbed by the mean pressure in the projection -- and
      the AB2/CN split is second-order consistent for *any* choice of
      the implicit functional, since the explicit part is always the
      exact remainder ``get_rhs - l_bf``.

    ``implicitness`` *c* is the Crank-Nicolson split weight
    (``c = 0.5`` = second-order trapezoidal): in ``"iterative-cn"`` it
    weights both the viscous *and* the nonlinear term (see the geometry
    ``_imm_iteration``); in ``"cnab2"`` it weights the viscous term (and,
    wall-bounded, the implicit base-flow coupling), while the explicit
    AB2 self-advection is independent of *c*.
    """

    scheme: Literal["iterative-cn", "cnab2"] = "iterative-cn"
    dt: float = Field(gt=0, default=0.01)
    implicitness: float = Field(ge=0, le=1, default=0.5)
    corrector_tolerance: float = Field(gt=0, default=1e-5)
    max_corrector_iterations: int = Field(ge=1, default=10)
    # Fold the instantaneous mean-flow coupling ``L_mf`` into the
    # implicit (FFT-free) coupling term of the wall-bounded CN/AB2
    # scheme; no effect on ``"iterative-cn"`` or triply-periodic flows.
    # See the class docstring.
    implicit_mean_coupling: bool = True


class Termination(BaseModel):
    """Stopping criteria for the simulation."""

    max_sim_time: float | None = None
    max_wall_time: timedelta | None = None  # ISO 8601 format for durations
    # Laminarization (relaminarization) check.  When enabled, the run
    # terminates once the perturbation kinetic energy ``E'`` drops
    # below ``laminarization_threshold`` (the flow has returned to
    # laminar).  ``E'`` is read on the host every
    # ``outs.it_error_check`` steps (the corrector-error sync point),
    # so detection lags by up to that many steps.  For Dean (total
    # field) ``E'`` is the kinetic energy of the deviation from the
    # analytical laminar profile.  Disabled in all tests.
    check_laminarization: bool = True
    laminarization_threshold: float = Field(gt=0, default=1e-9)


class Solver(BaseModel):
    """Linear algebraic solver configurations."""

    # ``"banded"``: SPIKE block-partitioned solver (memory-efficient,
    # exploits the known stencil bandwidth of D1, D2).
    # ``"dense"``: full ``Ny x Ny`` LU factors per Fourier mode
    # (legacy path, kept for verification against the banded path).
    # ``"pallas"``: one-program-per-mode sequential banded sweep via a
    # Pallas/Triton kernel (GPU); CPU runs the same banded math in pure
    # JAX.  Uses a no-pivot banded LU, falling back per operator group
    # to the pivoted ``"banded"`` (SPIKE) solver when unstable or when
    # ``pallas_force_pivoting`` is set (see ``solvers.py``).  Implemented
    # for all wall-bounded geometries (cartesian, cylindrical, annular);
    # the operators are assembled directly in banded storage by each
    # geometry's ``_build_{Lk,Hk}_band_gpu`` via the shared
    # ``solvers._assemble_banded_operator`` helper.
    backend: Literal["banded", "dense", "pallas"] = "banded"
    # ``"pallas"`` backend only: force the pivoted SPIKE fallback for
    # every operator group instead of the no-pivot banded LU.  Default
    # ``False`` decides per group from a setup-time stability residual
    # (a diagnostic line is printed either way).
    pallas_force_pivoting: bool = False
    # ``"pallas"`` backend only: Pallas mode-tile size along the ``k_z``
    # mode axis (one Pallas program solves a ``bm0 x bm1`` tile of
    # Fourier modes, vectorising the banded sweep across the tile).  ``1``
    # is one program per mode; ``> 1`` coalesces mode loads and fills more
    # SIMD lanes.  The default ``2`` is the H100 tuning (4 warps/program
    # with ``k = 2``).  Partial boundary tiles are padded to full tiles
    # inside the kernel (a masked partial-tile band load miscompiles on
    # real Triton -- see ``solvers._pallas_banded_solve``).  Must be a
    # power of two.
    pallas_block_m0: int = Field(ge=1, default=2)
    # ``"pallas"`` backend only: Pallas mode-tile size along the
    # contiguous ``k_x`` mode axis (the innermost, coalesced axis).  The
    # default ``32`` is the H100 tuning -- one warp wide, so a warp's band
    # load fully coalesces.  Same internal padding to full tiles as
    # ``pallas_block_m0``.  Must be a power of two.
    pallas_block_m1: int = Field(ge=1, default=32)
    # ``"pallas"`` backend only: no-pivot banded-LU stability threshold
    # (max relative solve residual ``||A x - b|| / ||b||`` over modes,
    # measured once at setup).  Above it, the operator group falls back
    # to the pivoted SPIKE solver.
    pallas_stability_tol: float = Field(gt=0, default=1e-6)
    # ``"pallas"`` backend only: Triton ``num_warps`` for the kernel
    # (warps per Pallas program).  ``None`` lets Triton choose; ``1``
    # forces the whole mode tile into a single warp (a cross-warp
    # diagnostic knob, exercised by ``scripts/pallas_tiling_diagnostic.py``).
    pallas_num_warps: int | None = Field(default=None, ge=1)
    # ``"pallas"`` backend only: Triton ``num_stages`` (software
    # pipelining depth) for the kernel.  ``None`` lets Triton choose.
    pallas_num_stages: int | None = Field(default=None, ge=1)
    # Target SPIKE block size `$m$`.  When set, `$P = N_y / m$`.
    # When ``None`` (default), the block partition that minimises
    # total per-mode SPIKE storage under the selected
    # reduced-system form (``block_thomas``) is chosen
    # automatically.  Use ``scripts/spike_partition_info.py`` to
    # explore the memory / latency trade-off for a resolution.
    spike_block_size: int | None = None
    # Use block-Thomas ``lax.scan`` solves for the SPIKE reduced
    # system: `$O(P p^2)$` memory, but `$2(P-1)$` *sequential*
    # scan steps of kernel-launch latency per solve, and its
    # memory-optimal partitions drift to large `$P$` / small
    # stage-1 blocks.  The default ``False`` solves the reduced
    # system as one batched dense LU solve (`$O(P^2 p^2)$`
    # memory): no sequential launches and larger stage-1 blocks,
    # costing ~28-30% more total SPIKE factor storage at the
    # respective memory-optimal partitions (p = 4).  Set ``True``
    # for memory-tight runs.
    block_thomas: bool = False


class Parameters(BaseModel):
    """Top-level parameter container aggregating all categories."""

    dist: Distribution | None = Distribution()
    phys: Physics = Physics()
    geo: Geometry = Geometry()
    res: Resolution = Resolution()
    init: Initiation = Initiation()
    outs: Outputs = Outputs()
    step: TimeStepping = TimeStepping()
    stop: Termination | None = Termination()
    solver: Solver = Solver()


class CLIParameters(
    BaseSettings,
    Parameters,
    cli_parse_args=True,
    cli_avoid_json=True,
    cli_hide_none_type=True,
    cli_prog_name="dnsjax",
):
    """Command-line arguments override parameters.toml (if present),
    which overrides the default parameters."""


@dataclass
class DerivedParameters:
    """Parameters derived from the user-facing configuration.

    ``ly`` is fixed by the geometry (4 for triply-periodic, 2 for
    Cartesian/cylindrical, ``2*r2`` for the annulus; cosmetic there --
    only read by triply-periodic code).
    ``volume_fac`` is also fixed by the geometry
        (1 for periodic, 2 for Cartesian, 0.5 for cylindrical, and
        ``(r2^2 - r1^2)/2`` for the annulus).
    ``ccf_A``, ``ccf_B`` are the circular-Couette base-flow coefficients
    ``U_theta = ccf_A * r + ccf_B / r`` and ``r_inner``, ``r_outer`` the
    non-dim annular radii (set for system == "taylor-couette").
    ``u_grid`` is the resolved moving-frame speed (always a concrete
    float; see ``params.phys.u_grid`` and ``update_parameters``).
    """

    ly: float = 4
    volume_fac: float = 1
    tilt_rad: float = 0
    cos_tilt: float = 0
    sin_tilt: float = 0
    wall_normal_grid: list[float] | None = None
    ccf_A: float = 0
    ccf_B: float = 0
    r_inner: float = 0
    r_outer: float = 0
    u_grid: float = 0


params: Parameters = Parameters()
derived_params: DerivedParameters = DerivedParameters()


def read_parameters(path: Path) -> Parameters:
    """Load a ``Parameters`` instance from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Parameters(**raw)


# Parameters that must be known to configure JAX *before* a snapshot is
# read (precision, platform, device mesh); they are never inherited from
# a snapshot's embedded parameters (resume is device- and
# precision-agnostic, and the precision mismatch is caught by
# ``snapshot.validate_snapshot_params``).
_SNAPSHOT_SKIP_FIELDS: tuple[tuple[str, str], ...] = (
    ("dist", "np0"),
    ("dist", "np1"),
    ("dist", "platform"),
    ("res", "double_precision"),
)


def read_snapshot_params(snapshot_path: Path) -> Parameters | None:
    """Build a ``Parameters`` from a snapshot's embedded parameters.

    Reads the ``_dnsjax_meta.json`` member of the snapshot tar (via the
    standard-library :mod:`dnsjax.snapshot_meta`; no JAX import, so it is
    safe to call before the distributed backend is configured) and
    returns the embedded ``params`` as a :class:`Parameters`, with the
    JAX-setup fields in :data:`_SNAPSHOT_SKIP_FIELDS` removed so they are
    not inherited.

    Returns ``None`` when *snapshot_path* is not a dnsjax snapshot file
    (legacy ``.npz`` snapshots, a laminar start, or a missing path), so
    the caller simply skips the snapshot layer.  Unknown fields in the
    stored dump are ignored (Pydantic ``extra="ignore"``), making this
    robust to parameter-schema drift across versions.
    """
    from .snapshot_meta import is_snapshot_file, read_snapshot_meta

    if not is_snapshot_file(snapshot_path):
        return None
    meta = read_snapshot_meta(snapshot_path)
    snap = meta.get("params")
    if not snap:
        return None
    for section, key in _SNAPSHOT_SKIP_FIELDS:
        if section in snap and key in snap[section]:
            snap[section].pop(key)
    return Parameters.model_validate(snap)


# Parameter sections that *define* the trajectory: on resume, a change to
# any of their fields (other than the JAX-setup skip fields) marks a new
# trajectory (reset ``it``/``t``/``isnap``) rather than a continuation.
_TRAJECTORY_SECTIONS: tuple[str, ...] = ("phys", "geo", "res")


def trajectory_defining_changes(snapshot_params: dict) -> list[str]:
    """List the trajectory-defining params the run overrides on resume.

    Compares the snapshot's embedded ``phys``/``geo``/``res`` parameters
    (``snapshot_params`` is the ``params`` sub-dict of
    ``_dnsjax_meta.json``) against the live global :data:`params`,
    skipping the JAX-setup fields in :data:`_SNAPSHOT_SKIP_FIELDS` (of
    which only ``res.double_precision`` lies in these sections).  Returns
    a human-readable ``"section.key: old -> new"`` description per
    differing field; an empty list means the resume is a pure
    *continuation* (inherit ``it``/``t``/``isnap``).  The comparison is
    against the snapshot's stored ``model_dump(mode="json")`` (taken
    after :func:`update_parameters`, so geometry-forced fields such as
    cylindrical ``lz = 2*pi`` match and do not register as changes).

    ``geo.axis_gap`` is compared only for cylindrical-family systems
    (it shapes only the cylindrical radial grid).  A pre-``axis_gap``
    pipe snapshot therefore flags ``geo.axis_gap: None -> 1`` -- its
    grid really did change under the current default -- resume with
    ``--geo.axis_gap 0`` for a continuation on the legacy grid, or
    ``init.force_resume`` to continue interpolated.
    """
    skip = set(_SNAPSHOT_SKIP_FIELDS)
    changes: list[str] = []
    for section in _TRAJECTORY_SECTIONS:
        snap = snapshot_params.get(section, {})
        cur = getattr(params, section).model_dump(mode="json")
        for key in sorted(set(snap) | set(cur)):
            if (section, key) in skip:
                continue
            if (section, key) == ("geo", "axis_gap") and (
                params.phys.system not in cylindrical_systems
            ):
                continue
            if snap.get(key) != cur.get(key):
                changes.append(
                    f"{section}.{key}: {snap.get(key)!r} -> {cur.get(key)!r}"
                )
    return changes


def update_parameters(params_new: Parameters) -> None:
    """Merge *params_new* into the global ``params``
    and recompute derived values.

    Only fields that were explicitly set in *params_new* are applied, so
    unset fields retain their previous values.
    """
    for category, dict in params_new.model_dump(exclude_unset=True).items():
        if dict is not None:
            for key, value in dict.items():
                if value is not None:
                    setattr(getattr(params, category), key, value)

    # Set derived parameters:
    system = params.phys.system
    if system in periodic_systems:
        derived_params.volume_fac = 1  # sum over ky comes as density
    elif system in cartesian_systems:
        derived_params.volume_fac = 2  # int_{-1]^{1} dy.
    elif system in cylindrical_systems:
        # Force a full 2*pi spanwise extent for the cylindrical
        # geometry, overriding any user-supplied value (the
        # azimuthal modes are integer harmonics over 2*pi).
        params.geo.lz = 2 * pi
        # Cylindrical area normalisation: int_0^1 r dr.
        derived_params.volume_fac = 0.5
    elif system in annular_systems:
        # Annular geometry (two concentric cylinders).  Shared by
        # shear-driven Taylor-Couette (perturbation form) and
        # force-driven Dean flow (total-field form).  Validate the
        # radius ratio eta and derive the non-dim radii on the gap
        # d = r2 - r1 = 1, the azimuthal extent, and the area norm.
        eta = params.geo.eta
        if eta is None:
            raise ValueError(f"{system} requires geo.eta (radius ratio r1/r2)")
        # Non-dim radii on the gap d = r2 - r1 = 1.
        r1 = eta / (1 - eta)
        r2 = 1 / (1 - eta)
        derived_params.r_inner = r1
        derived_params.r_outer = r2
        # Force a full 2*pi azimuthal extent (integer harmonics).
        params.geo.lz = 2 * pi
        # Annular area normalisation: int_{r1}^{r2} r dr.
        derived_params.volume_fac = (r2**2 - r1**2) / 2

        if system == "taylor-couette":
            # Validate the (re1, re2) control parameters and derive the
            # circular-Couette base flow U_theta = A0 r + B0/r.
            re1, re2 = params.phys.re1, params.phys.re2
            if re1 is None or re2 is None:
                raise ValueError(
                    "taylor-couette requires phys.re1 and phys.re2"
                )
            if re1 < 0:
                raise ValueError(
                    "taylor-couette: re1 must be >= 0 (sign convention)"
                )
            if re1 > 0:
                re_ref = re1  # Case 1: inner-driven
            elif re2 > 0:
                re_ref = re2  # Case 2: outer-driven (re1 == 0)
            else:
                raise ValueError(
                    "taylor-couette needs re1 > 0, or re1 == 0 and re2 > 0 "
                    f"(got re1={re1}, re2={re2})"
                )
            # Set the reference Reynolds number so every downstream 1/re
            # viscous/IMM/stats path is reused unchanged.
            params.phys.re = re_ref
            # Gap-scaled circular-Couette coefficients (divided by Re_ref):
            #   A0 = (re2 - eta re1) / [(1+eta) Re_ref]
            #   B0 = eta (re1 - eta re2) / [(1+eta)(1-eta)^2 Re_ref]
            derived_params.ccf_A = (re2 - eta * re1) / ((1 + eta) * re_ref)
            derived_params.ccf_B = (
                eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
            )
        # Dean flow uses phys.re directly (both walls stationary); its
        # azimuthal body force lives in flows.wall_bounded.dean.
    else:
        raise NotImplementedError

    # Resolve the moving-frame speed U_grid.  A user-set value wins (but
    # the moving frame is only implemented for wall-bounded systems);
    # otherwise default to the laminar bulk velocity in the grid
    # direction so the mean advection is removed.
    if params.phys.u_grid is not None:
        if system in periodic_systems:
            raise ValueError(
                "phys.u_grid (moving frame) is only supported for "
                "wall-bounded systems"
            )
        derived_params.u_grid = params.phys.u_grid
    else:
        derived_params.u_grid = {
            "pipe": 0.5,
            "plane-poiseuille": 2.0 / 3.0,
        }.get(system, 0.0)

    # Select tilting parameters to exact precision for special angles
    if abs(params.geo.tilt_degree) == 0:
        derived_params.tilt_rad = 0
        derived_params.cos_tilt = 1
        derived_params.sin_tilt = 0
    elif abs(params.geo.tilt_degree - 180) == 0:
        derived_params.tilt_rad = pi
        derived_params.cos_tilt = -1
        derived_params.sin_tilt = 0
    elif abs(params.geo.tilt_degree - 90) == 0:
        derived_params.tilt_rad = pi / 2
        derived_params.cos_tilt = 0
        derived_params.sin_tilt = 1
    elif abs(params.geo.tilt_degree + 90) == 0:
        derived_params.tilt_rad = -pi / 2
        derived_params.cos_tilt = 0
        derived_params.sin_tilt = -1
    else:
        derived_params.tilt_rad = pi * params.geo.tilt_degree / 180
        derived_params.cos_tilt = cos(derived_params.tilt_rad)
        derived_params.sin_tilt = sin(derived_params.tilt_rad)

    if (
        params.geo.wall_grid is not None
        and not Path(params.geo.wall_grid).is_file()
    ):
        raise FileNotFoundError(
            f"Wall grid file not found: {params.geo.wall_grid}"
        )

    if params.geo.wall_grid is not None and params.geo.grid_type is not None:
        raise ValueError(
            "Cannot set both wall_grid and grid_type"
            " (wall_grid takes precedence; remove one)"
        )


def validate_parameters() -> None:
    """Validate cross-field constraints on the merged global ``params``.

    Run once, after every configuration layer (snapshot, TOML, CLI) has
    been applied -- not as a Pydantic validator, which would fire on each
    partial layer and reject, e.g., a ``it_corrector`` set in TOML while
    ``it_error_check`` is still at its default.
    """
    o = params.outs
    if o.it_corrector is not None and o.it_error_check > o.it_corrector:
        raise ValueError(
            f"outs.it_error_check ({o.it_error_check}) must be <= "
            f"outs.it_corrector ({o.it_corrector}) so the corrector "
            "convergence is checked at least as often as it is logged."
        )

    # axis_gap shapes only the cylindrical radial grid (the pipe's
    # coordinate axis); reject a non-default value elsewhere to catch
    # confusion early (wall_grid / tanh grids simply ignore it).
    if (
        params.geo.axis_gap != 1
        and params.phys.system not in cylindrical_systems
    ):
        raise ValueError(
            "geo.axis_gap applies only to the cylindrical geometry "
            f"(system {params.phys.system!r} has no axis)."
        )

    # The Pallas kernel tiles the mode plane in ``bm0 x bm1`` blocks;
    # Triton block loads require power-of-two tile dims.
    for name in ("pallas_block_m0", "pallas_block_m1"):
        v = getattr(params.solver, name)
        if v & (v - 1) != 0:
            raise ValueError(
                f"solver.{name} ({v}) must be a power of two "
                "(Triton block-load constraint)."
            )


@dataclass
class PaddedResolution:
    """Grid sizes after 3/2-rule oversampling for dealiasing.

    The oversampled (padded) grid is used when evaluating nonlinear terms
    in physical space.  Each direction is expanded by a factor of
    ``oversampling_factor / 2`` (typically 3/2).
    """

    nx_padded: int = params.phys.oversampling_factor * params.res.nx // 2
    ny_padded: int | None = None
    if params.phys.system in periodic_systems:
        ny_padded = (
            params.phys.oversampling_factor * params.res.ny // 2
            if params.phys.oversample_y
            else params.res.ny
        )
    nz_padded: int = params.phys.oversampling_factor * params.res.nz // 2

    def set_padded_resolution(self, parameters: Parameters) -> None:
        """Recompute padded sizes from *parameters*."""
        if (
            params.phys.system in periodic_systems
            and not parameters.phys.oversample_y
        ):
            print("WARNING: y is *not* oversampled!")

        self.nx_padded = (
            parameters.phys.oversampling_factor * params.res.nx // 2
        )
        if params.phys.system in periodic_systems:
            self.ny_padded = (
                params.phys.oversampling_factor * params.res.ny // 2
                if params.phys.oversample_y
                else params.res.ny
            )
        else:
            self.ny_padded = None
        self.nz_padded = (
            parameters.phys.oversampling_factor * params.res.nz // 2
        )


padded_res: PaddedResolution = PaddedResolution()
