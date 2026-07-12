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
from dataclasses import dataclass, field
from datetime import timedelta
from math import cos, isclose, pi, sin
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .harmonics import parse_mode_pairs

monochromatic_systems: list[str] = ["kolmogorov", "waleffe"]
periodic_systems: list[str] = ["decaying-box", *monochromatic_systems]

cartesian_systems: list[str] = ["plane-couette", "plane-poiseuille"]
cylindrical_systems: list[str] = ["pipe"]
annular_systems: list[str] = ["taylor-couette", "dean"]
# Viscoelastic flows living on the annular *geometry* (grid on
# [r1, r2], u_+/u_- velocity formulation) but integrating a coupled
# 9-component state (3 velocity + 6 symmetric conformation-tensor
# components).  Kept separate from ``annular_systems`` so the
# eta-based annular derivation and the 3-component annular IC
# generators / analysis reader do not accidentally catch them; the
# annular-*geometry* routing sites explicitly include this list.
viscoelastic_systems: list[str] = ["viscoelastic-dean"]
walled_systems: list[str] = [
    *cartesian_systems,
    *cylindrical_systems,
    *annular_systems,
    *viscoelastic_systems,
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
    space, so FD solves are unchanged.

    When ``np0 == 1`` (default), the decomposition collapses
    to the original 1D scheme (only `$k_x$` / `$z$`
    distributed).

    Divisibility
    ------------
    No divisibility requirements: every axis a mesh
    direction splits is auto-padded, and each adjustment is
    reported by a startup diagnostic (padding costs FFT and
    solve work, so it is never silent).  The spectral `$k_z$`
    / `$k_x$` axes are zero-padded to the next multiple of
    ``np0`` / ``np1``; for wall-bounded flows the physical
    `$y$` axis is likewise zero-padded (stripped after the
    `$y \leftrightarrow k_z$` reshard); the padded physical
    sizes -- ``nz_padded`` (split by ``np1``) and, for
    periodic flows, ``ny_padded`` (split by ``np0``) -- are
    rounded up to the next FFT-friendly (7-smooth) multiple
    (:func:`round_up_padded_smooth`; marginally more
    oversampling, physically neutral).  Divisible,
    smooth-padding sizes avoid the padding overhead entirely.
    Note that CGL grids traditionally
    use ``ny = 2^k + 1`` (``N + 1`` collocation points for
    ``N`` Chebyshev polynomials), but any ``ny >= 2`` is
    valid: the code uses finite differences, not spectral
    Chebyshev transforms, and the Clenshaw--Curtis
    quadrature handles both even and odd ``ny``.

    Process topology
    ----------------
    ``np0 * np1`` counts *devices*, not processes: the mesh
    only requires ``jax.device_count() == np0 * np1``, so a
    multi-GPU run may be one process per device (the usual
    ``mpirun``/``srun -n N`` launch) **or a single process
    addressing all devices** -- launch one task with
    ``JAX_LOCAL_DEVICE_IDS=0,1,...`` spanning the GPUs
    (overrides the SLURM one-device-per-task heuristic),
    e.g. ``srun -n 1 --overlap`` inside a 4-GPU allocation
    with ``JAX_LOCAL_DEVICE_IDS=0,1,2,3 --dist.np0 2
    --dist.np1 2``.  Both topologies produce identical
    global meshes, trajectories, and snapshots (resume is
    np-agnostic), and both are validated on real multi-GPU
    hardware (``scripts/solver_benchmark.py``, 2026-07).
    The single-process form avoids cross-process NCCL
    entirely -- the reliable choice on single-node
    allocations whose multi-process collective stack is
    broken (observed: JAX 0.10.2 + NCCL 2.30 H100 nodes
    hang in the first execution of a large multi-collective
    program while every small-program collective works).
    Multi-node runs necessarily remain multi-process; use
    one task per node spanning that node's GPUs.

    In multi-process launches, never narrow per-task GPU
    visibility (SLURM ``--gpus-per-task`` and the like):
    NCCL's cuMem cross-process P2P import requires the peer
    device to be visible to the importing process and fails
    hard otherwise (``ncclP2pImportShareableBuffer ...
    Cuda failure 101 'invalid device ordinal'``).  Leave
    every job GPU visible to every task and select the
    per-task device explicitly (``JAX_LOCAL_DEVICE_IDS``;
    JAX's SLURM detection uses ``[SLURM_LOCALID]`` by
    default, which is correct under full visibility).
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
    # Viscoelastic (sPTT) control parameters (system ==
    # "viscoelastic-dean"; see ``flows.wall_bounded.viscoelastic_dean``
    # and ``geometries.wall_bounded.annular_viscoelastic``).  All
    # ``None`` for other systems and default to the reference
    # configuration when unset for the viscoelastic system (applied in
    # the ``update_parameters`` viscoelastic branch):
    #   el   -- elasticity number El (defines Re := Wi/El, so
    #           ``phys.re`` is derived, not set directly).  Default 80.
    #   wi   -- Weissenberg number.  Default 105.
    #   beta -- solvent-to-total viscosity ratio in (0, 1]; the solvent
    #           viscosity is nu = beta/Re (``derived_params.nu``) and
    #           the polymer stress carries (1-beta)/(Re Wi).  Default
    #           0.8.
    #   epsilon -- sPTT extensibility parameter (>= 0).  Default 0.001.
    #   kappa   -- artificial stress diffusivity (>= 0); kappa = 0 makes
    #              the conformation transport purely hyperbolic (no
    #              wall BC on c).  Default 5e-5.
    el: float | None = Field(default=None, gt=0)
    wi: float | None = Field(default=None, gt=0)
    beta: float | None = Field(default=None, gt=0, le=1)
    epsilon: float | None = Field(default=None, ge=0)
    kappa: float | None = Field(default=None, ge=0)
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
    3. Default (``grid_type`` unset): ``update_parameters`` resolves
       it to a concrete value -- full CGL (``"cgl"``) for the
       Cartesian / annular families, and for the cylindrical family
       **half-CGL** under the default ``iterative-cn`` scheme or
       **rigged-CGL** (``"cgl"``) under ``cnab2`` (see below).
       Because the resolved value is concrete, snapshots embed the
       grid they actually ran and a resume pins it -- the
       scheme-dependent default never silently re-grids an old
       trajectory (see :func:`_snapshot_grid_type`).

    Setting both ``wall_grid`` and ``grid_type`` is an error.

    ``grid_type`` values: ``"cgl"`` the plain Chebyshev-Gauss-Lobatto
    grid (Cartesian / annular; for the cylindrical family it is a
    synonym for the rigged-CGL grid); ``"half-cgl"`` the half-CGL
    radial grid (**cylindrical family only, and only with**
    ``step.scheme == "iterative-cn"``); ``"tanh"`` a tanh-stretched
    grid.

    **Cylindrical radial grids.**  Both keep the ``ny`` outermost
    *positive* points of an auxiliary CGL grid on `$[-1, 1]$`, so the
    near-axis spacing is `$\Delta r \approx \pi/(2 n_y)$` and no
    degree of freedom lives in `$[0, r_0)$` (parity ghosts close the
    FD stencils across the axis; the quadrature covers the segment):

    - **rigged-CGL** (the ``cnab2`` default): the positive half of a
      `$(2 n_y + 1)$`-point grid.  The odd total's centre point falls
      exactly on the coordinate-singular axis and is dropped, so the
      innermost point sits at `$r_0 \approx \Delta r$`
      (`$= \sin(\pi/(2 n_y))$`).
    - **half-CGL** (the ``iterative-cn`` default): the positive half
      of a `$2 n_y$`-point grid, staggered so
      `$r_0 \approx \Delta r/2$`
      (`$= \sin(\pi/(2\,(2 n_y - 1)))$`) -- half the rigged value.

    Rationale: the near-axis *azimuthal* advection CFL
    `$\propto 1/r_0$` is a stability artifact of explicit stepping
    evaluated at grid points only, so it relaxes `$\propto r_0$` at a
    truncation-level accuracy cost.  The rigged grid's `$2\times$`
    larger `$r_0$` doubles the admissible explicit-``cnab2`` ``dt``
    (measured), which is why it is the ``cnab2`` default; the tighter
    half-CGL axis makes ``cnab2`` blow up at low ``dt`` (near-axis
    explicit instability), so half-CGL is restricted to the
    implicitly-iterated ``iterative-cn`` scheme, which integrates it
    cleanly, gains its finer near-axis resolution, and defaults to
    it.  See ``build_radial_cgl_grid`` in ``cylindrical.py``.
    """

    lx: float = Field(gt=0, default=4.0)
    lz: float = Field(gt=0, default=4.0)
    tilt_degree: float = Field(gt=-180, le=180, default=0)
    # Radius ratio eta = r1/r2 for the annular (Taylor-Couette)
    # geometry.  Non-dim radii r1 = eta/(1-eta), r2 = 1/(1-eta) on the
    # gap d = r2 - r1 = 1.  Required for system == "taylor-couette".
    eta: float | None = Field(default=None, gt=0, lt=1)
    # Inner radius delta for the viscoelastic annular geometry
    # (system == "viscoelastic-dean").  The gap is fixed at 2 (half-gap
    # length unit), so r1 = delta, r2 = delta + 2.  Default 11 (applied
    # in the ``update_parameters`` viscoelastic branch).
    delta: float | None = Field(default=None, gt=0)
    wall_grid: Path | None = None
    grid_type: Literal["cgl", "half-cgl", "tanh"] | None = None
    grid_stretch: float = Field(gt=0, default=1.5)


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
    # Amplitude of the random symmetric-tensor perturbation added to the
    # laminar conformation for the viscoelastic random IC
    # (system == "viscoelastic-dean"); shares ``random_smoothness`` for
    # the spectral envelope and is radially windowed to zero at both
    # walls (the reference restart recipe).  Ignored by every other
    # system (which has no conformation field).
    random_conformation_amplitude: float = 700.0
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
    # when the corrector diagnostic is enabled.  Also the cadence of
    # the non-finite (NaN/inf) guard on the synced corrector error
    # and perturbation energy (a hit aborts the run with exit code 3;
    # the ``__main__`` module docstring documents the full guard).
    it_error_check: int = Field(ge=1, default=10)
    # Spectral-mode probe stream: record the complex wall-normal
    # profiles ``u_hat(y)`` of the listed global spectral modes every
    # ``it_probes`` steps into a binary ``probes.bin`` (+ a
    # ``probes.json`` schema sidecar).  ``probe_modes`` is an
    # ``"i2,i3;i2,i3;..."`` list of stored-layout indices (axis 2 =
    # complex slot, axis 3 = real-FFT slot -- the transient-growth
    # CLI ``--modes`` convention); the mean mode ``(0,0)`` is allowed
    # (it records the instantaneous mean profile).  Wall-bounded
    # systems only; both fields set together (``validate_parameters``).
    # ``it_probes`` trades time resolution for disk (a record is
    # ``8 + K*C*ny*2`` values): sample densely enough for the fastest
    # statistics of interest -- the buffered writer makes any cadence
    # cheap at runtime, so disk volume is the only real constraint.
    # Format, buffering, and the reader: the :mod:`dnsjax.probes` and
    # :mod:`dnsjax.analysis.response.probes` docstrings.
    probe_modes: str | None = None
    it_probes: int | None = Field(default=None, ge=1)
    # Rows buffered on device before flushing ``stats.dat`` /
    # ``steps.dat`` / ``corrector.dat`` / ``probes.bin`` to disk.
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

      Wall-bounded split corrector (``split_corrector``).  In the
      rotational perturbation form the iterated RHS contains the
      *linear* coupling -- ``L_bf``, the frame term, and (per
      ``implicit_mean_coupling``) ``L_mf`` -- i.e. exactly the
      FFT-free ``_l_bf`` the CN/AB2 scheme makes implicit.  The split
      corrector iterates that part FFT-free: each outer (FFT)
      iteration freezes the pure self-advection ``N_nl = get_rhs -
      l_bf`` at the last full evaluation, converges the coupling by
      re-evaluating only ``l_bf`` (matvecs / a mean-mode ``psum``, no
      transform), then refreshes the full RHS once and corrects.  The
      composite fixed point is the same CN equation and the loop
      always exits on a fresh-RHS correction, so trajectories agree
      within ``corrector_tolerance`` and the reported ``num_c`` /
      ``error`` keep their meaning (extra FFT evaluations / last
      fresh-RHS correction norm) -- ``corrector.dat`` is comparable
      across the setting.  It is an **opt-in** (``split_corrector``,
      default **off**): at realistic ``dt`` the corrector converges in
      ~1--2 iterations for *every* flow (measured, including the
      total-field Dean and high-Wi viscoelastic-dean -- unsplit ``c``
      stays 2--3, far from the cap, at ``dt = 0.01``), so the unsplit
      corrector is both correct and faster (the split is measured a
      few % slower at production sizes on GPU, up to tens of % on small
      problems).  The split only pays off once ``dt`` is pushed far
      enough that the unsplit corrector approaches
      ``max_corrector_iterations`` -- e.g. Dean at ``dt = 0.15`` (an
      unrealistically large step), where the unsplit corrector hits the
      cap (``c = 10``) and fails while the split converges it FFT-free.
      When enabled: the tail launches an implicit solve only while the
      coupling estimate still moves the state
      (``c dt ||l_bf(u_j) - l_bf(u_{j-1})|| > tol``, a cheap test), so a
      fluctuation-driven iteration adds one ``l_bf`` evaluation and a
      norm, not a solve -- and a step whose first correction already
      meets tolerance costs 2 FFT evaluations either way.  A split
      corrector that fails to reach tolerance automatically redoes the
      step with the unsplit corrector (``lax.cond``, stdout
      diagnostic), pinning the worst case to the unsplit path.
      Triply-periodic flows have no coupling (``l_bf_fn = None``) and
      always run unsplit.
    - ``"cnab2"``: Crank-Nicolson viscous (implicitness *c*) + 2nd-order
      Adams-Bashforth nonlinear (explicit ``1.5 N^n - 0.5 N^{n-1}``).
      **One** expensive nonlinear/FFT evaluation per step; the previous
      nonlinear RHS is carried by the main loop, seeded by a discarded
      priming ``step_cnab2(state, zeros)`` call while ``iterative-cn``
      takes the very first integration step (see ``step_cnab2`` in
      ``timestep.py``).  Explicit
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
      apply here.  The first step self-starts with ``iterative-cn``,
      and if the coupling corrector fails to converge on a step (its
      Picard rate reaches 1 -- only at ``dt`` well past the advective
      limit, e.g. plane-Couette ``dt >~ 0.2``) that step automatically
      falls back to a full ``iterative-cn`` step (a stdout diagnostic
      is printed).  The
      residual ``dt`` bound is then the ordinary explicit self-advection
      CFL of the *fluctuations* on the clustered grid (stationary-wall
      flows -- Poiseuille, pipe, Dean -- are bounded only by this, their
      ``L_bf`` being mild as ``U -> 0`` at the wall).  Where it binds is
      geometry-specific: for the **pipe** it is the *near-axis
      azimuthal* advection (the innermost radial node
      ``r_0 ~ pi/(2 ny)`` on the default rigged-CGL grid makes
      ``CFL_th = dt |u_th(r_0)| nz/(2 pi r_0)`` the dominant
      column -- linear in ``nz`` and in the fluctuation amplitude, and
      a *weak* AB2 imaginary-axis instability, so it needs sustained
      ``CFL_th >~ 0.5`` and pass/fail is trajectory-marginal near the
      boundary; the **rigged-CGL** radial grid sits at
      ``r_0 ~ Delta r`` -- twice the half-CGL ``Delta r/2`` --
      which raises the admissible cnab2 ``dt`` (measured ``dt* ~
      0.0125 -> 0.0175`` at the 32^3 / Re = 1800 reference config;
      ``iterative-cn`` rides ``CFL_th ~ 1.5--2`` there at growing
      corrector cost) and is why it is the ``cnab2``-default radial
      grid, whereas the tighter half-CGL grid
      (``geo.grid_type = 'half-cgl'``, the ``iterative-cn`` default)
      destabilises cnab2 and is
      restricted to ``iterative-cn``); Cartesian flows feel the
      near-wall ``dy ~ 1/N^2`` spacing instead.  A strongly
      non-normal base flow
      (counter-rotating Taylor-Couette) amplifies the explicit
      self-advection error further into a delayed blow-up needing
      ~8x smaller ``dt``; the coupling corrector converges throughout
      (it is *not* a corrector failure), so the fallback typically
      does not fire -- at coarser resolution the induced corrector
      stress can trip it into rescuing single steps (``ny = 32``
      completes via fallbacks; ``ny = 48`` diverges with the
      corrector still converged).  These are inherent
      explicit-nonlinear
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
      term, removing its advective-CFL contribution.  Decisive for
      Dean: at ``nz = 64, dt = 0.15`` (mean-flow ``CFL_th ~ 0.5``)
      coupling-off NaNs by ``t ~ 4`` while coupling-on runs clean,
      matching ``iterative-cn`` at ~4 FFT-free Picard iterations per
      step vs its ~4 FFT evaluations; neutral where the limit is
      fluctuation-driven (the pipe near-axis) and only mildly slowing
      the counter-rotating-TC blow-up.  The double-counted
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
    # FFT-free coupling term ``_l_bf`` shared by the wall-bounded
    # CN/AB2 scheme and the split ``iterative-cn`` corrector
    # (``split_corrector``); no effect on triply-periodic flows.
    # See the class docstring.
    implicit_mean_coupling: bool = True
    # Wall-bounded ``"iterative-cn"`` only, **opt-in** (default off):
    # iterate the linear coupling ``_l_bf`` FFT-free between full-RHS
    # corrector refreshes (same CN fixed point; coupling-driven
    # corrector iterations stop costing one FFT evaluation each).  At
    # realistic ``dt`` the corrector converges in ~1--2 iterations for
    # every flow (measured, incl. Dean and high-Wi viscoelastic-dean),
    # so the unsplit corrector is both correct and faster -- the split
    # only pays off if ``dt`` is pushed far enough that the unsplit
    # corrector approaches its iteration cap.  No effect on cnab2 or
    # triply-periodic flows.  See the class docstring.
    split_corrector: bool = False


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
    """Numerical-kernel execution configuration.

    Linear-solver backends (pallas / dense) and pseudo-spectral
    transform batching.  These knobs select *how* the numerics are
    executed (speed / memory trade-offs), never the results.
    """

    # ``"pallas"`` (the default for all wall-bounded systems): the
    # production backend -- a one-program-per-mode sequential banded
    # sweep via a Pallas/Triton kernel on GPU (single- and multi-GPU
    # validated 2026-07), the same banded math as a sequential
    # pure-JAX sweep on CPU, and the smallest operator storage
    # (``O(N_y p)`` no-pivot banded factors per mode vs the dense
    # ``O(N_y^2)``).  Operators are assembled directly in banded
    # storage by each geometry's ``_build_{Lk,Hk}_band_gpu`` via the
    # shared ``solvers._assemble_banded_operator`` helper, then
    # factored and stability-checked once at setup by
    # ``solvers._build_pallas_operator`` (an unstable no-pivot LU is
    # a hard error; a merely ill-conditioned operator prints a notice
    # -- see ``pallas_stability_tol``).  The solve is a
    # shard_map-local region (each device runs the kernel on its
    # local mode-plane block; no communication).
    # ``"dense"``: full ``Ny x Ny`` pivoted LU factors per Fourier
    # mode.  The *reference* backend: the mathematically readable
    # formulation of the operators and the regression oracle the
    # Pallas path is tested against -- not a production backend (a
    # wall-bounded run selecting it prints a warning).
    # Triply-periodic systems have no wall-normal matrix solves (the
    # implicit step is diagonal in spectral space): ``"dense"`` (the
    # reference semantics) is their only backend, the default
    # resolves to it in ``update_parameters()``, and ``"pallas"`` is
    # rejected there.
    backend: Literal["pallas", "dense"] = "pallas"
    # ``"pallas"`` backend only: Pallas mode-tile size along the ``k_z``
    # mode axis (one Pallas program solves a ``bm0 x bm1`` tile of
    # Fourier modes, vectorising the banded sweep across the tile).  ``1``
    # is one program per mode; ``> 1`` coalesces mode loads and fills more
    # SIMD lanes.  The default ``2`` is the H100 tuning (4 warps/program
    # with ``k = 2``).  The mode plane is padded up to whole tiles (a
    # masked partial-tile band load miscompiles on real Triton -- see
    # ``solvers._pallas_banded_solve``): factors once at construction,
    # the RHS per solve.  The padded modes cost memory and solve work
    # proportional to the roundup fraction -- negligible at DNS mode
    # counts, but worth shrinking the tile for when the plane is small
    # relative to it (e.g. ``nx/2 < 32``).  Must be a power of two.
    pallas_block_m0: int = Field(ge=1, default=2)
    # ``"pallas"`` backend only: Pallas mode-tile size along the
    # contiguous ``k_x`` mode axis (the innermost, coalesced axis).  The
    # default ``32`` is the H100 tuning -- one warp wide, so a warp's band
    # load fully coalesces.  Same internal padding to full tiles as
    # ``pallas_block_m0``.  Must be a power of two.
    pallas_block_m1: int = Field(ge=1, default=32)
    # ``"pallas"`` backend only: solve-residual threshold for the
    # setup-time no-pivot banded-LU stability check (max relative
    # residual ``||A x - b|| / ||b||`` over modes, measured once per
    # operator group).  Above it, ``solvers._build_pallas_operator``
    # prints an ill-conditioning notice (benign LU element growth) or
    # raises on genuine no-pivot instability (see
    # ``solvers._NO_PIVOT_GROWTH_TOL``).
    pallas_stability_tol: float = Field(gt=0, default=1e-6)
    # Chunk count for the batched inverse transform of the
    # pseudo-spectral RHS, all flows: the 6-field velocity+vorticity
    # batch of ``rhs.get_nonlin`` and the ~36-field fused viscoelastic
    # batch (``_get_rhs_core`` in
    # ``geometries/wall_bounded/annular_viscoelastic.py``), both via
    # ``fft.chunked_transform``.  A memory/throughput trade-off: ``1``
    # (default) keeps one fused batch -- throughput-optimal (one FFT
    # dispatch and one reshard round per pipeline stage).  ``k > 1``
    # splits the batch into ``k`` balanced groups, cutting the
    # transform-stage transient (the padded intermediate buffers; see
    # the ``fft.py`` memory note) by ~``k`` at the cost of ``k``x the
    # FFT dispatches (and ``k`` smaller reshard rounds per stage on
    # multi-device runs).  Results are identical (per-field transforms
    # are independent).  Raise it only when that transient sets the
    # device-memory peak: chiefly the viscoelastic batch (it dominates
    # that step's peak); the Newtonian 6-field transient is ~6x
    # smaller.  Forward transforms stay fused.
    rhs_transform_chunks: int = Field(ge=1, default=1)


class StochasticForcing(BaseModel):
    r"""White-in-time stochastic mode forcing (state kicks), optional.

    Enabled when ``modes`` / ``profiles`` / ``amplitude`` /
    ``it_force`` are all set (``validate_parameters`` enforces
    all-or-none): every ``it_force`` steps the main loop adds to each
    listed spectral mode (plus its real-FFT conjugate partner) a
    random superposition of the stored channel profiles -- a sequence
    of independent state increments ("kicks"), the discrete-time
    realisation of white-in-time forcing.  The drawn coefficients
    stream to ``forcing.bin``/``forcing.json`` next to the other
    diagnostics; the cross-covariance of the probe stream with them
    identifies the mode's linear operator
    (:mod:`dnsjax.analysis.response.ssi`).

    Kicks rather than a body-force term: a forcing term inside the
    nonlinear RHS would be AB2-extrapolated under ``cnab2``
    (colouring the noise) or corrector-iterated under
    ``iterative-cn``, and would trace into the jitted steppers; a
    loop-level kick keeps both schemes untouched and makes the
    per-kick response exactly the solver's own propagator.  Full
    conventions (timing relative to probes/snapshots, resume
    continuation, amplitude guidance): the :mod:`dnsjax.forcing`
    module docstring.

    Wall-bounded, non-viscoelastic systems only.  The whole section
    is **trajectory-defining**: resuming with changed forcing starts
    a new trajectory (like a ``phys`` change).
    """

    # ``"i2,i3;..."`` global spectral modes to force -- the
    # ``outs.probe_modes`` convention; the mean mode (0,0) is
    # rejected (real + constrained under bulk-velocity driving).
    # Forced modes should normally also be probed, or the response
    # cannot be identified (a startup note reminds when they are not).
    modes: str | None = None
    # npz with per-mode channel profiles ``cont_modes_{i2}_{i3}``
    # (``(m, C, Ny)`` complex, unit energy norm -- the
    # ``operator_tools.save_modes_npz`` format, typically the leading
    # controllability modes) on **this run's** wall-normal grid
    # (exact match required; regrid offline if needed).
    profiles: str | None = None
    # Leading channels used per mode (default: all stored).  Fewer
    # channels = fewer directions identified but less injected energy
    # and a smaller operator to fit.
    n_channels: int | None = Field(default=None, ge=1)
    # Kick coefficient scale eps: each kick adds
    # ``eps * sum_j w_j profile_j`` with ``w_j ~ CN(0,1)`` i.i.d., so
    # the expected injected energy is ``eps^2`` per channel per kick.
    # Pick eps in the linear-response window (halving it must leave
    # the identified operator unchanged); the predicted stationary
    # forced energy for planning: ``dnsjax.analysis.response.ssi.
    # predicted_forced_variance``.
    amplitude: float | None = Field(default=None, gt=0)
    # Kick cadence in steps (``Delta_f = it_force * dt``).  Must be a
    # multiple of ``outs.it_probes`` when probing, so every kick
    # coincides with a (pre-kick) probe sample.  Larger values give
    # cleaner per-kick responses; smaller values more statistics per
    # run time.
    it_force: int | None = Field(default=None, ge=1)
    # Seed of the coefficient PRNG (host-side, identical on every
    # rank; a resumed run skips the already-recorded draws, so the
    # coefficient stream continues exactly as if uninterrupted).
    seed: int = 0


class Parameters(BaseModel):
    """Top-level parameter container aggregating all categories."""

    dist: Distribution | None = Distribution()
    phys: Physics = Physics()
    geo: Geometry = Geometry()
    res: Resolution = Resolution()
    init: Initiation = Initiation()
    outs: Outputs = Outputs()
    step: TimeStepping = TimeStepping()
    force: StochasticForcing = StochasticForcing()
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
    ``nu`` is the (solvent) kinematic viscosity multiplying the velocity
    Laplacian: ``1/re`` for the Newtonian systems, ``beta/re`` for the
    viscoelastic system (whose polymer stress carries the remaining
    ``(1-beta)/(re wi)``).
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
    nu: float = 0


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


def _snapshot_grid_type(snapshot_params: dict) -> str | None:
    r"""Effective ``geo.grid_type`` of a snapshot's embedded params.

    ``update_parameters`` resolves the wall-normal grid default to a
    concrete value, so current snapshots embed the grid they ran
    (``"cgl"`` / ``"half-cgl"`` / ``"tanh"``).  Older wall-bounded
    snapshots embed ``null`` (or, before ``grid_type`` existed, the
    retired ``geo.axis_gap`` key) although they ran a definite grid:
    the then-default full/rigged CGL (both spelled ``"cgl"``), or the
    grid ``axis_gap`` selected (``0`` = half-CGL, ``1`` = rigged).
    This helper maps a stored params dict to that effective value, so
    :func:`read_snapshot_params` can pin the original grid on resume
    and :func:`trajectory_defining_changes` compares grids rather than
    storage conventions.  Returns the stored value unchanged when it
    is concrete, and ``None`` for periodic systems and custom
    ``wall_grid`` runs (where ``grid_type`` is genuinely unset).
    """
    geo = snapshot_params.get("geo") or {}
    grid_type = geo.get("grid_type")
    if grid_type is not None:
        return grid_type
    system = (snapshot_params.get("phys") or {}).get("system")
    if system not in walled_systems or geo.get("wall_grid") is not None:
        return None
    if geo.get("axis_gap") == 0:  # retired pre-``grid_type`` field
        return "half-cgl"
    return "cgl"


def read_snapshot_params(snapshot_path: Path) -> Parameters | None:
    """Build a ``Parameters`` from a snapshot's embedded parameters.

    Reads the ``_dnsjax_meta.json`` member of the snapshot tar (via the
    standard-library :mod:`dnsjax.snapshot_meta`; no JAX import, so it is
    safe to call before the distributed backend is configured) and
    returns the embedded ``params`` as a :class:`Parameters`, with the
    JAX-setup fields in :data:`_SNAPSHOT_SKIP_FIELDS` and the whole
    ``solver`` section removed so they are not inherited.

    Returns ``None`` when *snapshot_path* is not a dnsjax snapshot file
    (legacy ``.npz`` snapshots, a laminar start, or a missing path), so
    the caller simply skips the snapshot layer.  Unknown fields in the
    stored dump are ignored (Pydantic ``extra="ignore"``), making this
    robust to parameter-schema drift across versions.

    A stored-``null`` wall-bounded ``geo.grid_type`` is backfilled to
    its effective value (:func:`_snapshot_grid_type`), so the snapshot
    layer pins the grid the trajectory actually ran -- the
    scheme-dependent grid default of ``update_parameters`` cannot
    silently re-grid an old run on resume.
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
    # Solver knobs are execution-only (they select *how* the numerics
    # run, never the results), so they are never inherited from a
    # snapshot -- which also keeps snapshots that embed a retired
    # backend/field resumable.
    snap.pop("solver", None)
    grid_type = _snapshot_grid_type(snap)
    if grid_type is not None:
        snap.setdefault("geo", {})["grid_type"] = grid_type
    return Parameters.model_validate(snap)


# Parameter sections that *define* the trajectory: on resume, a change to
# any of their fields (other than the JAX-setup skip fields) marks a new
# trajectory (reset ``it``/``t``/``isnap``) rather than a continuation.
# ``force`` belongs here: stochastic kicks alter the dynamics exactly
# like a ``phys`` change would.
_TRAJECTORY_SECTIONS: tuple[str, ...] = ("phys", "geo", "res", "force")


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

    The wall-normal grid choice lives in ``geo.grid_type``
    (``"cgl"`` = full CGL, or rigged-CGL for the cylindrical family;
    ``"half-cgl"`` = the cylindrical half grid) and is compared by its
    *effective* value: when the current side is concrete (the resolved
    wall-bounded default), the stored side is normalised through
    :func:`_snapshot_grid_type`, so an old snapshot's stored ``null``
    (which ran the then-default ``"cgl"``) matches a current explicit
    ``"cgl"`` and continues cleanly, while a genuine grid switch
    (rigged <-> half-CGL) is flagged.  Legacy keys a snapshot carries
    but the current model no longer defines (e.g. the retired
    ``geo.axis_gap``) are otherwise ignored -- they are absent from
    the current ``model_dump`` and skipped below.
    """
    skip = set(_SNAPSHOT_SKIP_FIELDS)
    changes: list[str] = []
    for section in _TRAJECTORY_SECTIONS:
        snap = snapshot_params.get(section, {})
        model = getattr(params, section)
        cur = model.model_dump(mode="json")
        defaults = type(model)().model_dump(mode="json")
        for key in sorted(set(snap) | set(cur)):
            if (section, key) in skip:
                continue
            if key not in cur:
                # Legacy field no longer in the model (e.g. the
                # retired geo.axis_gap): not trajectory-defining.
                continue
            # A key (or whole section, e.g. ``force`` on old
            # snapshots) absent from the stored dump predates the
            # field: the old run effectively ran the model default,
            # so compare against that rather than flagging every
            # legacy resume when a schema addition has a non-null
            # default.
            old = snap[key] if key in snap else defaults.get(key)
            if (
                section == "geo"
                and key == "grid_type"
                and cur.get(key) is not None
            ):
                # Old snapshots store ``null`` although they ran a
                # definite grid; compare effective values (see
                # ``_snapshot_grid_type``).  Skipped when the current
                # side is ``None`` (periodic / ``wall_grid`` runs, or
                # a not-yet-resolved offline ``params``), where a
                # stored ``null`` genuinely matches.
                old = _snapshot_grid_type(snapshot_params)
            if old != cur.get(key):
                changes.append(f"{section}.{key}: {old!r} -> {cur.get(key)!r}")
    return changes


# ``(section, key)`` of every field explicitly provided through any
# configuration layer (snapshot / TOML / CLI), accumulated across all
# :func:`update_parameters` calls.  Read by per-system defaults that
# must distinguish "left at the class default" from "explicitly set to
# the class default" (currently the viscoelastic ``geo.lx`` axial
# period and the ``phys.re`` consistency check).
_user_set_fields: set[tuple[str, str]] = set()


def update_parameters(params_new: Parameters) -> None:
    """Merge *params_new* into the global ``params``
    and recompute derived values.

    Only fields that were explicitly set in *params_new* are applied, so
    unset fields retain their previous values.  The ``(section, key)`` of
    every explicitly-set field is recorded (across all layers) in the
    module-level :data:`_user_set_fields`, so a per-system default (e.g.
    the viscoelastic axial period ``geo.lx``) can be applied only when
    the user / snapshot / TOML never set it.
    """
    for category, dict in params_new.model_dump(exclude_unset=True).items():
        if dict is not None:
            for key, value in dict.items():
                if value is not None:
                    _user_set_fields.add((category, key))
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
    elif system in viscoelastic_systems:
        # Viscoelastic (sPTT) flow on the annular geometry.  Adopt an
        # external reference normalisation: a half-gap length unit
        # (gap = 2), so r1 = delta, r2 = delta + 2, the Reynolds number
        # is derived as Re := Wi/El, and the axial period defaults to
        # 2*pi.  Any unset control parameter falls back to the
        # reference configuration.
        if params.phys.el is None:
            params.phys.el = 80.0
        if params.phys.wi is None:
            params.phys.wi = 105.0
        if params.phys.beta is None:
            params.phys.beta = 0.8
        if params.phys.epsilon is None:
            params.phys.epsilon = 0.001
        if params.phys.kappa is None:
            params.phys.kappa = 5.0e-5
        if params.geo.delta is None:
            params.geo.delta = 11.0
        # Axial period: default to 2*pi (reference value) unless the user /
        # snapshot / TOML set geo.lx explicitly (it is a genuine domain
        # length, unlike the azimuthal lz which is always forced 2*pi).
        if ("geo", "lx") not in _user_set_fields:
            params.geo.lx = 2 * pi
        r1 = params.geo.delta
        r2 = params.geo.delta + 2.0
        derived_params.r_inner = r1
        derived_params.r_outer = r2
        # Force a full 2*pi azimuthal extent (integer harmonics).
        params.geo.lz = 2 * pi
        # Annular area normalisation: int_{r1}^{r2} r dr.
        derived_params.volume_fac = (r2**2 - r1**2) / 2
        # Re := Wi/El.  A directly-set phys.re is accepted only when it
        # matches (a snapshot resume replays a consistent value); set it
        # so every downstream 1/re path is reused.
        re_derived = params.phys.wi / params.phys.el
        if ("phys", "re") in _user_set_fields and not isclose(
            params.phys.re, re_derived, rel_tol=1e-9
        ):
            raise ValueError(
                "viscoelastic-dean derives phys.re := wi/el "
                f"(= {re_derived:g}); do not set phys.re directly "
                f"(got {params.phys.re}).  Set phys.wi / phys.el instead."
            )
        params.phys.re = re_derived
    else:
        raise NotImplementedError

    # Solvent viscosity multiplying the velocity Laplacian: 1/re for the
    # Newtonian systems; beta/re for the viscoelastic system (the polymer
    # stress carries the remaining (1-beta)/(re wi)).  Read by the
    # annular geometry's operator builders and IMM.
    if system in viscoelastic_systems:
        derived_params.nu = params.phys.beta / params.phys.re
    else:
        derived_params.nu = 1.0 / params.phys.re

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

    # Resolve the solver backend's per-family default.  Triply-periodic
    # systems have no wall-normal matrix solves (the implicit step is
    # diagonal in spectral space), so ``"dense"`` (the reference
    # semantics) is their only backend; wall-bounded systems default to
    # the ``"pallas"`` production backend.  Re-resolved on every layer
    # application, so a later ``system`` override cannot inherit a
    # stale family default.
    if ("solver", "backend") not in _user_set_fields:
        params.solver.backend = (
            "dense" if system in periodic_systems else "pallas"
        )
    elif system in periodic_systems and params.solver.backend == "pallas":
        raise ValueError(
            "solver.backend='pallas' is not available for "
            "triply-periodic systems: their implicit solves are "
            "diagonal in spectral space, with no banded structure to "
            "solve; 'dense' (the reference semantics) is the only "
            "backend there."
        )

    # Resolve the wall-normal grid's per-family default when the user /
    # snapshot / TOML never set ``geo.grid_type``: half-CGL for the
    # cylindrical family under ``iterative-cn`` (which integrates its
    # tighter axis cleanly and gains the finer near-axis resolution),
    # rigged-CGL (``"cgl"``) for cylindrical cnab2 (the tighter
    # half-CGL axis destabilises the explicit scheme -- see the
    # ``Geometry`` docstring), and the full CGL grid (``"cgl"``) for
    # the Cartesian / annular families.  Resolving to a *concrete*
    # value here (rather than interpreting ``None`` at grid
    # construction) makes every snapshot embed the grid it actually
    # ran, so a resume pins the trajectory's grid independently of
    # later defaults.  Re-resolved on every layer application (a later
    # ``system`` / ``scheme`` override cannot inherit a stale
    # default); assigned directly -- not via the merge loop -- so the
    # field never enters ``_user_set_fields``.  A custom
    # ``geo.wall_grid`` file forces ``grid_type = None`` (setting both
    # is an error); periodic systems have no wall-normal grid and stay
    # ``None``.
    if ("geo", "grid_type") not in _user_set_fields:
        if params.geo.wall_grid is not None or system not in walled_systems:
            params.geo.grid_type = None
        elif (
            system in cylindrical_systems
            and params.step.scheme == "iterative-cn"
        ):
            params.geo.grid_type = "half-cgl"
        else:
            params.geo.grid_type = "cgl"

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

    # Spectral-mode probe stream: both knobs together, wall-bounded
    # only, and every probed index must address a *true* (unpadded)
    # mode -- axis 2 carries ``nz - 1`` complex modes, axis 3 carries
    # ``nx // 2`` real-FFT modes (Nyquist omitted; ``harmonics``).
    if (o.probe_modes is None) != (o.it_probes is None):
        raise ValueError(
            "outs.probe_modes and outs.it_probes must be set together "
            "(one selects the modes, the other the cadence)."
        )
    if o.probe_modes is not None:
        if params.phys.system not in walled_systems:
            raise ValueError(
                "outs.probe_modes: the spectral-mode probe stream "
                "supports wall-bounded systems only (system "
                f"{params.phys.system!r})."
            )
        n2, n3 = params.res.nz - 1, params.res.nx // 2
        for i2, i3 in parse_mode_pairs(o.probe_modes):
            if i2 >= n2 or i3 >= n3:
                raise ValueError(
                    f"outs.probe_modes: mode ({i2},{i3}) out of range "
                    f"(axis 2 has {n2} modes from res.nz = "
                    f"{params.res.nz}, axis 3 has {n3} modes from "
                    f"res.nx = {params.res.nx})."
                )

    # Stochastic mode forcing (the ``force`` section): all-or-none
    # knobs, wall-bounded non-viscoelastic only, true-mode indices,
    # no mean-mode kick, and kick/probe sample alignment.
    f = params.force
    f_set = {
        "modes": f.modes,
        "profiles": f.profiles,
        "amplitude": f.amplitude,
        "it_force": f.it_force,
    }
    missing = [k for k, v in f_set.items() if v is None]
    if missing and len(missing) < len(f_set):
        raise ValueError(
            "force.modes, force.profiles, force.amplitude and "
            "force.it_force enable the stochastic forcing together; "
            f"missing: {', '.join(missing)}."
        )
    if f.modes is not None:
        if (
            params.phys.system not in walled_systems
            or params.phys.system in viscoelastic_systems
        ):
            raise ValueError(
                "force.modes: stochastic forcing supports the "
                "wall-bounded velocity systems only (system "
                f"{params.phys.system!r})."
            )
        n2, n3 = params.res.nz - 1, params.res.nx // 2
        force_pairs = parse_mode_pairs(f.modes)
        for i2, i3 in force_pairs:
            if i2 >= n2 or i3 >= n3:
                raise ValueError(
                    f"force.modes: mode ({i2},{i3}) out of range "
                    f"(axis 2 has {n2} modes, axis 3 has {n3} modes)."
                )
            if (i2, i3) == (0, 0):
                raise ValueError(
                    "force.modes: the (0,0) mean mode cannot be "
                    "forced (its coefficient is real, and under "
                    "bulk-velocity driving it is constrained)."
                )
        if o.it_probes is not None:
            if f.it_force % o.it_probes != 0:
                raise ValueError(
                    f"force.it_force ({f.it_force}) must be a "
                    f"multiple of outs.it_probes ({o.it_probes}) so "
                    "every kick coincides with a (pre-kick) probe "
                    "sample."
                )
            probed = set(parse_mode_pairs(o.probe_modes))
            unprobed = [m for m in force_pairs if m not in probed]
            if unprobed:
                # A note, not an error: forcing one mode while probing
                # another is a legitimate cross-mode experiment.
                print(
                    f"[force] note: forced mode(s) {unprobed} are not "
                    "in outs.probe_modes; their own response will not "
                    "be recorded."
                )
        else:
            print(
                "[force] note: no probe stream configured "
                "(outs.probe_modes); the forced responses will not be "
                "recorded, so the run cannot feed the SSI "
                "identification."
            )

    # The half-CGL radial grid is a cylindrical-only option, and its
    # tighter near-axis point makes the explicit cnab2 scheme blow up
    # at low dt (near-axis instability); restrict it to iterative-cn,
    # which integrates it cleanly (and defaults to it).  The rigged
    # grid (the cnab2 default) has no such restriction.
    if params.geo.grid_type == "half-cgl":
        if params.phys.system not in cylindrical_systems:
            raise ValueError(
                "geo.grid_type='half-cgl' applies only to the "
                "cylindrical geometry (system "
                f"{params.phys.system!r} has no axis); use the "
                "default 'cgl' grid."
            )
        if params.step.scheme != "iterative-cn":
            raise ValueError(
                "geo.grid_type='half-cgl' requires "
                "step.scheme='iterative-cn' (the tighter half-CGL "
                "axis destabilises the explicit "
                f"{params.step.scheme!r} scheme at low dt); use the "
                "rigged-CGL grid ('cgl', the cnab2 default) instead."
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

    # The dense backend is a readability/regression reference, not a
    # production path; nudge wall-bounded production runs back to the
    # default.  Plain ``print``: this runs before JAX is configured.
    if (
        params.solver.backend == "dense"
        and params.phys.system in walled_systems
    ):
        print(
            "[solver] backend='dense' selected: full Ny x Ny per-mode "
            "LU factors -- a reference backend kept for readability "
            "and regression checks; prefer the default 'pallas' "
            "backend for production runs."
        )


def round_up_padded(n_padded: int, divisor: int) -> int:
    r"""Round a padded FFT size up to a multiple of *divisor*.

    Returns the smallest `$m \ge n_\mathrm{padded}$` with
    `$m \equiv 0 \pmod d$` (``d = max(divisor, 1)``), so the padded
    physical axis splits evenly across *divisor* devices.  The padded
    region carries only zero (dealiased) modes, so rounding up is
    physically neutral (it costs marginally more FFT work).  The pad
    parity is unconstrained: the :func:`dnsjax.fft.zeropad_fft` /
    ``truncate_fft`` wrap-order mode placement is exact for even and
    odd pads alike, so no combination of grid size and mesh axis is
    rejected.
    """
    d = max(divisor, 1)
    return -(-n_padded // d) * d


#: Primes with specialised FFT radix kernels (cuFFT, pocketfft/XLA).
#: A transform length with only these factors takes the fast kernels;
#: any larger prime factor falls back to a markedly slower generic
#: (Bluestein-type) algorithm.
_FFT_SMOOTH_PRIMES = (2, 3, 5, 7)


def is_fft_smooth(n: int) -> bool:
    r"""True when *n* has no prime factor beyond `$\{2, 3, 5, 7\}$`.

    Such 7-smooth transform lengths take the specialised FFT radix
    kernels; padded FFT sizes are therefore rounded up to them
    (:func:`round_up_padded_smooth`).
    """
    if n <= 0:
        return False
    for prime in _FFT_SMOOTH_PRIMES:
        while n % prime == 0:
            n //= prime
    return n == 1


def round_up_padded_smooth(n_padded: int, divisor: int) -> int:
    r"""Round a padded FFT size up to a 7-smooth multiple of *divisor*.

    The smallest `$m \ge n_\mathrm{padded}$` that is a multiple of
    ``max(divisor, 1)`` **and** 7-smooth (:func:`is_fft_smooth`), so
    the FFT at the padded size takes the fast radix-2/3/5/7 kernels
    while the axis still splits evenly across *divisor* devices.
    Physically neutral for the same reason as
    :func:`round_up_padded` -- the extra slots carry only zero
    (dealiased) modes -- and 7-smooth numbers are dense at practical
    sizes, so the bump beyond the divisibility rounding is a few
    percent at most.  When *divisor* itself has a prime factor beyond
    7 no multiple of it can be smooth; the plain divisibility rounding
    is returned instead.
    """
    d = max(divisor, 1)
    m = round_up_padded(n_padded, d)
    if not is_fft_smooth(d):
        return m
    while not is_fft_smooth(m):
        m += d
    return m


def _rounding_note(
    name: str, old: int, new: int, divisor: int, axis: str
) -> str:
    """Compose the startup-diagnostic line for one padded-size bump."""
    reasons = []
    if old % max(divisor, 1) != 0:
        reasons.append(f"{axis} divisibility")
    if not is_fft_smooth(old) and is_fft_smooth(new):
        reasons.append("FFT-friendly size")
    return f"{name} rounded from {old} to {new} ({', '.join(reasons)})."


@dataclass
class PaddedResolution:
    """Grid sizes after 3/2-rule oversampling for dealiasing.

    The oversampled (padded) grid is used when evaluating nonlinear terms
    in physical space.  Each direction is expanded by a factor of
    ``oversampling_factor / 2`` (typically 3/2); every FFT axis is
    then rounded up to a mesh-divisible, FFT-friendly 7-smooth length
    (:meth:`apply_rounding`), with every adjustment recorded in
    :attr:`notes` for the startup diagnostics printed by
    :mod:`dnsjax.sharding`.
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
    notes: list[str] = field(default_factory=list)

    def apply_rounding(self, parameters: Parameters) -> None:
        r"""Round padded sizes for mesh divisibility and FFT speed.

        Rounds every FFT axis up with :func:`round_up_padded_smooth`:
        ``nz_padded`` and the periodic ``ny_padded`` to 7-smooth
        multiples of the mesh direction that shards them (``np1`` for
        `$z$`, ``np0`` for `$y$`), and ``nx_padded`` (the real-FFT
        axis, never sharded in physical space -- divisor 1) to the
        next 7-smooth length.  Both roundings insert only zero
        (dealiased) modes, so they are physically neutral; smoothness
        keeps every transform on the fast radix-2/3/5/7 kernels
        regardless of the base resolution.  Idempotent; re-applied at
        :mod:`dnsjax.sharding` import for entry points that set
        ``params.dist`` after (or without)
        :meth:`set_padded_resolution`.  Each adjustment appends a
        diagnostic line to :attr:`notes`, which
        :mod:`dnsjax.sharding` prints once on the main process.
        """
        dist = parameters.dist
        np0 = dist.np0 if dist is not None else 1
        np1 = dist.np1 if dist is not None else 1

        nx_new = round_up_padded_smooth(self.nx_padded, 1)
        if nx_new != self.nx_padded:
            self.notes.append(
                _rounding_note("nx_padded", self.nx_padded, nx_new, 1, "")
            )
            self.nx_padded = nx_new
        if self.ny_padded is not None:
            ny_new = round_up_padded_smooth(self.ny_padded, np0)
            if ny_new != self.ny_padded:
                self.notes.append(
                    _rounding_note(
                        "ny_padded", self.ny_padded, ny_new, np0, "np0"
                    )
                )
                self.ny_padded = ny_new
        nz_new = round_up_padded_smooth(self.nz_padded, np1)
        if nz_new != self.nz_padded:
            self.notes.append(
                _rounding_note("nz_padded", self.nz_padded, nz_new, np1, "np1")
            )
            self.nz_padded = nz_new

    def set_padded_resolution(self, parameters: Parameters) -> None:
        r"""Recompute padded sizes from *parameters*.

        The natural sizes are ``oversampling_factor * n // 2`` (`$y$`
        unpadded for wall-bounded flows and only optionally oversampled
        for periodic ones); :meth:`apply_rounding` then rounds every
        FFT axis up to a mesh-divisible, FFT-friendly 7-smooth length
        (``nx_padded`` is exempt from the divisibility part only:
        real-FFT axis, never sharded in physical space).
        """
        self.notes = []
        if (
            parameters.phys.system in periodic_systems
            and not parameters.phys.oversample_y
        ):
            print("WARNING: y is *not* oversampled!")

        self.nx_padded = (
            parameters.phys.oversampling_factor * parameters.res.nx // 2
        )
        if parameters.phys.system in periodic_systems:
            self.ny_padded = (
                parameters.phys.oversampling_factor * parameters.res.ny // 2
                if parameters.phys.oversample_y
                else parameters.res.ny
            )
        else:
            self.ny_padded = None
        self.nz_padded = (
            parameters.phys.oversampling_factor * parameters.res.nz // 2
        )
        self.apply_rounding(parameters)


padded_res: PaddedResolution = PaddedResolution()
