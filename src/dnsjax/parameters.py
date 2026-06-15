"""Simulation parameter management via Pydantic models and TOML files.

Configuration is layered: hard-coded defaults -> ``parameters.toml`` (if
present) -> command-line arguments.  The global singletons ``params``,
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
annular_systems: list[str] = ["taylor-couette"]
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
    # Kolmogorov: sine forcing
    # Waleffe: cosine forcing + Ry symmetry (not yet implemented)
    system: Literal[*periodic_systems, *walled_systems] = "kolmogorov"
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


class Geometry(BaseModel):
    """Domain size and optional tilt angle for the forcing direction.

    Wall-normal grid selection (precedence order):

    1. ``wall_grid`` (file path): load a custom grid from file.
    2. ``grid_type``: generate a named grid at startup.
    3. Default: CGL (Cartesian) or half-CGL (cylindrical).

    Setting both ``wall_grid`` and ``grid_type`` is an error.
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


class Resolution(BaseModel):
    """Grid resolution (number of Fourier modes before dealiasing)."""

    # Number of grid points = (before oversampling) # of Fourier modes
    nx: int = Field(ge=1, default=128)
    ny: int = Field(ge=1, default=128)
    nz: int = Field(ge=1, default=128)
    fd_order: int = Field(ge=2, default=4)
    double_precision: bool = True  # use double-precision floating point


class Initiation(BaseModel):
    """Initial condition: start from laminar or load a snapshot."""

    start_from_laminar: bool = True
    snapshot: Path | None = None
    t0: float = 0  # Initial value of time
    it0: int = 0  # Initial value of number of time steps taken


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
    # How often (in steps) to sync the corrector error to the host
    # for the convergence check.  Between checks the host enqueues
    # steps ahead of the device (JAX async dispatch); corrector
    # divergence is therefore detected up to ``it_error_check``
    # steps late, each late step bounded by
    # ``max_corrector_iterations``.  1 restores a per-step check
    # (and a per-step host-device sync).
    it_error_check: int = Field(ge=1, default=10)
    # Rows buffered on device before flushing ``stats.dat`` /
    # ``steps.dat`` to disk.
    nbuffer: int = Field(ge=1, default=100)
    stats_precision: int = Field(ge=1, le=17, default=9)
    # How processes write a snapshot's shared chunk files:
    #   "concurrent": all processes write their disjoint byte ranges
    #                 at once (fast; POSIX/parallel filesystems).
    #   "serial":     rank-ordered (token-passing) writes, one
    #                 process at a time -- safe on filesystems such
    #                 as NFS where concurrent writes can corrupt
    #                 data.  No effect for single-process runs.
    snapshot_write_mode: Literal["concurrent", "serial"] = "concurrent"


class TimeStepping(BaseModel):
    """Time integration parameters.

    The ``implicitness`` parameter *c* controls the implicit/explicit split
    of the viscous term.  ``c = 0.5`` gives a standard Crank-Nicolson
    scheme (second-order).
    """

    dt: float = Field(gt=0, default=0.01)
    implicitness: float = Field(ge=0, le=1, default=0.5)
    corrector_tolerance: float = Field(gt=0, default=1e-5)
    max_corrector_iterations: int = Field(ge=1, default=10)


class Termination(BaseModel):
    """Stopping criteria for the simulation."""

    max_sim_time: float | None = None
    max_wall_time: timedelta | None = None  # ISO 8601 format for durations


class Solver(BaseModel):
    """Linear algebraic solver configurations."""

    # ``"banded"``: SPIKE block-partitioned solver (memory-efficient,
    # exploits the known stencil bandwidth of D1, D2).
    # ``"dense"``: full ``Ny x Ny`` LU factors per Fourier mode
    # (legacy path, kept for verification against the banded path).
    backend: Literal["banded", "dense"] = "banded"
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

    ``ly`` is fixed by the geometry (4 for triply-periodic, 2 for walled;
    only read by triply-periodic code).
    ``volume_fac`` is also fixed by the geometry
        (1 for periodic, 2 for Cartesian, 0.5 for cylindrical, and
        ``(r2^2 - r1^2)/2`` for the annulus).
    ``ccf_A``, ``ccf_B`` are the circular-Couette base-flow coefficients
    ``U_theta = ccf_A * r + ccf_B / r`` and ``r_inner``, ``r_outer`` the
    non-dim annular radii (set for system == "taylor-couette").
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


params: Parameters = Parameters()
derived_params: DerivedParameters = DerivedParameters()


def read_parameters(path: Path) -> Parameters:
    """Load a ``Parameters`` instance from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Parameters(**raw)


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
        derived_params.ly = 4
        derived_params.volume_fac = 1
    elif system in cartesian_systems:
        derived_params.ly = 2
        derived_params.volume_fac = 2
    elif system in cylindrical_systems:
        derived_params.ly = 2  # Diameter = 2*radius
        # Force a full 2*pi spanwise extent for the cylindrical
        # geometry, overriding any user-supplied value (the
        # azimuthal modes are integer harmonics over 2*pi).
        params.geo.lz = 2 * pi
        # To compansate for the (1/Lz) factor
        derived_params.volume_fac = 0.5
    elif system in annular_systems:
        # Taylor-Couette: validate the (re1, re2, eta) control
        # parameters and derive the circular-Couette base flow
        # U_theta = A0 r + B0/r on the annulus [r1, r2] (gap = 1).
        re1, re2, eta = params.phys.re1, params.phys.re2, params.geo.eta
        if eta is None:
            raise ValueError(
                "taylor-couette requires geo.eta (radius ratio r1/r2)"
            )
        if re1 is None or re2 is None:
            raise ValueError("taylor-couette requires phys.re1 and phys.re2")
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
        # Non-dim radii on the gap d = r2 - r1 = 1.
        r1 = eta / (1 - eta)
        r2 = 1 / (1 - eta)
        derived_params.r_inner = r1
        derived_params.r_outer = r2
        # Gap-scaled circular-Couette coefficients (divided by Re_ref):
        #   A0 = (re2 - eta re1) / [(1+eta) Re_ref]
        #   B0 = eta (re1 - eta re2) / [(1+eta)(1-eta)^2 Re_ref]
        derived_params.ccf_A = (re2 - eta * re1) / ((1 + eta) * re_ref)
        derived_params.ccf_B = (
            eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
        )
        derived_params.ly = 2 * r2  # cosmetic (unread by wall-bounded)
        # Force a full 2*pi azimuthal extent (integer harmonics).
        params.geo.lz = 2 * pi
        # Annular area normalisation: volume_fac = int_{r1}^{r2} r dr.
        derived_params.volume_fac = (r2**2 - r1**2) / 2
    else:
        raise NotImplementedError

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
