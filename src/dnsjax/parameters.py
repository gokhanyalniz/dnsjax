"""Simulation parameter management via Pydantic models and TOML files.

Configuration is layered: hard-coded defaults -> ``parameters.toml`` (if
present) -> command-line arguments.  The global singletons ``params``,
``derived_params``, and ``padded_res`` are mutated in-place by
:func:`update_parameters` so that every module sees the same state.
"""

import tomllib
from dataclasses import dataclass
from datetime import timedelta
from math import pi
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

monochromatic_systems: list[str] = ["kolmogorov", "waleffe"]
periodic_systems: list[str] = ["decaying-box", *monochromatic_systems]

cartesian_systems: list[str] = ["plane-couette"]
walled_systems: list[str] = [*cartesian_systems, "pipe"]

# TODO: Add physical sanity checks


class Distribution(BaseModel):
    """Device distribution and backend platform."""

    np: int = Field(ge=1, default=1)
    platform: Literal["cpu", "cuda", "rocm", "tpu"] = "cpu"


class Physics(BaseModel):
    """Physical parameters: Reynolds number, flow system, dealiasing."""

    re: float = Field(gt=0, default=1000)  # Reynolds number
    # Kolmogorov: sine forcing
    # Waleffe: cosine forcing + Ry symmetry (not yet implemented)
    system: Literal[*periodic_systems, *walled_systems] = "kolmogorov"
    # (n + 1) / 2 oversampling in each direction
    # to dealias the n'th order nonlinearity
    # oversampling_factor = n + 1
    oversampling_factor: int = Field(ge=2, default=3)
    oversample_y: bool = True


class Geometry(BaseModel):
    """Domain size and optional tilt angle for the forcing direction."""

    lx: float = Field(gt=0, default=4.0)
    lz: float = Field(gt=0, default=4.0)
    tilt_degree: float = Field(gt=-180, le=180, default=0)


class Resolution(BaseModel):
    """Grid resolution (number of Fourier modes before dealiasing)."""

    # Number of grid points = (before oversampling) # of Fourier modes
    nx: int = Field(ge=1, default=128)
    ny: int = Field(ge=1, default=128)
    nz: int = Field(ge=1, default=128)
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


class TimeStepping(BaseModel):
    """Time integration parameters.

    The ``implicitness`` parameter *c* controls the implicit/explicit split
    of the viscous term.  ``c = 0.5`` gives a standard Crank-Nicolson
    scheme (second-order); ``c = 0.501`` is recommended for improved
    stability with negligible accuracy loss.
    """

    dt: float = Field(gt=0, default=0.01)
    implicitness: float = Field(ge=0, le=1, default=0.5)
    corrector_tolerance: float = Field(gt=0, default=1e-5)
    max_corrector_iterations: int = Field(ge=1, default=10)


class Termination(BaseModel):
    """Stopping criteria for the simulation."""

    max_sim_time: float | None = None
    max_wall_time: timedelta | None = None  # ISO 8601 format for durations


class Debugging(BaseModel):
    """Debug and diagnostic flags."""

    time_functions: bool = True
    measure_corrections: bool = False
    correct_divergence: bool = True


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
    debug: Debugging | None = Debugging()


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

    ``ly`` is fixed by the flow system (4 for periodic, 2 for walled).
    Tilt flags are set from ``tilt_degree``.
    """

    ly: float = 2
    tilt_rad: float = 0
    tilt: bool = False
    tilt_90: bool = False


params: Parameters = Parameters()
derived_params: DerivedParameters = DerivedParameters()


def read_parameters(path: Path) -> Parameters:
    """Load a ``Parameters`` instance from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Parameters(**raw)


def update_parameters(params_new: Parameters) -> None:
    """Merge *params_new* into the global ``params`` and recompute derived values.

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
    elif system in cartesian_systems or system == "pipe":
        derived_params.ly = 2
    else:
        raise NotImplementedError

    derived_params.tilt_rad = pi * params.geo.tilt_degree / 180

    if (
        abs(params.geo.tilt_degree) == 0
        or abs(params.geo.tilt_degree - 180) == 0
    ):
        derived_params.tilt = False
        derived_params.tilt_90 = False
    elif (
        abs(params.geo.tilt_degree + 90) == 0
        or abs(params.geo.tilt_degree - 90) == 0
    ):
        derived_params.tilt = True
        derived_params.tilt_90 = True
    else:
        derived_params.tilt = True
        derived_params.tilt_90 = False


@dataclass
class PaddedResolution:
    """Grid sizes after 3/2-rule oversampling for dealiasing.

    The oversampled (padded) grid is used when evaluating nonlinear terms
    in physical space.  Each direction is expanded by a factor of
    ``oversampling_factor / 2`` (typically 3/2).
    """

    nx_padded: int = params.phys.oversampling_factor * params.res.nx // 2
    ny_padded: int = (
        params.res.ny
        if not params.phys.oversample_y
        else params.phys.oversampling_factor * params.res.ny // 2
    )
    nz_padded: int = params.phys.oversampling_factor * params.res.nz // 2

    def set_padded_resolution(self, parameters: Parameters) -> None:
        """Recompute padded sizes from *parameters*."""
        if not parameters.phys.oversample_y:
            print("WARNING: y is *not* oversampled!")

        self.nx_padded = (
            parameters.phys.oversampling_factor * params.res.nx // 2
        )
        self.ny_padded = (
            parameters.res.ny
            if not params.phys.oversample_y
            else parameters.phys.oversampling_factor * params.res.ny // 2
        )
        self.nz_padded = (
            parameters.phys.oversampling_factor * params.res.nz // 2
        )


padded_res: PaddedResolution = PaddedResolution()
