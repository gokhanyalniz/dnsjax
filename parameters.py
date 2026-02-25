import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Distribution(BaseModel):
    # Set Np0=1, Np1=N for slab decomposition
    Np0: int = Field(ge=1, default=1)
    Np1: int = Field(ge=1, default=1)
    platform: Literal["cpu", "cuda", "rocm", "tpu"] = "cpu"


class Physics(BaseModel):
    Re: float = Field(gt=0, default=1000)  # Reynolds number
    # Kolmogorov: sine forcing
    # Waleffe: cosine forcing + Ry symmetry (not yet implemented)
    forcing: Literal["kolmogorov", "waleffe"] | None = None
    # (n + 1) / 2 oversampling in each direction
    # to dealias the n'th order nonlinearity
    # oversampling_factor = n + 1
    oversampling_factor: int = Field(ge=2, default=3)


class Geometry(BaseModel):
    Lx: float = Field(gt=0, default=4.0)
    Lz: float = Field(gt=0, default=4.0)
    # ... in units where Ly = 4.0 is fixed.
    # Ly *is* fixed, but still kept here for generality.
    # Do not change for Kolmogorov/Waleffe flow!
    Ly: float = Field(gt=0, default=4.0)


class Resolution(BaseModel):
    # Number of grid points = (before oversampling) # of Fourier modes
    Nx: int = Field(ge=1, default=128)
    Ny: int = Field(ge=1, default=128)
    Nz: int = Field(ge=1, default=128)
    double_precision: bool = True  # use double-precision floating point


class Initiation(BaseModel):
    start_from_laminar: bool = True
    snapshot: Path | None = None
    t0: float = 0  # Initial value of time
    it0: int = 0  # Initial value of number of time steps taken


class Outputs(BaseModel):
    # All outputs are with respect to the number of time steps taken
    it_stats: int | None = None  # How often to compute stats


class TimeStepping(BaseModel):
    dt: float = Field(gt=0, default=0.01)
    implicitness: float = Field(ge=0, le=1, default=0.5)
    corrector_tolerance: float = Field(gt=0, default=1e-5)
    max_corrector_iterations: int = Field(ge=1, default=10)


class Termination(BaseModel):
    max_sim_time: float | None = None
    max_wall_time: timedelta | None = None  # ISO 8601 format for durations


class Debugging(BaseModel):
    time_functions: bool = False


class Parameters(BaseModel):
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


params = Parameters()


def read_parameters(path: Path) -> Parameters:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Parameters(**raw)


def update_parameters(params_new: Parameters):
    for category, dict in params_new.model_dump(exclude_unset=True).items():
        if dict is not None:
            for key, value in dict.items():
                if value is not None:
                    setattr(getattr(params, category), key, value)


@dataclass
class PaddedResolution:
    Nx_half: int = params.res.Nx // 2
    Ny_half: int = params.res.Ny // 2
    Nz_half: int = params.res.Nz // 2

    Nx_padded: int = params.phys.oversampling_factor * params.res.Nx // 2
    Ny_padded: int = params.phys.oversampling_factor * params.res.Ny // 2
    Nz_padded: int = params.phys.oversampling_factor * params.res.Nz // 2

    def set_padded_resolution(self, parameters: Parameters):
        self.Nx_half = parameters.res.Nx // 2
        self.Ny_half = parameters.res.Ny // 2
        self.Nz_half = parameters.res.Nz // 2

        self.Nx_padded = parameters.phys.oversampling_factor * self.Nx_half
        self.Ny_padded = parameters.phys.oversampling_factor * self.Ny_half
        self.Nz_padded = parameters.phys.oversampling_factor * self.Nz_half


padded_res = PaddedResolution()
