import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Distribution(BaseModel):
    # Set np0=1, np1=N for slab decomposition
    np0: int = Field(ge=1, default=1)
    np1: int = Field(ge=1, default=1)
    platform: Literal["cpu", "cuda", "rocm", "tpu"] = "cpu"


class Physics(BaseModel):
    re: float = Field(gt=0, default=1000)  # Reynolds number
    # Kolmogorov: sine forcing
    # Waleffe: cosine forcing + Ry symmetry (not yet implemented)
    forcing: Literal["none", "kolmogorov", "waleffe"] = "kolmogorov"
    # (n + 1) / 2 oversampling in each direction
    # to dealias the n'th order nonlinearity
    # oversampling_factor = n + 1
    oversampling_factor: int = Field(ge=2, default=3)
    oversample_y: bool = True


class Geometry(BaseModel):
    lx: float = Field(gt=0, default=4.0)
    lz: float = Field(gt=0, default=4.0)
    # ... in units where Ly = 4.0 is fixed.
    # Ly *is* fixed, but still kept here for generality.
    # Do not change for Kolmogorov/Waleffe flow!
    ly: float = Field(gt=0, default=4.0)


class Resolution(BaseModel):
    # Number of grid points = (before oversampling) # of Fourier modes
    nx: int = Field(ge=1, default=128)
    ny: int = Field(ge=1, default=128)
    nz: int = Field(ge=1, default=128)
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
    time_functions: bool = True
    measure_corrections: bool = False
    correct_divergence: bool = True


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
    nx_half: int = params.res.nx // 2
    ny_half: int = params.res.ny // 2
    nz_half: int = params.res.nz // 2

    nx_padded: int = params.phys.oversampling_factor * params.res.nx // 2
    ny_padded: int = (
        params.res.ny
        if not params.phys.oversample_y
        else params.phys.oversampling_factor * params.res.ny // 2
    )
    nz_padded: int = params.phys.oversampling_factor * params.res.nz // 2

    def set_padded_resolution(self, parameters: Parameters):
        if not parameters.phys.oversample_y:
            print("WARNING: y is *not* oversampled!")
        self.nx_half = parameters.res.nx // 2
        self.ny_half = parameters.res.ny // 2
        self.nz_half = parameters.res.nz // 2

        self.nx_padded = parameters.phys.oversampling_factor * self.nx_half
        self.ny_padded = (
            parameters.res.ny
            if not params.phys.oversample_y
            else parameters.phys.oversampling_factor * self.ny_half
        )
        self.nz_padded = parameters.phys.oversampling_factor * self.nz_half


padded_res = PaddedResolution()
