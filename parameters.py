from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


# Settings class adapted from:
# https://docs.pydantic.dev/latest/concepts/pydantic_settings
class Parameters(BaseSettings):
    class Physics(BaseModel):
        Re: float = Field(gt=0, default=630)  # Reynolds number
        # Kolmogorov: sine forcing
        # Waleffe: cosine forcing + Ry symmetry (not yet implemented)
        forcing: Literal["none", "kolmogorov", "waleffe"] = "none"
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
        Nx: int = Field(ge=1, default=48)
        Ny: int = Field(ge=1, default=48)
        Nz: int = Field(ge=1, default=48)

    class Initiation(BaseModel):
        start_from_laminar: bool = True
        snapshot: Path | None = None
        t0: float = 0  # Initial value of time
        it0: int = 0  # Initial value of number of time steps taken

    class Outputs(BaseModel):
        # All outputs are with respect to the number of time steps taken
        it_stats: int = 20  # How often to compute stats

    class TimeStepping(BaseModel):
        dt: float = Field(gt=0, default=0.02)
        implicitness: float = Field(ge=0, le=1, default=0.5)
        corrector_tolerance: float = Field(gt=0, default=1e-5)
        max_corrector_iterations: int = Field(ge=1, default=10)

    class Termination(BaseModel):
        max_sim_time: float = 10

    class Debugging(BaseModel):
        time_functions: bool = False

    phys: Physics
    geo: Geometry
    res: Resolution
    init: Initiation
    outs: Outputs
    step: TimeStepping
    stop: Termination
    debug: Debugging

    model_config = SettingsConfigDict(toml_file="parameters.toml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


params = Parameters()
