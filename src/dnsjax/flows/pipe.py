from dataclasses import dataclass

from jax import numpy as jnp
from parameters import (
    params,
)
from sharding import sharding


@dataclass
class PipeFlow:
    ys = -jnp.cos(
        jnp.arange(2 * params.res.ny + 1, dtype=sharding.float_type)
        * jnp.pi
        / (2 * params.res.ny)
    )[-params.res.ny :]

    base_flow = 1 - ys**2
    dy_base_flow = -2 * ys
    ekin_lam = 1 / 6
    input_lam = 2 / params.phys.re
    dissip_lam = 2 / params.phys.re
    pres_grad_lam = 4 / params.phys.re
    ubulk_lam = 1 / 2


flow = PipeFlow()
