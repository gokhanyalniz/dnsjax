from dataclasses import dataclass

from jax import numpy as jnp
from parameters import (
    params,
)
from sharding import sharding


@dataclass
class PlaneChannelFlow:
    ys = -jnp.cos(
        jnp.arange(params.res.ny, dtype=sharding.float_type)
        * jnp.pi
        / (params.res.ny - 1)
    )

    base_flow = 1 - ys**2
    dy_base_flow = -2 * ys
    ekin_lam = 4 / 15
    input_lam = 4 / (3 * params.phys.re)
    dissip_lam = 4 / (3 * params.phys.re)
    pres_grad_lam = 2 / params.phys.re
    ubulk_lam = 2 / 3


flow = PlaneChannelFlow()
