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

    base_flow = jnp.copy(ys)
    dy_base_flow = jnp.ones(params.res.ny, dtype=sharding.float_type)
    ekin_lam = 1 / 6
    input_lam = 1 / params.phys.re
    dissip_lam = 1 / params.phys.re
    pres_grad_lam = 0
    ubulk_lam = 0


flow = PlaneChannelFlow()
