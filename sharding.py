from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import params


@dataclass
class Sharding:
    N_DEVICES = params.dist.Np0 * params.dist.Np1
    N_DEVICES_REPORTED = len(jax.devices())
    if N_DEVICES_REPORTED != N_DEVICES:
        jax.distributed.shutdown()
        exit(
            f"# of devices visible ({N_DEVICES_REPORTED}) "
            f"is not equal to Np0 x Np1 = {N_DEVICES}."
        )

    MESH = jax.make_mesh(
        [params.dist.Np0, params.dist.Np1],
        axis_names=("Z", "X"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )

    phys_shard = NamedSharding(MESH, P(None, "Z", "X", None))
    spec_shard = NamedSharding(MESH, P(None, "Z", "X", None))
    scalar_phys_shard = NamedSharding(MESH, P("Z", "X", None))
    scalar_spec_shard = NamedSharding(MESH, P("Z", "X", None))

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64


sharding = Sharding()
