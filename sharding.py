import jax
from jax import numpy as jnp
from jax.sharding import AxisType

from parameters import params

N_DEVICES = params.dist.Np0 * params.dist.Np0
if len(jax.devices()) != N_DEVICES:
    jax.distributed.shutdown()
    exit("# of devices not equal to Np0 x Np1.")

MESH = jax.make_mesh(
    [params.dist.Np0, params.dist.Np1],
    axis_names=("Z", "X"),
    axis_types=(AxisType.Auto, AxisType.Auto),
)

if params.res.double_precision:
    float_type = jnp.float64
    complex_type = jnp.complex128
else:
    float_type = jnp.float32
    complex_type = jnp.complex64
