import sys
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import AxisType, explicit_axes
from jax.sharding import PartitionSpec as P

from parameters import padded_res, params


@dataclass
class Sharding:
    n_devices = params.dist.np
    main_device = bool(jax.process_index() == 0)

    devices = jax.devices()
    n_devices_reported = len(devices)
    if n_devices_reported != n_devices:
        if main_device:
            print(
                f"# of devices visible ({n_devices_reported}) "
                f"is not equal to np = {n_devices}.",
                flush=True,
            )
        sys.exit(1)

    print(
        f"Working with {n_devices} {params.dist.platform} devices:",
        *devices,
        flush=True,
    )

    axis_names = ("gpus",)
    mesh = jax.make_mesh(
        (params.dist.np,),
        axis_names=axis_names,
        axis_types=(AxisType.Auto,),
    )

    jax.set_mesh(mesh)

    spec_vector_shard = P(None, None, None, *axis_names)
    spec_scalar_shard = P(None, None, *axis_names)

    phys_vector_shard = P(None, None, *axis_names, None)
    phys_scalar_shard = P(None, *axis_names, None)

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    phys_type = float_type
    int4_substitute = jnp.int8

    spec_shape = (
        params.res.ny - 1,
        params.res.nz - 1,
        params.res.nx // 2,
    )

    phys_shape = (
        padded_res.ny_padded,
        padded_res.nz_padded,
        padded_res.nx_padded,
    )

    def constrain_spec_vector(self, vector):
        return with_sharding_constraint(vector, self.spec_vector_shard)

    def constrain_spec_scalar(self, scalar):
        return with_sharding_constraint(scalar, self.spec_scalar_shard)

    def constrain_phys_vector(self, vector):
        return with_sharding_constraint(vector, self.phys_vector_shard)

    def constrain_phys_scalar(self, scalar):
        return with_sharding_constraint(scalar, self.phys_scalar_shard)

    def exit(self, code=1):
        sys.exit(code)

    def print(self, *args, **kwargs):
        if self.main_device:
            print(*args, **kwargs, flush=True)


sharding = Sharding()


@partial(explicit_axes, axes=sharding.axis_names)
def get_zeros(shape, dtype, out_sharding):
    x = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=out_sharding,
    )
    return x
