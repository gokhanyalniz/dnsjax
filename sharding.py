import sys
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from parameters import padded_res, params


@dataclass
class Sharding:
    n_devices = params.dist.np0 * params.dist.np1
    main_device = bool(jax.process_index() == 0)

    devices = jax.devices()
    n_devices_reported = len(devices)
    if n_devices_reported != n_devices:
        if main_device:
            print(
                f"# of devices visible ({n_devices_reported}) "
                f"is not equal to np0 x np1 = {n_devices}.",
                flush=True,
            )
        sys.exit(1)

    print(
        f"Working with {n_devices} {params.dist.platform} devices:",
        *devices,
        flush=True,
    )

    axis_names = ("z", "x")
    if params.dist.np1 == 1 and params.dist.np0 > 1:
        print(
            "(np0 > 1, np1 = 1) distribution is not supported. "
            "Running with (np1 = 1, np0) instead.",
            flush=True,
        )
        mesh = jax.make_mesh(
            (params.dist.np1, params.dist.np0),
            axis_names=axis_names,
            axis_types=(AxisType.Auto, AxisType.Auto),
        )
    else:
        mesh = jax.make_mesh(
            (params.dist.np0, params.dist.np1),
            axis_names=axis_names,
            axis_types=(AxisType.Auto, AxisType.Auto),
        )

    jax.set_mesh(mesh)

    vector_shard = P(None, *axis_names, None)
    scalar_shard = P(*axis_names, None)

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    phys_type = float_type
    int4_substitute = jnp.int8

    spec_shape = (
        padded_res.nz_padded,
        padded_res.nx_padded,
        padded_res.ny_padded,
    )
    phys_shape = (
        padded_res.ny_padded,
        padded_res.nz_padded,
        padded_res.nx_padded,
    )

    def constrain_vector(self, vector):
        return with_sharding_constraint(vector, self.vector_shard)

    def constrain_scalar(self, scalar):
        return with_sharding_constraint(scalar, self.scalar_shard)

    def exit(self, code=1):
        sys.exit(code)

    def print(self, *args, **kwargs):
        if self.main_device:
            print(*args, **kwargs, flush=True)


sharding = Sharding()
