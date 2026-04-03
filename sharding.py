import sys
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from parameters import padded_res, params, periodic_systems


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
        axis_types=(AxisType.Explicit,),
    )

    jax.set_mesh(mesh)

    vector_mean_mode = ((0, 1, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    scalar_mean_mode = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

    spec_vector_shard = P(None, None, None, *axis_names)
    spec_scalar_shard = P(None, None, *axis_names)

    phys_vector_shard = P(None, None, *axis_names, None)
    phys_scalar_shard = P(None, *axis_names, None)

    no_shard = P(None)

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    if params.phys.system in periodic_systems:
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
    else:
        spec_shape = (
            params.res.ny,
            params.res.nz - 1,
            params.res.nx // 2,
        )

        phys_shape = (
            params.res.ny,
            padded_res.nz_padded,
            padded_res.nx_padded,
        )

    def exit(self, code=1):
        sys.exit(code)

    def print(self, *args, **kwargs):
        if self.main_device:
            print(*args, **kwargs, flush=True)


sharding = Sharding()
