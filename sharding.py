from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from parameters import padded_res, params


@dataclass
class Sharding:
    n_devices = params.dist.Np0 * params.dist.Np1

    rank = jax.process_index()
    main_device = False
    if rank == 0:
        main_device = True

    devices = jax.devices()
    n_devices_reported = len(devices)
    if n_devices_reported != n_devices:
        jax.distributed.shutdown()
        if main_device:
            print(
                f"# of devices visible ({n_devices_reported}) "
                f"is not equal to Np0 x Np1 = {n_devices}.",
                flush=True,
            )
        exit()
    elif main_device:
        print(
            f"Working with {n_devices} {params.dist.platform} devices:",
            *devices,
        )

    axis_names = ("z", "x")
    mesh = jax.make_mesh(
        (params.dist.Np0, params.dist.Np1),
        axis_names=axis_names,
        axis_types=(AxisType.Auto, AxisType.Auto),
    )

    jax.set_mesh(mesh)

    vector_shard = P(None, *axis_names, None)
    scalar_shard = P(*axis_names, None)

    phys_shard = vector_shard
    spec_shard = vector_shard
    scalar_phys_shard = scalar_shard
    scalar_spec_shard = scalar_shard

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    phys_type = float_type

    spec_shape = (
        padded_res.Nz_padded,
        padded_res.Nx_padded,
        padded_res.Ny_padded,
    )
    phys_shape = (
        padded_res.Ny_padded,
        padded_res.Nz_padded,
        padded_res.Nx_padded,
    )


sharding = Sharding()
