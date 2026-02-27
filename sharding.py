from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import params


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

    mesh = jax.make_mesh(
        (params.dist.Np0, params.dist.Np1),
        axis_names=("Z", "X"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    # Create a mesh with size 1 on the first axis
    mesh_X = jax.make_mesh(
        (params.dist.Np1,),
        axis_names=("X",),
        axis_types=(AxisType.Explicit,),
        devices=devices[:: params.dist.Np0],
    )
    mesh_Z = jax.make_mesh(
        (params.dist.Np0,),
        axis_names=("Z",),
        axis_types=(AxisType.Explicit,),
        devices=devices[:: params.dist.Np1],
    )

    # OK now - UNUSED axis has size 1
    X_shard = NamedSharding(mesh_X, P("X"))
    Z_shard = NamedSharding(mesh_Z, P("Z"))

    phys_shard = NamedSharding(mesh, P(None, "Z", "X", None))
    spec_shard = NamedSharding(mesh, P(None, "Z", "X", None))
    scalar_phys_shard = NamedSharding(mesh, P("Z", "X", None))
    scalar_spec_shard = NamedSharding(mesh, P("Z", "X", None))

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64


sharding = Sharding()
