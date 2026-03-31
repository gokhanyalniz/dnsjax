from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding, explicit_axes
from jax.sharding import PartitionSpec as P

from fft import _irfft3d, _rfft3d
from parameters import padded_res, params
from sharding import sharding


@dataclass
class Fourier:
    def real_harmonics(n):
        # Don't include the Nyquist mode
        return jnp.arange(0, n // 2, dtype=int)

    def complex_harmonics(n):
        return (jnp.arange(n, dtype=int) + n // 2) % n - n // 2

    qx = jax.device_put(
        real_harmonics(params.res.nx).reshape([1, 1, -1]),
        NamedSharding(sharding.mesh, P(None, None, "gpus")),
    )
    qy = complex_harmonics(params.res.ny).reshape([-1, 1, 1])
    qz = complex_harmonics(params.res.nz).reshape([1, -1, 1])

    kx = qx * 2 * jnp.pi / params.geo.lx
    ky = qy * 2 * jnp.pi / params.geo.ly
    kz = qz * 2 * jnp.pi / params.geo.lz

    # The Nyquist modes and the zero mode are to be discarded
    # TODO: Do this without keeping a velocity-sized array in memory
    active_modes = jnp.where(
        (jnp.abs(qx) < padded_res.nx_half)
        & (jnp.abs(qy) < padded_res.ny_half)
        & (jnp.abs(qz) < padded_res.nz_half)
        & ~((qx == 0) & (qy == 0) & (qz == 0)),
        True,
        True,
    )

    metric = jnp.where(qx == 0, 1, 2)

    @partial(explicit_axes, axes=sharding.axis_names)
    def get_kvec():
        kvec = jnp.zeros(
            (3, *sharding.spec_shape),
            dtype=sharding.float_type,
            out_sharding=sharding.spec_vector_shard,
        )
        return kvec

    kvec = get_kvec(in_sharding=sharding.spec_vector_shard)

    kvec = kvec.at[0].set(kx)
    kvec = kvec.at[1].set(ky)
    kvec = kvec.at[2].set(kz)

    # Zero the inactive modes to (potentially) save computation
    kvec = active_modes * kvec
    lapl = -(kx**2 + ky**2 + kz**2) * active_modes
    inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)


fourier = Fourier()


@jit(
    out_shardings=sharding.spec_vector_shard,
)
@partial(vmap, in_axes=(0, None))
def phys_to_spec(velocity_phys, active_modes):
    # WARNING: With active_modes:
    # - Nyquist modes
    # - *and* the mean [(0, 0, 0)] mode
    # get zeroed!

    velocity_spec = (
        _rfft3d(
            velocity_phys,
        )
        * active_modes
    )

    return velocity_spec


@jit(
    out_shardings=sharding.phys_vector_shard,
)
@vmap
def spec_to_phys(velocity_spec):

    velocity_phys = _irfft3d(
        velocity_spec,
    )

    return velocity_phys
