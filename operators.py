from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax import vmap
from jax.sharding import NamedSharding, explicit_axes
from jax.sharding import PartitionSpec as P
from jaxdecomp.fft import pfft3d, pifft3d

from parameters import padded_res, params
from sharding import sharding


@dataclass
class Fourier:
    def harmonics(n):
        return (jnp.arange(n, dtype=int) + n // 2) % n - n // 2

    qx = jax.device_put(
        harmonics(padded_res.nx_padded).reshape([1, -1, 1]),
        NamedSharding(sharding.mesh, P(None, "x", None)),
    )
    qy = harmonics(padded_res.ny_padded).reshape([1, 1, -1])
    qz = jax.device_put(
        harmonics(padded_res.nz_padded).reshape([-1, 1, 1]),
        NamedSharding(sharding.mesh, P("z", None, None)),
    )

    kx = qx * 2 * jnp.pi / params.geo.lx
    ky = qy * 2 * jnp.pi / params.geo.ly
    kz = qz * 2 * jnp.pi / params.geo.lz

    # Aliased modes, the Nyquist modes, and the zero mode are to be discarded
    active_modes = jnp.where(
        (jnp.abs(qx) < padded_res.nx_half)
        & (jnp.abs(qy) < padded_res.ny_half)
        & (jnp.abs(qz) < padded_res.nz_half)
        & ~((qx == 0) & (qy == 0) & (qz == 0)),
        True,
        False,
    )

    @partial(explicit_axes, axes=sharding.axis_names)
    def get_kvec():
        kvec = jnp.zeros(
            (3, *sharding.spec_shape),
            dtype=sharding.float_type,
            out_sharding=sharding.vector_shard,
        )
        return kvec

    kvec = get_kvec(in_sharding=sharding.vector_shard)

    kvec = kvec.at[0].set(kx)
    kvec = kvec.at[1].set(ky)
    kvec = kvec.at[2].set(kz)

    # Zero the dealiased modes to (potentially) save computation
    kvec = active_modes * kvec
    lapl = -(kx**2 + ky**2 + kz**2) * active_modes
    inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)


fourier = Fourier()


@partial(vmap, in_axes=(0, None))
def phys_to_spec(velocity_phys, active_modes):
    # WARNING: With active_modes:
    # - aliased modes
    # - Nyquist modes
    # - *and* the mean [(0, 0, 0)] mode
    # get zeroed!
    velocity_spec = (
        pfft3d(
            jax.lax.with_sharding_constraint(
                velocity_phys, sharding.scalar_shard
            ),
            norm="forward",
        )
        * active_modes
    )

    return sharding.constrain_scalar(velocity_spec)


@vmap
def spec_to_phys(velocity_spec):
    velocity_phys = pifft3d(
        jax.lax.with_sharding_constraint(velocity_spec, sharding.scalar_shard),
        norm="forward",
    ).real

    return sharding.constrain_scalar(velocity_phys)
