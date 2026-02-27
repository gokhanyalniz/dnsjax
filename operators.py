from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding, auto_axes
from jax.sharding import PartitionSpec as P
from jaxdecomp.fft import pfft3d, pifft3d

from parameters import padded_res, params
from sharding import sharding


@dataclass
class Fourier:
    @partial(vmap, in_axes=(0, None))
    def harmonics(k, n):
        return (k + n // 2) % n - n // 2

    qx = harmonics(
        jnp.arange(
            padded_res.Nx_padded,
            dtype=int,
            out_sharding=sharding.X_shard,
        ),
        padded_res.Nx_padded,
    )[jnp.newaxis, :, jnp.newaxis]
    qy = harmonics(
        jnp.arange(padded_res.Ny_padded, dtype=int), padded_res.Ny_padded
    )[jnp.newaxis, jnp.newaxis, :]
    qz = harmonics(
        jnp.arange(
            padded_res.Nz_padded,
            dtype=int,
            out_sharding=sharding.Z_shard,
        ),
        padded_res.Nz_padded,
    )[:, jnp.newaxis, jnp.newaxis]

    # qx = jax.device_put(
    #     harmonics(padded_res.Nx_padded).reshape([1, -1, 1]),
    #     NamedSharding(sharding.mesh, P(None, "X", None)),
    # )
    # qy = harmonics(padded_res.Ny_padded).reshape([1, 1, -1])
    # qz = jax.device_put(
    #     harmonics(padded_res.Nz_padded).reshape([-1, 1, 1]),
    #     NamedSharding(sharding.mesh, P("Z", None, None)),
    # )

    kx = qx * 2 * jnp.pi / params.geo.Lx
    ky = qy * 2 * jnp.pi / params.geo.Ly
    kz = qz * 2 * jnp.pi / params.geo.Lz

    # All aliased modes and the Nyquist modes are to be discarded
    dealias = jnp.where(
        (jnp.abs(qx) < padded_res.Nx_half)
        & (jnp.abs(qy) < padded_res.Ny_half)
        & (jnp.abs(qz) < padded_res.Nz_half),
        True,
        False,
    )

    # nabla = jax.device_put(
    #     jnp.zeros(
    #         (
    #             3,
    #             padded_res.Nz_padded,
    #             padded_res.Nx_padded,
    #             padded_res.Ny_padded,
    #         ),
    #         dtype=sharding.complex_type,
    #     ),
    #     sharding.spec_shard,
    # )
    nabla = jnp.zeros(
        (3, padded_res.Nz_padded, padded_res.Nx_padded, padded_res.Ny_padded),
        dtype=sharding.complex_type,
        out_sharding=sharding.spec_shard,
    )

    nabla = nabla.at[0].set(1j * kx)
    nabla = nabla.at[1].set(1j * ky)
    nabla = nabla.at[2].set(1j * kz)

    # Zero the dealiased modes to (potentially) save computation
    nabla = dealias * nabla
    lapl = (-(kx**2) - ky**2 - kz**2) * dealias
    inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)

    zero_mean = jnp.where((qx == 0) & (qy == 0) & (qz == 0), False, True)


fourier = Fourier()


@jit(donate_argnums=0)
@partial(vmap, in_axes=(0, None))
@auto_axes
def phys_to_spec(velocity_phys, dealias):
    velocity_spec = (
        pfft3d(
            jax.lax.with_sharding_constraint(
                velocity_phys, sharding.scalar_phys_shard
            ),
            norm="forward",
        )
        * dealias
    )

    return jax.lax.with_sharding_constraint(
        velocity_spec, sharding.scalar_spec_shard
    )


@jit(donate_argnums=0)
@vmap
@auto_axes
def spec_to_phys(velocity_spec):
    velocity_phys = pifft3d(
        jax.lax.with_sharding_constraint(
            velocity_spec, sharding.scalar_spec_shard
        ),
        norm="forward",
    )
    velocity_phys = velocity_phys.real.at[
        ...
    ].get() + 1j * velocity_phys.imag.at[...].set(0)

    return jax.lax.with_sharding_constraint(
        velocity_phys, sharding.scalar_phys_shard
    )
