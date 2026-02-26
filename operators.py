from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, vmap
from jax import numpy as jnp
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
        harmonics(padded_res.Nx_padded).reshape([1, -1, 1]),
        NamedSharding(sharding.mesh, P(None, "X", None)),
    )
    qy = harmonics(padded_res.Ny_padded).reshape([1, 1, -1])
    qz = jax.device_put(
        harmonics(padded_res.Nz_padded).reshape([-1, 1, 1]),
        NamedSharding(sharding.mesh, P("Z", None, None)),
    )

    kx = qx * 2 * jnp.pi / params.geo.Lx
    ky = qy * 2 * jnp.pi / params.geo.Ly
    kz = qz * 2 * jnp.pi / params.geo.Lz

    # Aliased modes, the Nyquist modes, and the zero mode are to be discarded
    active_modes = jnp.where(
        (jnp.abs(qx) < padded_res.Nx_half)
        & (jnp.abs(qy) < padded_res.Ny_half)
        & (jnp.abs(qz) < padded_res.Nz_half),
        # & ~((qx == 0) & (qy == 0) & (qz == 0)),
        True,
        False,
    )

    zero_mean = jnp.where((qx == 0) & (qy == 0) & (qz == 0), False, True)

    @partial(explicit_axes, axes=("Z", "X"))
    def get_nabla():
        nabla = jnp.zeros(
            (
                3,
                padded_res.Nz_padded,
                padded_res.Nx_padded,
                padded_res.Ny_padded,
            ),
            dtype=sharding.complex_type,
            out_sharding=P(None, "Z", "X", None),
        )
        return nabla

    nabla = get_nabla(in_sharding=sharding.spec_shard)

    nabla = nabla.at[0].set(1j * kx)
    nabla = nabla.at[1].set(1j * ky)
    nabla = nabla.at[2].set(1j * kz)

    # Zero the dealiased modes to (potentially) save computation
    nabla = active_modes * nabla
    lapl = (-(kx**2) - ky**2 - kz**2) * active_modes
    inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)


fourier = Fourier()


@jit(donate_argnums=0)
@partial(vmap, in_axes=(0, None))
def phys_to_spec(velocity_phys, active_modes):
    velocity_spec = (
        pfft3d(
            jax.lax.with_sharding_constraint(
                velocity_phys, sharding.scalar_phys_shard
            ),
            norm="forward",
        )
        * active_modes
    )

    return jax.lax.with_sharding_constraint(
        velocity_spec, sharding.scalar_spec_shard
    )


@jit(donate_argnums=0)
@vmap
def spec_to_phys(velocity_spec):
    velocity_phys = pifft3d(
        jax.lax.with_sharding_constraint(
            velocity_spec, sharding.scalar_spec_shard
        ),
        norm="forward",
    ).real

    return jax.lax.with_sharding_constraint(
        velocity_phys, sharding.scalar_phys_shard
    )
