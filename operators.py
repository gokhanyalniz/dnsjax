from dataclasses import dataclass
from functools import partial

import jax
import jaxdecomp
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import params
from sharding import sharding

NX_HALF = params.res.Nx // 2
NY_HALF = params.res.Ny // 2
NZ_HALF = params.res.Nz // 2

NX_PADDED = params.phys.oversampling_factor * NX_HALF
NY_PADDED = params.phys.oversampling_factor * NY_HALF
NZ_PADDED = params.phys.oversampling_factor * NZ_HALF


@dataclass
class Fourier:
    def harmonics(n):
        i = jnp.arange(n, dtype=int)
        k = (i + n // 2) % n - n // 2
        return k

    QX = jax.device_put(
        harmonics(NX_PADDED).reshape([1, -1, 1]),
        NamedSharding(sharding.MESH, P(None, "X", None)),
    )
    QY = harmonics(NY_PADDED).reshape([1, 1, -1])
    QZ = jax.device_put(
        harmonics(NZ_PADDED).reshape([-1, 1, 1]),
        NamedSharding(sharding.MESH, P("Z", None, None)),
    )

    KX = QX * 2 * jnp.pi / params.geo.Lx
    KY = QY * 2 * jnp.pi / params.geo.Ly
    KZ = QZ * 2 * jnp.pi / params.geo.Lz

    # All aliased modes and the Nyquist modes are to be discarded
    DEALIAS = jnp.where(
        (jnp.abs(QX) < NX_HALF)
        & (jnp.abs(QY) < NY_HALF)
        & (jnp.abs(QZ) < NZ_HALF),
        True,
        False,
    )

    NABLA = jax.device_put(
        jnp.zeros(
            (3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=sharding.complex_type
        ),
        sharding.spec_shard,
    )

    NABLA = NABLA.at[0].set(1j * KX)
    NABLA = NABLA.at[1].set(1j * KY)
    NABLA = NABLA.at[2].set(1j * KZ)

    # Zero the dealiased modes to (potentially) save computation
    NABLA = DEALIAS * NABLA
    LAPL = (-(KX**2) - KY**2 - KZ**2) * DEALIAS
    INV_LAPL = jnp.where(LAPL < 0, 1 / LAPL, 0)

    ZERO_MEAN = jnp.where((QX == 0) & (QY == 0) & (QZ == 0), False, True)


fourier = Fourier()


@jit(donate_argnums=0)
@partial(vmap, in_axes=(0, None))
def phys_to_spec(velocity_phys, dealias):
    velocity_spec = (
        jaxdecomp.fft.pfft3d(
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
def spec_to_phys(velocity_spec):
    velocity_phys = jaxdecomp.fft.pifft3d(
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
