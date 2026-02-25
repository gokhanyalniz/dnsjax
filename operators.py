from dataclasses import dataclass
from functools import partial

import jax
import jaxdecomp
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import params
from sharding import MESH, complex_type

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
        NamedSharding(MESH, P(None, "X", None)),
    )
    QY = harmonics(NY_PADDED).reshape([1, 1, -1])
    QZ = jax.device_put(
        harmonics(NZ_PADDED).reshape([-1, 1, 1]),
        NamedSharding(MESH, P("Z", None, None)),
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
        jnp.zeros((3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=complex_type),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    # for ix in range(NX_PADDED):
    #     NABLA = NABLA.at[0, :, ix, :].set(1j * KX[0, ix, 0])
    # for iy in range(NY_PADDED):
    #     NABLA = NABLA.at[1, :, :, iy].set(1j * KY[0, 0, iy])
    # for iz in range(NZ_PADDED):
    #     NABLA = NABLA.at[2, iz, :, :].set(1j * KZ[iz, 0, 0])
    NABLA = NABLA.at[0].set(1j * KX)
    NABLA = NABLA.at[1].set(1j * KY)
    NABLA = NABLA.at[2].set(1j * KZ)

    # Zero the dealiased modes to (potentially) save computation
    NABLA = DEALIAS * NABLA
    LAPL = (-(KX**2) - KY**2 - KZ**2) * DEALIAS
    INV_LAPL = jnp.where(LAPL < 0, 1 / LAPL, 0)

    ZERO_MEAN = jnp.where((QX == 0) & (QY == 0) & (QZ == 0), False, True)


fourier = Fourier()


@partial(jit, donate_argnums=0)
def phys_to_spec_scalar(scalar_phys):
    scalar_spec = jax.lax.with_sharding_constraint(
        jaxdecomp.fft.pfft3d(
            jax.lax.with_sharding_constraint(
                scalar_phys, NamedSharding(MESH, P("Z", "X", None))
            ),
            norm="forward",
        )
        * fourier.DEALIAS,
        NamedSharding(MESH, P("Z", "X", None)),
    )
    return scalar_spec


@partial(jit, donate_argnums=0)
def spec_to_phys_scalar(scalar_spec):
    scalar_phys = jaxdecomp.fft.pifft3d(
        jax.lax.with_sharding_constraint(
            scalar_spec, NamedSharding(MESH, P("Z", "X", None))
        ),
        norm="forward",
    )
    scalar_phys = scalar_phys.real.at[...].get() + 1j * scalar_phys.imag.at[
        ...
    ].set(0)
    return jax.lax.with_sharding_constraint(
        scalar_phys, NamedSharding(MESH, P("Z", "X", None))
    )


@partial(jit, donate_argnums=0)
@vmap
def phys_to_spec_vector(velocity_phys):
    velocity_spec = jax.lax.with_sharding_constraint(
        jaxdecomp.fft.pfft3d(
            jax.lax.with_sharding_constraint(
                velocity_phys, NamedSharding(MESH, P("Z", "X", None))
            ),
            norm="forward",
        )
        * fourier.DEALIAS,
        NamedSharding(MESH, P("Z", "X", None)),
    )
    return velocity_spec


@partial(jit, donate_argnums=0)
@vmap
def spec_to_phys_vector(velocity_spec):
    velocity_phys = jaxdecomp.fft.pifft3d(
        jax.lax.with_sharding_constraint(
            velocity_spec, NamedSharding(MESH, P("Z", "X", None))
        ),
        norm="forward",
    )
    velocity_phys = velocity_phys.real.at[
        ...
    ].get() + 1j * velocity_phys.imag.at[...].set(0)

    return jax.lax.with_sharding_constraint(
        velocity_phys, NamedSharding(MESH, P("Z", "X", None))
    )
