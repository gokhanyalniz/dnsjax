from functools import partial

import jax
import jaxdecomp
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import (
    LX,
    LY,
    LZ,
    NX,
    NY,
    NZ,
    OVERSAMP_FAC,
)
from sharding import MESH

NX_HALF = NX // 2
NY_HALF = NY // 2
NZ_HALF = NZ // 2

NXX = OVERSAMP_FAC * NX_HALF
NYY = OVERSAMP_FAC * NY_HALF
NZZ = OVERSAMP_FAC * NZ_HALF

DX = LX / NXX
DY = LY / NYY
DZ = LZ / NZZ

QX = (
    jnp.fft.fftfreq(NXX, d=1 / NXX, dtype=jnp.float64)
    .astype(int)
    .reshape([1, -1, 1])
)
QY = (
    jnp.fft.fftfreq(NYY, d=1 / NYY, dtype=jnp.float64)
    .astype(int)
    .reshape([1, 1, -1])
)
QZ = (
    jnp.fft.fftfreq(NZZ, d=1 / NZZ, dtype=jnp.float64)
    .astype(int)
    .reshape([-1, 1, 1])
)

KX = QX * 2 * jnp.pi / LX
KY = QY * 2 * jnp.pi / LY
KZ = QZ * 2 * jnp.pi / LZ

# All aliased modes and the Nyquist modes are to be discarded
DEALIAS = jnp.where(
    (jnp.abs(QX) < NX_HALF)
    & (jnp.abs(QY) < NY_HALF)
    & (jnp.abs(QZ) < NZ_HALF),
    True,
    False,
)


@partial(jit, donate_argnums=0)
def phys_to_spec_scalar(scalar_phys):
    scalar_spec = jax.lax.with_sharding_constraint(
        jaxdecomp.fft.pfft3d(
            jax.lax.with_sharding_constraint(
                scalar_phys, NamedSharding(MESH, P("Z", "X", None))
            ),
            norm="forward",
        )
        * DEALIAS,
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
        * DEALIAS,
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
