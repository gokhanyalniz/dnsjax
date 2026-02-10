import jax
import jaxdecomp
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
# from jax_array_info import sharding_vis

from parameters import (
    AMP,
    FORCING,
    LX,
    LY,
    LZ,
    NX,
    NY,
    NZ,
    QF,
    SUBSAMP_FAC,
)
from sharding import MESH

NX_HALF = NX // 2
NY_HALF = NY // 2
NZ_HALF = NZ // 2

NXX = SUBSAMP_FAC * NX_HALF
NYY = SUBSAMP_FAC * NY_HALF
NZZ = SUBSAMP_FAC * NZ_HALF

DX = LX / NXX
DY = LY / NYY
DZ = LZ / NZZ

KX = (jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, -1, 1])
KY = (jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, 1, -1])
KZ = (jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi).reshape([-1, 1, 1])

QX = jnp.fft.fftfreq(NXX, d=1 / NXX, dtype=jnp.float64).astype(int).reshape([1, -1, 1])
QY = jnp.fft.fftfreq(NYY, d=1 / NYY, dtype=jnp.float64).astype(int).reshape([1, 1, -1])
QZ = jnp.fft.fftfreq(NZZ, d=1 / NZZ, dtype=jnp.float64).astype(int).reshape([-1, 1, 1])

DEALIAS = jnp.where(
    (jnp.abs(QX) < NX_HALF) & (jnp.abs(QY) < NY_HALF) & (jnp.abs(QZ) < NZ_HALF),
    1.0,
    0.0,
)

KVEC = jnp.zeros((3, NZZ, NXX, NYY), dtype=jnp.float64)  # TODO: Avoid creating this

for ix in range(NXX):
    KVEC = KVEC.at[0, :, ix, :].set(KX[0, ix, 0])
for iy in range(NYY):
    KVEC = KVEC.at[1, :, :, iy].set(KY[0, 0, iy])
for iz in range(NZZ):
    KVEC = KVEC.at[2, iz, :, :].set(KZ[iz, 0, 0])

LAPL = -(KX**2) - KY**2 - KZ**2
INV_LAPL = jnp.where(LAPL < 0, 1 / LAPL, 0)

if FORCING == 0:
    FORCE = 0
elif FORCING == 1:
    FORCE = jnp.where((QX == 0) & (QY == QF) & (QZ == 0), -1j * 0.5 * AMP, 0)
    FORCE = jnp.where((QX == 0) & (QY == -QF) & (QZ == 0), 1j * 0.5 * AMP, FORCE)
elif FORCING == 2:
    FORCE = jnp.where((QX == 0) & (QY == QF) & (QZ == 0), 0.5 * AMP, 0)
    FORCE = jnp.where((QX == 0) & (QY == -QF) & (QZ == 0), 0.5 * AMP, FORCE)


@jit
def phys_to_spec_scalar(scalar_phys):
    # TODO: Once real-to-complex FFT is implemented, drop the astype
    return jax.lax.with_sharding_constraint(
        jaxdecomp.fft.pfft3d(
            jax.lax.with_sharding_constraint(
                scalar_phys, NamedSharding(MESH, P("Z", "X", None))
            ).astype(jnp.complex128),
            norm="forward",
        )
        * DEALIAS,
        NamedSharding(MESH, P("Z", "X", None)),
    )


@jit
def spec_to_phys_scalar(scalar_spec):
    return jax.lax.with_sharding_constraint(
        jaxdecomp.fft.pifft3d(
            jax.lax.with_sharding_constraint(
                scalar_spec, NamedSharding(MESH, P("Z", "X", None))
            ),
            norm="forward",
        ).real,
        NamedSharding(MESH, P("Z", "X", None)),
    )


phys_to_spec_vector = jax.vmap(phys_to_spec_scalar)

spec_to_phys_vector = jax.vmap(spec_to_phys_scalar)
