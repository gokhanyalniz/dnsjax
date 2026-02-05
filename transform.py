import jax
import jaxdecomp
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


# TODO: Pin sharding for gradients
def phys_to_spec_scalar(scalar_phys):
    scalar_spec = jaxdecomp.fft.pfft3d(scalar_phys.astype(jnp.complex128))

    # Dealias + zero the Nyquist mode
    scalar_spec = jnp.where(
        (jnp.abs(QX) < NX_HALF) & (jnp.abs(QY) < NY_HALF) & (jnp.abs(QZ) < NZ_HALF),
        scalar_spec,
        0,
    )
    # TODO: Once real-to-complex FFT is implemented, drop the astype
    return scalar_spec


def spec_to_phys_scalar(scalar_spec):
    return jaxdecomp.fft.pifft3d(scalar_spec).real


def phys_to_spec_vector(vector_phys):
    return jnp.array([phys_to_spec_scalar(vector_phys[n]) for n in range(3)])


def spec_to_phys_vector(vector_spec):
    return jnp.array([spec_to_phys_scalar(vector_spec[n]) for n in range(3)])


KX = jax.device_put(
    (jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, -1, 1]),
    NamedSharding(MESH, P(None, "X", None)),
)
KY = (jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, 1, -1])
KZ = jax.device_put(
    (jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi).reshape([-1, 1, 1]),
    NamedSharding(MESH, P("Z", None, None)),
)


QX = jax.device_put(
    jnp.fft.fftfreq(NXX, d=1 / NXX, dtype=jnp.float64).astype(int).reshape([1, -1, 1]),
    NamedSharding(MESH, P(None, "X", None)),
)
QY = jnp.fft.fftfreq(NYY, d=1 / NYY, dtype=jnp.float64).astype(int).reshape([1, 1, -1])
QZ = jax.device_put(
    jnp.fft.fftfreq(NZZ, d=1 / NZZ, dtype=jnp.float64).astype(int).reshape([-1, 1, 1]),
    NamedSharding(MESH, P("Z", None, None)),
)

KVEC = jax.device_put(
    jnp.zeros((3, NZZ, NXX, NYY), dtype=jnp.float64),
    NamedSharding(MESH, P(None, "Z", "X", None)),
)

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
