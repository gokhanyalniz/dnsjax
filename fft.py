# The FFT structure keeps the whole wavenumbers rather than just one half!
import jax

# ruff: disable[E402]
# This needs to be done first
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
# ruff: enable[E402]
import jaxdecomp
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# from jax_array_info import sharding_vis
from parameters import (
    AMP,
    DX,
    DY,
    DZ,
    FORCING,
    KF,
    NXX,
    NYY,
    NZZ,
    RE,
)
from sharding import MESH


# TODO: Pin sharding for gradients
def sx2k(sfieldx):
    # TODO: Dealias!
    # Once real-to-complex FFT is implemented, drop the astype
    return jaxdecomp.fft.pfft3d(sfieldx.astype(jnp.complex128))


def sk2x(sfieldk):
    return jaxdecomp.fft.pifft3d(sfieldk).real


def vx2k(vfieldx):
    return jnp.array([sx2k(vfieldx[n]) for n in range(3)])


def vk2x(vfieldk):
    return jnp.array([sk2x(vfieldk[n]) for n in range(3)])


KX = jax.device_put(
    (jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, -1, 1]),
    NamedSharding(MESH, P(None, "X", None)),
)
KY = (jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, 1, -1])
KZ = jax.device_put(
    (jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi).reshape([-1, 1, 1]),
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
INV_LAPL = jnp.where(LAPL > 0, 1 / LAPL, 0)

if FORCING == 0:
    FORCE = 0
elif FORCING == 1:
    FORCE = jnp.where((KX == 0) & (KY == KF), -1j * 0.5 * AMP / (4 * RE), 0)
elif FORCING == 2:
    FORCE = jnp.where((KX == 0) & (KY == KF), 0.5 * AMP / (4 * RE), 0)
