# The FFT structure keeps the whole wavenumbers rather than just one half!
import jax
import jaxdecomp
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import (
    AMP,
    DX,
    DY,
    DZ,
    FORCING,
    GLOBAL_SHAPE,
    KF,
    LOCAL_SHAPE,
    NXX,
    NYY,
    NZZ,
    PDIMS,
    RE,
)

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
rank = jax.process_index()

mesh = jax.make_mesh(PDIMS, axis_names=("X", "Z"))
sharding = NamedSharding(mesh, P("X", "Z"))


def sx2k(vfieldx):
    # TODO: Dealias!
    return jaxdecomp.fft.pfft3d(vfieldx)


def sk2x(vfieldk):
    return jaxdecomp.fft.pifft3d(vfieldk).real

def vx2k(vfieldx):
    return jnp.array([sx2k(vfieldx[n]) for n in range(3)])

def vk2x(vfieldk):
    return jnp.array([sk2x(vfieldk[n]) for n in range(3)])

_XARRAY = jax.make_array_from_callback(
    GLOBAL_SHAPE,
    sharding,
    data_callback=lambda _: jax.random.normal(jax.random.PRNGKey(rank), LOCAL_SHAPE),
)

print("XARRAY", _XARRAY.shape)

_KARRAY = sx2k(_XARRAY)
print("KARRAY", _KARRAY.shape)
_K_ARRAY_STRUCTURE = jax.tree.structure(_KARRAY)

KX = jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi
KY = jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi
KZ = jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi

KX = jax.tree.unflatten(_K_ARRAY_STRUCTURE, (KX,))
KY = jax.tree.unflatten(_K_ARRAY_STRUCTURE, (KY,))
KZ = jax.tree.unflatten(_K_ARRAY_STRUCTURE, (KZ,))

KX = KX.reshape([1, 1, -1])
KY = KY.reshape([1, -1, 1])
KZ = KZ.reshape([-1, 1, 1])

print("KARRAYS", KX.shape, KY.shape, KZ.shape)

# KUNIT = jnp.ones((_KARRAY.shape))
# KVEC = jnp.array([KUNIT*KX, KUNIT*KY, KUNIT*KZ])
# print("KVEC", KVEC.shape)
KVEC = [KX, KY, KZ]

# TODO: do these in a way einsum works

LAPL = -(KX**2) - KY**2 - KZ**2
INV_LAPL = jnp.where(LAPL > 0, 1 / LAPL, 0)

print("LAPL", LAPL.shape)

if FORCING == 0:
    FORCE = 0
elif FORCING == 1:
    FORCE = jnp.where((KX == 0) & (KY == KF), -1j * 0.5 * AMP / (4 * RE), 0)
elif FORCING == 2:
    FORCE = jnp.where((KX == 0) & (KY == KF), 0.5 * AMP / (4 * RE), 0)

print("FORCE", FORCE.shape)
