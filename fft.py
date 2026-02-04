# The FFT structure keeps the whole wavenumbers rather than just one half!
import jax
import jaxdecomp
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax import lax

from sharding import SHARDING, RANK, MESH
from parameters import (
    AMP,
    DX,
    DY,
    DZ,
    FORCING,
    K_GLOBAL_SHAPE,
    KF,
    K_LOCAL_SHAPE,
    NXX,
    NYY,
    NZZ,
    RE,
)

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

# TODO: Replace hacky way to create an X array
_SXARRAY = jax.make_array_from_callback(
    K_GLOBAL_SHAPE,
    SHARDING,
    data_callback=lambda _: jax.random.normal(jax.random.PRNGKey(RANK), K_LOCAL_SHAPE),
)

# jax.debug.visualize_array_sharding(_SXARRAY[:,:,0], use_color=False)
# exit()
# print("XARRAY", _SXARRAY.shape)

# TODO: Replace hacky way to create a K array
_SKARRAY = sx2k(_SXARRAY)

# jax.debug.visualize_array_sharding(_SKARRAY[:,:,0], use_color=False)
# exit()
# print("KARRAY", _SKARRAY.shape)

_SK_ARRAY_STRUCTURE = jax.tree.structure(_SKARRAY)

KX = jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi
KY = jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi
KZ = jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi

KX = jax.tree.unflatten(_SK_ARRAY_STRUCTURE, (KX,))
KY = jax.tree.unflatten(_SK_ARRAY_STRUCTURE, (KY,))
KZ = jax.tree.unflatten(_SK_ARRAY_STRUCTURE, (KZ,))

KX = KX.reshape([1, -1, 1])
KY = KY.reshape([1, 1, -1])
KZ = KZ.reshape([-1, 1, 1])

# jax.debug.visualize_array_sharding(KX[:,:,0], use_color=False)
# exit()

x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(MESH, P('X', 'Z')))
jax.debug.visualize_array_sharding(y)

# Didn't work
# KX = lax.with_sharding_constraint((jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, -1, 1]),NamedSharding(MESH, P(None,'X',None)))
# KY = jnp.array((jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, 1, -1]))
# KZ = lax.with_sharding_constraint((jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi).reshape([-1, 1, 1]), NamedSharding(MESH, P('Z',None,None)))

# Didn't work
# KX = lax.with_sharding_constraint(jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi, NamedSharding(MESH, P('X'))).reshape([1, -1, 1])
# KY = jnp.array((jnp.fft.fftfreq(NYY, d=DY, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, 1, -1]))
# KZ = lax.with_sharding_constraint(jnp.fft.fftfreq(NZZ, d=DZ, dtype=jnp.float64) * 2 * jnp.pi, NamedSharding(MESH, P('Z'))).reshape([1, -1, 1])

# KX = jax.device_put((jnp.fft.fftfreq(NXX, d=DX, dtype=jnp.float64) * 2 * jnp.pi).reshape([1, -1, 1]),NamedSharding(MESH, P(None,'X',None)))

# jax.debug.visualize_array_sharding(KX[:,:,0], use_color=False)
# exit()

# jax.debug.visualize_array_sharding(KX[:,:,0], use_color=False)
# exit()
# print("KARRAYS", KX.shape, KY.shape, KZ.shape)

# KUNIT = jnp.ones((_KARRAY.shape))
# KVEC = jnp.array([KUNIT*KX, KUNIT*KY, KUNIT*KZ])
# print("KVEC", KVEC.shape)
KVEC = [KX, KY, KZ]

# TODO: do these in a way einsum works

LAPL = -KX**2 - KY**2 - KZ**2
INV_LAPL = jnp.where(LAPL > 0, 1 / LAPL, 0)

# jax.debug.visualize_array_sharding(INV_LAPL[:,:,0], use_color=False)

print("LAPL", LAPL.shape)

if FORCING == 0:
    FORCE = 0
elif FORCING == 1:
    FORCE = jnp.where((KX == 0) & (KY == KF), -1j * 0.5 * AMP / (4 * RE), 0)
elif FORCING == 2:
    FORCE = jnp.where((KX == 0) & (KY == KF), 0.5 * AMP / (4 * RE), 0)

print("FORCE", FORCE.shape)
