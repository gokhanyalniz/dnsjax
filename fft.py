from parameters import *

import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P, NamedSharding
import jaxdecomp

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
rank = jax.process_index()

mesh = jax.make_mesh(PDIMS, axis_names=('X', 'Z'))
sharding = NamedSharding(mesh, P('X', 'Z'))

def sx2k(vfieldx):
    # TODO: Dealias!
    return jaxdecomp.fft.pfft3d(vfieldx)

def sk2x(vfieldk):
    return jaxdecomp.fft.pifft3d(vfieldk).real

_XARRAY = jax.make_array_from_callback(
    GLOBAL_SHAPE,
    sharding,
    data_callback=lambda _: jax.random.normal(
        jax.random.PRNGKey(rank), LOCAL_SHAPE)
)

_KARRAY = sx2k(_XARRAY)
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

# TODO: do these in a way einsum works
KVEC = [KX, KY, KZ]
NABLA = [1j*k for k in KVEC]

LAPL = -KX**2 -KY**2 -KZ**2
INV_LAPL = np.where(LAPL > 0, 1/LAPL, 0)