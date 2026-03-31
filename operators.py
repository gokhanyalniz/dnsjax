from dataclasses import dataclass

import jax
from jax import jit, vmap
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from fft import _irfft3d, _rfft3d
from parameters import params
from sharding import get_zeros, sharding


@dataclass
class Fourier:
    def real_harmonics(n):
        # Omit the Nyquist mode
        return jnp.arange(0, n // 2, dtype=int)

    def complex_harmonics(n):
        qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
        # Omit the Nyquist mode
        qs_out = jnp.zeros(n - 1, dtype=int)
        qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
        qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
        return qs_out

    qx = jax.device_put(
        real_harmonics(params.res.nx).reshape([1, 1, -1]),
        NamedSharding(sharding.mesh, P(None, None, "gpus")),
    )
    qy = complex_harmonics(params.res.ny).reshape([-1, 1, 1])
    qz = complex_harmonics(params.res.nz).reshape([1, -1, 1])

    kx = qx * 2 * jnp.pi / params.geo.lx
    ky = qy * 2 * jnp.pi / params.geo.ly
    kz = qz * 2 * jnp.pi / params.geo.lz

    metric = jnp.where(qx == 0, 1, 2)

    kvec = get_zeros(
        shape=(3, *sharding.spec_shape),
        dtype=sharding.float_type,
        in_sharding=sharding.spec_vector_shard,
        out_sharding=sharding.spec_vector_shard,
    )

    kvec = kvec.at[0].set(kx)
    kvec = kvec.at[1].set(ky)
    kvec = kvec.at[2].set(kz)

    lapl = -(kx**2 + ky**2 + kz**2)
    inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)


fourier = Fourier()


@jit(
    out_shardings=sharding.spec_vector_shard,
)
@vmap
def phys_to_spec(velocity_phys):
    velocity_spec = _rfft3d(
        velocity_phys,
    )

    return velocity_spec


@jit(
    out_shardings=sharding.phys_vector_shard,
)
@vmap
def spec_to_phys(velocity_spec):

    velocity_phys = _irfft3d(
        velocity_spec,
    )

    return velocity_phys
