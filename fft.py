from jax import numpy as jnp
from jax import shard_map

from parameters import params
from sharding import get_zeros, sharding

norm = "forward"


def zeropad_fft(a: jnp.ndarray, n: int, axis: int) -> jnp.ndarray:
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    N = a.shape[axis] + 1  # Add the omitted Nyquist mode
    if n < N:
        raise ValueError(f"Target size {n} is smaller than input size {N}.")
    if (n - N) % 2 != 0:
        raise ValueError(f"Difference (n - N) = {n - N} cannot be odd.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = get_zeros(
        shape=out_shape,
        dtype=a.dtype,
        in_sharding=sharding.spec_scalar_shard,
        out_sharding=sharding.spec_scalar_shard,
    )

    idx_in = [slice(None)] * 3
    idx_out = [slice(None)] * 3

    # positive modes
    idx_in[axis] = slice(None, N // 2)
    idx_out[axis] = slice(None, N // 2)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    # negative modes (skip the Nyquist modes)
    idx_in[axis] = slice(N // 2, None)
    idx_out[axis] = slice(n - N // 2 + 1, None)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    return out


def truncate_fft(a: jnp.ndarray, n: int, axis: int) -> jnp.ndarray:
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    N = a.shape[axis]
    if n > N:
        raise ValueError(f"Target size {n} is larger than input size {N}.")
    if (N - n) % 2 != 0:
        raise ValueError(f"Difference (N - n) = {N - n} cannot be odd.")

    out_shape = list(a.shape)
    out_shape[axis] = n - 1  # Omit the Nyquist mode
    out = get_zeros(
        shape=out_shape,
        dtype=a.dtype,
        in_sharding=sharding.spec_scalar_shard,
        out_sharding=sharding.spec_scalar_shard,
    )

    idx_in = [slice(None)] * 3
    idx_out = [slice(None)] * 3

    # positive modes
    idx_in[axis] = slice(None, n // 2)
    idx_out[axis] = slice(None, n // 2)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    # negative modes (skip the Nyquist modes)
    idx_in[axis] = slice(N - n // 2 + 1, None)
    idx_out[axis] = slice(n // 2, None)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    return out


def zeropad_rfft(a: jnp.ndarray, n: int) -> jnp.ndarray:
    axis = 2
    N = a.shape[axis]
    if n < N:
        raise ValueError(f"Target mode count {n} is smaller than input {N}.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = get_zeros(
        shape=out_shape,
        dtype=a.dtype,
        in_sharding=sharding.phys_scalar_shard,
        out_sharding=sharding.phys_scalar_shard,
    )

    idx = [slice(None)] * 3
    idx[axis] = slice(None, N)
    out = out.at[tuple(idx)].set(a)

    return out


def truncate_rfft(a: jnp.ndarray, n: int) -> jnp.ndarray:
    axis = 2
    N = a.shape[axis]
    if n > N:
        raise ValueError(f"Target mode count {n} is larger than input {N}.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = get_zeros(
        shape=out_shape,
        dtype=a.dtype,
        in_sharding=sharding.phys_scalar_shard,
        out_sharding=sharding.phys_scalar_shard,
    )

    idx_in = [slice(None)] * 3
    idx_out = [slice(None)] * 3
    idx_in[axis] = slice(None, n)
    idx_out[axis] = slice(None, n)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    return out


def _rfft3d(x):
    # Transform in x (z is sharded)
    # Truncates the Nyquist mode as well
    y = truncate_rfft(
        shard_map(
            lambda a: jnp.fft.rfft(a, axis=2, norm="forward"),
            mesh=sharding.mesh,
            in_specs=sharding.phys_scalar_shard,
            out_specs=sharding.phys_scalar_shard,
        )(x),
        params.res.nx // 2,
    )

    # Reshard
    y = sharding.constrain_spec_scalar(y)

    # Transform in z (x is sharded)
    y = truncate_fft(
        shard_map(
            lambda a: jnp.fft.fft(a, axis=1, norm="forward"),
            mesh=sharding.mesh,
            in_specs=sharding.spec_scalar_shard,
            out_specs=sharding.spec_scalar_shard,
        )(y),
        params.res.nz,
        1,
    )

    # Transform in y (x is sharded)
    y = truncate_fft(
        shard_map(
            lambda a: jnp.fft.fft(a, axis=0, norm="forward"),
            mesh=sharding.mesh,
            in_specs=sharding.spec_scalar_shard,
            out_specs=sharding.spec_scalar_shard,
        )(y),
        params.res.ny,
        0,
    )

    return y


def _irfft3d(x):

    # Transform in y (x is sharded)
    y = zeropad_fft(x, params.res.ny, 0)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=0, norm="forward"),
        mesh=sharding.mesh,
        in_specs=sharding.spec_scalar_shard,
        out_specs=sharding.spec_scalar_shard,
    )(y)

    # Transform in z (x is sharded)
    y = zeropad_fft(y, params.res.nz, 1)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=1, norm="forward"),
        mesh=sharding.mesh,
        in_specs=sharding.spec_scalar_shard,
        out_specs=sharding.spec_scalar_shard,
    )(y)

    # Reshard
    y = sharding.constrain_phys_scalar(y)

    # Transform in x (z is sharded)
    y = zeropad_rfft(y, params.res.nx // 2 + 1)
    y = shard_map(
        lambda a: jnp.fft.irfft(
            a,
            axis=2,
            norm=norm,
        ),
        mesh=sharding.mesh,
        in_specs=sharding.phys_scalar_shard,
        out_specs=sharding.phys_scalar_shard,
    )(y)

    return y
