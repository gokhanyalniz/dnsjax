"""3D real FFT with 3/2-rule dealiasing via zero-padding and truncation.

The forward transform (physical -> spectral) is ``_rfft3d``; the inverse
is ``_irfft3d``.  Both operate on scalar fields of shape
``(ny, nz, nx)`` and use ``shard_map`` for per-device FFTs with an
explicit reshard between the two sharding layouts (physical: z-sharded;
spectral: kx-sharded).

Dealiasing
----------
The 3/2-rule expands each direction by a factor of ``oversampling_factor
/ 2`` before transforming to physical space (``zeropad_*``), and
truncates back after the forward transform (``truncate_*``).  Nyquist
modes are omitted in all stored spectral arrays (``n - 1`` modes for a
full-complex axis, ``n // 2`` modes for the real-FFT axis).

Normalisation
-------------
All transforms use ``norm="forward"``, which divides by *N* on the
forward transform and applies no factor on the inverse.
"""

from jax import Array, shard_map
from jax import numpy as jnp
from jax.sharding import reshard

from .parameters import padded_res, params
from .sharding import sharding

norm: str = "forward"


def zeropad_fft(a: Array, n: int, axis: int) -> Array:
    """Zero-pad a full-complex spectral array along *axis* to length *n*.

    Inserts zeros between the positive and negative Fourier modes,
    reinstating the (previously omitted) Nyquist mode as zero.  This is
    the spectral-space equivalent of interpolation to a finer grid.

    Parameters
    ----------
    a:
        Input array with ``a.shape[axis] == N - 1`` stored modes (Nyquist
        omitted), where *N* is the original full mode count.
    n:
        Target length (>= *N*).  Must satisfy ``(n - N) % 2 == 0``.
    axis:
        Axis along which to pad (0 for y, 1 for z).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    N = a.shape[axis] + 1  # Add the omitted Nyquist mode
    if n < N:
        raise ValueError(f"Target size {n} is smaller than input size {N}.")
    if (n - N) % 2 != 0:
        raise ValueError(f"Difference (n - N) = {n - N} cannot be odd.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = jnp.zeros(
        shape=out_shape,
        dtype=a.dtype,
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


def truncate_fft(a: Array, n: int, axis: int) -> Array:
    """Truncate a full-complex FFT output along *axis*, dropping aliased modes.

    Keeps the lowest ``n // 2`` positive and ``n // 2 - 1`` negative
    modes, discarding all higher modes including the Nyquist mode.  The
    output has ``n - 1`` stored modes.

    Parameters
    ----------
    a:
        Full FFT output with ``a.shape[axis] == N`` modes.
    n:
        Target mode count (<= *N*).  Must satisfy ``(N - n) % 2 == 0``.
    axis:
        Axis along which to truncate (0 for y, 1 for z).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    N = a.shape[axis]
    if n > N:
        raise ValueError(f"Target size {n} is larger than input size {N}.")
    if (N - n) % 2 != 0:
        raise ValueError(f"Difference (N - n) = {N - n} cannot be odd.")

    out_shape = list(a.shape)
    out_shape[axis] = n - 1  # Omit the Nyquist mode
    out = jnp.zeros(
        shape=out_shape,
        dtype=a.dtype,
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


def zeropad_rfft(a: Array, n: int) -> Array:
    """Zero-pad a real-FFT spectral array along axis 2 (kx) to *n* modes.

    Unlike ``zeropad_fft``, only positive frequencies exist in a real FFT,
    so padding simply appends zeros at the high-frequency end.
    """
    axis = 2
    N = a.shape[axis]
    if n < N:
        raise ValueError(f"Target mode count {n} is smaller than input {N}.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = jnp.zeros(
        shape=out_shape,
        dtype=a.dtype,
        out_sharding=sharding.phys_scalar_shard,
    )

    idx = [slice(None)] * 3
    idx[axis] = slice(None, N)
    out = out.at[tuple(idx)].set(a)

    return out


def truncate_rfft(a: Array, n: int) -> Array:
    """Truncate a real-FFT output along axis 2 (kx) to *n* modes.

    Keeps only the lowest *n* non-negative frequencies.
    """
    axis = 2
    N = a.shape[axis]
    if n > N:
        raise ValueError(f"Target mode count {n} is larger than input {N}.")

    out_shape = list(a.shape)
    out_shape[axis] = n
    out = jnp.zeros(
        shape=out_shape,
        dtype=a.dtype,
        out_sharding=sharding.phys_scalar_shard,
    )

    idx_in = [slice(None)] * 3
    idx_out = [slice(None)] * 3
    idx_in[axis] = slice(None, n)
    idx_out[axis] = slice(None, n)
    out = out.at[tuple(idx_out)].set(a[tuple(idx_in)])

    return out


def _rfft2d(x: Array) -> Array:
    """Forward 2D real FFT in x and z (wall-bounded): physical -> spectral.

    Transform order is x -> z.  The y-axis is left untouched (grid-point
    space).  After each step, aliased modes are truncated.  A reshard
    (z-sharded -> kx-sharded) happens between the x- and z-transforms.

    Parameters
    ----------
    x:
        Real-valued scalar field of shape ``(ny, nz_padded, nx_padded)``,
        z-sharded.

    Returns
    -------
    :
        Complex spectral coefficients of shape ``(ny, nz-1, nx//2)``,
        kx-sharded.  Nyquist modes are omitted on z and x.
    """
    # Step 1: real FFT in x (z is sharded across devices)
    y = truncate_rfft(
        shard_map(
            lambda a: jnp.fft.rfft(a, axis=2, norm="forward"),
            mesh=sharding.mesh,
            in_specs=sharding.phys_scalar_shard,
            out_specs=sharding.phys_scalar_shard,
        )(x),
        params.res.nx // 2,
    )

    # Step 2: reshard from z-sharded (physical) to kx-sharded (spectral)
    y = reshard(y, sharding.spec_scalar_shard)

    # Step 3: complex FFT in z (kx is sharded), then truncate aliased modes
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

    return y


def _irfft2d(x: Array) -> Array:
    """Inverse 2D real FFT in x and z (wall-bounded): spectral -> physical.

    Transform order is z -> x (reverse of ``_rfft2d``).  Before each step,
    the spectral array is zero-padded to the oversampled grid size for
    dealiasing.  A reshard (kx-sharded -> z-sharded) happens between the
    z-transform and the x-transform.  The y-axis is untouched.

    Parameters
    ----------
    x:
        Complex spectral coefficients of shape ``(ny, nz-1, nx//2)``,
        kx-sharded.

    Returns
    -------
    :
        Real-valued scalar field of shape ``(ny, nz_padded, nx_padded)``,
        z-sharded.
    """
    # Step 1: zero-pad z to oversampled size, then inverse FFT in z
    # (kx is sharded)
    y = zeropad_fft(x, padded_res.nz_padded, 1)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=1, norm="forward"),
        mesh=sharding.mesh,
        in_specs=sharding.spec_scalar_shard,
        out_specs=sharding.spec_scalar_shard,
    )(y)

    # Step 2: reshard from kx-sharded (spectral) to z-sharded (physical)
    y = reshard(y, sharding.phys_scalar_shard)

    # Step 3: zero-pad kx to oversampled size, then inverse real FFT in x
    # (z is sharded)
    y = zeropad_rfft(y, padded_res.nx_padded // 2 + 1)
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


def _rfft3d(x: Array) -> Array:
    """Forward 3D real FFT: physical space -> spectral space.

    Transform order is x -> z -> y.  After each step, aliased modes are
    truncated.  A reshard (z-sharded -> kx-sharded) happens between the
    x-transform and the z-transform.

    Parameters
    ----------
    x:
        Real-valued scalar field of shape ``(ny_padded, nz_padded,
        nx_padded)``, z-sharded.

    Returns
    -------
    :
        Complex spectral coefficients of shape ``(ny-1, nz-1, nx//2)``,
        kx-sharded.  Nyquist modes are omitted on all axes.
    """
    # Step 1: real FFT in x (z is sharded across devices)
    y = truncate_rfft(
        shard_map(
            lambda a: jnp.fft.rfft(a, axis=2, norm="forward"),
            mesh=sharding.mesh,
            in_specs=sharding.phys_scalar_shard,
            out_specs=sharding.phys_scalar_shard,
        )(x),
        params.res.nx // 2,
    )

    # Step 2: reshard from z-sharded (physical) to kx-sharded (spectral)
    y = reshard(y, sharding.spec_scalar_shard)

    # Step 3: complex FFT in z (kx is sharded), then truncate aliased modes
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

    # Step 4: complex FFT in y (kx is sharded), then truncate aliased modes
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


def _irfft3d(x: Array) -> Array:
    """Inverse 3D real FFT: spectral space -> physical space.

    Transform order is y -> z -> x (reverse of ``_rfft3d``).  Before each
    step, the spectral array is zero-padded to the oversampled grid size
    for dealiasing.  A reshard (kx-sharded -> z-sharded) happens between
    the z-transform and the x-transform.

    Parameters
    ----------
    x:
        Complex spectral coefficients of shape ``(ny-1, nz-1, nx//2)``,
        kx-sharded.

    Returns
    -------
    :
        Real-valued scalar field of shape ``(ny_padded, nz_padded,
        nx_padded)``, z-sharded.
    """
    # Step 1: zero-pad y to oversampled size, then inverse FFT in y
    # (kx is sharded)
    y = zeropad_fft(x, padded_res.ny_padded, 0)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=0, norm="forward"),
        mesh=sharding.mesh,
        in_specs=sharding.spec_scalar_shard,
        out_specs=sharding.spec_scalar_shard,
    )(y)

    # Step 2: zero-pad z to oversampled size, then inverse FFT in z
    # (kx is sharded)
    y = zeropad_fft(y, padded_res.nz_padded, 1)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=1, norm="forward"),
        mesh=sharding.mesh,
        in_specs=sharding.spec_scalar_shard,
        out_specs=sharding.spec_scalar_shard,
    )(y)

    # Step 3: reshard from kx-sharded (spectral) to z-sharded (physical)
    y = reshard(y, sharding.phys_scalar_shard)

    # Step 4: zero-pad kx to oversampled size, then inverse real FFT in x
    # (z is sharded)
    y = zeropad_rfft(y, padded_res.nx_padded // 2 + 1)
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
