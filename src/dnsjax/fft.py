r"""2D and 3D real FFT with 3/2-rule dealiasing via zero-padding
and truncation, plus double-parallelisation reshards.

For the 2D case:
The forward transform (physical -> spectral) is ``_rfft2d``; the inverse
is ``_irfft2d``. They operate on scalar fields of layout ``[y, z, x]``
and ``[y, kz, kx]`` respectively.  The spectral layout ``[y, kz, kx]``
is the same as the public-facing convention.

For the 3D case:
The forward transform (physical -> spectral) is ``_rfft3d``; the inverse
is ``_irfft3d``.  They operate on scalar fields of layout ``[y, z, x]``,
and ``[ky, kz, kx]`` respectively.

``shard_map`` is used for per-device FFTs.  Two reshards shuttle data
between the three sharding stages of the pipeline:

1. **phys** ``P(a0, a1, None)`` — ``[y_{np0}, z_{np1}, x]``
2. **mid**  ``P(a0, None, a1)`` — ``[y_{np0}, z, kx_{np1}]``
   (after the `$z \leftrightarrow k_x$` reshard, Ns-way)
3. **spec** ``P(None, a0, a1)`` — ``[y, kz_{np0}, kx_{np1}]``
   (after the `$y \leftrightarrow k_z$` reshard, Nr-way)

When ``np0 == 1`` the mid and spec layouts are identical and the
second reshard is skipped.  When ``np1 == 1`` the phys and mid
layouts are identical and the first reshard is skipped.

Spectral padding
~~~~~~~~~~~~~~~~
If the true mode count (`$n_z - 1$` or `$n_x / 2$`) is not
divisible by the mesh axis, zero-valued padding modes are
carried at the high-frequency end of the stored arrays.  They
are appended by the dealiasing truncation itself (forward) and
skipped by the oversampling zero-pad's input slices (inverse)
-- the ``pad`` / ``strip`` arguments of the ``truncate_*`` /
``zeropad_*`` helpers -- so divisibility padding costs no extra
array pass.  The padding amount is read from
``sharding.nz_spec_pad`` and ``sharding.nx_spec_pad``.

Dealiasing
----------
The 3/2-rule expands each direction by a factor of oversampling_factor / 2
before transforming to physical space (``zeropad_*``), and
truncates back after the forward transform (``truncate_*``).  Nyquist
modes are omitted in all stored spectral arrays (`$n - 1$` modes for a
full-complex axis, `$n / 2$` modes for the real-FFT axis).

Memory (deferred optimisation)
------------------------------
Beyond its input and output, each transform materialises one to two
batch-sized intermediates per padded axis: the ``zeropad_*`` /
``truncate_*`` concatenate output and the per-axis (i)FFT result,
plus the reshard copies.  For the batched RHS transforms (6 fields
Newtonian, ~36 viscoelastic) these stage buffers dominate the
per-step working set.  Deferred optimisations: fuse the zero-pad
into the adjacent FFT stage (transform over the padded length while
reading only the unpadded input) and/or chunk batched transforms
generally (today only the viscoelastic RHS chunks, via
``solver.rhs_transform_chunks``).

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


# ── Physical y padding / stripping helpers ──────────────────


def _pad_y(a: Array, out_shard) -> Array:
    """Append ``ny_y_pad`` zero rows along axis 0 (y)."""
    pad = jnp.zeros(
        (sharding.ny_y_pad, a.shape[1], a.shape[2]),
        dtype=a.dtype,
        out_sharding=out_shard,
    )
    return jnp.concatenate([a, pad], axis=0)


def _strip_y(a: Array) -> Array:
    """Remove the trailing ``ny_y_pad`` rows along axis 0."""
    return a[: a.shape[0] - sharding.ny_y_pad, :, :]


# ── Dealiasing padding / truncation ─────────────────────────


def zeropad_fft(
    a: Array, n: int, axis: int, out_shard, strip: int = 0
) -> Array:
    """Zero-pad a full-complex spectral array along *axis* to length *n*.

    Inserts zeros between the positive and negative Fourier modes,
    reinstating the (previously omitted) Nyquist mode as zero.  This is
    the spectral-space equivalent of interpolation to a finer grid; the
    wrap-order mode placement is exact for any parity of the pad
    `$n - N$` and any parity of `$N$`.  Built as a single
    ``concatenate`` of the two kept slices around a zeros block (one
    output write pass, mirroring :func:`truncate_fft`; the zero-init +
    two scatter passes it replaces wrote the padded array roughly
    twice).  The padded axis is locally stored in the stage where each
    pad happens, so the zeros block created with *out_shard*
    concatenates without a reshard.

    Parameters
    ----------
    a:
        Input array with ``a.shape[axis] == N - 1 + strip`` stored
        modes (Nyquist omitted), where *$N$* is the original full mode
        count.
    n:
        Target length (`$\\ge N$`, any parity).
    axis:
        Axis along which to pad (0 for y, 1 for z).
    out_shard:
        Partition spec for the zeros block (and thus the output).
    strip:
        Trailing zero-valued divisibility-padding modes on *axis*
        (``sharding.nz_spec_pad``) to drop while padding; skipped by
        the input slices, so stripping costs no extra pass.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    stored = a.shape[axis] - strip
    N = stored + 1  # Add the omitted Nyquist mode
    if n < N:
        raise ValueError(f"Target size {n} is smaller than input size {N}.")

    mid_shape = list(a.shape)
    mid_shape[axis] = n - N + 1  # inserted zeros incl. the Nyquist slot
    mid = jnp.zeros(shape=mid_shape, dtype=a.dtype, out_sharding=out_shard)

    idx_pos = [slice(None)] * 3
    idx_neg = [slice(None)] * 3
    # positive modes; negative modes (the Nyquist slot is in ``mid``,
    # the trailing divisibility padding is skipped)
    idx_pos[axis] = slice(None, N // 2)
    idx_neg[axis] = slice(N // 2, stored)
    return jnp.concatenate(
        [a[tuple(idx_pos)], mid, a[tuple(idx_neg)]], axis=axis
    )


def truncate_fft(
    a: Array, n: int, axis: int, pad: int = 0, out_shard=None
) -> Array:
    """Truncate a full-complex FFT output along *axis*, dropping
    aliased modes.

    Keeps the lowest `$\\lfloor n/2 \\rfloor$` positive and
    `$n - 1 - \\lfloor n/2 \\rfloor$` negative modes, discarding all
    higher modes including the Nyquist mode.  The output has `$n - 1$`
    stored modes -- the layout of
    :func:`dnsjax.harmonics.complex_harmonics` for even and odd *n*
    alike (an odd *n* retains an asymmetric band, one more negative
    than positive mode; even sizes remain the recommended
    resolutions).  Formed by concatenating the kept slices (one copy;
    no zero-init plus scatters); the wrap-order placement is exact for
    any parity of `$N - n$`.  The truncated axis is locally stored in
    every pipeline stage, so the input sharding carries over to the
    output.

    Parameters
    ----------
    a:
        Full FFT output with ``a.shape[axis] == N`` modes.
    n:
        Target mode count (`$\\le N$`, any parity).
    axis:
        Axis along which to truncate (0 for y, 1 for z).
    pad:
        Zero-valued divisibility-padding modes
        (``sharding.nz_spec_pad``) to append after the kept modes;
        rides in the same ``concatenate``, so padding costs no extra
        pass.
    out_shard:
        Partition spec for the ``pad`` zeros block (required when
        ``pad > 0``).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1; got {axis}.")
    N = a.shape[axis]
    if n > N:
        raise ValueError(f"Target size {n} is larger than input size {N}.")

    idx_pos = [slice(None)] * 3
    idx_neg = [slice(None)] * 3
    # positive modes; negative modes (skip the Nyquist modes)
    n_neg = (n - 1) - n // 2
    idx_pos[axis] = slice(None, n // 2)
    idx_neg[axis] = slice(N - n_neg, None)
    parts = [a[tuple(idx_pos)], a[tuple(idx_neg)]]
    if pad:
        pad_shape = list(a.shape)
        pad_shape[axis] = pad
        parts.append(
            jnp.zeros(shape=pad_shape, dtype=a.dtype, out_sharding=out_shard)
        )
    return jnp.concatenate(parts, axis=axis)


def zeropad_rfft(a: Array, n: int, out_shard, strip: int = 0) -> Array:
    """Zero-pad a real-FFT spectral array along axis 2 (kx) to *n* modes.

    Unlike ``zeropad_fft``, only positive frequencies exist in a real FFT,
    so padding simply appends a zeros block at the high-frequency end
    (single ``concatenate``, one output write pass -- see
    :func:`zeropad_fft`; the kx axis is locally stored in this pipeline
    stage).  ``strip`` trailing divisibility-padding modes
    (``sharding.nx_spec_pad``) are dropped by the input slice at no
    extra pass.
    """
    axis = 2
    N = a.shape[axis] - strip
    if n < N:
        raise ValueError(f"Target mode count {n} is smaller than input {N}.")
    kept = a[:, :, :N] if strip else a
    if n == N:
        return kept

    tail_shape = list(a.shape)
    tail_shape[axis] = n - N
    tail = jnp.zeros(shape=tail_shape, dtype=a.dtype, out_sharding=out_shard)
    return jnp.concatenate([kept, tail], axis=axis)


def truncate_rfft(a: Array, n: int, pad: int = 0, out_shard=None) -> Array:
    """Truncate a real-FFT output along axis 2 (kx) to *n* modes.

    Keeps only the lowest *n* non-negative frequencies (a plain
    slice; the kx axis is locally stored in this pipeline stage,
    so the input sharding carries over).  ``pad`` zero-valued
    divisibility-padding modes (``sharding.nx_spec_pad``) are
    appended in the same ``concatenate`` at no extra pass.
    """
    N = a.shape[2]
    if n > N:
        raise ValueError(f"Target mode count {n} is larger than input {N}.")
    kept = a[:, :, :n]
    if not pad:
        return kept
    pad_shape = list(a.shape)
    pad_shape[2] = pad
    tail = jnp.zeros(shape=pad_shape, dtype=a.dtype, out_sharding=out_shard)
    return jnp.concatenate([kept, tail], axis=2)


# ── 2D FFT (wall-bounded) ───────────────────────────────────


def _rfft2d(x: Array) -> Array:
    r"""Forward 2D real FFT in x and z (wall-bounded):
    physical -> spectral.

    Pipeline: x-FFT `$\to$` [reshard #1: `$z \leftrightarrow
    k_x$`] `$\to$` z-FFT `$\to$` [reshard #2:
    `$y \leftrightarrow k_z$`].

    Reshard #1 is skipped when ``np1 == 1`` (layouts are
    identical).  Reshard #2 is skipped when ``np0 == 1``.

    Parameters
    ----------
    x:
        Real-valued scalar field of shape
        ``(ny + ny_y_pad, nz_padded, nx_padded)``, with
        ``P(a0, a1, None)`` sharding.

    Returns
    -------
    :
        Complex spectral coefficients of shape
        ``(ny, nz_spec, nx_spec)`` with
        ``P(None, a0, a1)`` sharding.
    """
    phys = sharding._fft_phys_scalar_shard
    mid = sharding._fft_mid_scalar_shard
    spec = sharding._fft_spec_scalar_shard

    # ---- Step 1: real FFT in x (y sharded by np0, z by np1) --
    # (kx divisibility padding appended within the truncate)
    y = truncate_rfft(
        shard_map(
            lambda a: jnp.fft.rfft(a, axis=2, norm="forward"),
            mesh=sharding.mesh,
            in_specs=phys,
            out_specs=phys,
        )(x),
        params.res.nx // 2,
        pad=sharding.nx_spec_pad,
        out_shard=phys,
    )

    # ---- Reshard #1: z <-> kx (Ns-way, skipped when np1==1) --
    if sharding.a1 is not None:
        y = reshard(y, mid)

    # ---- Step 2: complex FFT in z, then truncate aliased modes
    # (kz divisibility padding appended within the truncate)
    y = truncate_fft(
        shard_map(
            lambda a: jnp.fft.fft(a, axis=1, norm="forward"),
            mesh=sharding.mesh,
            in_specs=mid,
            out_specs=mid,
        )(y),
        params.res.nz,
        1,
        pad=sharding.nz_spec_pad,
        out_shard=mid,
    )

    # ---- Reshard #2: y <-> kz (Nr-way, skipped when np0==1) --
    if sharding.a0 is not None:
        y = reshard(y, spec)

    # Strip y-padding (appended zeros for np0 divisibility).
    if sharding.ny_y_pad:
        y = _strip_y(y)

    return y


def _irfft2d(x: Array) -> Array:
    r"""Inverse 2D real FFT in x and z (wall-bounded):
    spectral -> physical.

    Reverse pipeline: [reshard #2: `$k_z \leftrightarrow y$`]
    `$\to$` z-IFFT `$\to$` [reshard #1:
    `$k_x \leftrightarrow z$`] `$\to$` x-IFFT.

    Parameters
    ----------
    x:
        Complex spectral coefficients of shape
        ``(ny, nz_spec, nx_spec)`` with
        ``P(None, a0, a1)`` sharding.

    Returns
    -------
    :
        Real-valued scalar field of shape
        ``(ny + ny_y_pad, nz_padded, nx_padded)`` with
        ``P(a0, a1, None)`` sharding.
    """
    phys = sharding._fft_phys_scalar_shard
    mid = sharding._fft_mid_scalar_shard
    spec = sharding._fft_spec_scalar_shard

    # Pad y for np0 divisibility (stripped after forward reshard).
    if sharding.ny_y_pad:
        x = _pad_y(x, spec)

    # ---- Reshard #2 reverse: kz <-> y (skipped when np0==1) --
    if sharding.a0 is not None:
        x = reshard(x, mid)
    else:
        mid = spec  # layouts are identical when np0==1

    # ---- Step 1: zero-pad z then inverse FFT in z ------------
    # (kz divisibility padding skipped by the zero-pad slices)
    y = zeropad_fft(
        x, padded_res.nz_padded, 1, mid, strip=sharding.nz_spec_pad
    )
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=1, norm="forward"),
        mesh=sharding.mesh,
        in_specs=mid,
        out_specs=mid,
    )(y)

    # ---- Reshard #1 reverse: kx <-> z (skipped when np1==1) --
    if sharding.a1 is not None:
        y = reshard(y, phys)

    # ---- Step 2: zero-pad kx then inverse real FFT in x ------
    # (kx divisibility padding skipped by the zero-pad slice)
    y = zeropad_rfft(
        y,
        padded_res.nx_padded // 2 + 1,
        phys,
        strip=sharding.nx_spec_pad,
    )
    y = shard_map(
        lambda a: jnp.fft.irfft(a, axis=2, norm=norm),
        mesh=sharding.mesh,
        in_specs=phys,
        out_specs=phys,
    )(y)

    return y


# ── 3D FFT (triply-periodic) ────────────────────────────────


def _rfft3d(x: Array) -> Array:
    r"""Forward 3D real FFT: physical space -> spectral space.

    Pipeline: x-FFT `$\to$` [reshard #1] `$\to$` z-FFT
    `$\to$` [reshard #2] `$\to$` y-FFT.

    Parameters
    ----------
    x:
        Real-valued scalar field of shape
        ``(ny_padded, nz_padded, nx_padded)`` with
        ``P(a0, a1, None)`` sharding.

    Returns
    -------
    :
        Complex spectral coefficients of shape
        ``(ny-1, nz_spec, nx_spec)`` with
        ``P(None, a0, a1)`` sharding.
    """
    phys = sharding._fft_phys_scalar_shard
    mid = sharding._fft_mid_scalar_shard
    spec = sharding._fft_spec_scalar_shard

    # ---- Step 1: real FFT in x --------------------------------
    # (kx divisibility padding appended within the truncate)
    y = truncate_rfft(
        shard_map(
            lambda a: jnp.fft.rfft(a, axis=2, norm="forward"),
            mesh=sharding.mesh,
            in_specs=phys,
            out_specs=phys,
        )(x),
        params.res.nx // 2,
        pad=sharding.nx_spec_pad,
        out_shard=phys,
    )

    # ---- Reshard #1: z <-> kx (skipped when np1==1) -----------
    if sharding.a1 is not None:
        y = reshard(y, mid)

    # ---- Step 2: complex FFT in z, then truncate ---------------
    # (kz divisibility padding appended within the truncate)
    y = truncate_fft(
        shard_map(
            lambda a: jnp.fft.fft(a, axis=1, norm="forward"),
            mesh=sharding.mesh,
            in_specs=mid,
            out_specs=mid,
        )(y),
        params.res.nz,
        1,
        pad=sharding.nz_spec_pad,
        out_shard=mid,
    )

    # ---- Reshard #2: y <-> kz (skipped when np0==1) -----------
    if sharding.a0 is not None:
        y = reshard(y, spec)

    # ---- Step 3: complex FFT in y, then truncate ---------------
    y = truncate_fft(
        shard_map(
            lambda a: jnp.fft.fft(a, axis=0, norm="forward"),
            mesh=sharding.mesh,
            in_specs=spec,
            out_specs=spec,
        )(y),
        params.res.ny,
        0,
    )

    return y


def _irfft3d(x: Array) -> Array:
    r"""Inverse 3D real FFT: spectral space -> physical space.

    Reverse pipeline: y-IFFT `$\to$` [reshard #2] `$\to$`
    z-IFFT `$\to$` [reshard #1] `$\to$` x-IFFT.

    Parameters
    ----------
    x:
        Complex spectral coefficients of shape
        ``(ny-1, nz_spec, nx_spec)`` with
        ``P(None, a0, a1)`` sharding.

    Returns
    -------
    :
        Real-valued scalar field of shape
        ``(ny_padded, nz_padded, nx_padded)`` with
        ``P(a0, a1, None)`` sharding.
    """
    phys = sharding._fft_phys_scalar_shard
    mid = sharding._fft_mid_scalar_shard
    spec = sharding._fft_spec_scalar_shard

    # ---- Step 1: zero-pad y then inverse FFT in y -------------
    y = zeropad_fft(x, padded_res.ny_padded, 0, spec)
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=0, norm="forward"),
        mesh=sharding.mesh,
        in_specs=spec,
        out_specs=spec,
    )(y)

    # ---- Reshard #2 reverse: kz <-> y (skipped when np0==1) ---
    if sharding.a0 is not None:
        y = reshard(y, mid)

    # ---- Step 2: zero-pad z then inverse FFT in z -------------
    # (kz divisibility padding skipped by the zero-pad slices)
    y = zeropad_fft(
        y, padded_res.nz_padded, 1, mid, strip=sharding.nz_spec_pad
    )
    y = shard_map(
        lambda a: jnp.fft.ifft(a, axis=1, norm="forward"),
        mesh=sharding.mesh,
        in_specs=mid,
        out_specs=mid,
    )(y)

    # ---- Reshard #1 reverse: kx <-> z (skipped when np1==1) ---
    if sharding.a1 is not None:
        y = reshard(y, phys)

    # ---- Step 3: zero-pad kx then inverse real FFT in x -------
    # (kx divisibility padding skipped by the zero-pad slice)
    y = zeropad_rfft(
        y,
        padded_res.nx_padded // 2 + 1,
        phys,
        strip=sharding.nx_spec_pad,
    )
    y = shard_map(
        lambda a: jnp.fft.irfft(a, axis=2, norm=norm),
        mesh=sharding.mesh,
        in_specs=phys,
        out_specs=phys,
    )(y)

    return y
