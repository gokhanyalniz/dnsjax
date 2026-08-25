"""Shared spectral utilities: FFT wrappers and wavenumber helpers.

Provides wavenumber generation functions (``real_harmonics``,
``complex_harmonics``) and vmapped FFT wrappers for 3D
(triply-periodic) and 2D (wall-bounded) transforms.

Geometry-specific ``Fourier`` dataclasses live in the
corresponding geometry modules
(``geometries.triply_periodic``,
``geometries.wall_bounded.cartesian``,
``geometries.wall_bounded.cylindrical``).
"""

from jax import Array, jit, vmap
from jax import numpy as jnp

from .fft import _irfft2d, _irfft3d, _rfft2d, _rfft3d
from .harmonics import complex_harmonics as _complex_harmonics_np
from .harmonics import real_harmonics as _real_harmonics_np
from .sharding import sharding


def real_harmonics(n: int) -> Array:
    r"""Non-negative integer wavenumbers for a real-FFT axis.

    Thin device-array wrapper over
    :func:`dnsjax.harmonics.real_harmonics` (the JAX-free source of
    truth, shared with :mod:`dnsjax.analysis`): the Nyquist mode is
    omitted, leaving `$n / 2$` modes `$[0, 1, \dots, n/2 - 1]$`.
    """
    return jnp.asarray(_real_harmonics_np(n))


def complex_harmonics(n: int) -> Array:
    r"""Full-complex integer wavenumbers with the Nyquist mode omitted.

    Thin device-array wrapper over
    :func:`dnsjax.harmonics.complex_harmonics` (the JAX-free source of
    truth, shared with :mod:`dnsjax.analysis`): `$n - 1$` wavenumbers in
    FFT order `$[0, 1, \dots, n/2-1, -n/2+1, \dots, -1]$`.
    """
    return jnp.asarray(_complex_harmonics_np(n))


def pad_harmonics(harmonics: Array, n: int, pad: int) -> Array:
    """Append placeholder wavenumbers for divisibility-padding slots.

    Spectral axes are padded so the mode count divides the device
    mesh (see :mod:`dnsjax.sharding`).  The padding slots receive
    the beyond-resolution wavenumbers
    `$[n/2, n/2 + 1, \\dots, n/2 + \\text{pad} - 1]$` (the omitted
    Nyquist magnitude and up; both generators above have maximum
    magnitude `$n/2 - 1$`).  They must be nonzero integers so the
    per-mode operators assembled at padding slots stay regular
    (only `$k^2 = 0$` systems are singular); the values are
    otherwise arbitrary because padded fields are identically zero
    (the forward FFT re-zeroes the padding slots on every
    evaluation; see :mod:`dnsjax.fft`).

    Parameters
    ----------
    harmonics:
        True wavenumbers from ``real_harmonics`` or
        ``complex_harmonics``.
    n:
        Full mode count along the axis.
    pad:
        Number of padding slots to append.

    Returns
    -------
    :
        ``harmonics`` with ``pad`` placeholder wavenumbers appended.
    """
    if not pad:
        return harmonics
    placeholder = jnp.arange(n // 2, n // 2 + pad, dtype=int)
    return jnp.concatenate([harmonics, placeholder])


@jit
@vmap
def phys_to_spec(velocity_phys: Array) -> Array:
    """Forward 3D real FFT, vmapped over velocity components.

    Used for triply-periodic flows, where all three directions are
    Fourier-expanded.

    Parameters
    ----------
    velocity_phys:
        Physical field of shape ``(3, ny_padded, nz_padded, nx_padded)``,
        sharded on the z axis.

    Returns
    -------
    :
        Spectral field of shape ``(3, ny-1, nz_spec, nx_spec)`` in
        ``[ky, kz, kx]`` layout, sharded on the kx axis.
    """
    return _rfft3d(velocity_phys)


@jit
@vmap
def spec_to_phys(velocity_spec: Array) -> Array:
    """Inverse 3D real FFT, vmapped over velocity components.

    Used for triply-periodic flows.

    Parameters
    ----------
    velocity_spec:
        Spectral field of shape ``(3, ny-1, nz_spec, nx_spec)`` in
        ``[ky, kz, kx]`` layout, sharded on the kx axis.

    Returns
    -------
    :
        Physical field of shape ``(3, ny_padded, nz_padded, nx_padded)``,
        sharded on the z axis.
    """
    return _irfft3d(velocity_spec)


if sharding.np0 > 1:
    # With 2D sharding, y is distributed by np0.  Merging the
    # component axis with y produces an ambiguous reshape that
    # JAX cannot automatically resolve.  Use vmap instead.
    @jit
    def phys_to_spec_2d(velocity_phys: Array) -> Array:
        r"""Forward 2D real FFT in `$(x, z)$`, vmapped over
        components.

        Parameters
        ----------
        velocity_phys:
            Physical field of shape
            ``(C, ny + ny_y_pad, nz_padded, nx_padded)``
            in ``[y, z, x]`` layout.

        Returns
        -------
        :
            Spectral field of shape
            ``(C, ny, nz_spec, nx_spec)`` in
            ``[y, kz, kx]`` layout.
        """
        return vmap(_rfft2d)(velocity_phys)

    @jit
    def spec_to_phys_2d(velocity_spec: Array) -> Array:
        r"""Inverse 2D real FFT in `$(x, z)$`, vmapped over
        components.

        Parameters
        ----------
        velocity_spec:
            Spectral field of shape
            ``(C, ny, nz_spec, nx_spec)`` in
            ``[y, kz, kx]`` layout.

        Returns
        -------
        :
            Physical field of shape
            ``(C, ny + ny_y_pad, nz_padded, nx_padded)``
            in ``[y, z, x]`` layout.
        """
        return vmap(_irfft2d)(velocity_spec)

else:
    # np0 == 1: y is replicated, so merging the component axis
    # with y is unambiguous, and the transform runs once on the
    # flattened batch instead of once per component.  It is *not*
    # fewer collectives -- ``vmap`` over the ``shard_map`` pipeline
    # already batches the reshard, and both forms compile to the
    # same op counts at the same mesh (measured on a 1x4 CPU mesh:
    # 6 all-to-all, 6 transpose, 2 fft either way).  What the fold
    # saves is the per-component ``vmap`` dispatch: 12.3 vs 14.1 ms
    # at 1x1, 8.03 vs 8.32 ms at 1x4.
    @jit
    def phys_to_spec_2d(velocity_phys: Array) -> Array:
        r"""Forward 2D real FFT in `$(x, z)$`, batched over
        components.

        The leading (component) axis is folded into the y-axis
        before the transform and unfolded afterwards, so the
        pipeline runs once on the flattened batch rather than once
        per component (see the branch comment above: the collective
        count is unchanged; the saving is the ``vmap`` dispatch).

        Parameters
        ----------
        velocity_phys:
            Physical field of shape
            ``(C, ny, nz_padded, nx_padded)`` in
            ``[y, z, x]`` layout.

        Returns
        -------
        :
            Spectral field of shape
            ``(C, ny, nz_spec, nx_spec)`` in
            ``[y, kz, kx]`` layout.
        """
        C, ny = velocity_phys.shape[0], velocity_phys.shape[1]
        flat = velocity_phys.reshape(C * ny, *velocity_phys.shape[2:])
        spec_flat = _rfft2d(flat)
        return spec_flat.reshape(C, ny, *spec_flat.shape[1:])

    @jit
    def spec_to_phys_2d(velocity_spec: Array) -> Array:
        r"""Inverse 2D real FFT in `$(x, z)$`, batched over
        components.

        The leading (component) axis is folded into the y-axis
        before the transform and unfolded afterwards, so the
        pipeline runs once on the flattened batch rather than once
        per component (see the branch comment above: the collective
        count is unchanged; the saving is the ``vmap`` dispatch).

        Parameters
        ----------
        velocity_spec:
            Spectral field of shape
            ``(C, ny, nz_spec, nx_spec)`` in
            ``[y, kz, kx]`` layout.

        Returns
        -------
        :
            Physical field of shape
            ``(C, ny, nz_padded, nx_padded)`` in
            ``[y, z, x]`` layout.
        """
        C, ny = velocity_spec.shape[0], velocity_spec.shape[1]
        flat = velocity_spec.reshape(C * ny, *velocity_spec.shape[2:])
        phys_flat = _irfft2d(flat)
        return phys_flat.reshape(C, ny, *phys_flat.shape[1:])
