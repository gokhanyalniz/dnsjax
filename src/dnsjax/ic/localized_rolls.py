r"""Localized-roll ("turbulent spot") initial-condition generators.

Builds a deterministic divergence-free **localized** perturbation of the
base flow for every flow system, returned as a sharded spectral state
ready to time step.  This is the implementation behind the
in-process ``init.localized_rolls`` start mode (``dnsjax.__main__``);
there is no offline script (unlike the random field).

**Fixed-physical structure (a spot, surrounded by laminar flow).** The
perturbation is a compact structure of *fixed physical size*, localized
in **every homogeneous direction**, peak-normalized so that
`$\max|\mathbf{u}'| = A$` (the ``amplitude`` argument) at any domain
size.  Growing a box length therefore just adds laminar flow around the
spot, so the volume-averaged statistics scale as `$1 / (L_x L_z)$` -- the
total perturbation energy is domain-independent.  Box-proportional
sizing would forfeit both properties: a width set as a *fraction* of the
box, or a single cross-plane Fourier mode (whose `$1/\beta$`
streamfunction prefactor with `$\beta = 2\pi / L$` makes the
cross-stream velocity grow `$\propto L$`), blows the perturbation up
with the domain instead of holding it fixed.

**Construction (Cartesian; `$x$` streamwise, `$z$` spanwise, `$y$`
wall-normal).** With zero streamwise velocity and a `$y$`-`$z$`
streamfunction `$\psi = G(y)\,\Psi(z)\,X(x)$`,

.. math::
    u_x = 0, \quad
    u_y = G(y)\,\Psi'(z)\,X(x), \quad
    u_z = -G'(y)\,\Psi(z)\,X(x),

which is divergence-free for **any** profiles (`$u_x = 0$` lets `$X$`
factor out of the divergence, and the `$y$`-`$z$` pair is a
streamfunction) -- an argument that never mentions walls, which is why
the triply-periodic member below is the same construction with one
factor swapped.  `$G = (1 - y^2)^2$` (peak 1, value + derivative zero at
both walls).  `$X(x)$` is a fixed-physical-width streamwise localization
and `$\Psi(z)$` a spanwise roll (wavelength `$\lambda$`) under a
fixed-physical-width envelope, so the spot is localized in both `$x$` and
`$z$`.  The spanwise derivative `$\Psi'$` is built **spectrally** as
`$\mathrm{i}k_z\,\hat\Psi$` so the discrete divergence is truncation-level
(projected out by the first corrector step).  Pipe and
annular use the analogous per-geometry streamfunction (see the
per-generator docstrings).

**Triply-periodic.** `$y$` is Fourier rather than a wall-normal grid, so
the only wall-specific ingredient -- `$G$`, whose shape exists to satisfy
the no-slip conditions -- is replaced by the same fixed-physical
localization the homogeneous directions use, and `$G'$` becomes the exact
`$\mathrm{i}k_y\hat G$`.  The divergence then cancels analytically
(round-off, not truncation-level) and the mean mode is zero structurally;
see :func:`generate_periodic_rolls`.

**Mean-free by construction**, and (wall-bounded) there is nothing
admissible to keep.  The `$(0,0)$` content the roll pair would otherwise
carry is `$-G'(y)$` times two scalars, and with `$G = (1-y^2)^2$` that is
a **cubic** in `$y$` -- while the mean mode's own conservation laws
(:mod:`dnsjax.ic.mean_mode`) require `$\delta(\pm 1) = 0$` *and*
`$\delta''(\pm 1) = 0$`, whose only cubic solution is
`$\delta \equiv 0$`.  The spot's natural mean content is therefore
*entirely* inadmissible: any compatible profile put in its place would
be a different function, not a scaled roll.  So the mode stays zero
whatever ``init.random_mean_flow`` is set to: the `$(k_x, k_z) =
(0, 0)$` mode of every component is **identically zero**, and a spot
never changes the field's bulk velocity or wall shear (the Dean
laminar profile is the only mean-mode content, added separately).  Two
independent mechanisms give that, one per component of each
streamfunction pair:

- the *cross-stream* component carries a spectral derivative
  `$\mathrm{i}k\,\hat f$` of its roll factor, which vanishes at
  `$k = 0$` exactly;
- the *roll* component would otherwise inherit its roll factor's DC bin,
  so every roll factor is made mean-free (:func:`_mean_free` on the
  signal, for the peak normalization; :func:`_zero_dc` on its spectrum,
  for an exact zero).  That bin is **not** generically zero: the roll is
  odd about the box centre under an even envelope, so all discrete pairs
  `$j \leftrightarrow n-j$`
  cancel except the self-paired `$j = 0$` box edge, leaving
  `$-\sin(\pi L/\lambda)\,e^{-(L/\pi\sigma)^2} / n$` -- machine-zero
  only when `$L/\lambda \in \mathbb{Z}$`, and 2 % of the peak
  coefficient at e.g. `$L = 2\pi$`, `$\lambda = 4$`.

In the triply-periodic family the mean mode is a single bin rather than
a `$y$` profile, and the *first* mechanism alone settles it on both
components at once (`$u_z$`'s `$\mathrm{i}k_y$` is a spectral derivative
there too), so the second is kept only for parity and for the host-side
peak.  The pipe would need neither in exact arithmetic -- its
azimuthal factors are exactly one `$m = \pm 1$` period and its `$u_z$`
is identically zero -- but its DC bins are round-off rather than zero,
so they are zeroed too.  Removing a roll factor's mean cannot disturb the
discrete divergence: on the affected `$k = 0$` plane the component's own
divergence contribution is `$\mathrm{i}\,0\,u = 0$` and its partner is
already identically zero there.  Guard:
``tests/test_localized_rolls.py``.

**Separable, sharded construction (no replication).** Each component is
an outer product `$(\text{wall-normal profile}) \otimes (\text{complex-}
\text{axis spectrum}) \otimes (\text{real-axis spectrum})$` built
**directly in spectral space, sharded**, from small 1-D factors via the
dnsjax broadcast-of-sharded-factors idiom (the same pattern that builds
``Fourier.k2`` / ``mean_mask``): the complex-FFT-axis factor is placed on
the ``np0`` mesh axis, the real-FFT-axis factor on ``np1``, and the
broadcast product is sharded -- so **no full array is ever materialised**
and the field is device-count-independent.  The peak normalization is a
one-time host-side computation on the small 1-D factor signals (also
replication-free).  No-slip is **exact** where there are walls (the wall
slice of every profile is identically zero, never transformed); the
triply-periodic member passes a `$k_y$` spectrum in that same slot,
which is unsharded on every mesh exactly as a wall-normal profile is.
Reality is automatic: every 1-D factor spectrum is the forward FFT of a
*real* signal, hence conjugate-symmetric, and separable products
preserve that Hermitian symmetry -- no enforcement step (unlike the
random field's ``enforce_hermitian_slice``).  The 1-D spectra are
zero-padded to the mesh-padded mode counts, so padding modes stay
identically zero on any device mesh.

**Import-order discipline**: only NumPy and the JAX-free
``harmonics`` / ``parameters`` leaves are imported at module top; ``jax``,
``sharding``, and the geometry modules are imported lazily inside each
generator, so importing this module is safe before JAX is configured.
Wavenumber arrays are recomputed host-side (never fetched from the
multi-device ``fourier`` singleton).
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

# Wavenumber sequences come from the JAX-free ``harmonics`` leaf: the
# builders must never fetch the ``fourier`` singleton's wavenumber
# arrays, which are global multi-device arrays (not addressable per
# process under ``mpirun``).
from ..flows.registry import (
    annular_systems,
    annular_viscoelastic_systems,
    cartesian_systems,
    cylindrical_systems,
    cylindrical_viscoelastic_systems,
    periodic_systems,
)
from ..harmonics import complex_harmonics, real_harmonics
from ..parameters import derived_params, params

if TYPE_CHECKING:
    # ``Array`` is used only in (stringised) annotations, so it never
    # needs importing at runtime -- keeping this module importable
    # before JAX is configured (see the module docstring).
    from jax import Array

# ── 1-D physical signals and host wavenumbers ────────────────────
#
# All signals are sampled at the box's ``n`` grid points; the box length
# ``L`` enters only through the *physical* width / wavelength, so the
# sampled shape -- and hence the perturbation -- has a fixed physical
# size independent of ``L`` (growing ``L`` adds laminar around the spot).
# Spectra apply dnsjax's *forward* transform (``norm="forward"``, Nyquist
# dropped) so dnsjax's inverse reproduces the sampled signal exactly (see
# :mod:`dnsjax.fft`).


def _envelope(n: int, sigma: float, L: float) -> np.ndarray:
    r"""Fixed-physical-width periodic localization, length n.

    `$e^{-(L / \pi\sigma)^2 \sin^2(\pi (s/L - 1/2))}$` sampled at
    `$s = jL/n$`, centred at the box mid-point `$L/2$`.  Near the centre
    this is `$e^{-((s - L/2)/\sigma)^2}$`, so `$\sigma$` is the physical
    `$e$`-folding half-width regardless of `$L$` (for `$\sigma \gtrsim L$`
    it degrades smoothly to ``1``, i.e. no localization).
    """
    a = (L / (pi * sigma)) ** 2
    j = np.arange(n)
    return np.exp(-a * np.sin(pi * (j / n - 0.5)) ** 2)


def _roll(n: int, wavelength: float, L: float) -> np.ndarray:
    r"""Cross-roll oscillation `$\sin(2\pi (s - L/2)/\lambda)$`, length n.

    A roll pattern of fixed physical wavelength `$\lambda$`, sampled at
    `$s = jL/n$` and centred at the box mid-point so it aligns with the
    localization envelope.
    """
    s = np.arange(n) * (L / n) - 0.5 * L
    return np.sin(2.0 * pi * s / wavelength)


def _mean_free(signal: np.ndarray) -> np.ndarray:
    r"""Subtract a periodic signal's mean.

    Applied to every **roll** factor (see the "Mean-free by
    construction" note in the module docstring).  A constant offset is
    invisible to the spectral derivative (`$\mathrm{i}k\,\hat f$` at
    `$k = 0$`), so the partner component and the discrete divergence are
    unchanged; what this *does* fix is the host-side peak normalization,
    which reads the physical signal and must see the field that is
    actually built.  Floating-point subtraction leaves an `$O(\epsilon)$`
    residue in the DC bin, so the spectrum still goes through
    :func:`_zero_dc` -- that is what makes the mean mode exact.
    """
    return signal - signal.mean()


def _zero_dc(spectrum: np.ndarray) -> np.ndarray:
    r"""Zero a roll factor's DC bin **exactly**.

    The `$(k_x, k_z) = (0, 0)$` mode of the assembled component is this
    bin times the other factors, so setting it to a hard zero is what
    makes "the spot does not move the bulk velocity or the wall shear"
    an exact statement rather than an `$O(\epsilon)$` one.  Only *roll*
    factors get this: an envelope's DC bin is its mean, the whole point
    of a localization, and zeroing it would turn the envelope into an
    oscillation about zero.
    """
    out = spectrum.copy()
    out[0] = 0.0
    return out


def _sin_signal(n: int) -> np.ndarray:
    r"""One full sine wave `$\sin(2\pi j / n)$`, length n (azimuthal m=1)."""
    return np.sin(2 * pi * np.arange(n) / n)


def _cos_signal(n: int) -> np.ndarray:
    r"""One full cosine wave `$\cos(2\pi j / n)$`, length n (azimuthal m=1)."""
    return np.cos(2 * pi * np.arange(n) / n)


def _complex_k(n: int, L: float) -> np.ndarray:
    """Host complex-axis wavenumbers, true length ``n - 1``."""
    return complex_harmonics(n) * (2.0 * pi / L)


def _real_k(n: int, L: float) -> np.ndarray:
    """Host real-axis wavenumbers ``[0, .., n//2-1] * 2 pi / L``."""
    return real_harmonics(n) * (2.0 * pi / L)


def _ddz(signal: np.ndarray, L: float) -> np.ndarray:
    r"""Spectral derivative `$\partial_s$` of a length-n periodic signal.

    Host-numpy `$\mathrm{ifft}(\mathrm{i}k\,\mathrm{fft}(f))$` on the box
    `$[0, L)$`; used only to evaluate the peak velocity of the
    spectrally-differentiated streamfunction factor (the field itself
    uses the dnsjax spectrum).
    """
    n = signal.shape[0]
    k = np.fft.fftfreq(n, d=L / n) * (2.0 * pi)
    return np.fft.ifft(1j * k * np.fft.fft(signal)).real


def _real_axis_spectrum(signal: np.ndarray) -> np.ndarray:
    """dnsjax real-FFT forward of a 1-D signal.

    ``rfft`` with ``norm="forward"``, Nyquist dropped: the ``n // 2``
    non-negative modes in :func:`operators.real_harmonics` order.
    """
    n = signal.shape[0]
    return np.fft.rfft(signal, norm="forward")[: n // 2]


def _complex_axis_spectrum(signal: np.ndarray) -> np.ndarray:
    """dnsjax complex-FFT forward of a 1-D signal.

    ``fft`` with ``norm="forward"``, Nyquist dropped and reordered to
    :func:`operators.complex_harmonics` order
    ``[0, 1, .., n//2-1, -n//2+1, .., -1]`` (length ``n - 1``).
    """
    n = signal.shape[0]
    spec = np.fft.fft(signal, norm="forward")
    return np.concatenate([spec[: n // 2], spec[n // 2 + 1 :]])


# ── Peak normalization (replication-free, host-side) ─────────────


def _peak_velocity(
    components: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> float:
    r"""Max `$|\mathbf{u}'|$` of a separable lab-basis field, host-side.

    Each *component* is the triple of physical 1-D factors
    ``(wall-normal profile, complex-axis signal, real-axis signal)`` of
    one **lab-frame** velocity component (`$u_x, u_y, u_z$` Cartesian;
    `$u_z, u_r, u_\theta$` cyl/annular).  The component physical field is
    their outer product, so
    `$\max|\mathbf{u}'| = \sqrt{\max_{\mathbf{x}} \sum_c u_c^2}$` is a
    tiny host-numpy reduction over the ``(Ny, n_c, n_r)`` grid (no JAX,
    no replication).  The wall-normal padding rows are absent from these
    1-D signals, so they cannot inflate the peak.
    """
    sq = None
    for prof, csig, rsig in components:
        field = prof[:, None, None] * csig[None, :, None] * rsig[None, None, :]
        sq = field**2 if sq is None else sq + field**2
    return float(np.sqrt(sq.max()))


# ── Sharded separable assembly (broadcast of sharded 1-D factors) ─


def _separable_scalar(
    prof: np.ndarray,
    complex_spectrum: np.ndarray,
    real_spectrum: np.ndarray,
) -> Array:
    r"""Outer-product one sharded spectral scalar from 1-D factors.

    Builds ``prof[:, None, None] * complex[None, :, None] *
    real[None, None, :]`` as a ``(Ny, nz_spec, nx_spec)`` array on
    ``spec_scalar_shard`` (``Ny -> Nky = ny - 1`` for the
    triply-periodic family, whose leading axis is spectral) via the
    dnsjax broadcast-of-sharded-factors idiom (cf. ``Fourier.k2`` /
    ``mean_mask``): the complex-FFT-axis
    factor is placed on the ``np0`` mesh axis and the real-FFT-axis
    factor on ``np1``, so the broadcast product is sharded and **no full
    array is materialised**.  The 1-D spectra are zero-padded to the
    mesh-padded mode counts (``nz_spec`` / ``nx_spec``), so padding modes
    stay zero.

    Parameters
    ----------
    prof:
        Leading-axis factor, already at its full length and
        unsharded: a **real wall-normal profile** ``(Ny,)`` / ``(Nr,)``
        for the wall-bounded geometries, or the **complex** `$k_y$`
        spectrum ``(ny - 1,)`` for the triply-periodic one.  That axis
        is never mesh-padded, so unlike the two below it needs no
        zero-fill.
    complex_spectrum:
        Complex-FFT-axis (`$k_z$` / `$m$`, ``np0``) spectrum, true
        length ``nz - 1``.
    real_spectrum:
        Real-FFT-axis (`$k_x$` / `$k_{z,\mathrm{ax}}$`, ``np1``)
        spectrum, true length ``nx // 2``.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    from ..sharding import sharding

    npc = np.complex128 if params.res.double_precision else np.complex64

    # Replicated wall-normal profile (Ny / Nr is local on every device).
    prof_f = jax.device_put(
        prof.reshape(-1, 1, 1).astype(npc), sharding.no_shard
    )
    # Complex-FFT-axis factor (k_z / m), sharded on np0, padding zero.
    cfac = np.zeros(sharding.nz_spec, dtype=npc)
    cfac[: complex_spectrum.shape[0]] = complex_spectrum
    cfac_f = jax.device_put(cfac.reshape(1, -1, 1), P(None, sharding.a0, None))
    # Real-FFT-axis factor (k_x / k_z,ax), sharded on np1, padding zero.
    rfac = np.zeros(sharding.nx_spec, dtype=npc)
    rfac[: real_spectrum.shape[0]] = real_spectrum
    rfac_f = jax.device_put(rfac.reshape(1, 1, -1), P(None, None, sharding.a1))
    return prof_f * cfac_f * rfac_f


# ── Cartesian wall-bounded rolls ─────────────────────────────────


def generate_cartesian_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Localized-spot rolls for Cartesian wall-bounded flow.

    Components `$(u_x, u_y, u_z)$`, axes `$[y, k_z, k_x]$`.  Streamwise
    `$x$` (real `$k_x$` axis) carries a fixed-width localization
    `$X(x)$`; spanwise `$z$` (complex `$k_z$` axis) a roll `$\Psi(z)$` of
    wavelength `$\lambda$` under a fixed-width envelope; `$y \in [-1, 1]$`
    the wall-normal profile `$G = (1 - y^2)^2$`.  The streamfunction
    `$\psi = G\,\Psi\,X$` gives the divergence-free triple

    .. math::
        u_x = 0, \quad
        u_y = G \otimes (\mathrm{i}k_z\hat\Psi) \otimes \hat X, \quad
        u_z = -G' \otimes \hat\Psi \otimes \hat X,

    with `$G' = -4 y (1 - y^2)$` (zero at the walls).  All three
    components vanish exactly at `$y = \pm 1$`; the field is
    peak-normalized so `$\max|\mathbf{u}'| = A$`.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.cartesian import build_cartesian_grid
    from ..sharding import sharding

    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    lx, lz = params.geo.lx, params.geo.lz
    ys, *_ = build_cartesian_grid(
        ny,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(ys)]

    ys_np = np.asarray(ys)
    g = (1.0 - ys_np**2) ** 2  # peak 1; value + derivative zero at walls
    gp = -4.0 * ys_np * (1.0 - ys_np**2)  # G'(y); zero at walls

    x_sig = _envelope(nx, width, lx)  # streamwise localization X(x)
    psi_sig = _roll(nz, wavelength, lz) * _envelope(nz, width, lz)  # Psi(z)
    psi_sig = _mean_free(psi_sig)  # kills the (0, 0) mode of u_z
    psi_dz = _ddz(psi_sig, lz)  # Psi'(z) (for the peak)

    x_spec = _real_axis_spectrum(x_sig)
    psi_spec = _zero_dc(_complex_axis_spectrum(psi_sig))
    psi_dz_spec = 1j * _complex_k(nz, lz) * psi_spec  # spectral derivative

    u_y = _separable_scalar(g, psi_dz_spec, x_spec)
    u_z = _separable_scalar(-gp, psi_spec, x_spec)
    u_x = jnp.zeros_like(u_y)
    state = jnp.stack([u_x, u_y, u_z]).astype(sharding.complex_type)

    peak = _peak_velocity([(g, psi_dz, x_sig), (gp, psi_sig, x_sig)])
    return state * (amplitude / peak)


# ── Cylindrical (pipe) rolls ─────────────────────────────────────


def generate_cylindrical_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Axially-localized puff for pipe flow (`$\lambda$` unused).

    Components `$(u_z, u_r, u_\theta)$`, axes `$[r, m, k_z]$`.  The
    azimuthal cross-section is the `$m = \pm 1$` roll pair (filling the
    pipe, fixed `$2\pi$` -- so no spanwise blow-up); axial `$z$` (real
    `$k_z$` axis) carries a fixed-width localization `$X(z)$`.  The
    reference cross-plane roll (`$g = (1 - r^2)^2$`):

    .. math::
        u_r &= g \otimes \sin(m) \otimes \hat X \\
        u_\theta &= (g + r g') \otimes \cos(m) \otimes \hat X \\
        u_z &= 0

    with `$g + r g' = (1 - r^2)(1 - 5 r^2)$`.  Both radial profiles
    vanish at the wall `$r = 1$`; the field is peak-normalized so
    `$\max|\mathbf{u}'| = A$`.  ``wavelength`` is ignored (the
    azimuthal structure is the fixed `$m = \pm 1$` mode).
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.cylindrical import build_cylindrical_grid
    from ..sharding import sharding

    nx, nr, nz = params.res.nx, params.res.ny, params.res.nz
    lx = params.geo.lx  # axial length
    rs, *_ = build_cylindrical_grid(
        nr,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    g = (1.0 - rs_np**2) ** 2  # peak 1; zero at the wall r = 1
    g_r = (1.0 - rs_np**2) * (1.0 - 5.0 * rs_np**2)  # g + r g'; zero at r=1

    sin_m = _sin_signal(nz)  # azimuthal m = +-1
    cos_m = _cos_signal(nz)
    x_sig = _envelope(nx, width, lx)  # axial localization
    x_spec = _real_axis_spectrum(x_sig)

    # ``_zero_dc``: one exact ``m = +-1`` period already leaves only a
    # round-off DC bin, but "the spot does not move the bulk velocity"
    # is an exact statement here as in the other two geometries.
    u_r = _separable_scalar(g, _zero_dc(_complex_axis_spectrum(sin_m)), x_spec)
    u_theta = _separable_scalar(
        g_r, _zero_dc(_complex_axis_spectrum(cos_m)), x_spec
    )
    u_z = jnp.zeros_like(u_r)
    state = jnp.stack([u_z, u_r, u_theta]).astype(sharding.complex_type)

    peak = _peak_velocity([(g, sin_m, x_sig), (g_r, cos_m, x_sig)])
    return state * (amplitude / peak)


# ── Annular (Taylor-Couette / Dean) rolls ────────────────────────


def generate_annular_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Localized-spot rolls for Taylor-Couette / Dean flow.

    Components `$(u_z, u_r, u_\theta)$`,
    axes `$[r, m, k_z]$`.  The streamwise/spanwise roles **swap** versus
    pipe: streamwise azimuthal `$\theta$` (complex `$m$` axis) carries a
    fixed-width localization `$A(\theta)$` (physical width converted to an
    arc at mid-radius); spanwise axial `$z$` (real `$k_z$` axis) a roll
    `$Z(z)$` of wavelength `$\lambda$` under a fixed-width envelope;
    `$r \in [r_1, r_2]$` the wall-normal profile.  A Stokes streamfunction
    `$\Phi = P(r)\,Z(z)\,A(\theta)$` in the `$r$`-`$z$` plane
    (`$P = ((r - r_1)(r_2 - r))^2$`) gives the divergence-free pair

    .. math::
        u_r &= -(P/r) \otimes \hat A \otimes (\mathrm{i}k_z\hat Z) \\
        u_z &= (P'/r) \otimes \hat A \otimes \hat Z \\
        u_\theta &= 0

    with `$P' = 2 (r - r_1)(r_2 - r)(r_1 + r_2 - 2 r)$`.  `$P$` and `$P'$`
    vanish at both walls, so all components are exactly zero at
    `$r = r_1, r_2$`.  The field is peak-normalized so
    `$\max|\mathbf{u}'| = A$`.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.annular import build_annular_grid
    from ..sharding import sharding

    nx, nr, nz = params.res.nx, params.res.ny, params.res.nz
    lx = params.geo.lx  # axial length (spanwise; real axis)
    r1, r2 = derived_params.r_inner, derived_params.r_outer
    rs, *_ = build_annular_grid(
        nr,
        params.res.fd_order,
        r1,
        r2,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    q = (rs_np - r1) * (r2 - rs_np)  # zero at both walls
    pp = 2.0 * q * (r1 + r2 - 2.0 * rs_np)  # P' = d/dr (q^2); zero at walls
    p_over_r = q**2 / rs_np  # P / r
    pp_over_r = pp / rs_np  # P' / r

    r_mid = 0.5 * (r1 + r2)
    # Azimuthal localization over the wedge extent l_z = 2*pi/m0 (m0 = 1
    # full circle); the physical angular half-width width/r_mid is
    # preserved regardless of the period (see ``_envelope``).
    az_sig = _envelope(nz, width / r_mid, params.geo.lz)
    z_sig = _roll(nx, wavelength, lx) * _envelope(nx, width, lx)  # Z(z_ax)
    z_sig = _mean_free(z_sig)  # kills the (0, 0) mode of u_z
    z_dz = _ddz(z_sig, lx)  # Z'(z_ax) (for the peak)

    az_spec = _complex_axis_spectrum(az_sig)  # envelope: DC bin kept
    z_spec = _zero_dc(_real_axis_spectrum(z_sig))
    z_dz_spec = 1j * _real_k(nx, lx) * z_spec  # spectral derivative

    u_r = _separable_scalar(-p_over_r, az_spec, z_dz_spec)
    u_z = _separable_scalar(pp_over_r, az_spec, z_spec)
    u_theta = jnp.zeros_like(u_r)
    state = jnp.stack([u_z, u_r, u_theta]).astype(sharding.complex_type)

    peak = _peak_velocity(
        [(p_over_r, az_sig, z_dz), (pp_over_r, az_sig, z_sig)]
    )
    return state * (amplitude / peak)


# ── Triply-periodic rolls ────────────────────────────────────────


def generate_periodic_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Localized-spot rolls for triply-periodic flow.

    Components `$(u_x, u_y, u_z)$`, axes `$[k_y, k_z, k_x]$`.  Same
    streamfunction `$\psi = G(y)\,\Psi(z)\,X(x)$` as
    :func:`generate_cartesian_rolls`, with the one wall-specific
    ingredient replaced: the shear direction `$y$` is Fourier here, not
    a wall-normal grid, so instead of `$G = (1 - y^2)^2$` (chosen so
    value *and* derivative vanish at the walls) it takes the same
    fixed-physical localization :func:`_envelope` the other two
    homogeneous directions already use, and `$G'$` becomes the exact
    spectral `$\mathrm{i}k_y\hat G$`:

    .. math::
        u_x = 0, \quad
        u_y = \hat G \otimes (\mathrm{i}k_z\hat\Psi) \otimes \hat X,
        \quad
        u_z = -(\mathrm{i}k_y\hat G) \otimes \hat\Psi \otimes \hat X.

    The spot is therefore localized in **all three** directions -- `$y$`
    is homogeneous here, so "localized in every homogeneous direction"
    (module docstring) includes it -- and, since :func:`_envelope`
    centres on the box mid-point while `$L_y$` is fixed at the base
    flow's own period, it sits on a shear extremum of
    `$U = \sin(2\pi y/L_y)$` (the profile's zero, where `$|U'|$` peaks).
    Like the Cartesian generator this puts no perturbation in the
    streamwise component and ignores ``geo.tilt_degree``.

    Two properties are **stronger** than in the wall-bounded case, both
    because `$G'$` is now spectral rather than an analytic derivative
    evaluated against a finite-difference operator:

    - the discrete divergence cancels **analytically**, mode by mode
      (`$\mathrm{i}k_y u_y + \mathrm{i}k_z u_z$` is the same product of
      the same five factors with opposite sign), so it is round-off
      rather than truncation-level;
    - the mean mode is zero **structurally**: `$(k_y, k_z, k_x) =
      (0,0,0)$` is a single bin, `$u_y$` carries `$\mathrm{i}k_z$` and
      `$u_z$` carries `$\mathrm{i}k_y$`, and both vanish there.
      :func:`_mean_free` / :func:`_zero_dc` on the roll factor are kept
      for parity with the other generators (and for the host-side peak),
      but nothing here depends on them.

    Reality is automatic for the same reason as elsewhere: every 1-D
    factor is the forward transform of a *real* signal, and a separable
    product of conjugate-symmetric factors satisfies the periodic
    family's joint `$f(k_y,k_z,0) = \overline{f(-k_y,-k_z,0)}$`
    condition on the `$k_x = 0$` plane (which the random-field generator
    needs ``_periodic_hermitian_raw`` to arrange).  That pairing needs
    **even** mode counts, which ``validate_parameters`` guarantees for
    every Fourier axis (:mod:`dnsjax.harmonics`).
    """
    from jax import numpy as jnp

    from ..geometries.triply_periodic.triply_periodic import ly
    from ..sharding import sharding

    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    lx, lz = params.geo.lx, params.geo.lz

    g_sig = _envelope(ny, width, ly)  # G(y): shear-direction spot
    gp_sig = _ddz(g_sig, ly)  # G'(y) (for the peak)
    x_sig = _envelope(nx, width, lx)  # streamwise localization X(x)
    psi_sig = _roll(nz, wavelength, lz) * _envelope(nz, width, lz)  # Psi(z)
    psi_sig = _mean_free(psi_sig)
    psi_dz = _ddz(psi_sig, lz)  # Psi'(z) (for the peak)

    x_spec = _real_axis_spectrum(x_sig)
    g_spec = _complex_axis_spectrum(g_sig)
    g_dy_spec = 1j * _complex_k(ny, ly) * g_spec  # G'(y), spectral
    psi_spec = _zero_dc(_complex_axis_spectrum(psi_sig))
    psi_dz_spec = 1j * _complex_k(nz, lz) * psi_spec  # Psi'(z), spectral

    u_y = _separable_scalar(g_spec, psi_dz_spec, x_spec)
    u_z = _separable_scalar(-g_dy_spec, psi_spec, x_spec)
    u_x = jnp.zeros_like(u_y)
    state = jnp.stack([u_x, u_y, u_z]).astype(sharding.complex_type)

    peak = _peak_velocity([(g_sig, psi_dz, x_sig), (gp_sig, psi_sig, x_sig)])
    return state * (amplitude / peak)


# ── Dispatch ─────────────────────────────────────────────────────


def generate_localized_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Generate the localized-rolls IC for the configured flow system.

    Dispatches to the geometry-specific generator for
    ``params.phys.system`` and returns the sharded spectral state (on
    ``sharding.spec_vector_shard``), ready to time step -- the same
    object type that ``init_state`` / ``load_snapshot`` return.  The
    perturbation is a fixed-physical localized spot with
    `$\max|\mathbf{u}'| =$` *amplitude*; *width* is the physical
    localization half-width (flow units) and *wavelength* the cross-roll
    spanwise wavelength (ignored by the pipe, whose cross-section is the
    fixed `$m = \pm 1$` mode).  For the total-field Dean flow the
    analytical laminar profile is added to the perturbation (reusing
    :func:`dnsjax.ic.random_field.add_dean_laminar`); every other
    system returns the perturbation directly.  The triply-periodic
    family has no wall-normal direction and takes
    :func:`generate_periodic_rolls`, whose `$y$` factor is a
    localization rather than a wall profile.

    Requires JAX to be configured and the parameter singletons set (the
    geometry ``fourier`` singleton is built lazily by the dispatched
    generator's import).
    """
    system = params.phys.system
    if system in cartesian_systems:
        return generate_cartesian_rolls(amplitude, width, wavelength)
    if system in periodic_systems:
        return generate_periodic_rolls(amplitude, width, wavelength)
    # Rheology before geometry: the viscoelastic systems are members of
    # their geometry's list too (see ``flows.registry``).  Both take a
    # velocity-only rolls perturbation on the total-field IC, so the
    # laminar velocity profile and the laminar conformation are added.
    if system in annular_viscoelastic_systems:
        from .random_field import add_viscoelastic_laminar

        return add_viscoelastic_laminar(
            generate_annular_rolls(amplitude, width, wavelength)
        )
    if system in cylindrical_viscoelastic_systems:
        from .random_field import add_viscoelastic_pipe_laminar

        return add_viscoelastic_pipe_laminar(
            generate_cylindrical_rolls(amplitude, width, wavelength)
        )
    if system in cylindrical_systems:
        return generate_cylindrical_rolls(amplitude, width, wavelength)
    if system in annular_systems:
        state = generate_annular_rolls(amplitude, width, wavelength)
        if system == "dean":
            from .random_field import add_dean_laminar

            state = add_dean_laminar(state)
        return state
    raise ValueError(f"Localized rolls are not defined for system: {system}")
