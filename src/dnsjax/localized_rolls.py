r"""Localized-roll ("turbulent spot") initial-condition generators.

Builds a deterministic divergence-free **localized** perturbation of the
base flow for every wall-bounded flow system, returned as a sharded
spectral state ready to time step.  This is the implementation behind the
in-process ``init.localized_rolls`` start mode (``dnsjax.__main__``);
there is no offline script (unlike the random field).

**Fixed-physical structure (a spot, surrounded by laminar flow).** The
perturbation is a compact structure of *fixed physical size*, localized
in **every homogeneous direction**, peak-normalized so that
`$\max|\mathbf{u}'| = A$` (the ``amplitude`` argument) at any domain
size.  Growing a box length therefore just adds laminar flow around the
spot, so the volume-averaged statistics scale as `$1 / (L_x L_z)$` -- the
total perturbation energy is domain-independent.  This replaces an
earlier construction whose streamwise width was a *fraction* of the box
and whose single cross-plane Fourier mode made the cross-stream velocity
grow `$\propto L$` (the `$1/\beta$` streamfunction prefactor with
`$\beta = 2\pi / L$`), which both broke the scaling and blew the flow up
in large domains.

**Construction (Cartesian; `$x$` streamwise, `$z$` spanwise, `$y$`
wall-normal).** With zero streamwise velocity and a `$y$`-`$z$`
streamfunction `$\psi = G(y)\,\Psi(z)\,X(x)$`,

.. math::
    u_x = 0, \quad
    u_y = G(y)\,\Psi'(z)\,X(x), \quad
    u_z = -G'(y)\,\Psi(z)\,X(x),

which is divergence-free for **any** profiles (`$u_x = 0$` lets `$X$`
factor out of the divergence, and the `$y$`-`$z$` pair is a
streamfunction).  `$G = (1 - y^2)^2$` (peak 1, value + derivative zero at
both walls).  `$X(x)$` is a fixed-physical-width streamwise localization
and `$\Psi(z)$` a spanwise roll (wavelength `$\lambda$`) under a
fixed-physical-width envelope, so the spot is localized in both `$x$` and
`$z$`.  The spanwise derivative `$\Psi'$` is built **spectrally** as
`$\mathrm{i}k_z\,\hat\Psi$` so the discrete divergence is truncation-level
(projected out by the first corrector step), exactly as before.  Pipe and
annular use the analogous per-geometry streamfunction (see the
per-generator docstrings); the roll excites only the `$\pm 1$` spanwise
mode for the pipe (its azimuthal cross-section), so the perturbation is
**mean-free** for every flow (the Dean laminar profile is the only
mean-mode content, added separately).

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
replication-free).  No-slip is **exact** (the wall slice of every profile
is identically zero, never transformed).

**Import-order discipline**: only NumPy and the (JAX-free)
``parameters`` singletons are imported at module top; ``jax``,
``sharding``, and the geometry modules are imported lazily inside each
generator, so importing this module is safe before JAX is configured.
Wavenumber arrays are recomputed host-side (never fetched from the
multi-device ``fourier`` singleton).
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

from .parameters import (
    annular_systems,
    cartesian_systems,
    cylindrical_systems,
    derived_params,
    params,
)

if TYPE_CHECKING:
    # ``Array`` is used only in (stringised) annotations, so it never
    # needs importing at runtime -- keeping this module importable
    # before JAX is configured (see the module docstring).
    from jax import Array

__all__ = [
    "generate_localized_rolls",
    "generate_cartesian_rolls",
    "generate_cylindrical_rolls",
    "generate_annular_rolls",
]


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


def _sin_signal(n: int) -> np.ndarray:
    r"""One full sine wave `$\sin(2\pi j / n)$`, length n (azimuthal m=1)."""
    return np.sin(2 * pi * np.arange(n) / n)


def _cos_signal(n: int) -> np.ndarray:
    r"""One full cosine wave `$\cos(2\pi j / n)$`, length n (azimuthal m=1)."""
    return np.cos(2 * pi * np.arange(n) / n)


def _complex_harmonics(n: int) -> np.ndarray:
    """Host :func:`operators.complex_harmonics`: FFT order
    ``[0, .., n//2-1, -n//2+1, .., -1]`` with the Nyquist mode dropped."""
    qs = (np.arange(n, dtype=int) + n // 2) % n - n // 2
    return np.concatenate([qs[: n // 2], qs[n // 2 + 1 :]])


def _complex_k(n: int, L: float) -> np.ndarray:
    """Host complex-axis wavenumbers, true length ``n - 1``."""
    return _complex_harmonics(n) * (2.0 * pi / L)


def _real_k(n: int, L: float) -> np.ndarray:
    """Host real-axis wavenumbers ``[0, .., n//2-1] * 2 pi / L``."""
    return np.arange(0, n // 2, dtype=int) * (2.0 * pi / L)


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
    `$u_r, u_\theta, u_z$` cyl/annular).  The component physical field is
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
    ``spec_scalar_shard`` via the dnsjax broadcast-of-sharded-factors
    idiom (cf. ``Fourier.k2`` / ``mean_mask``): the complex-FFT-axis
    factor is placed on the ``np0`` mesh axis and the real-FFT-axis
    factor on ``np1``, so the broadcast product is sharded and **no full
    array is materialised**.  The 1-D spectra are zero-padded to the
    mesh-padded mode counts (``nz_spec`` / ``nx_spec``), so padding modes
    stay zero.

    Parameters
    ----------
    prof:
        Real wall-normal profile, shape ``(Ny,)`` (or ``(Nr,)``).
    complex_spectrum:
        Complex-FFT-axis (`$k_z$` / `$m$`, ``np0``) spectrum, true
        length ``nz - 1``.
    real_spectrum:
        Real-FFT-axis (`$k_x$` / `$k_{z,\mathrm{ax}}$`, ``np1``)
        spectrum, true length ``nx // 2``.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    from .sharding import sharding

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

    from .geometries.wall_bounded.cartesian import build_cartesian_grid
    from .sharding import sharding

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
    psi_dz = _ddz(psi_sig, lz)  # Psi'(z) (for the peak)

    x_spec = _real_axis_spectrum(x_sig)
    psi_spec = _complex_axis_spectrum(psi_sig)
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

    Components `$(u_z, u_+, u_-)$` with `$u_\pm = u_r \pm i u_\theta$`,
    axes `$[r, m, k_z]$`.  The azimuthal cross-section is the `$m = \pm 1$`
    roll pair (filling the pipe, fixed `$2\pi$` -- so no spanwise
    blow-up); axial `$z$` (real `$k_z$` axis) carries a fixed-width
    localization `$X(z)$`.  The reference cross-plane roll
    (`$g = (1 - r^2)^2$`):

    .. math::
        u_r &= g \otimes \sin(m) \otimes \hat X \\
        u_\theta &= (g + r g') \otimes \cos(m) \otimes \hat X \\
        u_z &= 0

    with `$g + r g' = (1 - r^2)(1 - 5 r^2)$`.  Both radial profiles vanish
    at the wall `$r = 1$`, so `$u_\pm$` is zero there; the field is
    peak-normalized so `$\max|\mathbf{u}'| = A$`.  ``wavelength`` is
    ignored (the azimuthal structure is the fixed `$m = \pm 1$` mode).
    """
    from jax import numpy as jnp

    from .geometries.wall_bounded.cylindrical import build_cylindrical_grid
    from .sharding import sharding

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

    u_r = _separable_scalar(g, _complex_axis_spectrum(sin_m), x_spec)
    u_theta = _separable_scalar(g_r, _complex_axis_spectrum(cos_m), x_spec)
    u_plus = u_r + 1j * u_theta
    u_minus = u_r - 1j * u_theta
    u_z = jnp.zeros_like(u_r)
    state = jnp.stack([u_z, u_plus, u_minus]).astype(sharding.complex_type)

    peak = _peak_velocity([(g, sin_m, x_sig), (g_r, cos_m, x_sig)])
    return state * (amplitude / peak)


# ── Annular (Taylor-Couette / Dean) rolls ────────────────────────


def generate_annular_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    r"""Localized-spot rolls for Taylor-Couette / Dean flow.

    Components `$(u_z, u_+, u_-)$` with `$u_\pm = u_r \pm i u_\theta$`,
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
    `$r = r_1, r_2$`.  Since `$u_\theta = 0$`, `$u_+ = u_- = u_r$`; the
    field is peak-normalized so `$\max|\mathbf{u}'| = A$`.
    """
    from jax import numpy as jnp

    from .geometries.wall_bounded.annular import build_annular_grid
    from .sharding import sharding

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
    az_sig = _envelope(nz, width / r_mid, 2.0 * pi)  # azimuthal localization
    z_sig = _roll(nx, wavelength, lx) * _envelope(nx, width, lx)  # Z(z_ax)
    z_dz = _ddz(z_sig, lx)  # Z'(z_ax) (for the peak)

    az_spec = _complex_axis_spectrum(az_sig)
    z_spec = _real_axis_spectrum(z_sig)
    z_dz_spec = 1j * _real_k(nx, lx) * z_spec  # spectral derivative

    u_r = _separable_scalar(-p_over_r, az_spec, z_dz_spec)
    u_z = _separable_scalar(pp_over_r, az_spec, z_spec)
    state = jnp.stack([u_z, u_r, u_r]).astype(sharding.complex_type)

    peak = _peak_velocity(
        [(p_over_r, az_sig, z_dz), (pp_over_r, az_sig, z_sig)]
    )
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
    :func:`dnsjax.random_field.add_dean_laminar`); every other system
    returns the perturbation directly.  Defined for wall-bounded systems
    only.

    Requires JAX to be configured and the parameter singletons set (the
    geometry ``fourier`` singleton is built lazily by the dispatched
    generator's import).
    """
    system = params.phys.system
    if system in cartesian_systems:
        return generate_cartesian_rolls(amplitude, width, wavelength)
    if system in cylindrical_systems:
        return generate_cylindrical_rolls(amplitude, width, wavelength)
    if system in annular_systems:
        state = generate_annular_rolls(amplitude, width, wavelength)
        if system == "dean":
            from .random_field import add_dean_laminar

            state = add_dean_laminar(state)
        return state
    raise ValueError(f"Localized rolls are not defined for system: {system}")
