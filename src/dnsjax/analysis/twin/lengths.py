r"""Integral length scales of the difference field (JAX-free).

The paper's fig. 10 diagnostic (Egerique-de-la-Concha & Hwang, *J.
Fluid Mech.* **1036**, A52, 2026): the wall-normal and spanwise
integral length scales

.. math::
    l = \int_0^{r_0} f(r)\, \mathrm{d}r,
    \qquad f(r) = C(r) / C(0),

of the streak difference field `$\Delta u_1$` (the `$k_x = 0$`,
`$k_z \ne 0$` modes of `$\Delta\mathbf{u}$`), per velocity
component, evaluated at a wall-normal anchor (the channel centre in
the paper).  Their saturation at the geometry-permitted scale marks
the onset of the linear-growth phase.

`$r_0$` is the **first zero crossing** of `$f$` (linearly
interpolated), falling back to the domain edge when `$f$` stays
positive.  On a periodic axis the crossing is unavoidable and the
convention necessary: a zero-mean periodic signal's autocorrelation
integrates to *exactly zero* over the half period (every harmonic
`$\int_0^{L_z/2}\cos(2\pi m r/L_z)\,\mathrm{d}r = 0$`), so the
paper's `$\int_0^\infty$` can only mean the decaying-part integral.

Correlations come straight from the stored spectra of a snapshot
*pair* (``state{isnap}.tar`` + ``state{isnap}_twin.tar``), with the
`$z$` average as the ensemble:

- spanwise: `$C_z(r) = \sum_m |\hat{u}_m(y_0)|^2 \cos(k_m r)$`
  (Wiener-Khinchin on the periodic axis);
- wall-normal: `$C_y(y_0, y_j) = \sum_m \mathrm{Re}[\hat{u}_m(y_0)\,
  \hat{u}_m^*(y_j)]$`, one integral toward each wall on the
  nonuniform grid, the two sides averaged.

:func:`integral_lengths_from_modes` is the pure-NumPy core (unit
tested against hand-built spectra); :func:`integral_lengths` wraps it
for a snapshot pair.  Cartesian wall-bounded snapshots only (the
stored layout is ``(y, k_z, k_x)`` per component and the base flow
cancels in the pair difference).

Assemble the pair with :func:`partner_of` rather than by hand: the
difference field is meaningful only between two snapshots of the *same*
lockstep write, and :func:`integral_lengths` enforces that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import read_state
from .._core import CARTESIAN_SYSTEMS, read_meta


def partner_of(reference: str | Path) -> Path:
    """The twin partner's path for a reference ``state*.tar``.

    Mirrors ``dnsjax.twin.driver._partner_path`` -- the naming rule the
    driver writes with -- so callers do not hand-assemble the pair.
    The file is not required to exist.
    """
    reference = Path(reference)
    return reference.with_name(f"{reference.stem}_twin{reference.suffix}")


def _integrate_to_first_zero(f: np.ndarray, r: np.ndarray) -> float:
    """Trapezoid integral of ``f(r)`` up to its first zero crossing.

    ``f[0]`` must be 1 (the normalised correlation); the crossing is
    linearly interpolated between the bracketing samples.  With no
    crossing the integral runs to ``r[-1]`` (the domain edge).
    """
    below = np.nonzero(f <= 0.0)[0]
    if below.size == 0:
        return float(np.trapezoid(f, r))
    j = int(below[0])
    if j == 0:
        return 0.0
    # Linear interpolation of the crossing inside [r[j-1], r[j]].
    r0 = r[j - 1] + (r[j] - r[j - 1]) * f[j - 1] / (f[j - 1] - f[j])
    partial = float(np.trapezoid(f[:j], r[:j]))
    return partial + 0.5 * f[j - 1] * (r0 - r[j - 1])


def integral_lengths_from_modes(
    du1: np.ndarray,
    y: np.ndarray,
    kz: np.ndarray,
    lz: float,
    y0: float = 0.0,
    n_r: int = 512,
) -> dict:
    """Integral lengths from the streak modes (the pure-NumPy core).

    *du1* is ``(3, ny, n_m)`` complex -- the `$k_x = 0$` column of
    the difference field with the mean mode dropped -- with *kz* the
    ``(n_m,)`` physical wavenumbers and *y* the wall-normal grid.
    Returns ``{"y0", "l_y" (3,), "l_z" (3,), "variance" (3,)}``;
    ``variance`` is the anchor-height spanwise variance `$C(0)$`
    (zero variance yields ``nan`` lengths).
    """
    j0 = int(np.argmin(np.abs(np.asarray(y) - y0)))
    r = np.linspace(0.0, lz / 2.0, n_r)
    l_y = np.full(3, np.nan)
    l_z = np.full(3, np.nan)
    variance = np.zeros(3)
    for c in range(3):
        power = np.abs(du1[c, j0]) ** 2
        c0 = float(power.sum())
        variance[c] = c0
        if c0 <= 0.0:
            continue
        f_z = (power[None, :] * np.cos(np.outer(r, kz))).sum(axis=1) / c0
        l_z[c] = _integrate_to_first_zero(f_z, r)

        c_y = np.real(du1[c] @ np.conj(du1[c, j0]))
        f_y = c_y / c_y[j0]
        sides = []
        for sel in (slice(j0, None), slice(j0, None, -1)):
            rr = np.abs(np.asarray(y)[sel] - y[j0])
            if rr.shape[0] >= 2:
                sides.append(_integrate_to_first_zero(f_y[sel], rr))
        l_y[c] = float(np.mean(sides))
    return {
        "y0": float(y[j0]),
        "l_y": l_y,
        "l_z": l_z,
        "variance": variance,
    }


def integral_lengths(
    reference: str | Path,
    partner: str | Path,
    y0: float = 0.0,
) -> dict:
    r"""Integral lengths of `$\Delta u_1$` for a snapshot pair.

    Reads both snapshots' stored spectra, forms the difference's
    `$k_x = 0$` column (mean mode dropped), and dispatches to
    :func:`integral_lengths_from_modes` at the anchor ``y0``.

    The two snapshots must be the same lockstep write -- same system
    and same `$(t, \mathrm{it})$`, as ``dnsjax-twin`` itself requires
    of a resumed pair.  Differencing across times subtracts two
    unrelated states: the result looks like a difference field and the
    lengths that follow are noise, so the mismatch raises rather than
    returning a plausible number.  :func:`partner_of` gives the
    partner path for a reference snapshot.
    """
    t_ref, t_twin = read_meta(reference), read_meta(partner)
    if (t_ref["t"], t_ref["it"]) != (t_twin["t"], t_twin["it"]):
        raise ValueError(
            f"the pair is not at the same time: {Path(reference).name} "
            f"is at (t={t_ref['t']}, it={t_ref['it']}) but "
            f"{Path(partner).name} is at (t={t_twin['t']}, "
            f"it={t_twin['it']}); pair a reference with its own "
            "partner (see partner_of)."
        )
    ref = read_state(reference, return_physical=False, return_spectral=True)
    twin = read_state(partner, return_physical=False, return_spectral=True)
    system = ref.params.phys.system
    if system not in CARTESIAN_SYSTEMS:
        raise ValueError(
            "integral_lengths supports the Cartesian wall-bounded "
            f"snapshots only (system {system!r})."
        )
    if twin.params.phys.system != system:
        raise ValueError("the pair stores different systems.")
    delta = np.stack(
        [t - r for r, t in zip(ref.spectral, twin.spectral, strict=True)]
    )
    y, kz, _ = ref.spectral_coords
    # k_x = 0 column, mean mode (kz index 0) excluded.
    return integral_lengths_from_modes(
        delta[:, :, 1:, 0],
        np.asarray(y),
        np.asarray(kz)[1:],
        float(ref.params.geo.lz),
        y0=y0,
    )
