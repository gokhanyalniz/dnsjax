r"""Integer wavenumber sequences for the spectral axes (JAX-free).

These NumPy generators are the single source of truth for the Fourier
mode numbering used throughout the solver and by the JAX-free analysis
tooling (:mod:`dnsjax.analysis`).  :mod:`dnsjax.operators` re-exports
them wrapped in ``jnp.asarray`` so the runtime keeps device arrays,
while host-side / external code (which must run without JAX) imports
the NumPy versions directly.  :func:`parse_mode_pairs`, the shared
parser of ``"i2,i3;..."`` spectral-mode lists (the probes / forcing /
transient-growth surfaces), also lives here, as does
:func:`inverse_metric_harmonics`, the curved pipe's `$1/h$` in
azimuthal harmonics, which the solver's flux read and the analysis
package's toroidal operators share.

The conventions match the storage layout: the Nyquist mode is always
omitted, so a real-FFT axis carries `$n / 2$` modes and a full-complex
axis carries `$n - 1$` modes (see :mod:`dnsjax.fft`).

**`$n$` is even, and the generators enforce it.**  A Nyquist mode
exists only at an even mode count -- it is the single self-aliasing slot
holding `$+n/2 = -n/2$`.  At an odd `$n$` the frequencies are
`$0, \pm 1, \dots, \pm(n-1)/2$`, all in conjugate pairs bar the mean,
so there is nothing to drop and omitting a slot anyway would remove a
genuine harmonic, stranding its partner on a complex axis.
``parameters.validate_parameters`` refuses an odd Fourier count at the
parameter layer; these generators refuse one too, so a count
mis-derived anywhere fails loudly instead of silently describing a
layout the solver never produces.

Odd counts are refused rather than supported because they would buy
at most one mode, and the least trustworthy one.  On a complex axis
the stranded partner sits in a slot that is stored, sharded, padded and
solved for but cannot hold physical content: a unit coefficient there
survives one spectral-physical round trip at half amplitude, so it is
silently damped on every nonlinear evaluation.  On the real-FFT axis
the top wavenumber is merely lost (``nx = 9`` resolves no more than
``nx = 8`` while paying for a larger padded grid).  An even count is
what every other constraint wants anyway: an integral 3/2 dealiasing
size, FFT-friendly padded sizes, mesh divisibility and power-of-two
Pallas tiles.
"""

import math

import numpy as np
from numpy import ndarray


def _require_even(n: int, who: str) -> None:
    """Reject an odd Fourier mode count (see the module docstring)."""
    if n % 2:
        raise ValueError(
            f"{who}: the mode count must be even, got {n}.  At an odd "
            "count there is no Nyquist mode to omit, so this layout "
            "would drop a genuine harmonic and strand its conjugate "
            "partner; parameters.validate_parameters refuses one."
        )


def real_harmonics(n: int) -> ndarray:
    r"""Non-negative integer wavenumbers for a real-FFT axis.

    The Nyquist mode is omitted, leaving `$n / 2$` modes.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        Wavenumber array `$[0, 1, \dots, n/2 - 1]$`, shape
        ``(n // 2,)``.

    Raises
    ------
    ValueError
        If *n* is odd (see the module docstring).
    """
    _require_even(n, "real_harmonics")
    # Omits the Nyquist mode
    return np.arange(0, n // 2, dtype=int)


def parse_mode_pairs(spec: str) -> list[tuple[int, int]]:
    r"""Parse an ``"i2,i3;i2,i3;..."`` spectral-mode list.

    Each pair is a global spectral index: ``i2`` on the complex
    (axis-2) slot and ``i3`` on the real-FFT (axis-3) slot of the
    stored spectral layout -- the same convention as the
    transient-growth CLI ``--modes`` argument.  Whitespace around
    numbers and separators is ignored.  Purely syntactic (this module
    is a JAX-free leaf): no range check against a resolution --
    callers validate bounds themselves.

    Raises ``ValueError`` on malformed pairs, negative indices, or
    duplicates.
    """
    pairs: list[tuple[int, int]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            raise ValueError(
                f"empty mode entry in {spec!r} (expected 'i2,i3;i2,i3')"
            )
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"malformed mode {item!r} in {spec!r} (expected 'i2,i3')"
            )
        try:
            i2, i3 = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"non-integer mode {item!r} in {spec!r}"
            ) from None
        if i2 < 0 or i3 < 0:
            raise ValueError(f"negative mode index in {item!r}")
        if (i2, i3) in pairs:
            raise ValueError(f"duplicate mode ({i2},{i3}) in {spec!r}")
        pairs.append((i2, i3))
    return pairs


def stored_mode_counts(m: int) -> tuple[int, int]:
    r"""Split a stored full-complex axis into its `$\pm$` blocks.

    A stored full-complex axis of ``m = n - 1`` modes is in FFT wrap
    order (:func:`complex_harmonics`): ``n_pos`` non-negative
    wavenumbers `$[0, \dots, n/2-1]$` first, then ``n_neg`` negative
    ones `$[-n/2+1, \dots, -1]$`.

    This is the arithmetic a resolution change needs: growing an axis
    **inserts** zeros between the two blocks (the high-`$|k|$` end),
    shrinking drops the outermost modes of each -- symmetrically, so
    the `$k_x = 0$` plane's Hermitian pairing survives.  Appending or
    truncating at the array end, correct for a real-FFT axis, would
    corrupt the negative block here.

    Parameters
    ----------
    m:
        Stored mode count along the axis (``n - 1``).

    Returns
    -------
    :
        ``(n_pos, n_neg)``, summing to *m*.
    """
    n_pos = (m + 1) // 2
    return n_pos, m - n_pos


def complex_harmonics(n: int) -> ndarray:
    r"""Full-complex integer wavenumbers with the Nyquist mode omitted.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        `$n - 1$` wavenumbers in FFT order:
        `$[0, 1, \dots, n/2-1, -n/2+1, \dots, -1]$`.

    Raises
    ------
    ValueError
        If *n* is odd (see the module docstring).
    """
    _require_even(n, "complex_harmonics")
    qs = (np.arange(n, dtype=int) + n // 2) % n - n // 2
    # Omits the Nyquist mode
    qs_out = np.zeros(n - 1, dtype=int)
    qs_out[: n // 2] = qs[: n // 2]
    qs_out[n // 2 :] = qs[n // 2 + 1 :]
    return qs_out


def inverse_metric_harmonics(
    kappa: float, rs: ndarray, m_vals: ndarray, power: int = 1
) -> ndarray:
    r"""Exact azimuthal harmonics of the curved pipe's `$h^{-p}$`.

    `$h = 1 + \kappa r\cos\theta$` is the toroidal metric
    (:mod:`dnsjax.geometries.wall_bounded.cylindrical_curved`).  With
    `$\epsilon = \kappa r$`,

    .. math::
        \frac{1}{1 + \epsilon\cos\theta}
        = \frac{1}{\sqrt{1-\epsilon^2}}
          \Big[1 + 2\sum_{n\ge1} q^n \cos n\theta\Big], \qquad
        q = \frac{-\epsilon}{1 + \sqrt{1-\epsilon^2}},

    so the complex coefficients of `$1/h$` are
    `$c_m = q^{|m|}/\sqrt{1-\epsilon^2}$`.  Those of `$1/h^2$` follow
    by differentiating `$1/(\lambda + \epsilon\cos\theta)$` in
    `$\lambda$` at `$\lambda = 1$`:
    `$q^{|m|}\,(1 + |m|\sqrt{1-\epsilon^2})/(1-\epsilon^2)^{3/2}$`.
    Both are real, even in `$m$` and `$O(r^{|m|})$` at the axis, hence
    in the `$(-1)^m$` parity class like the fields they weight, and
    exact to machine precision (``tests/test_curved_pipe.py`` checks
    them against an FFT).

    Parameters
    ----------
    kappa:
        Curvature `$\kappa = a/R_c$`, below 1.
    rs:
        Radii, shape ``(n_r,)``.
    m_vals:
        Azimuthal wavenumbers.
    power:
        `$p$`, 1 or 2.

    Returns
    -------
    :
        Shape ``(len(rs), len(m_vals))``.
    """
    if power not in (1, 2):
        raise ValueError(f"power must be 1 or 2, got {power}.")
    eps = kappa * np.asarray(rs)[:, None]
    root = np.sqrt(1.0 - eps**2)
    q = -eps / (1.0 + root)
    abs_m = np.abs(np.asarray(m_vals))[None, :]
    if power == 1:
        return q**abs_m / root
    return q**abs_m * (1.0 + abs_m * root) / root**3


def curvature_dealiasing_pad(kappa: float, tol: float) -> int:
    r"""Azimuthal points beyond `$3M$` that dealias `$1/h^2$` to *tol*.

    The curved pipe's nonlinear term carries quadratic products over
    `$h^2$` (the
    :mod:`~dnsjax.geometries.wall_bounded.cylindrical_curved` module
    docstring, "Dealiasing").  On a collocation grid of
    `$N = 3M + J$` azimuthal points, a product `$P$` with harmonics
    `$|n| \le 2M$` times `$1/h^2$` folds back onto the stored
    `$|m| \le M$` only through harmonics `$|j| \ge J$` of `$1/h^2$`, so
    the aliased coefficients obey

    .. math::
        \|e\|_1 \le 2 \sum_{j \ge J} |c^{(2)}_j|\;\|\hat P\|_1 ,

    with `$c^{(2)}_j$` the harmonics of
    :func:`inverse_metric_harmonics` (``power=2``) at the wall, where
    `$\epsilon = \kappa$` is largest.  The bound holds for any `$P$`,
    resolved or not, which is why `$J$` depends on `$\kappa$` and *tol*
    alone.  This returns the smallest `$J$` whose bound, relative to
    `$c^{(2)}_0$`, is at most *tol*; the tail sums in closed form,
    `$2 x^J [1/(1-x) + s\,(J - (J-1)x)/(1-x)^2]$` with
    `$x = |q|$` and `$s = \sqrt{1-\kappa^2}$`.

    Parameters
    ----------
    kappa:
        Curvature `$\kappa = a/R_c$`, in `$[0, 1)$`.
    tol:
        Target relative aliasing, e.g. the working precision's unit
        roundoff.

    Returns
    -------
    :
        `$J \ge 1$`.
    """
    if not 0.0 <= kappa < 1.0:
        raise ValueError(f"kappa must lie in [0, 1), got {kappa}.")
    s = math.sqrt(1.0 - kappa**2)
    x = kappa / (1.0 + s)

    def bound(j: int) -> float:
        geometric = 1.0 / (1.0 - x)
        linear = s * (j - (j - 1) * x) / (1.0 - x) ** 2
        return 2.0 * x**j * (geometric + linear)

    j = 1
    while bound(j) > tol:
        j += 1
    return j
