r"""The curved pipe's azimuthal harmonics (JAX-free).

The exact harmonics of the toroidal metric's `$1/h$` and `$1/h^2$`
(:func:`inverse_metric_harmonics`), which the solver's flux read and
the analysis package's toroidal metric share, and the dealiasing pad of
`$1/h^2$` (:func:`curvature_dealiasing_pad`), which the curved-pipe
spec's padding hook applies before the core's mesh and FFT rounding.
"""

import math

import numpy as np
from numpy import ndarray


def inverse_metric_harmonics(
    kappa: float, rs: ndarray, m_vals: ndarray, power: int = 1
) -> ndarray:
    r"""Exact azimuthal harmonics of the curved pipe's `$h^{-p}$`.

    `$h = 1 + \kappa r\cos\theta$` is the toroidal metric
    (:mod:`dnsjax_lab.geometries.wall_bounded.cylindrical_curved`).  With
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
    exact to machine precision (``lab/tests/test_curved_pipe.py`` checks
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
    :mod:`~dnsjax_lab.geometries.wall_bounded.cylindrical_curved` module
    docstring, "Dealiasing").  On a collocation grid of
    `$N = 3M + J$` azimuthal points, a product `$P$` with harmonics
    `$|n| \le 2M$` times `$1/h^2$` folds back onto the stored
    `$|m| \le M$` only through harmonics `$|j| \ge J$` of `$1/h^2$`, so
    the aliased coefficients obey

    .. math::
        \|e\|_1 \le 2 \sum_{j \ge J} |c^{(2)}_j|\;\|\hat P\|_1 ,

    with `$c^{(2)}_j$` the harmonics of
    :func:`inverse_metric_harmonics` (``power=2``) at
    the wall, where
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
