r"""The curved pipe's azimuthal harmonics (JAX-free).

The dealiasing pad of the toroidal metric's `$1/h^2$`
(:func:`curvature_dealiasing_pad`), which the curved-pipe spec's
padding hook applies before the core's mesh and FFT rounding.
"""

import math


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
    :func:`dnsjax.harmonics.inverse_metric_harmonics` (``power=2``) at
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
