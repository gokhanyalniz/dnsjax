r"""Log-coordinate moments of a `$(y, k)$` density, and their budget.

JAX-free.  A twin stream stores energies -- and budget rates -- per
discrete wavenumber band and wall-normal node.  Read as a distribution
over `$(\xi, \eta) = (\ln\lambda, \ln y)$`, any such field `$f$` with
fixed cell weights `$w$` (quadrature weights, a premultiplier, the
trapezoidal widths a map is painted with) has a mass, a centroid and a
covariance,

.. math::
    M = \sum w f , \qquad
    \mu_\xi = M^{-1} \sum w f\,\xi , \qquad
    C_{\xi\eta} = M^{-1} \sum w f\,(\xi - \mu_\xi)(\eta - \mu_\eta) ,

and :func:`log_moment_sums` returns the six raw sums
`$\sum w f\,\{1, \xi, \eta, \xi^2, \eta^2, \xi\eta\}$` they are built
from.  Those sums are **linear** in `$f$`, so they can be accumulated
over chunks of records or over ensemble members first and turned into
moments last (:func:`log_moments`): the moments of the ensemble-mean
field, exactly, without ever holding it.  The moments themselves are
not linear, which is why the sums are the unit of accumulation.

What the moments say (:class:`LogMoments`):

- **where**: the centroid `$(\mu_\xi, \mu_\eta)$`;
- **how large**: the spreads `$\sigma_\xi$`, `$\sigma_\eta$` and
  `$\sqrt{\det C}$` -- the one-sigma ellipse has area
  `$\pi\sqrt{\det C}$`;
- **how tilted**: the correlation `$\rho$` and the ridge slope
  `$b = C_{\xi\eta}/C_{\eta\eta}$`.  The latter is also the slope of
  the mass-weighted least-squares line through the conditional mean
  `$\langle\xi \mid \eta\rangle$`, row by row, so `$b = 1$` is
  `$\lambda \propto y$` along the ridge.  The principal-axis angle is
  kept for drawing the ellipse only: it is ill-conditioned when the
  ellipse is nearly round (`$C_{\xi\xi} \approx C_{\eta\eta}$` and
  `$\rho$` small), where `$\rho$` and `$b$` are not.

Every constant factor of `$f$` cancels from every moment, each being a
ratio over the mass; a coordinate-dependent factor of the weights does
not -- it is what decides the weighting.

Their budget (:func:`moment_rates`).  If `$\partial_t f = \sum_B B$`
with the weights fixed in time, then for any function `$\varphi$` of
the cell

.. math::
    \frac{\mathrm{d}\langle\varphi\rangle}{\mathrm{d}t}
      = M^{-1} \sum w\,(\varphi - \langle\varphi\rangle)\,
        \partial_t f ,

so every moment's rate splits term by term: `$\mathrm{d}\ln M /
\mathrm{d}t$` from `$\sum w B / M$`, the centroid from `$\varphi = \xi,
\eta$`, a variance from `$\varphi = (\xi - \mu_\xi)^2$` less
`$C_{\xi\xi}$`, and the covariance likewise.  The identity holds for
**any** fixed weights, so a budget closes on whichever density the
caller chose -- the energy itself, or a map's painted density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The raw sums :func:`log_moment_sums` stacks, in order: the mass and
#: the first and second moments in `$\xi = \ln\lambda$` and
#: `$\eta = \ln y$`.
SUM_NAMES: tuple[str, ...] = ("m", "x", "y", "xx", "yy", "xy")


def log_moment_sums(
    field: np.ndarray,
    weights: np.ndarray,
    ln_lam: np.ndarray,
    ln_y: np.ndarray,
) -> np.ndarray:
    r"""The six raw sums of *field* over its trailing `$(n_y, n_k)$` axes.

    ``(..., 6)``, in :data:`SUM_NAMES` order: `$\sum w f\,\{1, \xi,
    \eta, \xi^2, \eta^2, \xi\eta\}$`, with `$\xi$` = *ln_lam* along the
    last axis and `$\eta$` = *ln_y* along the one before it.  *weights*
    broadcasts against those two axes.  A non-finite cell contributes
    nothing, as it is left unpainted on a map.

    Linear in *field*: a sum of these over chunks or members is the
    sums of the summed field (module docstring).
    """
    wf = np.asarray(field, dtype=np.float64) * np.asarray(
        weights, dtype=np.float64
    )
    wf = np.where(np.isfinite(wf), wf, 0.0)
    x = np.asarray(ln_lam, dtype=np.float64)[None, :]
    y = np.asarray(ln_y, dtype=np.float64)[:, None]
    parts = (wf, wf * x, wf * y, wf * x * x, wf * y * y, wf * x * y)
    return np.stack([p.sum(axis=(-2, -1)) for p in parts], axis=-1)


@dataclass(frozen=True)
class LogMoments:
    r"""Mass, centroid and covariance of a density in `$(\ln\lambda, \ln y)$`.

    All in natural-log units, leading axes as the sums had them; divide
    a spread by `$\ln 10$` for decades.  ``nan`` where the mass is not
    positive.
    """

    mass: np.ndarray
    mean_lam: np.ndarray
    mean_y: np.ndarray
    var_lam: np.ndarray
    var_y: np.ndarray
    cov: np.ndarray

    @property
    def sd_lam(self) -> np.ndarray:
        r"""`$\sigma_\xi$`."""
        return np.sqrt(np.maximum(self.var_lam, 0.0))

    @property
    def sd_y(self) -> np.ndarray:
        r"""`$\sigma_\eta$`."""
        return np.sqrt(np.maximum(self.var_y, 0.0))

    @property
    def sqrt_det(self) -> np.ndarray:
        r"""`$\sqrt{\det C}$`: the one-sigma ellipse's area over `$\pi$`."""
        return np.sqrt(
            np.maximum(self.var_lam * self.var_y - self.cov**2, 0.0)
        )

    @property
    def rho(self) -> np.ndarray:
        r"""The correlation `$C_{\xi\eta}/(\sigma_\xi\sigma_\eta)$`."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.cov / (self.sd_lam * self.sd_y)

    @property
    def slope(self) -> np.ndarray:
        r"""The ridge slope `$b = C_{\xi\eta}/C_{\eta\eta}$`."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.cov / self.var_y

    @property
    def angle(self) -> np.ndarray:
        r"""The major axis's angle from the `$\xi$` axis, in radians.

        For drawing only (module docstring): it swings freely when the
        ellipse is nearly round.
        """
        return 0.5 * np.arctan2(2.0 * self.cov, self.var_lam - self.var_y)


def log_moments(sums: np.ndarray) -> LogMoments:
    """The moments the raw *sums* (:func:`log_moment_sums`) describe."""
    s = np.asarray(sums, dtype=np.float64)
    mass = s[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(mass > 0.0, 1.0 / mass, np.nan)
        mean_lam = s[..., 1] * inv
        mean_y = s[..., 2] * inv
        return LogMoments(
            mass=mass,
            mean_lam=mean_lam,
            mean_y=mean_y,
            var_lam=s[..., 3] * inv - mean_lam**2,
            var_y=s[..., 4] * inv - mean_y**2,
            cov=s[..., 5] * inv - mean_lam * mean_y,
        )


@dataclass(frozen=True)
class MomentRates:
    r"""One term's contribution to the rate of every moment.

    ``mass`` is to `$\mathrm{d}\ln M/\mathrm{d}t$`; the rest to the
    rates of the :class:`LogMoments` fields of the same names.  Linear
    in the term, so contributions add up to the rate of the summed
    term.
    """

    mass: np.ndarray
    mean_lam: np.ndarray
    mean_y: np.ndarray
    var_lam: np.ndarray
    var_y: np.ndarray
    cov: np.ndarray


def moment_rates(
    energy_sums: np.ndarray, term_sums: np.ndarray
) -> MomentRates:
    r"""What one term does to the moments of the distribution it changes.

    *energy_sums* are the raw sums of the density whose moments move,
    and *term_sums* those of one term of its rate of change, under the
    **same** weights and coordinates -- the rate identity of the module
    docstring with `$\varphi$` each of the moments in turn:

    .. math::
        \dot\mu_\xi = M^{-1}\textstyle\sum w B\,(\xi - \mu_\xi) , \qquad
        \dot C_{\xi\eta} = M^{-1}\textstyle\sum w B\,
          [(\xi - \mu_\xi)(\eta - \mu_\eta) - C_{\xi\eta}] .

    Written in terms of the raw sums, so it shares their leading axes.
    """
    e = log_moments(energy_sums)
    b = np.asarray(term_sums, dtype=np.float64)
    b0, bx, by, bxx, byy, bxy = (b[..., i] for i in range(6))
    mx, my = e.mean_lam, e.mean_y
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(e.mass > 0.0, 1.0 / e.mass, np.nan)
        return MomentRates(
            mass=b0 * inv,
            mean_lam=(bx - mx * b0) * inv,
            mean_y=(by - my * b0) * inv,
            var_lam=(bxx - 2.0 * mx * bx + (mx**2 - e.var_lam) * b0) * inv,
            var_y=(byy - 2.0 * my * by + (my**2 - e.var_y) * b0) * inv,
            cov=(bxy - mx * by - my * bx + (mx * my - e.cov) * b0) * inv,
        )
