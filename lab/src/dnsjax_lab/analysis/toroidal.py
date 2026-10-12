r"""The curved pipe's toroidal metric, for the snapshot analysis (JAX-free).

The curved pipe's streamwise direction is the centreline arclength
`$s$` of a pipe bent into a torus of curvature `$\kappa$`: its scale
factors along `$(r, \theta, s)$` are `$(1, r, h)$`, with
`$h = 1 + \kappa r\cos\theta$`.  The curved-pipe spec names
:func:`metric` as its ``analysis_metric``, so :mod:`dnsjax.analysis`
applies :class:`ToroidalMetric` as the snapshot's axial scale factor
(:class:`dnsjax.analysis.AxialMetric`):

- the divergence it forms, `$h^{-2}\,[h\,\nabla_0\cdot(u_s, h u_r,
  h u_\theta)]$`, is the solver's continuity row
  (``CurvedCylindricalFlow.divergence_defect``) divided by `$h^2$`, so
  it vanishes exactly where the solver's does; the product with `$h$`
  drops the top mode's outflow, which would set a division by `$h$`
  alone apart by `$O(\kappa^2)$` of that mode's amplitude;
- its curl expands the solver's carried state
  `$(h u_s, u_r, u_\theta)$` (identity 1 of the ``cylindrical_curved``
  module docstring) and divides the `$h$` out of the transverse pair,
  as the solver's own statistics do;
- its integrals over `$s$` measure the arclength `$h\,ds$`.
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from dnsjax.harmonics import complex_harmonics
from dnsjax_lab.harmonics import inverse_metric_harmonics


@dataclass(frozen=True)
class ToroidalMetric:
    r"""`$h = 1 + \kappa r\cos\theta$`, the curved pipe's axial scale factor.

    An :class:`dnsjax.analysis.AxialMetric` with
    `$\kappa$` = *curvature* (``geo.curvature``).
    """

    curvature: float

    def scale(self, r: ndarray, theta: ndarray) -> ndarray:
        r"""`$h$` at the points *r*, *theta* (broadcast together)."""
        return 1.0 + self.curvature * r * np.cos(theta)

    def product(self, field: ndarray, grid, info, power: int) -> ndarray:
        r"""Multiply a spectral curved-pipe field by `$h^p$`, exactly.

        *power* `$p$` is 1, -1 or -2.  Each is a convolution along the
        azimuthal axis (axis 1) with the harmonics `$c_j(r)$` of
        `$h^p$` on each radius of *grid* (axis 0): `$h$` has three,
        `$c_0 = 1$` and `$c_{\pm1} = \kappa r/2$`, and
        :func:`~dnsjax_lab.harmonics.inverse_metric_harmonics` gives
        those of `$1/h$` and `$1/h^2$` in closed form.  Every stored
        coefficient of the product is the full sum
        `$\sum_n c_{m-n}\hat f_n$` over the input's modes, so the
        result is exact on the stored set and drops only what lies
        beyond `$|m| = M$`, as the solver's truncating shift does for
        `$h$`.

        The solver forms the inverse powers by collocation on its
        dealiased grid of `$N$` azimuthal points instead, so the two
        agree to the aliased harmonics, `$O(|q|^{N - 2M})$` with
        `$q \approx -\kappa r/2$`: below roundoff on the default
        ``phys.curvature_padding`` grid, `$N \ge 3M + J(\kappa)$`.

        Every power preserves the radial parity class
        (`$\chi = r\cos\theta$` shifts `$m$` by one and multiplies by
        `$r$`), so the product takes the radial operator its input
        did.
        """
        field = np.asarray(field)
        rs = np.asarray(grid, dtype=float)
        m = complex_harmonics(info.n[1]) * info.azimuthal_m0
        offset = np.abs(m[:, None] - m[None, :])  # |m_a - m_b|, (n_m, n_m)
        n_harm = int(offset.max()) + 2  # at least c_0 and c_1
        if power == 1:
            coeffs = np.zeros((len(rs), n_harm))
            coeffs[:, 0] = 1.0
            coeffs[:, 1] = 0.5 * self.curvature * rs
        elif power in (-1, -2):
            coeffs = inverse_metric_harmonics(
                self.curvature, rs, np.arange(n_harm), -power
            )
        else:
            raise ValueError(f"power must be 1, -1 or -2, got {power}.")
        # One (n_m, n_m) Toeplitz matrix per radius: stacking them would
        # cost n_r times the memory.
        out = np.empty_like(field)
        for j in range(len(rs)):
            out[j] = coeffs[j][offset].astype(field.dtype) @ field[j]
        return out


def metric(params) -> ToroidalMetric | None:
    """The curved pipe's metric at ``geo.curvature``, ``None`` at 0.

    The curved-pipe spec's ``analysis_metric`` factory: at zero
    curvature `$h = 1$`, the straight pipe.
    """
    kappa = float(params.geo.curvature)
    return ToroidalMetric(kappa) if kappa else None
