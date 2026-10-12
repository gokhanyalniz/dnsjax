r"""The laminar profile of a force-driven annular flow (lab helper).

:func:`annular_forced_laminar_u_theta`, the closed-form azimuthal profile
an azimuthal body force sustains between two fixed cylinders, shared by
the Dean and viscoelastic Dean flows.
"""

import numpy as np
from jax import Array
from jax import numpy as jnp


def annular_forced_laminar_u_theta(
    rs: Array, r1: float, r2: float, C: float
) -> Array:
    r"""Analytical laminar azimuthal profile for an annular body force.

    Closed-form steady solution of the azimuthal momentum balance
    `$(1/\mathrm{Re})\,(\nabla^2 \mathbf{U})_\theta - \Pi_\theta = 0$`
    for a radius-dependent azimuthal driving whose applied force is
    `$-\Pi_\theta = C/(r\,\mathrm{Re})$` -- `$\Pi_\theta$` being the
    mean-mode azimuthal gradient in the codebase convention, so the
    force is its negative (:mod:`dnsjax.ic.mean_mode`) -- with no-slip
    walls
    `$U_\theta(r_1) = U_\theta(r_2) = 0$`.  The `$1/\mathrm{Re}$`
    cancels, so the profile is **Reynolds-independent**:

    .. math::
        U_\theta(r) = -\tfrac{C}{2}\,r\ln r + \alpha\,r
        + \frac{\beta}{r},

    where the wall conditions fix

    .. math::
        \alpha = \frac{C}{2}\,
        \frac{r_1^2 \ln r_1 - r_2^2 \ln r_2}{r_1^2 - r_2^2},
        \qquad
        \beta = \frac{C}{2}\,
        \frac{(r_1 r_2)^2 (\ln r_2 - \ln r_1)}{r_1^2 - r_2^2}.

    Shared by the two force-driven annular flows: Newtonian Dean
    (:func:`dnsjax.flows.wall_bounded.dean.dean_laminar_u_theta`,
    `$C = 2(r_1 + r_2)$`) and the viscoelastic
    sPTT flow (`$C = r_1 + r_2$` in half-gap units, whose mid-gap force is the
    plane channel's `$2/\mathrm{Re}$`;
    :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean`).  Pure function (no
    flow construction), so it is importable both by the ``start_from_laminar``
    state and by the flows' initial conditions (the total-field IC = laminar
    profile + perturbation).

    Parameters
    ----------
    rs:
        Radial grid on `$[r_1, r_2]$`, shape ``(Nr,)``.
    r1, r2:
        Inner / outer non-dim radii.
    C:
        Body-force coefficient: the applied force is
        `$-\Pi_\theta = C/(r\,\mathrm{Re})$`.
    """
    denom = r1**2 - r2**2
    alpha = (C / 2.0) * (r1**2 * np.log(r1) - r2**2 * np.log(r2)) / denom
    beta = (C / 2.0) * (r1 * r2) ** 2 * (np.log(r2) - np.log(r1)) / denom
    return -(C / 2.0) * rs * jnp.log(rs) + alpha * rs + beta / rs
