r"""CFL-based adaptive time-step controller (JAX-free).

Pure host-side arithmetic behind ``step.adaptive``: the main loop
(:mod:`dnsjax.__main__`) reads the measured total CFL every
``step.cfl_cadence`` steps and asks :func:`propose_dt` for the next
time step.  The CFL reported by :func:`dnsjax.measurements.get_cfl`
is linear in the step,

.. math::

    \mathrm{CFL}(\Delta t)
    = \Delta t \, \max_{\mathbf{x}} \sum_i \frac{|u_i|}{\Delta_i},

so the step that would exactly meet ``step.cfl_target`` is

.. math::

    \Delta t_{\mathrm{ideal}}
    = \Delta t \, \frac{\mathrm{CFL}_{\mathrm{target}}}
                       {\mathrm{CFL}}.

The proposal is then restricted, in this order: capped from above by
``dt_max`` and the per-evaluation growth ratio ``dt_max_change``,
floored from below by ``dt_min`` and the shrink ratio
``dt_min_change`` (the floors win over the caps in degenerate
configurations -- cap-then-floor), and finally passed through a
relative deadband: the current ``dt`` is kept unless the restricted
proposal moves it by more than ``dt_threshold * dt``, suppressing
rebuild churn from CFL noise.

The controller is host-side and rank-deterministic: the measured CFL
scalar is replicated across devices/ranks, so every rank computes the
same proposal with no communication.  Knob semantics, defaults, and
what an accepted change triggers (the on-device operator rebuild, the
variable-step AB2 weight): the ``TimeStepping`` docstring in
:mod:`dnsjax.parameters`.
"""

import math


def propose_dt(
    cfl: float,
    dt: float,
    *,
    cfl_target: float,
    dt_min: float,
    dt_max: float,
    dt_min_change: float,
    dt_max_change: float,
    dt_threshold: float,
) -> float:
    r"""Propose the next time step from the measured total CFL.

    Parameters
    ----------
    cfl:
        Measured total CFL of the last step (taken at *dt*).  A
        non-positive or non-finite value carries no advective signal
        and proposes unrestricted growth; the main loop aborts on a
        non-finite CFL *before* consulting the controller, so only
        the genuine zero-velocity case reaches that branch here.
    dt:
        The current time step.
    cfl_target, dt_min, dt_max, dt_min_change, dt_max_change, \
dt_threshold:
        The ``step.*`` controller knobs (see ``TimeStepping``).

    Returns
    -------
    :
        The accepted next time step: either *dt* unchanged (deadband)
        or the restricted ideal step.
    """
    if not math.isfinite(cfl) or cfl <= 0.0:
        dt_ideal = math.inf
    else:
        dt_ideal = dt * cfl_target / cfl
    new = min(dt_ideal, dt_max, dt_max_change * dt)
    new = max(new, dt_min, dt_min_change * dt)
    if abs(new - dt) <= dt_threshold * dt:
        return dt
    return new
