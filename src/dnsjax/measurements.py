r"""Physical-space measurements for the nonlinear-term hook.

Some diagnostics need the velocity in physical space, which
exists only transiently inside the nonlinear-term evaluation
(:func:`dnsjax.rhs.get_nonlin`), on the dealiased
(3/2-oversampled) grid.  Rather than re-transforming the state,
such measurements plug into ``get_nonlin`` through its optional
``measure_fn`` callback:

    ``measure_fn(velocity_phys, vorticity_phys)
    -> dict[str, Array]``

where both inputs have shape ``(3, ny_phys, nz_padded,
nx_padded)`` (layout ``[C, y, z, x]``) and the returned dict
maps column names to replicated scalars.  The dict structure
must be static (same keys every call) so the JIT-compiled
stepper has a fixed output pytree; ``__main__`` derives the
``steps.dat`` header from the keys of a warm-up call.  Note
that dicts returned through ``jit`` are canonicalised to
sorted key order, which sets the column order of
``steps.dat`` (as with ``get_stats`` and ``stats.dat``).

Future physical-space measurements should follow the same
pattern: a ``get_*`` function here taking the physical fields
plus precomputed flow data, wired through the geometry's
``_get_rhs_measured`` wrapper.

Currently implemented: the CFL (Courant-Friedrichs-Lewy)
numbers, :func:`get_cfl`.
"""

from jax import Array
from jax import numpy as jnp


def get_cfl(
    velocity_phys: Array,
    base_flow: Array,
    inv_spacing: Array,
    names: tuple[str, str, str],
    dt: Array,
) -> dict[str, Array]:
    r"""CFL numbers of the advecting velocity on the current grid.

    With the advecting velocity
    `$\mathbf{u} = \mathbf{u}' + \mathbf{U}_{\!adv}$` -- where
    *base_flow* is the advection base velocity, which in a moving
    frame of reference is `$\mathbf{U}_{\!adv} = \mathbf{U}
    - U_{grid}\hat{\mathbf{e}}_0$` -- and the local advection length
    scale `$\Delta_i$` of each direction
    (encoded in *inv_spacing*), computes the per-direction CFL
    numbers and the sum-form total

    .. math::
        \mathrm{CFL}_i = \Delta t \max_{\mathbf{x}}
        \frac{|u_i|}{\Delta_i}, \qquad
        \mathrm{CFL} = \Delta t \max_{\mathbf{x}}
        \sum_i \frac{|u_i|}{\Delta_i},

    the latter being the standard multi-dimensional advective
    stability bound (not recoverable from the per-direction
    maxima).  The recorded column set is decided here alone.

    Wall-bounded zero-padding rows of the physical `$y$` axis
    cannot contribute: *velocity_phys*, *base_flow*, and
    *inv_spacing* are all zero there.

    Parameters
    ----------
    velocity_phys:
        Perturbation velocity in physical space,
        ``(3, ny_phys, nz_padded, nx_padded)``.
    base_flow:
        Advection base velocity `$\mathbf{U}_{\!adv}$` in physical
        space, ``(3, ny_phys, 1, 1)``, same component order.  The
        wall-bounded geometries pass ``base_flow_adv_padded`` (the
        moving-frame `$\mathbf{U} - U_{grid}\hat{\mathbf{e}}_0$`),
        so the CFL reflects the frame-relative advection.
    inv_spacing:
        Inverse local advection length scale per component,
        ``(3, ny_phys, 1, 1)``: e.g. `$1/\Delta x$` (uniform
        directions), `$1/\Delta y_j$` (local wall-normal node
        spacing), `$1/(r_j \Delta\theta)$` (azimuthal arc).
    names:
        Per-direction column names matching the component
        order (e.g. ``("CFL_x", "CFL_y", "CFL_z")``).
    dt:
        The step's time step as a 0-d array -- the callers pass the
        ``flow.dt`` pytree leaf, so an adaptive-``dt`` run reports
        the live value without retracing.

    Returns
    -------
    :
        ``{names[0]: ..., names[1]: ..., names[2]: ...,
        "CFL": ..., "dt": dt}`` of replicated scalars.  The ``dt``
        column records the step size the CFL was evaluated at (and,
        under ``step.adaptive``, the varying step itself).
    """
    scaled = jnp.abs(velocity_phys + base_flow) * inv_spacing
    dir_max = dt * jnp.max(scaled, axis=(1, 2, 3))
    total = dt * jnp.max(jnp.sum(scaled, axis=0))
    return {
        names[0]: dir_max[0],
        names[1]: dir_max[1],
        names[2]: dir_max[2],
        "CFL": total,
        "dt": dt,
    }
