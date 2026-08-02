r"""Parameter spec for viscoelastic (sPTT) pipe flow.

Lengths are in pipe radii (`$r \in (0, 1]$`) and the velocity unit is
the `$\epsilon = 0$` laminar centreline velocity, so the Newtonian and
Oldroyd-B limits both reduce to `$W = 1 - r^2$` -- the same convention
as :mod:`~dnsjax.flows.wall_bounded.pipe`.

The Reynolds number is **derived**, ``Re := Wi/El``: the elasticity
number `$\mathrm{El} = \mathrm{Wi}/\mathrm{Re}$` is a property of the
fluid and the geometry rather than of the flow rate, so it is the
control parameter, and `$\mathrm{Re}$` follows.  The defaults below put
that at `$\mathrm{Wi} = 20$`, `$\mathrm{El} = 0.02$`, hence
`$\mathrm{Re} = 1000$` -- the Newtonian pipe's own default Reynolds
number, in the elasto-inertial range where a viscoelastic pipe is
usually run.  (The annular sPTT flow defaults to `$\mathrm{El} = 80$`,
`$\mathrm{Re} \approx 1.3$` instead: its subject is the inertialess,
strongly elastic regime, which is not this one.)  To sweep
`$\mathrm{Re}$` at fixed `$\mathrm{Wi}$`, vary ``el``.
"""

from math import pi

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CYLINDRICAL_GRIDS,
    DEFERRED_TILT,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)
from .pipe import _grid_default, _validate


def _derive(params, derived, user_set) -> None:
    # The el/wi/beta/epsilon/kappa defaults are materialized generically
    # from the FieldSpec overrides before this hook runs.
    #
    # Azimuthal wedge extent (m0 = 1 is the full circle); the tensor
    # spin components ride the same physical wavenumbers
    # m = m0 * harmonic as the velocity u_pm (their spin shifts
    # m_eff = m + s are intrinsic, not harmonics), so the wedge -- and
    # the axis parity it sets -- works exactly as in the Newtonian pipe.
    params.geo.lz = 2 * pi / params.geo.m0
    derived.volume_fac = 0.5  # int_0^1 r dr
    # Re := Wi/El -- derived, not a user parameter here: any directly
    # assigned value is simply overwritten (resumed snapshots replay a
    # consistent value).
    params.phys.re = params.phys.wi / params.phys.el


def _rehydrate(sections: dict) -> None:
    wedge_rehydrate(sections)
    phys = sections.setdefault("phys", {})
    wi, el = phys.get("wi"), phys.get("el")
    if wi is not None and el is not None:
        phys["re"] = wi / el


SPEC = FlowSpec(
    system="viscoelastic-pipe",
    family="cylindrical-viscoelastic",
    geometry_label="cylindrical",
    summary="viscoelastic (sPTT) pressure-driven circular pipe",
    flow_module="dnsjax.flows.wall_bounded.viscoelastic_pipe",
    # 3 velocity + 6 symmetric conformation-tensor components.
    n_components=9,
    fields=(
        *wall_fields(0.5, CYLINDRICAL_GRIDS),
        *cyl_annular_fields(),
        # Wi = 20, El = 0.02 => Re = 1000, the Newtonian pipe's default
        # (see the module docstring); the rheology below is the shared
        # sPTT reference (dilute solvent ratio, weak extensibility, a
        # trace of artificial conformation diffusion).
        FieldSpec("phys", "el", default=0.02),
        FieldSpec("phys", "wi", default=20.0),
        FieldSpec("phys", "beta", default=0.8),
        FieldSpec("phys", "epsilon", default=0.001),
        FieldSpec("phys", "kappa", default=5.0e-5),
        FieldSpec("init", "random_conformation_amplitude"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default=_grid_default,
    derive=_derive,
    validate=_validate,
    rehydrate=_rehydrate,
)
