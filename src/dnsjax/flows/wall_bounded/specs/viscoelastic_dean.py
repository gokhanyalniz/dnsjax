r"""Parameter spec for viscoelastic (sPTT) Dean flow.

The reference normalisation uses a half-gap length unit (gap = 2, so
``r1 = delta``, ``r2 = delta + 2``) and derives the Reynolds number as
``Re := Wi/El``; the unset control parameters default to the reference
configuration (the ``default`` overrides below, materialized by
``update_parameters``).
"""

from math import pi

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    DEFERRED_TILT,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)


def _derive(params, derived, user_set) -> None:
    # The el/wi/beta/epsilon/kappa/delta reference defaults are
    # materialized generically from the FieldSpec overrides before
    # this hook runs; only the geometry / Reynolds derivation remains.
    r1 = params.geo.delta
    r2 = r1 + 2.0
    derived.r_inner = r1
    derived.r_outer = r2
    # Azimuthal wedge extent (m0 = 1 is the full circle); the tensor
    # spin components ride the same physical wavenumbers
    # m = m0 * harmonic as the velocity u_pm (their spin shifts
    # m_eff = m + s are intrinsic, not harmonics), so the wedge works
    # exactly as in the Newtonian annulus.
    params.geo.lz = 2 * pi / params.geo.m0
    derived.volume_fac = (r2**2 - r1**2) / 2  # int_{r1}^{r2} r dr
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
    system="viscoelastic-dean",
    family="annular-viscoelastic",
    geometry_label="annular",
    summary="viscoelastic (sPTT) force-driven annular flow",
    flow_module="dnsjax.flows.wall_bounded.viscoelastic_dean",
    # 3 velocity + 6 symmetric conformation-tensor components.
    n_components=9,
    fields=(
        *wall_fields(0.0, CARTESIAN_ANNULAR_GRIDS),
        *cyl_annular_fields(axial_default=2 * pi),
        FieldSpec("geo", "delta", default=11.0),
        FieldSpec("phys", "el", default=80.0),
        FieldSpec("phys", "wi", default=105.0),
        FieldSpec("phys", "beta", default=0.8),
        FieldSpec("phys", "epsilon", default=0.001),
        FieldSpec("phys", "kappa", default=5.0e-5),
        FieldSpec("phys", "block_mean_spanwise_velocity"),
        FieldSpec("init", "random_conformation_amplitude"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default="cgl",
    derive=_derive,
    rehydrate=_rehydrate,
)
