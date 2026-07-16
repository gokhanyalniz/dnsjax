r"""Parameter spec for Taylor-Couette flow."""

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    DEFERRED_TILT,
    annular_base_derive,
    circular_couette_derive,
    circular_couette_rehydrate,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)


def _derive(params, derived, user_set) -> None:
    eta = annular_base_derive(params, derived)
    circular_couette_derive(params, derived, eta)


def _rehydrate(sections: dict) -> None:
    wedge_rehydrate(sections)
    circular_couette_rehydrate(sections)


SPEC = FlowSpec(
    system="taylor-couette",
    family="annular",
    geometry_label="annular",
    summary="rotating-cylinder annular flow",
    flow_module="dnsjax.flows.wall_bounded.taylor_couette",
    fields=(
        *wall_fields(0.0, CARTESIAN_ANNULAR_GRIDS),
        *cyl_annular_fields(),
        FieldSpec("geo", "eta"),
        FieldSpec("phys", "re1"),
        FieldSpec("phys", "re2"),
        FieldSpec("phys", "block_mean_spanwise_velocity"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default="cgl",
    derive=_derive,
    rehydrate=_rehydrate,
)
