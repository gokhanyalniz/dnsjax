r"""Parameter spec for Taylor-Couette flow."""

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    DEFERRED_LOWRES_PRESSURE,
    DEFERRED_MEAN_FLOW,
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


def _control_summary(params) -> str:
    return f"re1={params.phys.re1} re2={params.phys.re2} eta={params.geo.eta}"


SPEC = FlowSpec(
    system="taylor-couette",
    family="annular",
    geometry="annular",
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
    deferred=(DEFERRED_TILT, DEFERRED_MEAN_FLOW, DEFERRED_LOWRES_PRESSURE),
    grid_type_default="cgl",
    derive=_derive,
    rehydrate=_rehydrate,
    control_summary=_control_summary,
)
