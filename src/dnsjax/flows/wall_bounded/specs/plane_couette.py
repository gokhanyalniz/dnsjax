r"""Parameter spec for plane-Couette flow."""

from ....flow_spec import FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    cartesian_derive,
    cartesian_fields,
    wall_fields,
)


def _validate(params, derived) -> None:
    # Wall-driven flow with zero laminar bulk and zero mean pressure
    # gradient: constant-bulk driving is not a meaningful choice, so
    # ``driving`` is not on the plane-couette surface and a direct
    # assignment is rejected here.
    if params.phys.driving == "constant_bulk_velocity":
        raise ValueError(
            "plane-couette does not support "
            "phys.driving='constant_bulk_velocity' (wall-driven flow)."
        )


SPEC = FlowSpec(
    system="plane-couette",
    family="cartesian",
    geometry_label="cartesian",
    summary="wall-driven plane channel",
    flow_module="dnsjax.flows.wall_bounded.plane_couette",
    fields=(
        *wall_fields(0.0, CARTESIAN_ANNULAR_GRIDS),
        *cartesian_fields(),
    ),
    grid_type_default="cgl",
    derive=cartesian_derive,
    validate=_validate,
)
