r"""Parameter spec for plane-Poiseuille flow."""

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    cartesian_derive,
    cartesian_fields,
    wall_fields,
)

SPEC = FlowSpec(
    system="plane-poiseuille",
    family="cartesian",
    geometry_label="cartesian",
    summary="pressure/bulk-driven plane channel",
    flow_module="dnsjax.flows.wall_bounded.plane_poiseuille",
    fields=(
        *wall_fields(2.0 / 3.0, CARTESIAN_ANNULAR_GRIDS),
        *cartesian_fields(),
        FieldSpec("phys", "driving"),
    ),
    grid_type_default="cgl",
    derive=cartesian_derive,
)
