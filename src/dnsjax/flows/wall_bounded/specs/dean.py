r"""Parameter spec for (force-driven) Dean flow."""

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    DEFERRED_TILT,
    annular_base_derive,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)


def _derive(params, derived, user_set) -> None:
    # Dean uses phys.re directly (both walls stationary); its
    # azimuthal body force lives in ``flows.wall_bounded.dean``.
    annular_base_derive(params, derived)


SPEC = FlowSpec(
    system="dean",
    family="annular",
    geometry_label="annular",
    summary="force-driven curved-channel (annular) flow",
    flow_module="dnsjax.flows.wall_bounded.dean",
    fields=(
        *wall_fields(0.0, CARTESIAN_ANNULAR_GRIDS),
        *cyl_annular_fields(),
        FieldSpec("geo", "eta"),
        FieldSpec("phys", "re"),
        FieldSpec("phys", "block_mean_spanwise_velocity"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default="cgl",
    derive=_derive,
    rehydrate=wedge_rehydrate,
)
