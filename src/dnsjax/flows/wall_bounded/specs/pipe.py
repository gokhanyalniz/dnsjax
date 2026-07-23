r"""Parameter spec for pipe flow."""

from math import pi

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CYLINDRICAL_GRIDS,
    DEFERRED_TILT,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)


def _grid_default(scheme: str) -> str:
    # Half-CGL's tighter near-axis point destabilises explicit cnab2;
    # the rigged grid's 2x larger innermost radius doubles cnab2's
    # admissible dt.  See the ``Geometry`` docstring.
    return "half-cgl" if scheme == "iterative-cn" else "rigged-cgl"


def _derive(params, derived, user_set) -> None:
    # Azimuthal wedge extent (m0 = 1 is the full circle); see
    # ``geo.m0``.
    params.geo.lz = 2 * pi / params.geo.m0
    derived.volume_fac = 0.5  # int_0^1 r dr


def _validate(params, derived) -> None:
    if (
        params.geo.grid_type == "half-cgl"
        and params.step.scheme != "iterative-cn"
    ):
        raise ValueError(
            "geo.grid_type='half-cgl' requires "
            "step.scheme='iterative-cn' (the tighter half-CGL axis "
            f"destabilises the explicit {params.step.scheme!r} scheme "
            "at low dt); use the rigged-CGL grid ('rigged-cgl', the "
            "cnab2 default) instead."
        )


SPEC = FlowSpec(
    system="pipe",
    family="cylindrical",
    geometry_label="cylindrical",
    summary="pressure/bulk-driven circular pipe",
    flow_module="dnsjax.flows.wall_bounded.pipe",
    fields=(
        *wall_fields(0.5, CYLINDRICAL_GRIDS),
        *cyl_annular_fields(),
        # ``consistent_imm`` is omitted from ``wall_fields`` for the
        # cylindrical family (see its docstring); the pipe opts in via
        # the x = r^2 parity operators + 1-wall closure (see
        # ``cylindrical.build_parity_reduced_matrices`` /
        # ``_imm_iteration``).  ``pipe_axis_fit`` is the cylindrical-only
        # x = r^2 ``D1`` without the composed ``D2`` (accurate + random-
        # IC-stable; see ``Resolution.pipe_axis_fit``).
        FieldSpec("res", "consistent_imm"),
        FieldSpec("res", "pipe_axis_fit"),
        FieldSpec("phys", "re"),
        FieldSpec("phys", "driving"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default=_grid_default,
    derive=_derive,
    validate=_validate,
    rehydrate=wedge_rehydrate,
)
