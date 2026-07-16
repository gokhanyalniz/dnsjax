r"""Parameter spec for the decaying (unforced) periodic box."""

from ....flow_spec import FlowSpec
from ._family import periodic_deferred, periodic_derive, periodic_fields

SPEC = FlowSpec(
    system="decaying-box",
    family="triply-periodic",
    geometry_label="triply-periodic",
    summary="freely decaying box turbulence",
    flow_module="dnsjax.flows.triply_periodic.monochromatic",
    fields=periodic_fields(),
    deferred=periodic_deferred(),
    derive=periodic_derive,
)
