r"""Parameter spec for Waleffe (cosine-forced) box flow."""

from ....flow_spec import FlowSpec
from ._family import periodic_deferred, periodic_derive, periodic_fields

SPEC = FlowSpec(
    system="waleffe",
    family="triply-periodic",
    geometry_label="triply-periodic",
    summary="cosine-forced box flow (Ry symmetry not yet implemented)",
    flow_module="dnsjax.flows.triply_periodic.monochromatic",
    fields=periodic_fields(),
    deferred=periodic_deferred(),
    derive=periodic_derive,
)
