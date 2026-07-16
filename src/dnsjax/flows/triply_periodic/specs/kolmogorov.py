r"""Parameter spec for Kolmogorov (sine-forced) box flow."""

from ....flow_spec import FlowSpec
from ._family import periodic_deferred, periodic_derive, periodic_fields

SPEC = FlowSpec(
    system="kolmogorov",
    family="triply-periodic",
    geometry_label="triply-periodic",
    summary="sine-forced box flow",
    flow_module="dnsjax.flows.triply_periodic.monochromatic",
    fields=periodic_fields(),
    deferred=periodic_deferred(),
    derive=periodic_derive,
)
