r"""The lab's flow systems, registered with the core as a plugin.

``SPECS`` is the object of the ``dnsjax.flows`` entry point in
``lab/pyproject.toml``; the core lists these flows after its own within
each geometry (:mod:`dnsjax.flows.registry`).  JAX-free, like every
spec module.
"""

from dnsjax_lab.flows.wall_bounded.specs.curved_pipe import SPEC as CURVED_PIPE
from dnsjax_lab.flows.wall_bounded.specs.dean import SPEC as DEAN
from dnsjax_lab.flows.wall_bounded.specs.quasi_keplerian import (
    SPEC as QUASI_KEPLERIAN,
)
from dnsjax_lab.flows.wall_bounded.specs.viscoelastic_dean import (
    SPEC as VISCOELASTIC_DEAN,
)
from dnsjax_lab.flows.wall_bounded.specs.viscoelastic_pipe import (
    SPEC as VISCOELASTIC_PIPE,
)

SPECS = (
    CURVED_PIPE,
    VISCOELASTIC_PIPE,
    QUASI_KEPLERIAN,
    DEAN,
    VISCOELASTIC_DEAN,
)

#: The viscoelastic (sPTT) flows, one per geometry: their state holds
#: the conformation tensor beside the velocity.
VISCOELASTIC_SYSTEMS: frozenset[str] = frozenset(
    {VISCOELASTIC_PIPE.system, VISCOELASTIC_DEAN.system}
)
