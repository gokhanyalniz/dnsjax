r"""Wall-bounded flow parameter specs (JAX-free).

The core's wall-bounded flows; plugins register further ones under the
``dnsjax.flows`` entry-point group (:mod:`dnsjax.flows.registry`).
"""

from .pipe import SPEC as PIPE
from .plane_couette import SPEC as PLANE_COUETTE
from .plane_poiseuille import SPEC as PLANE_POISEUILLE
from .taylor_couette import SPEC as TAYLOR_COUETTE

SPECS = (
    PLANE_COUETTE,
    PLANE_POISEUILLE,
    PIPE,
    TAYLOR_COUETTE,
)
