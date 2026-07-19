r"""Wall-bounded flow parameter specs (JAX-free)."""

from .dean import SPEC as DEAN
from .pipe import SPEC as PIPE
from .plane_couette import SPEC as PLANE_COUETTE
from .plane_poiseuille import SPEC as PLANE_POISEUILLE
from .quasi_keplerian import SPEC as QUASI_KEPLERIAN
from .taylor_couette import SPEC as TAYLOR_COUETTE
from .viscoelastic_dean import SPEC as VISCOELASTIC_DEAN

SPECS = (
    PLANE_COUETTE,
    PLANE_POISEUILLE,
    PIPE,
    TAYLOR_COUETTE,
    QUASI_KEPLERIAN,
    DEAN,
    VISCOELASTIC_DEAN,
)
