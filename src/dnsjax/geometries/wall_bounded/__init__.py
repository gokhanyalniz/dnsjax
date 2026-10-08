"""Wall-bounded geometries: Cartesian, cylindrical and annular.

``cartesian``, ``cylindrical`` (with its toroidal variant
``cylindrical_curved``) and ``annular`` are the geometries and
``_base`` their shared layer.  The remaining modules hold the pipe
stepping both cylindrical geometries share, the viscoelastic
extensions, the legacy primitive influence-matrix paths and the
Cartesian static pressure (each module docstring says which).
"""

from ._base import (
    get_norm2,
    integrate_scalar,
)

__all__ = [
    "get_norm2",
    "integrate_scalar",
]
