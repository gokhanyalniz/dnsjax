"""Wall-bounded geometries: Cartesian, cylindrical and annular.

``cartesian``, ``cylindrical`` and ``annular`` are the geometries,
``_base`` their shared layer, and the remaining modules the stepping
and the viscoelastic extensions they share (each module docstring
says which).
"""

from ._base import (
    get_norm2,
    integrate_scalar,
)

__all__ = [
    "get_norm2",
    "integrate_scalar",
]
