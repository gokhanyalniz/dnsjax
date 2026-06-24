r"""JAX-free analysis API for dnsjax snapshots.

For physics/applied-maths researchers post-processing snapshots without
the solver runtime.  Depends only on NumPy, the standard library, and
dnsjax's JAX-free leaf modules (:mod:`dnsjax.fd`,
:mod:`dnsjax.snapshot_meta`, :mod:`dnsjax.harmonics`) -- importing
``dnsjax.analysis`` never imports JAX.

Read a snapshot::

    from dnsjax.analysis import read_state
    st = read_state("state00000.tar")
    (ux, uy, uz) = st.physical          # physical-space components
    (y, x, z) = st.physical_coords      # matching coordinate arrays
    re = st.params.phys.re

Operate on fields::

    from dnsjax.analysis import read_state, divergence, integrate
    st = read_state("state00000.tar", return_spectral=True)
    div = divergence(st.spectral, st.params, st.spectral_coords)

See :func:`read_state` and :mod:`dnsjax.analysis.snapshot_ops`.
"""

# Object-like view over embedded params/stats (re-exported for typing).
from ._core import Namespace, geometry_info, read_meta, read_stats
from .snapshot_export import StateData, read_state
from .snapshot_ops import (
    curl,
    derivative,
    divergence,
    gradient,
    integrate,
    to_physical,
    to_spectral,
)

__all__ = [
    "read_state",
    "StateData",
    "derivative",
    "gradient",
    "divergence",
    "curl",
    "integrate",
    "to_physical",
    "to_spectral",
    "read_meta",
    "read_stats",
    "geometry_info",
    "Namespace",
]
