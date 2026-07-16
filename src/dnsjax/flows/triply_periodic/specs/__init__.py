r"""Triply-periodic flow parameter specs (JAX-free)."""

from .decaying_box import SPEC as DECAYING_BOX
from .kolmogorov import SPEC as KOLMOGOROV
from .waleffe import SPEC as WALEFFE

SPECS = (DECAYING_BOX, KOLMOGOROV, WALEFFE)
