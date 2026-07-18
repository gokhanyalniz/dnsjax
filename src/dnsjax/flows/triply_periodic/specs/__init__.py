r"""Triply-periodic flow parameter specs (JAX-free)."""

from .kolmogorov import SPEC as KOLMOGOROV
from .waleffe import SPEC as WALEFFE

SPECS = (KOLMOGOROV, WALEFFE)
