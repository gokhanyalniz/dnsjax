"""Direct numerical simulation of incompressible flow in JAX.

Pseudo-spectral in the periodic directions and banded finite
differences in up to one wall-bounded direction, with an
influence-matrix treatment of the wall conditions.  The solver is the
``dnsjax`` console script (:mod:`dnsjax.__main__`) and the NumPy-only
snapshot API is :mod:`dnsjax.analysis`; further flows, parameter
fields and run-loop sections plug in through entry points
(:mod:`dnsjax.flows.registry`, :mod:`dnsjax.parameters`,
:mod:`dnsjax.extensions`).  This package module imports nothing, so
``import dnsjax.analysis`` stays free of JAX.
"""
