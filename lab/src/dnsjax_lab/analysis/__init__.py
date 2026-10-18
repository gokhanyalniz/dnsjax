"""The lab's snapshot analysis (JAX-free).

What the lab flows add to :mod:`dnsjax.analysis`: the curved pipe's
toroidal metric (:mod:`dnsjax_lab.analysis.toroidal`).  Importing this
package imports nothing; its subpackages are imported explicitly:
:mod:`dnsjax_lab.analysis.twin` (the twin streams' readers, JAX-free)
and :mod:`dnsjax_lab.analysis.response` (JAX, and SciPy lazily).
"""
