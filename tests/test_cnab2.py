#!/usr/bin/env python3
r"""CN/AB2 scheme guards (offline, in-process, no ``mpirun``).

Pins the structural claims of ``step.scheme == "cnab2"`` (see
``step_cnab2`` / ``_cnab2_lbf_core`` in ``timestep.py``) that the
subprocess smoke tests cannot check directly:

- **Split exactness**: the FFT-free base-flow coupling equals the
  base-flow part of the full nonlinear RHS to machine precision,
  ``get_rhs(u) - l_bf(u) == get_rhs(u; U = 0)`` (checked with
  ``step.implicit_mean_coupling`` off) -- the explicit AB2 forcing
  really is the pure self-advection `$u' \times \omega'$`.  The
  zero-base-flow reference reuses the geometry ``_get_rhs`` on a
  shallow flow copy with zeroed ``base_flow_padded`` /
  ``curl_base_flow_padded`` (identical transforms, so the difference
  isolates exactly the coupling terms).
- **Mean-flow coupling reference**: with ``implicit_mean_coupling`` on
  (the default), ``l_bf`` gains exactly `$L_{mf} = \mathbf{u}
  \times \bar{\boldsymbol{\omega}} + \bar{\mathbf{u}} \times
  \boldsymbol{\omega}$` -- checked against a manually-written cross
  product with the mean profiles indexed directly off the single
  device's ``[:, :, 0, 0]`` mode (validating the ``psum``-based
  ``extract_mean_mode``, the profile broadcast, and the
  cylindrical/annular basis conversion).  Cartesian additionally checks
  the mean of the spectral curl equals the ``D1`` derivative of the
  mean profile (curl is mode-diagonal, so mean-of-curl ==
  curl-of-mean).
- **Carry-seed independence**: the AB2-history output of
  ``step_cnab2(state, carry)`` (the second element) is independent of
  the ``carry`` argument -- the ``__main__`` priming call
  ``step_cnab2(state, zeros)`` seeds the true
  `$N_{nl}(u^0)$` regardless of the placeholder carry.
- **FFT-count guards** (jaxpr traversal): outside the ``lax.cond``
  fallback branch, one ``step_cnab2`` costs exactly one nonlinear
  RHS evaluation's FFTs (**1 FFT eval/step**); the implicit-coupling
  Picard ``while_loop`` body is **FFT-free**; the iterative-CN
  fallback (with its FFTs) exists only under a ``cond``.
  ``predict_and_fully_correct`` -- wall-bounded: the module stepper is
  the **unsplit** corrector (``step.split_corrector`` defaults off) --
  2 RHS evaluations plus one per corrector iteration, no ``cond``.
  Rebuilt with ``split_corrector = True`` (``_split_core`` in
  ``timestep.py``) it has the split shape: FFT-free coupling tail and
  solver loops as 0-FFT bodies, the outer refresh paying one RHS
  evaluation, and the unsplit-corrector fallback under a ``cond``.
  Both gate values are built explicitly here (the default is off, so
  the split shape is only reached via the forced-on rebuild).
  Triply-periodic ``step_cnab2`` (no ``l_bf_fn``) has no loop and no
  cond -- exactly one RHS evaluation total -- and its corrector is
  always unsplit.
- **Split-corrector equivalence** (wall-bounded): at a tight
  ``corrector_tolerance`` (1e-12; read at trace time, so no operator
  rebuild) the split and unsplit correctors converge the same step
  to the same CN fixed point (states agree to ~1e-10 relative) with
  the split outer loop genuinely entered (``num_c >= 1``) -- the FFT
  savings change the iteration path, never the converged answer.
Each system runs in its own subprocess (import-time singletons), on a
single forced-CPU device, mirroring ``tests/test_localized_rolls.py``
(the worker: ``tests/_cnab2.py``; the lab's flows have cases of their
own, ``lab/tests/test_cnab2.py``)::

    uv run python tests/test_cnab2.py                  # all systems
    uv run python tests/test_cnab2.py --system pipe    # one system
"""

from __future__ import annotations

import argparse

from _cnab2 import run_cases, worker

#: ``system -> case`` (the keys: :func:`_cnab2.worker`).
CASES: dict[str, dict] = {
    "plane-couette": {
        "stepper": (
            "dnsjax.geometries.wall_bounded.cartesian:build_cartesian_stepper"
        ),
    },
    "pipe": {
        "stepper": (
            "dnsjax.geometries.wall_bounded.cylindrical"
            ":build_cylindrical_stepper"
        ),
    },
    "taylor-couette": {
        "stepper": (
            "dnsjax.geometries.wall_bounded.annular:build_annular_stepper"
        ),
        "phys": {"re1": 100.0, "re2": 0.0},
        "geo": {"lx": 5.0, "lz": 5.0, "eta": 0.5},
    },
    "kolmogorov": {},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(CASES), default=None)
    parser.add_argument("--worker", choices=list(CASES), default=None)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, CASES[args.worker])
        return
    run_cases(__file__, [args.system] if args.system else list(CASES))


if __name__ == "__main__":
    main()
