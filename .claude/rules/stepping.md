---
paths:
  - "src/dnsjax/{timestep,adaptive,rhs,__main__}.py"
  - "src/dnsjax/geometries/**/*.py"
  - "src/dnsjax/twin/driver.py"
  - "src/dnsjax/analysis/transient_growth.py"
  - "tests/test_{cnab2,temporal_order,adaptive,autodiff,imm_continuity}.py"
---

# Time stepping (`[step]`)

- The two schemes, their `dt` limits, `implicitness`,
  `implicit_mean_coupling` and `split_corrector`: the
  `parameters.TimeStepping` docstring. `timestep.make_stepper` builds
  the steppers; each family binds its singletons around it
  (`_base.build_wall_bounded_stepper`,
  `triply_periodic.build_triply_periodic_stepper`).
- Anything that measures the formal temporal order pins
  `step.implicitness = 0.5`: the default 0.5001 damps Crank-Nicolson's
  stiff wall modes (`tests/test_temporal_order.py`).
- `step.corrector_iterations > 0` fixes the corrector count so a step
  reverse-differentiates. It is refused with `split_corrector` and
  forced to 0 by the transient-growth driver, and it turns the
  corrector error from a verdict into a diagnostic, so both drivers
  (`__main__.py`, `twin/driver.py`) gate their loop guard, their
  closing line and their exit code (`run_status.EXIT_CORRECTOR`) on
  it.
- "corrector failed to converge" at a *low* CFL is the corrector's
  contraction limit: reduce `dt`; it is not a blow-up. A correction
  that stalls at one value pass after pass, and keeps stalling as `dt`
  shrinks, is not that limit: an incremental update fed the constant
  `f(u^n)` re-adds itself. Every corrector uses the direct form
  (`triply_periodic._correct_component` says why).
- Buffer donation: `predict_and_fully_correct(_measured)` donates
  `state`, and `step_cnab2(_measured)` donates `state` and `carry`. A
  caller that reuses an input afterwards passes `jnp.copy` (as the
  `__main__` warm-up calls do).
