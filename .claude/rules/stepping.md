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
- `step.implicitness` defaults to 0.5001, not 0.5, to damp
  Crank-Nicolson's near-neutral stiff wall modes. Anything that
  measures the formal order pins 0.5 (`tests/test_temporal_order.py`).
- `curved-pipe` refuses `cnab2` and `split_corrector` (its spec's
  validate hook says why).
- `step.corrector_iterations > 0` fixes the corrector count so a step
  reverse-differentiates. It is refused with `split_corrector` and
  forced to 0 by the transient-growth driver, and it turns the
  corrector error from a verdict into a diagnostic, so both drivers
  (`__main__.py`, `twin/driver.py`) gate their loop guard and their
  closing line on it.
- "corrector failed to converge" at a *low* CFL is the corrector's
  contraction limit: reduce `dt`. It is not a blow-up.
- Adaptive `dt` (`step.adaptive`; the controller is `adaptive.py`):
  the builders' `set_dt` rebuilds the operators on device without a
  recompile.
- Buffer donation: `predict_and_fully_correct(_measured)` donates
  `state`, and `step_cnab2(_measured)` donates `state` and `carry`. A
  caller that reuses an input afterwards passes `jnp.copy` (as the
  `__main__` warm-up calls do).
- `phys.u_grid` (moving frame) is implicit in both schemes but does
  not relax cnab2's explicit self-advection CFL, and a changed
  `u_grid` on resume starts a new trajectory (the field's docs in
  `parameters.py`).
