# Flow systems

Adding a flow `X` (the human recipe: `docs/extending.md`):

1. `flows/<family>/specs/X.py`: a `FlowSpec` (templates: the existing
   specs and `_family.py`), added to that package's `SPECS` tuple. A
   spec imports only the standard library: no pydantic, no JAX, never
   `parameters.py`. Registration alone extends the `phys.system`
   literal, the `--help`/TOML surfaces, `--sample-toml`, the snapshot
   metadata, the flow dispatch and the `analysis/_core.py` sets.
2. `flows/<family>/X.py`: the `flow_module`, exporting the surface the
   `FlowSpec` docstring lists (`get_perturbation_energy` is the cheap
   `E'` read behind the laminarization check). A base-flow flow also
   exports `frozen_profile_flow(profile)` (via
   `_base.frozen_profile_flow`; `test_transient_growth.py` pins each).
3. Declare `n_components` for a state that is not three velocity
   components (and add its branch to `analysis/_core.geometry_info`),
   and `total_field=True` for a flow that integrates the total field.
4. Add a row to `tests/test_laminar_smoke.py` and one to
   `tests/test_random_smoke.py` (`Re` above onset, a small box); they
   cover the base flow, so write no per-field unit tests for it. Three
   couplings there are kept by hand: the laminar smoke picks its CFL
   columns by system-name prefix and needs its own check branch for a
   flow without a perturbation `E'`, and the random smoke lists the
   cylindrical/annular names that take `--res.nz/nr/ntheta`.

- Laminar and base profiles are closed-form (e.g.
  `annular.dean_laminar_u_theta`), never discrete solves. Such a
  profile is only a near-fixed point on the FD grid, so its laminar
  smoke measures the deviation from the profile.
- A flow dataclass is a registered pytree, so every array field is
  traced into the steppers. Keep data needed only outside them (e.g. a
  total-field flow's laminar profile) at module level and pass it to
  the jitted diagnostics as an argument.
- `taylor-couette` and `quasi-keplerian` bind the same
  `wall_bounded/_circular_couette.py` and differ only in their
  parameterization.
- The registry's geometry, rheology and total-field axes overlap: read
  the comment in `registry.py` before writing an ordered branch.
