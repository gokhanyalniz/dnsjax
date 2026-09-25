---
paths:
  - "src/dnsjax/ic/*.py"
  - "src/dnsjax/{seeding,bootstrap}.py"
  - "src/dnsjax/twin/driver.py"
  - "src/dnsjax/extensions/forcing.py"
  - "src/dnsjax/analysis/transient_growth.py"
  - "scripts/{snapshot_perturb,ensemble_setup,random_ic_calibrate}.py"
  - "tests/test_{mean_mode,localized_rolls,rolls_smoke,random_smoke,seeding,snapshot_perturb,forcing}.py"
---

# Initial conditions, seeds and the mean mode

- Start modes, their knobs and their precedence: the
  `parameters.Initiation` docstring. Construction: the
  `ic/random_field.py` (the default mode) and `ic/localized_rolls.py`
  module docstrings; why `init.random_smoothness` is deliberately not
  calibrated: `ic/random_field.py`.
- Every seed defaults to unset, meaning "draw one": `init.random_seed`,
  `twin.seed`, `force.seed`, `scripts/ensemble_setup.py --seed-base`.
  The contract (draw, agree across processes, print, record, refuse
  only a seed the run would draw with): `seeding.py`. The solver
  resolves them in `bootstrap.resolve_run_seeds`, between
  `configure_jax_runtime` and the `sharding` import; `dnsjax-twin`
  resolves `twin.seed` itself, after its paired-resume decision. The
  seeds in the benchmark and diagnostic scripts stay fixed on purpose.
- Only the Cartesian flows may perturb the `(kx, kz) = (0, 0)` mode,
  and only through the conservation laws in `ic/mean_mode.py`, so that
  a state and its perturbed copy are the same flow, driven
  identically. Never restate those laws elsewhere. By route:
  - opt-in, default off: `init.random_mean_flow` (plane Couette and
    plane Poiseuille) and `scripts/snapshot_perturb.py --perturb.mode
    "0,0"`, which checks the profile and refuses rather than reshaping
    it;
  - default on: `twin.mean_flow`, a `[twin]` field of its own because
    `init.*` is inherited from snapshots in both directions;
  - deferred (`DeferredSpec`) on every other flow, Kolmogorov
    included; rejected for `[force]` kicks and in transient growth;
  - localized rolls stay mean-free.

  Guards: `tests/test_mean_mode.py`, `test_localized_rolls.py`,
  `test_forcing.py`, `test_snapshot_perturb.py`.
