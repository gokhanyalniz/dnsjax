## Triply-periodic geometry

### Module layout

- `triply_periodic.py`: Fourier class, spectral differential operators
  (derivative/curl/div/grad/laplacian/inverse-laplacian), norms,
  `TriplyPeriodicFlow` base dataclass, `init_state`, the algebraic
  Helmholtz predict/correct, `correct_divergence`, and
  `build_triply_periodic_stepper`.

### Key differences from wall-bounded

- Helmholtz inversion is algebraic (pointwise multiply by `ildt_2`) --
  no matrix solves, so `dnsjax.solvers` is never involved and
  `solver.backend` is absent from the periodic parameter surfaces
  (the geometry never reads it). See
  `TriplyPeriodicFlow.__post_init__`. Adaptive dt (`set_dt`) just
  recomputes `ldt_1`/`ildt_2` (`_build_dt_leaves`) -- no stability
  check, fully continuous dt.
- Divergence correction is a two-stage projection, not an IMM: inside
  `_get_rhs` (pressure Poisson) and post-step (`correct_divergence` +
  mean-mode zeroing, fused into every stepper via `make_stepper`'s
  `finalize_fn` -- `_finalize_state`). See the `correct_divergence`
  docstring.
- Spectral layout: `(ny-1, nz_spec, nx_spec)` with `[ky, kz, kx]` --
  ky fully local, kz sharded by np0, kx sharded by np1.
- The 3D FFT extends the wall-bounded 2D reshard pipeline with a y-FFT
  step after reshard #2 brings the full y-extent local on each device
  (full pipeline: the `fft.py` module docstring). `np0` requires
  `ny_padded` divisibility; if unmet, `ny_padded` bumps to the next
  multiple.

### Flows

- `flows/triply_periodic/monochromatic.py`:
  MonochromaticFlow(TriplyPeriodicFlow) -- base flow and forcing for
  Kolmogorov / Waleffe; diagnostics (E, I, D, E').

### Tests

Kolmogorov is stepped by `tests/test_random_smoke.py` (random-IC
integration, incl. a cnab2 entry; `dt` capped at 0.005 there, a
corrector-rate limit), `tests/test_cnab2.py` (the plain no-corrector
AB2 path), and `tests/test_temporal_order.py` (the cnab2
self-convergence study). No operator-level unit tests yet; smoke-test
the other periodic flows manually via
`mpirun -np 1 .venv/bin/dnsjax --phys.system waleffe ...`.
