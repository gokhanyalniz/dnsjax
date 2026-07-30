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
- Physical layout: `[y, z, x]` (`y` sharded by np0, `z` by np1, `x`
  local); axes are (shear, spanwise, streamwise). Velocity components
  `(u_x, u_y, u_z)` = (streamwise, shearwise, spanwise) -- the same
  (streamwise, wall-normal/shear, spanwise) order as Cartesian (`y` is
  the mean-shear direction here, `L_y = 4`).
- The 3D FFT extends the wall-bounded 2D reshard pipeline with a y-FFT
  step after reshard #2 brings the full y-extent local on each device
  (full pipeline: the `fft.py` module docstring). `np0` requires
  `ny_padded` divisibility; if unmet, `ny_padded` bumps to the next
  7-smooth multiple (`round_up_padded_smooth`, `parameters.py`).

### Flows

- `flows/triply_periodic/monochromatic.py`:
  MonochromaticFlow(TriplyPeriodicFlow) -- base flow and forcing for
  Kolmogorov; diagnostics (E, I, D, E').

### Tests

Kolmogorov is stepped by `tests/test_random_smoke.py` (random-IC
integration: the base iterative-cn entry caps `dt` at 0.005 -- a
corrector-rate limit; the corrector-free cnab2 entry runs the default
`dt`; plus adaptive and nan-guard entries), `tests/test_cnab2.py`
(the plain no-corrector
AB2 path), and `tests/test_temporal_order.py` (the cnab2
self-convergence study); `tests/test_monochromatic.py` pins the
`get_stats` diagnostics (Parseval/physical-space enstrophy identity,
laminar limits).
