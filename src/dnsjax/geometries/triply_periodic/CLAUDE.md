## Triply-periodic geometry

### Module layout

- `triply_periodic.py`: Fourier class, spectral differential operators (derivative, curl, div, grad, laplacian, inverse_laplacian), norms, `TriplyPeriodicFlow` base dataclass, `init_state`, algebraic Helmholtz predict/correct, `correct_divergence`, `build_triply_periodic_stepper` factory

### Key differences from wall-bounded

- Helmholtz inversion is algebraic (pointwise multiply by `ildt_2`) -- no matrix solves, so `dnsjax.solvers` is never involved. `solver.backend` resolves to `"dense"` (the reference semantics) for periodic systems in `update_parameters()`; an explicit `"pallas"` is rejected there. See `TriplyPeriodicFlow.__post_init__` docstring.
- Divergence correction is a two-stage projection, not built into an IMM: inside `_get_rhs` (pressure Poisson) and post-step (`correct_divergence` + mean-mode zeroing, fused into every stepper via `make_stepper`'s `finalize_fn` -- `_finalize_state`, no separate per-step dispatch). See the `correct_divergence` docstring.
- Spectral layout: `(ny-1, nz_spec, nx_spec)` with `[ky, kz, kx]` -- ky fully local, kz sharded by np0, kx sharded by np1.

### Parallelization (double decomposition)

The 3D FFT extends the wall-bounded 2D reshard pipeline with an additional y-FFT step after reshard #2 brings the full y-extent local on each device:

- Forward: x-FFT -> [reshard #1: z<->kx] -> z-FFT -> [reshard #2: y<->kz] -> y-FFT
- Physical `(y_np0, z_np1, x)` -> spectral `(ky, kz_np0, kx_np1)`

See `fft.py` module docstring for the full reshard pipeline. `np0` requires `ny_padded` divisibility; if not met, `ny_padded` is automatically bumped to the next multiple.

### Flows

- `flows/triply_periodic/monochromatic.py`: MonochromaticFlow(TriplyPeriodicFlow) -- base flow and forcing for Kolmogorov / Waleffe / decaying-box; diagnostics (E, I, D, E')

### Tests

Kolmogorov is stepped by `tests/test_random_smoke.py` (random-IC
integration, incl. a cnab2 entry; `dt` capped at 0.005 there, a
corrector-rate limit), `tests/test_cnab2.py` (the plain no-corrector
AB2 path), and `tests/test_temporal_order.py` (the cnab2
self-convergence study). No operator-level unit tests yet; smoke-test
the other periodic flows manually via
`mpirun -np 1 python -m dnsjax --phys.system waleffe ...` (or
`decaying-box`).
