## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (norms, integration, init_state, `build_wall_bounded_stepper`, `extract_mean_mode`)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid, `CartesianFlow`, Kleiser-Schumann IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, half-CGL grid, `CylindricalFlow`, decoupled u+/u- formulation, parity-reduced FD, 1x1 IMM)

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps `timestep.make_stepper()` and binds the `fourier` and `flow` singletons, returning `(predict_and_correct, iterate_correction, init_state_bound, predict_and_fully_correct)`. Each geometry module provides a thin `build_*_stepper(flow)` that passes geometry-specific callables to it.

### Influence-matrix method (IMM)

See `_imm_iteration` in `cartesian.py` for the full 9-stage algorithm, mathematical equivalence (Schur complement / Woodbury), and the optional constant-bulk-velocity and block-mean-spanwise-velocity corrections.

- `params.solver.backend` selects the operator storage: `"banded"` (default, SPIKE algorithm -- see `solvers.py`) or `"dense"` (full `(Ny, Ny)` matrices via `DenseJAXSolver`).
- Both backends apply `Lk` and `Hk_minus` matvecs matrix-free via `_lk_matvec` / `_hk_minus_matvec`, reconstructing from shared `D1`/`D2` FD matrices.
- All IMM homogeneous data is derived from the GPU operator by `CartesianFlow._derive_imm_homogeneous_data`.

### Mean-mode writes vs gauge fixing

`Fourier.k2_is_zero` is also True at zero-padded dummy modes (any `np0 > 1` run pads kz). Nonzero writes targeting the mean mode (e.g. bulk-velocity corrections) must use `Fourier.mean_mask` (one-hot at the true (0,0) mode). `k2_is_zero` must **stay** in the operator pin rows, `_lk_matvec`, and the `M_inv` mean branches: dummy-mode systems are mean-mode-like and singular without them (Inf/NaN factors, and `NaN * 0 = NaN` then poisons all modes). The zero-projections (mean-mode `v`/`u_r` and `d_wall` zeroing) also keep `k2_is_zero` deliberately — they pin dummy modes to zero each iteration. See the `Fourier` docstrings in `cartesian.py` / `cylindrical.py`.

### Cylindrical geometry

Documented in the `cylindrical.py` module docstring: decoupled `u+`/`u-` velocity formulation (Willis 2017), effective azimuthal modes, parity-reduced FD matrices, half-CGL radial grid, 1x1 influence matrix, and constant-bulk-velocity enforcement.

### Custom wall-normal grids

Grid selection (precedence): (1) `params.geo.wall_grid` file path, (2) `params.geo.grid_type` (`"tanh"` / `"cgl"`), (3) default CGL/half-CGL. See `build_cartesian_grid` and `build_cylindrical_grid` docstrings for file format, validation, and integration weights. Tanh grid properties (conditioning, CFL) are documented in `fd.py` (`tanh_two_sided_grid`, `tanh_one_sided_grid`).

### Wall-normal interpolation

When loading a snapshot with a different wall-normal grid, `_interpolate_if_needed` in `__main__.py` applies the optimal method: CGL-to-CGL (Chebyshev coefficients), half-CGL-to-half-CGL (parity-aware), or general (barycentric Lagrange). See `fd.py` functions: `chebyshev_interpolation_matrix`, `half_cgl_interpolation_matrices`, `barycentric_interpolation_matrix`, `build_interpolation_matrix`.

### Optimization patterns

When the aim is to operate on a quantity derivable from the mean mode (streamwise *and* spanwise wavenumber equal to zero), first index to the mean mode, and then operate, when this indexing and the desired operations commute. You can use the function `extract_mean_mode` for this purpose.

### Flows

- `flows/wall_bounded/plane_couette.py`: PlaneCouetteFlow(CartesianFlow) -- base flow U(y) = y with tilt
- `flows/wall_bounded/plane_poiseuille.py`: PlanePoiseuilleFlow(CartesianFlow) -- base flow Us = 1-y^2 with tilt
- `flows/wall_bounded/pipe.py`: PipeFlow(CylindricalFlow) -- base flow Uz = 1 - r^2

### Tests

- `tests/test_cartesian.py`: Cartesian operator and matvec tests
- `tests/test_cylindrical.py`: cylindrical operator and matvec tests
- `tests/test_integration.py`: quadrature weight and interpolation matrix tests
- `tests/test_mean_mask.py`: `mean_mask` vs `k2_is_zero` under forced spectral padding
- `tests/test_laminar_smoke.py`: laminar time-stepping smoke tests for all wall-bounded flows
