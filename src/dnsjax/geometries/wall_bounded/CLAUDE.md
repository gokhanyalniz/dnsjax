## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (norms, integration, init_state, `build_wall_bounded_stepper`, `extract_mean_mode`)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid, `CartesianFlow`, Kleiser-Schumann IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, half-CGL grid, `CylindricalFlow`, decoupled u+/u- formulation, parity-reduced FD, 1x1 IMM)
- `annular.py`: annular geometry / concentric cylinders (Fourier, CGL grid on `[r1, r2]`, `AnnularFlow`, decoupled u+/u- formulation, 2x2 IMM)

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps `timestep.make_stepper()` and binds the `fourier` and `flow` singletons, returning `(predict_and_correct, iterate_correction, init_state_bound, predict_and_fully_correct, predict_and_fully_correct_measured)`. Each geometry module provides a thin `build_*_stepper(flow)` that passes geometry-specific callables to it, including the measured RHS variant (`_get_rhs_measured`, which computes the CFL via the `rhs.py` `measure_fn` hook from the per-flow `cfl_inv_spacing` profile).

### Influence-matrix method (IMM)

See `_imm_iteration` in `cartesian.py` for the full 9-stage algorithm, mathematical equivalence (Schur complement / Woodbury), and the optional constant-bulk-velocity and block-mean-spanwise-velocity corrections.

- `params.solver.backend` selects the operator storage: `"banded"` (default, SPIKE algorithm -- see `solvers.py`) or `"dense"` (full `(Ny, Ny)` matrices via `DenseJAXSolver`).
- Both backends apply `Lk` and `Hk_minus` matvecs matrix-free via `_lk_matvec` / `_hk_minus_matvec`, reconstructing from shared `D1`/`D2` FD matrices.
- All IMM homogeneous data is derived from the GPU operator by `CartesianFlow._derive_imm_homogeneous_data`.

### Mean mode and padding modes

Spectral padding slots (any `np0 > 1` run pads kz/m; `np1 > 1` pads the streamwise axis) carry nonzero beyond-resolution placeholder wavenumbers (`pad_harmonics` in `operators.py`), so the mean mode is the only k^2 = 0 mode and `Fourier.mean_mask` (one-hot at global index (0,0)) is the single mask: it selects the operator pin row (`_build_Lk_*`, `_lk_matvec`), the `M_inv` mean branch, and all mean-mode physics (the `v`/`u_r` projections, `d_wall` gauge zeroing, bulk-velocity writes). Padding modes need no special-casing: their per-mode operators are regular, the forward FFT re-zeroes their slots on every evaluation (zero RHS -> zero solution), and the IMM corrections vanish there. See the `Fourier` docstrings in `cartesian.py` / `cylindrical.py`.

### Cylindrical geometry

Documented in the `cylindrical.py` module docstring: decoupled `u+`/`u-` velocity formulation (Willis 2017), effective azimuthal modes, parity-reduced FD matrices, half-CGL radial grid, 1x1 influence matrix, and constant-bulk-velocity enforcement.

### Annular geometry

Fourier slot mapping (cylindrical and annular): `nx`→axial (real-FFT `k_z`), `nz`→azimuthal (complex `m`), `ny`→radial. So for Taylor-Couette the streamwise (azimuthal) resolution is `nz` and spanwise (axial) is `nx` — swapped vs. the Cartesian `nx`=streamwise convention (see the `Fourier` coordinate-mapping tables in `annular.py`/`cylindrical.py`).

Documented in the `annular.py` module docstring: the same decoupled `u+`/`u-` formulation and azimuthal/axial Fourier layout as cylindrical, but with **two walls** and **no `r=0` axis** (`r1 > 0`). Consequences: a single CGL grid affinely mapped to `[r1, r2]` (no parity-reduced FD, no ghost matrices, no `m_is_even` operator selection), Dirichlet/Neumann BCs at both walls, and a **2x2 influence matrix** (the cylindrical `u+`/`u-` divergence and pressure gradient combined with the Cartesian two-wall Schur reduction). Shear-driven (no mean pressure gradient): base coupling enters only through `base_flow`/`curl_base_flow` in the rotational-form `rhs.py` (no hand-coded coupling terms); the optional `block_mean_spanwise_velocity` zeroes the mean **axial** velocity. The mean mode keeps `u_r ≡ 0` (continuity + both no-slip walls) with `M_inv = 0` there (the wall divergence residual vanishes, so the IMM correction does too).

### Custom wall-normal grids

Grid selection (precedence): (1) `params.geo.wall_grid` file path, (2) `params.geo.grid_type` (`"tanh"` / `"cgl"`), (3) default CGL/half-CGL (annular: CGL mapped to `[r1, r2]`). See `build_cartesian_grid`, `build_cylindrical_grid`, and `build_annular_grid` docstrings for file format, validation, and integration weights. Tanh grid properties (conditioning, CFL) are documented in `fd.py` (`tanh_two_sided_grid`, `tanh_one_sided_grid`).

### Wall-normal interpolation

When loading a snapshot with a different wall-normal grid, `_interpolate_if_needed` in `__main__.py` applies the optimal method: CGL-to-CGL (Chebyshev coefficients; for annular, the affine map `[r1,r2] -> [-1,1]` is detected so the same Chebyshev path applies), half-CGL-to-half-CGL (parity-aware), or general (barycentric Lagrange). See `fd.py` functions: `chebyshev_interpolation_matrix`, `half_cgl_interpolation_matrices`, `barycentric_interpolation_matrix`, `build_interpolation_matrix`.

### Optimization patterns

When the aim is to operate on a quantity derivable from the mean mode (streamwise *and* spanwise wavenumber equal to zero), first index to the mean mode, and then operate, when this indexing and the desired operations commute. You can use the function `extract_mean_mode` for this purpose.

FD matvecs via `apply_y_matrix` batch over the leading component axis (a pure batch dim), so `D1`/`D2` GEMMs can be regrouped or deduplicated across IMM stages without changing numerics (bit-identical); the laminar smoke test's `err=0.00e+00` confirms such refactors.

### Flows

- `flows/wall_bounded/plane_couette.py`: PlaneCouetteFlow(CartesianFlow) -- base flow U(y) = y with tilt
- `flows/wall_bounded/plane_poiseuille.py`: PlanePoiseuilleFlow(CartesianFlow) -- base flow Us = 1-y^2 with tilt
- `flows/wall_bounded/pipe.py`: PipeFlow(CylindricalFlow) -- base flow Uz = 1 - r^2
- `flows/wall_bounded/taylor_couette.py`: TaylorCouetteFlow(AnnularFlow) -- circular-Couette base flow Uθ = A0 r + B0/r from `(re1, re2, eta)`; shear-driven stats (I_lam = D_lam = 4 B0^2 / (Re r1^2 r2^2))

### Tests

- `tests/test_cartesian.py`: Cartesian operator and matvec tests
- `tests/test_cylindrical.py`: cylindrical operator and matvec tests
- `tests/test_annular.py`: annular operator/matvec tests, 2x2 SPIKE-vs-dense parity, circular-Couette A0/B0 coefficient checks
- `tests/test_integration.py`: quadrature weight and interpolation matrix tests
- `tests/test_mean_mask.py`: placeholder padding wavenumbers and `mean_mask` as the unique k^2 = 0 mode under forced spectral padding
- `tests/test_laminar_smoke.py`: laminar time-stepping smoke tests for all wall-bounded flows (incl. taylor-couette)
