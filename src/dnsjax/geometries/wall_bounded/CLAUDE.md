## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (integrate_scalar, get_inprod/get_norm2/get_norm, get_pert_enstrophy, init_state, build_wall_bounded_stepper factory, phys_to_spec/spec_to_phys aliases, extract_mean_mode)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid, build_cartesian_grid, CartesianFlow, Kleiser-Schumann IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, half-CGL grid, build_half_cgl_grid, build_parity_reduced_matrices, build_cylindrical_grid, CylindricalFlow, get_pert_enstrophy_cyl, parity-reduced FD, decoupled u+/u- formulation, 1x1 IMM)

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps `timestep.make_stepper()` and binds the `fourier` and `flow` singletons into closures, returning `(predict_and_correct, iterate_correction, init_state_bound, predict_and_fully_correct)`. Each geometry module provides a thin `build_*_stepper(flow)` that passes its geometry-specific `_get_rhs` / `_predict` / `_correct` / `_norm` to `build_wall_bounded_stepper`. Flow modules call the builder at module level to expose the public interface consumed by `__main__`.

### Influence-matrix method (IMM)

- The pressure Poisson equation with preliminary Neumann BCs is solved via LU-factored matrices (`Lk`, `Hk`). The entire per-mode setup runs on the device: FD matrices `D1`/`D2` are built using JAX arrays with Python control flow (outside `@jit`) and distributed to devices once, after which `Lk` and `Hk` are assembled and factorised with no further host-device traffic.
- All IMM homogeneous data (`p1, p2, v1, v2, q1, q2, M_inv`) is derived by `CartesianFlow._derive_imm_homogeneous_data` from the already-factored GPU operator.
- `params.solver.backend` selects the operator-factor storage format:
  - `"banded"` (default): SPIKE algorithm (Polizzi & Sameh 2006) partitions each banded `(Ny, Ny)` operator into `P` contiguous blocks of size `m = Ny/P` (with `m >= 2p`, `p = params.res.fd_order`) and factors each block as a dense `(m, m)` LU via cuSOLVER's batched LU (`jax.scipy.linalg.lu_factor`). Spike matrices `V_i = A_i^{-1} B_i`, `W_i = A_i^{-1} C_i` capture off-block coupling. The reduced system of size `2Pp` is factored via either block-Thomas LU on its `P`-block tridiagonal structure (`params.solver.block_thomas = True`, default; memory `O(P*p^2)`) or dense LU (`False`; memory `O(P^2*p^2)`). At solve time, per-block LU solves, the reduced solve, and spike reconstruction are cuSOLVER-batched. No `(Nkz, Nkx, Ny, Ny)` array is ever materialised.
  - `"dense"`: builds the full `(Nkz, Nkx, Ny, Ny)` matrices on the GPU via `_build_Lk_dense_gpu`/`_build_Hk_dense_gpu`, LU-factors them on-device via `DenseJAXSolver`, then discards the originals -- a reference path kept for parity with the banded backend.
- Solver infrastructure: geometry-independent code (`DenseJAXSolver`, `PerModeBandedOperator`, `_spike_factor`, `_choose_block_partition`, `_extract_banded_corners`) lives in `solvers.py`; `_build_Lk_blocks_gpu`/`_build_Hk_blocks_gpu` in `cartesian.py` assemble the per-block dense operators and coupling corners using those helpers.
- Both backends apply `Lk` and `Hk_minus` matvecs matrix-free via `_lk_matvec` / `_hk_minus_matvec`, reconstructing the operator action on the fly from the shared `D2` / `D1` FD matrices (no per-mode operator matrices are stored).
- IMM iteration: homogeneous solutions (`p1`, `p2`) and influence matrix `M_inv` find the correct pressure BC from the normal derivative of wall-normal velocity at the wall; pressure is then solved with that BC, and velocity is updated with the corresponding pressure gradient. Operator factors and homogeneous data inherit the kx-sharded layout from the broadcast against `fourier.k2`.
- Constant bulk velocity: with `params.phys.driving == "constant_bulk_velocity"`, `CartesianFlow._precompute_bulk_response` solves `Hk h = 1` (zero Dirichlet wall BCs) at the mean mode; after each IMM iteration, the mean-mode streamwise velocity is corrected by `G * h` where `G = -Ub_pert / H_bulk` and `H_bulk = dot(y_weights, h) / 2`.
- Block mean spanwise velocity (Cartesian only): with `params.phys.block_mean_spanwise_velocity == True`, each IMM iteration additionally zeroes the perturbation bulk velocity in the spanwise direction `(-sin theta, 0, cos theta)`. Uses the same `h_bulk_response` / `H_bulk_inv` as the streamwise constant-bulk-velocity enforcement (the Helmholtz operator at the mean mode is identical for all horizontal velocity components). The two corrections are orthogonal and independent.
- Tilt: both Cartesian flows support tilted domains via `cos_tilt`/`sin_tilt` (from `derived_params.tilt_rad`), which rotate the streamwise direction in the (x, z) plane.

### Cylindrical geometry and decoupled velocity formulation

- The cylindrical Navier-Stokes vector Laplacian couples `u_r` and `u_theta` through `1/r^2` terms. Following Openpipeflow (Willis 2017), `cylindrical.py` decouples them via `u+ = u_r + i u_theta`, `u- = u_r - i u_theta`, reducing the vector problem to three scalar Helmholtz equations.
- Each component has an **effective azimuthal mode** `m_eff` that governs its scalar Laplacian structure (`D2 + (1/r)D1 - m_eff^2/r^2`): `m_eff = m+1` for `u+`, `m_eff = m-1` for `u-`, `m_eff = m` for `u_z`. Despite different `m_eff`, `u+` and `u-` share the **same parity** `(-1)^{m+1}` -- parity is kinematic (how a field transforms under `r -> -r` on the auxiliary grid), while `m_eff` determines the operator spectrum.
- Radial grid: half-CGL on `(0, 1]` with `Nr = ny` points, formed by taking the positive half of a `2Nr`-point CGL grid on `[-1, 1]`. No grid point falls at `r = 0`; regularity is enforced by parity-reduced FD matrices built by mirroring the grid and folding ghost unknowns: `D_reduced = D_pos +/- D_ghost_flipped`, where the sign depends on parity.
- Two base operators `A_base_even` and `A_base_odd` (`D2 + diag(1/r)*D1` with even/odd parity) differ only in the first ~p rows (near the centre).
- Three Helmholtz operators `Hk_plus`, `Hk_minus`, `Hk_z` are built per velocity component (with the appropriate `m_eff^2/r^2` diagonal shift), factored separately, then stacked into a single combined `Hk_op` with a leading component axis (order: plus, minus, z). Both `DenseJAXSolver` and `PerModeBandedOperator` support batched-operator dispatch: when the factor arrays have one extra leading dimension, `solve()` vmaps over both operator and RHS, issuing one batched kernel launch instead of three.
- The pipe has only one physical wall at `r = 1`, giving a `1x1` influence matrix (scalar `alpha` per mode) instead of the Cartesian `2x2`. Homogeneous data consists of 4 arrays (`p1`, `v_plus_1`, `v_minus_1`, `q_z_1`) plus scalar `M_inv`.
- SPIKE block construction reuses `solvers.py` with a parity-dependent first block: pre-built for both parities, selected per mode via `jnp.where`.
- Matrix-free matvecs decompose into a common part (`D_pos`) plus a parity-dependent ghost correction for the first ~p entries.
- Velocity ordering: state array stores `(u_z, u+, u-)`, matching the Cartesian convention of (streamwise, wall-normal, spanwise); the physical representation follows the same convention as `(u_z, u_r, u_theta)`. `_get_rhs` converts between the two for the nonlinear term, and `_curl_fn` implements the cylindrical curl in spectral space.
- Constant bulk velocity: with `params.phys.driving == "constant_bulk_velocity"`, each IMM iteration adds a uniform mean pressure gradient `G` to the mean-mode `u_z` Helmholtz RHS (via a Helmholtz-consistent post-solve correction `uz += G * h`, where `h = Hk_z^{-1} [1,...,1,0]` is precomputed) to enforce zero perturbation bulk velocity; `G = -Ub_pert / H_bulk` where `H_bulk = 2 int_0^1 h r dr`.

### Custom wall-normal grids

Wall-normal grid selection (precedence order):

1. `params.geo.wall_grid` (file path): load a custom grid from file.
2. `params.geo.grid_type`: `"tanh"` for tanh-stretched grid, `"cgl"` for default CGL/half-CGL. Combined with `params.geo.grid_stretch` (default 1.5).
3. Default: CGL (Cartesian) or half-CGL (cylindrical).

Setting both `wall_grid` and `grid_type` is an error.

**Built-in tanh grid** (`grid_type = "tanh"`): the default CGL grid clusters points as `O(1/N^2)` at the walls -- appropriate for Chebyshev spectral methods but suboptimal for order-`p` finite differences, which gain nothing from that clustering while inheriting `O(N^4)` second-derivative conditioning (Trefethen 2000; Weideman & Reddy 2000) and an `O(1/N^2)` convective CFL limit. A tanh-stretched grid with controlled wall spacing `~O(1/N)` achieves the same order-`p` accuracy with `O(N^2)` conditioning and an `O(1/N)` CFL limit. The stretching parameter `grid_stretch` (`s > 0`) controls wall clustering: larger values concentrate more points near the walls; as `s -> 0` the grid approaches uniform. Cartesian uses symmetric two-sided stretching (`tanh_two_sided_grid` in `fd.py`); cylindrical uses one-sided stretching toward the wall at `r=1` (`tanh_one_sided_grid`), excluding `r=0`.

**Custom file format**: one coordinate per line, in wall-to-interior order. Cartesian: first line = top wall (y=1), last line = bottom wall (y=-1). Cylindrical: first line = wall (r=1), last line = closest to centre. The code reverses to ascending order internally.
- **Validation**: file must exist (checked in `update_parameters`), have exactly `ny` values, and span the correct domain. Cartesian: [-1, 1]. Cylindrical: (0, 1] with all r > 0.
- **Integration weights**: CGL grids use Clenshaw-Curtis weights (spectral accuracy). Custom grids use `build_integration_weights` from `fd.py` (composite polynomial, order-p accuracy matching the FD stencil). Cylindrical always uses composite weights (both default and custom).
- **Operators**: `build_diff_matrices` (Fornberg's algorithm) and all downstream operator assembly (Lk, Hk, IMM) work on arbitrary monotonic grids. Parity-reduced matrices in cylindrical (`_build_parity_reduced_matrices`) mirror the grid and call `build_diff_matrices` on the auxiliary grid, which is valid for any sorted positive grid.
- **Snapshot metadata**: the grid is stored in `_dnsjax_meta.json` as `wall_normal_grid` (float array). On snapshot load, if the current grid differs from the snapshot's grid, the state is interpolated automatically.

### Wall-normal interpolation

When loading a snapshot with a different wall-normal grid (different `ny` or different point locations), `_interpolate_if_needed` in `__main__.py` applies the optimal interpolation. Utilities live in `fd.py`:

- **CGL-to-CGL** (`chebyshev_interpolation_matrix`): Chebyshev coefficient truncation/extension via DCT-I. Exact for polynomials of degree <= min(N_old, N_new) - 1. Spectrally optimal.
- **Half-CGL-to-half-CGL** (`half_cgl_interpolation_matrices`): parity-aware extension to full CGL grid, Chebyshev interpolation, restriction to positive half. Returns (T_even, T_odd) matrices. Parity depends on azimuthal mode m and velocity component: u_z uses (-1)^m, u_+/u_- use (-1)^{m+1}.
- **General** (`barycentric_interpolation_matrix`): barycentric Lagrange interpolation (Berrut & Trefethen 2004). Weights computed in log-space for stability.
- **Dispatch** (`build_interpolation_matrix`): selects optimal method based on grid type and geometry.
- **After interpolation**: wall BCs are enforced (zeroed). Residual divergence from the changed y-derivative operator is O(interpolation error) and is projected out by the first corrector step's pressure Poisson solve.

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
- `tests/test_laminar_smoke.py`: laminar time-stepping smoke tests for all wall-bounded flows
