## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (norms, integration,
  `init_state`, `apply_y_matrix`, `extract_mean_mode`, `pad_base_flow`,
  `base_flow_coupling`, `build_wall_bounded_stepper`)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid,
  `CartesianFlow`, Kleiser-Schumann IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, radial CGL grid --
  half-CGL default under `iterative-cn`, rigged-CGL under `cnab2`
  (`geo.grid_type`), `CylindricalFlow`,
  decoupled u+/u- formulation, parity-reduced FD, 1x1 IMM,
  `interpolate_to_axis` r=0 evaluation)
- `annular.py`: annular geometry / concentric cylinders (Fourier, CGL
  grid on `[r1, r2]`, `AnnularFlow`, decoupled u+/u- formulation,
  2x2 IMM, optional mean-mode azimuthal body force `pi_theta`,
  `annular_forced_laminar_u_theta` / `dean_laminar_u_theta`)
- `annular_viscoelastic.py`: viscoelastic (sPTT) extension of the
  annular geometry (`ViscoelasticAnnularFlow`): 9-component state
  (3 velocity + 6 conformation-tensor **spin** components, the tensor
  analogue of `u_± = u_r ± i u_θ`); the tensor Laplacian diagonalises
  per spin (`m_eff = m + s`), each diffusing through a scalar
  Helmholtz solve (`Hc`); one fused pseudo-spectral RHS
  (`solver.rhs_transform_chunks` splits its 36-field inverse
  transform to cap peak memory; see `_get_rhs_core`); both schemes
  supported (FFT-free viscoelastic `_l_bf`). Spin diagonalisation,
  state layout, reality structure, cnab2 split: module docstring.

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps
`timestep.make_stepper()` and binds the `fourier`/`flow` singletons,
returning `(predict_and_correct, iterate_correction, init_state_bound,
predict_and_fully_correct, predict_and_fully_correct_measured,
step_cnab2, step_cnab2_measured)`. Each geometry module provides a
thin `build_*_stepper(flow)` passing its callables: the measured RHS
(CFL via the `rhs.py` `measure_fn` hook) and `_l_bf` -- the FFT-free
linear base-flow coupling `L_bf = u'×curl(U) + U×ω'` that wall-bounded
cnab2 **and** the opt-in split iterative-cn corrector
(`step.split_corrector`, default off; `_split_core` in `timestep.py`)
make implicit, built from the shared `base_flow_coupling`
helper. With `step.implicit_mean_coupling` (default on) each `_l_bf`
also folds in the instantaneous mean-flow coupling `L_mf` by adding
the `extract_mean_mode` profiles onto the base/curl profiles (still
FFT-free; for total-field Dean it is the *only* coupling). Why the
coupling is stiff: the `TimeStepping` docstring.
`tests/test_cnab2.py` pins the split exactness, the
machine-precision `L_mf` oracle, the FFT counts, and the
split-vs-unsplit corrector fixed-point equivalence.

**Moving frame (`phys.u_grid`)**: the convective-form frame term
`+i k₀ U_grid u'` is added spectrally in each geometry's
`_get_rhs_core` *and* `_l_bf` (identical expression, CN-implicit in
both schemes); the CFL diagnostic advects with
`flow.base_flow_adv_padded = U − U_grid·ê₀` from `pad_base_flow`.
`get_nonlin` keeps the lab-frame `base_flow_padded` -- do **not**
shift the cross-product velocity (the rotational split's explicit
`c·∂_y u'` half is wall-stiff; that was the removed first
implementation's instability).

### Influence-matrix method (IMM)

See `_imm_iteration` in `cartesian.py` for the full 9-stage algorithm,
mathematical equivalence (Schur complement / Woodbury), and the
optional constant-bulk-velocity and block-mean-spanwise-velocity
corrections.

- `params.solver.backend` selects the operator storage: `"pallas"`
  (default; per-mode banded sweep, `PerModeBandedPallasOperator`) or
  the `"dense"` reference (`DenseJAXSolver`, mathematically readable
  and the parity-test oracle; a wall-bounded run selecting it prints
  a warning) -- see the `solvers.py` docstrings. The pallas build is
  wired in all three geometries: each `_build_{Lk,Hk}_band_gpu`
  assembles directly in banded storage via the shared
  `solvers._assemble_banded_operator` helpers (cartesian: single
  shared `Hk`, row-constant diagonal shift (per-mode `k²`); annular:
  three stacked `Hk`; cylindrical: parity-selected base band),
  factored by the setup-checked no-pivot banded LU
  (`solvers._build_pallas_operator`: hard error on genuine LU
  instability, notice-and-proceed on mere ill-conditioning).
- Both backends apply `Lk` and `Hk_minus` matvecs matrix-free via
  `_lk_matvec` / `_hk_minus_matvec`, reconstructing from shared
  `D1`/`D2` FD matrices.
- All IMM homogeneous data is derived from the GPU operator by
  `CartesianFlow._derive_imm_homogeneous_data`.

### Mean mode and padding modes

Spectral padding slots (any `np0 > 1` run pads kz/m; `np1 > 1` pads
the streamwise axis) carry nonzero beyond-resolution placeholder
wavenumbers (`pad_harmonics` in `operators.py`), so the mean mode is
the only k^2 = 0 mode and `Fourier.mean_mask` (one-hot at global index
(0,0)) is the single mask: it selects the operator pin row, the
`M_inv` mean branch, and all mean-mode physics. Padding modes need no
special-casing: their per-mode operators are regular, the forward FFT
re-zeroes their slots on every evaluation, and the IMM corrections
vanish there. See the `Fourier` docstrings in
`cartesian.py`/`cylindrical.py`.

### Cylindrical geometry

Documented in the `cylindrical.py` module docstring: decoupled
`u+`/`u-` formulation (Willis 2017), effective azimuthal modes,
parity-reduced FD matrices, radial CGL grid, 1x1 influence matrix,
constant-bulk-velocity enforcement. Radial grid construction
(half-CGL: the `iterative-cn` default; rigged-CGL: the `cnab2`
default; near-axis spacing, cnab2 rationale): the
`build_radial_cgl_grid` and `TimeStepping` docstrings.
Full-disc radial quadrature is **parity-specific** spectral
Clenshaw-Curtis (`fd.cgl_radial_quadrature_weights` →
`y_weights`/`y_weights_odd`; each diagnostic integrates with its known
parity -- even for `|u|²`/mean `u_z`/dissipation, odd for the mean
`u_θ`); custom/tanh grids and the JAX-free analysis package use the
parity-agnostic axis-augmented composite rule instead. There is no
grid point at `r = 0`; `interpolate_to_axis` evaluates centreline
values by Fornberg extrapolation with optional per-mode parity (see
its docstring).

### Annular geometry

Fourier slot mapping (cylindrical and annular): `nx`→axial (real-FFT
`k_z`), `nz`→azimuthal (complex `m`), `ny`→radial. So for
Taylor-Couette the streamwise (azimuthal) resolution is `nz` and
spanwise (axial) is `nx` -- swapped vs. the Cartesian `nx`=streamwise
convention (see the `Fourier` coordinate-mapping tables in
`annular.py`/`cylindrical.py`).

Same decoupled `u+`/`u-` formulation as cylindrical but **two walls**
and **no `r=0` axis** (`r1 > 0`): a single CGL grid affinely mapped to
`[r1, r2]` (no parity reduction), Dirichlet/Neumann BCs at both walls,
a **2x2 influence matrix**; the mean mode keeps `u_r ≡ 0` with
`M_inv = 0` there. `block_mean_spanwise_velocity` zeroes the mean
**axial** velocity. Two driving modes via the same infrastructure (see
the `annular.py` module docstring): **shear-driven** Taylor-Couette
(perturbation `u'`, coupling only via `base_flow`) and
**force-driven** Dean (total field, `base_flow = 0`, mean-mode
azimuthal body force `AnnularFlow.pi_theta` added in `_get_rhs_core`);
the **viscoelastic** third mode extends the total-field infrastructure
with the coupled conformation tensor (`annular_viscoelastic.py`
above).

### Custom wall-normal grids

Grid selection precedence: (1) `params.geo.wall_grid` file path
(always overrides generation), (2) `params.geo.grid_type` (`"cgl"` /
`"half-cgl"` / `"tanh"`; half-CGL is cylindrical + `iterative-cn`
only, enforced by `validate_parameters`), (3) default, resolved to a
concrete `grid_type` by `update_parameters` (so snapshots embed the
grid they ran and resumes pin it): full CGL (Cartesian/annular);
cylindrical half-CGL under `iterative-cn`, rigged-CGL under `cnab2`.
Quadrature: spectral
Clenshaw-Curtis on the CGL grids (`fd.clenshaw_curtis_weights`,
annular affine-mapped; cylindrical: the parity-specific
`fd.cgl_radial_quadrature_weights`), the parity-agnostic `fd_order`
composite rule on custom/tanh grids. File format, validation, and
weights: the `build_{cartesian,cylindrical,annular}_grid` docstrings;
tanh-grid properties: `fd.py` (`tanh_two_sided_grid`,
`tanh_one_sided_grid`).

### Wall-normal interpolation

When a loaded snapshot's wall-normal grid differs,
`_interpolate_if_needed` in `__main__.py` applies the optimal method:
Chebyshev coefficients for Cartesian/annular CGL, the spectral
per-mode **parity interpolation** for cylindrical half/rigged-CGL
grids (`fd.cgl_parity_interpolation_matrices`, near machine
precision), and the local Fornberg `local_interpolation_matrix`
fallback for custom/tanh/undetected grids. See its docstring and the
`fd.py` interpolation docstrings (`chebyshev_interpolation_matrix`,
`cgl_axis_gap`, `axis_extrapolation_weights`,
`build_interpolation_matrix`).

### Optimization patterns

When the aim is to operate on a quantity derivable from the mean mode,
first index to the mean mode (`extract_mean_mode`) and then operate,
when the indexing and the operation commute.

FD matvecs via `apply_y_matrix` batch over the leading component axis
(a pure batch dim), so `D1`/`D2` GEMMs can be regrouped or
deduplicated across IMM stages without changing numerics
(bit-identical); the laminar smoke test's `err=0.00e+00` confirms such
refactors.

**Matvec layout (transpose-free GEMMs)**: stacking matvec inputs
y-leading and passing `apply_y_matrix(..., component_axis=1)` (and the
matching `.solve` `component_axis` arg of both solver backends)
keeps the cuBLAS GEMMs transpose-free. The curl/divergence matvecs and
the cylindrical/annular Hk-construction stacks are y-leading;
Cartesian's Hk stays component-leading on purpose (converting would be
a net loss). Full rationale, the one remaining `vmap` output
transpose, and the deferred kernel refactor: the `apply_y_matrix`
docstring (`_base.py`), the `PerModeBandedPallasOperator.solve`
docstring (`solvers.py`), and the comment at the `_imm_iteration` Hk
stage (`cartesian.py`); `scripts/pallas_solve_profile.py` is the
profiling harness that attributed the transposes.

### Flows

- `plane_couette.py`: PlaneCouetteFlow(CartesianFlow) -- `U(y) = y`
  with tilt.
- `plane_poiseuille.py`: PlanePoiseuilleFlow(CartesianFlow) --
  `Us = 1 - y^2` with tilt.
- `pipe.py`: PipeFlow(CylindricalFlow) -- `Uz = 1 - r^2`.
- `taylor_couette.py`: TaylorCouetteFlow(AnnularFlow) --
  circular-Couette `Uθ = A0 r + B0/r` from `(re1, re2, eta)`;
  shear-driven stats (`I_lam = D_lam = 4 B0^2 / (Re r1^2 r2^2)`).
- `dean.py`: DeanFlow(AnnularFlow) -- force-driven from `(re, eta)`,
  `Π_θ = (2η+2)/(r Re (1−η))`; integrates the **total** field;
  `start_from_laminar` uses the analytical `dean_laminar_u_theta`;
  `E'` is the kinetic energy of the deviation from that profile.
- `viscoelastic_dean.py`: ViscoelasticDeanFlow(ViscoelasticAnnularFlow)
  -- force-driven sPTT Dean (Dedalus-native normalisation,
  `Π_θ = (r1+r2)/(Re r)`, radii `(δ, δ+2)`, solvent viscosity
  `ν = β/Re`); 9-component **total** field; `start_from_laminar` uses
  the analytical velocity + pointwise sPTT-equilibrium conformation
  (exact discrete fixed point at `ε = κ = 0`); stats add polymer work
  `W_p`, elastic energy `E_p`, mean trace `TrC` (energy balance
  `I ≈ D_s − W_p`).

**Transient-growth hook**: each base-flow flow above (all except
dean/viscoelastic-dean) exports `frozen_profile_flow(profile)`, used by
`dnsjax.analysis.transient_growth` to linearise around an arbitrary
wall-normal *total* profile. It builds the geometry's
`(base_flow, curl_base_flow)` pair from the profile (Cartesian:
`tilted_profile_arrays` tilt split; pipe: even-parity `D1_pos+D1_ghost`
radial derivative; TC: `ω_z = D1·U_θ + U_θ/r`) and returns a shallow
flow copy via `_base.frozen_profile_flow` -- the Lk/Hk/IMM operators
are profile-independent, so they are shared and the jitted stepper does
not retrace. `tests/test_transient_growth.py` pins each hook against
the builtin laminar coupling.

### Tests

Relevant files: `test_cartesian.py`, `test_cylindrical.py`,
`test_annular.py`, `test_viscoelastic.py` (per-geometry operators),
`test_integration.py`, `test_mean_mask.py`, `test_cnab2.py`,
`test_laminar_smoke.py`, `test_random_smoke.py`,
`test_rolls_smoke.py`, `test_localized_rolls.py`. One-line
descriptions in the root CLAUDE.md Tests section; detail in each
test's module docstring.
