## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (norms, integration,
  `init_state`, `apply_y_matrix`, `extract_mean_mode`, `pad_base_flow`,
  `base_flow_coupling`, `build_wall_bounded_stepper`)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid, `CartesianFlow`,
  Kleiser-Schumann IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, radial CGL grid,
  `CylindricalFlow`, decoupled u+/u- formulation, parity-reduced FD,
  1x1 IMM, `interpolate_to_axis` r=0 evaluation)
- `annular.py`: annular geometry / concentric cylinders (Fourier, CGL
  grid on `[r1, r2]`, `AnnularFlow`, decoupled u+/u- formulation, 2x2
  IMM, optional mean-mode azimuthal body force `pi_theta`)
- `annular_viscoelastic.py`: viscoelastic (sPTT) extension of the
  annular geometry (`ViscoelasticAnnularFlow`): 9-component state, one
  fused pseudo-spectral RHS, both schemes supported

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps
`timestep.make_stepper()`, binds the `fourier`/`flow` singletons, and
returns the stepping functions — plus the adaptive-dt hooks
`set_dt`/`reset_ab2_kappa`: a jitted rebuild of the dt-dependent
operator/IMM leaves (each geometry's `_build_dt_leaves`, unchecked
pallas factorization) swapped onto the flow in place, no stepper
recompilation. Each geometry provides a thin
`build_*_stepper(flow)` passing its measured RHS (CFL via the `rhs.py`
`measure_fn` hook) and `_l_bf` — the FFT-free linear base-flow coupling
(from the shared `base_flow_coupling` helper) that wall-bounded cnab2
and the opt-in split iterative-cn corrector make implicit. Why it is
stiff, and what `implicit_mean_coupling` folds in: the `TimeStepping`
docstring. Guards: `tests/test_cnab2.py`.

**Moving frame (`phys.u_grid`)**: the convective frame term is added
spectrally in each geometry's `_get_rhs_core` *and* `_l_bf`
(CN-implicit in both schemes); the CFL diagnostic advects with
`flow.base_flow_adv_padded` from `pad_base_flow`. `get_nonlin` keeps
the lab-frame `base_flow_padded` — do **not** shift the cross-product
velocity (the rotational split's explicit `c·∂_y u'` half is wall-stiff;
that instability removed the first implementation).

### Influence-matrix method (IMM)

`_imm_iteration` in `cartesian.py` documents the full 9-stage
algorithm, its Schur-complement/Woodbury equivalence, and the optional
constant-bulk-velocity / block-mean-spanwise-velocity corrections.

- `params.solver.backend` selects operator storage: `"pallas"` (default
  banded sweep) or the `"dense"` reference/oracle -- see the
  `solvers.py` docstrings. The pallas build is wired in all three
  geometries: each `_build_{Lk,Hk}_band_gpu` assembles directly in
  banded storage via the shared `solvers._assemble_banded_operator`.
- Both backends apply `Lk`/`Hk_minus` matvecs matrix-free
  (`_lk_matvec`/`_hk_minus_matvec`) from shared `D1`/`D2`; IMM
  homogeneous data comes from
  `CartesianFlow._derive_imm_homogeneous_data`.

### Mean mode and padding modes

Spectral padding slots carry nonzero placeholder wavenumbers
(`pad_harmonics` in `operators.py`), so the mean mode is the only `k²=0`
mode and `Fourier.mean_mask` is the single mask selecting the operator
pin row, the `M_inv` mean branch, and all mean-mode physics. Padding
modes need no special-casing. See the `Fourier` docstrings.

### Cylindrical geometry

The `cylindrical.py` module docstring documents the decoupled `u+`/`u-`
formulation, effective azimuthal modes, parity-reduced FD, the radial
CGL grid, the 1×1 influence matrix, and constant-bulk-velocity
enforcement. Cross-cutting gotchas:

- Full-disc radial quadrature is **parity-specific**
  (`fd.cgl_radial_quadrature_weights` → `y_weights`/`y_weights_odd`;
  each diagnostic integrates with its known parity), while custom/tanh
  grids and the JAX-free analysis package use the parity-agnostic
  composite rule.
- There is no `r=0` grid point: `interpolate_to_axis` evaluates the
  centreline via `fd.axis_extrapolation_weights`.

### Annular geometry

Fourier slot mapping (cylindrical and annular): `nx`→axial (real-FFT
`k_z`), `nz`→azimuthal (complex `m`), `ny`→radial. So for Taylor-Couette
the streamwise (azimuthal) resolution is `nz` and spanwise (axial) is
`nx` -- **swapped vs. the Cartesian `nx`=streamwise convention** (see the
`Fourier` coordinate-mapping tables in `annular.py`/`cylindrical.py`).

Same decoupled `u+`/`u-` formulation as cylindrical but **two walls**,
**no `r=0` axis** (`r1 > 0`), no parity reduction, and a **2×2
influence matrix**. Three driving modes share the infrastructure (see
the `annular.py` module docstring): shear-driven Taylor-Couette /
quasi-Keplerian (perturbation `u'`), force-driven Dean (total field,
mean-mode body force `AnnularFlow.pi_theta`), and the viscoelastic
total-field mode (`annular_viscoelastic.py`).

**Azimuthal wedge (`geo.m0`, annular and cylindrical)**: `geo.m0 > 1`
reduces the azimuthal domain to `θ ∈ [0, 2π/m0)` and resolves only
`m = m0·j`, genuinely cutting azimuthal cost/memory by `m0` at fixed
`nz` (all array/FFT sizes stay `nz`-driven; `m0` only scales wavenumber
values). Every `geo.lz` consumer follows automatically. Cylindrical
parity `m_is_even` tracks the *physical* `m0·j`, i.e. the correct r=0
axis-regularity per mode. Rejected for Cartesian / periodic /
viscoelastic in `validate_parameters`. The physical-space picture (why
the wedge is fully resolved rather than decimated) and the cost
argument: the `geo.m0` field docs (`parameters.py`).

### Custom wall-normal grids

Grid selection precedence: (1) `params.geo.wall_grid` file (always
overrides), (2) `params.geo.grid_type` — Cartesian/annular choices
`"cgl"`/`"tanh"`, cylindrical choices `"half-cgl"`/`"rigged-cgl"`/
`"half-tanh"` (half-CGL is `iterative-cn` only; each flow's surface
narrows the Literal), (3) the flow spec's default, resolved to a
concrete `grid_type` by `update_parameters` (so snapshots embed the
grid they ran): full CGL for Cartesian/annular, cylindrical half-CGL
under `iterative-cn` / rigged-CGL under `cnab2`. Quadrature is spectral
Clenshaw-Curtis on CGL grids, the `fd_order` composite rule on
custom/tanh grids. File format, validation, weights, and tanh-grid
properties: the `build_*_grid` and `fd.py` docstrings.

When a loaded snapshot's wall-normal grid differs,
`_interpolate_if_needed` (`__main__.py`) picks the optimal method; see
its docstring and the `fd.py` interpolation docstrings.

### Optimization patterns

- To operate on a mean-mode-derivable quantity, index the mean mode
  first (`extract_mean_mode`) then operate, when the two commute.
- `apply_y_matrix` FD matvecs batch over the leading component axis, so
  `D1`/`D2` GEMMs can be regrouped/deduplicated across IMM stages
  bit-identically (the laminar smoke `err=0.00e+00` confirms refactors).
- **Transpose-free GEMMs**: stacking matvec inputs y-leading with
  `apply_y_matrix(..., component_axis=1)` (and the matching `.solve`
  arg) keeps the cuBLAS GEMMs transpose-free (curl/divergence and the
  cyl/annular Hk stacks are y-leading; Cartesian's Hk stays
  component-leading on purpose). Rationale: the `apply_y_matrix`
  (`_base.py`) and `PerModeBandedPallasOperator.solve` (`solvers.py`)
  docstrings.

### Flows

- `plane_couette.py`: PlaneCouetteFlow(CartesianFlow) -- `U(y) = y`
  with tilt.
- `plane_poiseuille.py`: PlanePoiseuilleFlow(CartesianFlow) --
  `Us = 1 - y^2` with tilt.
- `pipe.py`: PipeFlow(CylindricalFlow) -- `Uz = 1 - r^2`.
- `taylor_couette.py`: circular-Couette `Uθ = A0 r + B0/r` from
  `(re1, re2, eta)` -- a thin binding of the shared
  `flows/wall_bounded/_circular_couette.py`
  (`CircularCouetteFlow(AnnularFlow)`, diagnostics, TG hook).
- `quasi_keplerian.py`: the same circular-Couette flow parameterized
  by `(re1, r_omega, eta)` on the quasi-Keplerian half-line `R_Ω < -1`
  (`re2` derived by its spec); binds the same `_circular_couette.py`
  machinery and differs only in its documented conventions.
- `dean.py`: DeanFlow(AnnularFlow) -- force-driven **total** field.
- `viscoelastic_dean.py`: ViscoelasticDeanFlow(ViscoelasticAnnularFlow)
  -- force-driven sPTT Dean, 9-component **total** field.

**Transient-growth hook**: each base-flow flow (all except
dean/viscoelastic-dean) exports `frozen_profile_flow(profile)`, used by
`dnsjax.analysis.transient_growth` to linearise around an arbitrary
wall-normal *total* profile via `_base.frozen_profile_flow` (operators
are profile-independent, so the jitted stepper does not retrace).
`tests/test_transient_growth.py` pins each hook against the builtin
laminar coupling.

### Tests

Relevant files: `test_cartesian.py`, `test_cylindrical.py`,
`test_annular.py`, `test_viscoelastic.py` (per-geometry operators),
`test_integration.py`, `test_mean_mask.py`, `test_cnab2.py`,
`test_laminar_smoke.py`, `test_random_smoke.py`,
`test_rolls_smoke.py`, `test_localized_rolls.py`. One-line
descriptions in the root CLAUDE.md Tests section; detail in each
test's module docstring.
