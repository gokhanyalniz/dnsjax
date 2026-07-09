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
  (3 velocity + 6 conformation-tensor spin components), one fused
  pseudo-spectral RHS, both schemes supported. Spin diagonalisation,
  state layout, reality structure, cnab2 split: module docstring.

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps
`timestep.make_stepper()`, binds the `fourier`/`flow` singletons, and
returns the stepping functions (see its docstring for the tuple). Each
geometry provides a thin `build_*_stepper(flow)` passing its measured
RHS (CFL via the `rhs.py` `measure_fn` hook) and `_l_bf` — the FFT-free
linear base-flow coupling `L_bf = u'×curl(U) + U×ω'` (from the shared
`base_flow_coupling` helper) that wall-bounded cnab2 and the opt-in
split iterative-cn corrector (`_split_core`) make implicit; with
`implicit_mean_coupling` (default on) `_l_bf` also folds in the
instantaneous mean-flow coupling `L_mf`. Why it is stiff: the
`TimeStepping` docstring. `tests/test_cnab2.py` pins the split
exactness, the `L_mf` oracle, FFT counts, and split-vs-unsplit
equivalence.

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

- `params.solver.backend` selects operator storage: `"pallas"`
  (default banded sweep) or the `"dense"` reference/oracle -- see the
  `solvers.py` docstrings. The pallas build is wired in all three
  geometries: each `_build_{Lk,Hk}_band_gpu` assembles directly in
  banded storage via the shared `solvers._assemble_banded_operator`
  (cartesian: one shared `Hk` + per-mode `k²` diagonal shift; annular:
  three stacked `Hk`; cylindrical: parity-selected band).
- Both backends apply `Lk`/`Hk_minus` matvecs matrix-free
  (`_lk_matvec`/`_hk_minus_matvec`) from shared `D1`/`D2`; IMM
  homogeneous data comes from
  `CartesianFlow._derive_imm_homogeneous_data`.

### Mean mode and padding modes

Spectral padding slots carry nonzero beyond-resolution placeholder
wavenumbers (`pad_harmonics` in `operators.py`), so the mean mode is
the only `k²=0` mode and `Fourier.mean_mask` (one-hot at global (0,0))
is the single mask selecting the operator pin row, the `M_inv` mean
branch, and all mean-mode physics. Padding modes need no special-casing
(regular per-mode operators; the forward FFT re-zeroes their slots; IMM
corrections vanish there). See the `Fourier` docstrings.

### Cylindrical geometry

The `cylindrical.py` module docstring documents the decoupled
`u+`/`u-` formulation (Willis 2017), effective azimuthal modes,
parity-reduced FD, radial CGL grid (half-CGL default under
`iterative-cn`, rigged-CGL under `cnab2`; see `build_radial_cgl_grid`),
1×1 influence matrix, and constant-bulk-velocity enforcement.
Cross-cutting gotchas: full-disc radial quadrature is
**parity-specific** (`fd.cgl_radial_quadrature_weights` →
`y_weights`/`y_weights_odd`; each diagnostic integrates with its known
parity), while custom/tanh grids and the JAX-free analysis package use
the parity-agnostic composite rule; there is no `r=0` grid point, so
`interpolate_to_axis` evaluates the centreline by Fornberg
extrapolation.

### Annular geometry

Fourier slot mapping (cylindrical and annular): `nx`→axial (real-FFT
`k_z`), `nz`→azimuthal (complex `m`), `ny`→radial. So for
Taylor-Couette the streamwise (azimuthal) resolution is `nz` and
spanwise (axial) is `nx` -- swapped vs. the Cartesian `nx`=streamwise
convention (see the `Fourier` coordinate-mapping tables in
`annular.py`/`cylindrical.py`).

Same decoupled `u+`/`u-` formulation as cylindrical but **two walls**,
**no `r=0` axis** (`r1 > 0`), no parity reduction, and a **2×2
influence matrix**. Three driving modes share the infrastructure (see
the `annular.py` module docstring): shear-driven Taylor-Couette
(perturbation `u'`), force-driven Dean (total field, mean-mode body
force `AnnularFlow.pi_theta`), and the viscoelastic total-field mode
with the coupled conformation tensor (`annular_viscoelastic.py`).

### Custom wall-normal grids

Grid selection precedence: (1) `params.geo.wall_grid` file (always
overrides), (2) `params.geo.grid_type` (`"cgl"`/`"half-cgl"`/`"tanh"`;
half-CGL is cylindrical + `iterative-cn` only), (3) default, resolved
to a concrete `grid_type` by `update_parameters` (so snapshots embed
the grid they ran): full CGL for Cartesian/annular, cylindrical
half-CGL under `iterative-cn` / rigged-CGL under `cnab2`. Quadrature is
spectral Clenshaw-Curtis on CGL grids, the `fd_order` composite rule on
custom/tanh grids. File format, validation, weights, and tanh-grid
properties: the `build_*_grid` and `fd.py` docstrings.

### Wall-normal interpolation

When a loaded snapshot's wall-normal grid differs,
`_interpolate_if_needed` (`__main__.py`) picks the optimal method
(Chebyshev for Cartesian/annular CGL, spectral parity interpolation for
cylindrical half/rigged-CGL, local Fornberg fallback for custom/tanh).
See its docstring and the `fd.py` interpolation docstrings.

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
  component-leading on purpose). Full rationale: the `apply_y_matrix`
  (`_base.py`) and `PerModeBandedPallasOperator.solve` (`solvers.py`)
  docstrings; `scripts/pallas_solve_profile.py` profiled it.

### Flows

- `plane_couette.py`: PlaneCouetteFlow(CartesianFlow) -- `U(y) = y`
  with tilt.
- `plane_poiseuille.py`: PlanePoiseuilleFlow(CartesianFlow) --
  `Us = 1 - y^2` with tilt.
- `pipe.py`: PipeFlow(CylindricalFlow) -- `Uz = 1 - r^2`.
- `taylor_couette.py`: TaylorCouetteFlow(AnnularFlow) --
  circular-Couette `Uθ = A0 r + B0/r` from `(re1, re2, eta)`.
- `dean.py`: DeanFlow(AnnularFlow) -- force-driven **total** field from
  `(re, eta)`; `start_from_laminar` uses `dean_laminar_u_theta`.
- `viscoelastic_dean.py`: ViscoelasticDeanFlow(ViscoelasticAnnularFlow)
  -- force-driven sPTT Dean, 9-component **total** field, radii
  `(δ, δ+2)`; stats add polymer work/elastic energy/mean trace.

**Transient-growth hook**: each base-flow flow (all except
dean/viscoelastic-dean) exports `frozen_profile_flow(profile)`, used by
`dnsjax.analysis.transient_growth` to linearise around an arbitrary
wall-normal *total* profile: it builds the geometry's
`(base_flow, curl_base_flow)` pair and returns a shallow flow copy via
`_base.frozen_profile_flow` (operators are profile-independent, so the
jitted stepper does not retrace). `tests/test_transient_growth.py` pins
each hook against the builtin laminar coupling.

### Tests

Relevant files: `test_cartesian.py`, `test_cylindrical.py`,
`test_annular.py`, `test_viscoelastic.py` (per-geometry operators),
`test_integration.py`, `test_mean_mask.py`, `test_cnab2.py`,
`test_laminar_smoke.py`, `test_random_smoke.py`,
`test_rolls_smoke.py`, `test_localized_rolls.py`. One-line
descriptions in the root CLAUDE.md Tests section; detail in each
test's module docstring.
