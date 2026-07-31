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
- `_viscoelastic_common.py`: the geometry-free half of the sPTT
  extension, shared by both viscoelastic geometries (9-component state
  layout, spin <-> physical maps, spin/Frobenius weights, pointwise
  physical-space RHS kernel, div-c curvature assembly, CN/AB2 mean
  conformation coupling, sPTT scalar root, narrow Laplacian BC wall
  row); its docstring states what is deliberately *not* here
- `annular_viscoelastic.py`: sPTT extension of the **annular** geometry
  (`ViscoelasticAnnularFlow`): 9-component state, one fused
  pseudo-spectral RHS, both schemes supported
- `cylindrical_viscoelastic.py`: sPTT extension of the **cylindrical**
  geometry (`ViscoelasticCylindricalFlow`), the same 9-component state
  through the pipe's parity-reduced radial operators -- the tensor's
  axis parity and its single-wall `H_c` are what differ; derivation
  and per-component table: its module docstring

**Component basis (cylindrical / annular only).** Two
representations, one boundary. The **solver basis** — decoupled
`u_± = u_r ± i u_θ` plus the conformation-spin components, which
diagonalize the implicit operators — is the state's in-memory form:
the carried state, the RHS, the cnab2 carry, and the interior of every
stepper. The **physical basis** `(u_z, u_r, u_θ)` (+ the physical
tensor) is what is observed or persisted: snapshots, diagnostics,
probes, forcing profiles, ICs, the analysis package, the TG export. A
given state crosses at most once, never back (the physical form is a
view, dropped after use), via `_base.to_pm_basis`/`from_pm_basis`
(aliased `to_solver_basis` / `from_solver_basis`, re-exported by the
flow modules) or the 9-component
`_viscoelastic_common.to_spin_basis`/`from_spin_basis` (shared by both
viscoelastic geometries). `__main__`
owns the field-level crossings; `probes.py`/`forcing.py` convert their
own mode columns instead. **Anything that hands a freshly built (i.e.
physical) state to a stepper must convert first** — `__main__`'s
post-IC line and `transient_growth._linear_step` are the templates.
`_get_rhs_core`/`_l_bf` convert internally (the real FFT needs
per-component Hermitian symmetry — the `cylindrical.py` docstring), so
physical-space fields and the CFL measurement are always physical
components.
**Cartesian carries physical `(u, v, w)` in both `res.consistent_imm`
states** and exports no basis pair; every consumer finds it by
`getattr` and falls back to the identity.

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
velocity (wall-stiff; the why and the history: `pad_base_flow`).

### Influence-matrix method (IMM)

Every geometry has the same two-way split: `_imm_iteration` is a
trace-time dispatcher over `_imm_iteration_vp` (primitive, flag-off)
and `_imm_iteration_vw` (the reconstruction scheme, flag-on).
`cartesian._imm_iteration` carries the shared derivation — why there
are two schemes (the discrete-continuity residual) and the five
measured repairs, four of them retired — and the other two dispatchers
add only their geometry's amendment to that record.
`cartesian._imm_iteration_vp` documents the primitive 9-stage
algorithm, its Schur-complement/Woodbury equivalence, and the optional
constant-bulk-velocity / block-mean-spanwise-velocity corrections
(shared via `_apply_bulk_corrections`).
`annular._imm_iteration_vw` carries the **cylindrical** algebra
(the `(Φ, ω_r)` pair, the mandatory conservative curl, the exact
`L_v,mod` recovery, the mean packing) and the retired-route record for
decoupling the pair; `cylindrical._imm_iteration_vw` adds only what
the axis forces (the spin quad, parity classes, the band splice).

- `params.solver.backend` selects operator storage: `"pallas"` (default
  banded sweep) or the `"dense"` reference/oracle -- see the
  `solvers.py` docstrings. The pallas build is wired in all three
  geometries: each `_build_{Lk,Hk}_band_gpu` (plus
  `annular_viscoelastic._build_Hc_band_gpu`) assembles directly in
  banded storage via the shared `solvers._assemble_banded_operator`,
  with the band width **measured** from the assembled operator
  (`fd.matrix_half_bandwidth`, both flag states), never assumed to be
  `fd_order`.
- Both backends apply the `Lk` matvec matrix-free (`_lk_matvec` in
  each geometry; only Cartesian also names `_hk_minus_matvec`, the
  others build `H_k^-` inline) from shared `D1`/`D2`; IMM homogeneous
  data comes from each geometry's `_derive_imm_homogeneous_data`
  (+ the `_derive_vw_homogeneous_data` twins under
  `res.consistent_imm`).
- `res.consistent_imm` (default off, offered by every wall-bounded
  flow) makes the
  discrete continuity identity hold by **one
  mechanism** in all three geometries: advance the wall-normal
  velocity + vorticity, reconstruct the tangential pair, never form a
  pressure — so the residual is machine-eps and *flat* under
  refinement, for any operator or grid. Per geometry: Cartesian
  advances `(φ, v, ω_y)` (4 → 3 solves); annular the `u_r`–`ω_r` pair
  on one shared Helmholtz (`m_eff² = m²+1`; 4 → 3 solves, 3 band
  families); the pipe the spin quad `(Φ±, ω±)` over the *existing*
  `H_k±` families (5 solves; only its two free wall differences ride
  the corrector iterate). **No geometry
  changes what it carries** (scalars re-derived per pass and
  reconstructed away), so snapshots, probes/forcing, analysis and
  resume are all flag-independent. Derivation, per-geometry efficacy,
  momentum prices, and the five rejected routes: the
  `Resolution.consistent_imm` docs (`parameters.py`); the shared
  scheme record: `cartesian._imm_iteration` (+ `_imm_iteration_vw`);
  the cylindrical algebra: `annular._imm_iteration_vw`; the pipe's
  free wall values, why its solve count and cost go the other way, and
  the instability that lagging them caused:
  `cylindrical._imm_iteration_vw`. Guards:
  `tests/test_imm_continuity.py` (continuity + the momentum ledger),
  `tests/test_random_smoke.py` (the nonlinear stability gate),
  `tests/test_temporal_order.py` (the order the flag must not cost).
- **Retired 2026-07-26**: `res.pipe_axis_fit`, the pipe's opt-in
  `x = r²` axis-regular radial fit — not needed (divergence exactness
  is fit-independent) and worse on every global metric despite a real
  pointwise near-axis gain. Record + all measurements:
  `cylindrical.build_parity_reduced_matrices`.

### Mean mode and padding modes

`pad_harmonics` (`operators.py`) keeps padding-slot wavenumbers
nonzero **so that** the mean mode is the only `k²=0` mode and
`Fourier.mean_mask` is a one-hot — the cross-module invariant every
pin-row/mean-mode consumer relies on, and padding modes need no
special-casing. Detail: the `Fourier` docstrings; guard:
`tests/test_mean_mask.py`.

### Cylindrical geometry

The `cylindrical.py` module docstring documents the decoupled `u+`/`u-`
formulation, effective azimuthal modes, parity-reduced FD, the radial
CGL grid, and the 1×1 influence matrix; constant-bulk-velocity
enforcement: `CylindricalFlow._precompute_bulk_response`.
Cross-cutting gotchas:

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

**Velocity component order (the annular exception).** The physical
triad is `(u_z, u_r, u_θ)`, so the annulus does **not** follow the
`(streamwise, wall-normal, spanwise)` component order the other three
geometries obey: streamwise `u_θ` is component 2, spanwise `u_z` is
component 0. Why (axial-first shared with the pipe; a reorder is
left-handed and was rejected): the `annular.py` docstring.

Same decoupled `u+`/`u-` formulation as cylindrical but **two walls**,
**no `r=0` axis** (`r1 > 0`), no parity reduction, and a **2×2
influence matrix**. Two driving modes plus the viscoelastic
extension share the infrastructure (the `annular.py` "Driving"
section): shear-driven Taylor-Couette / quasi-Keplerian (perturbation
`u'`), force-driven Dean (total field, mean-mode body force
`AnnularFlow.pi_theta`), and the viscoelastic total-field mode
(`annular_viscoelastic.py`).

**Azimuthal wedge (`geo.m0`, every cylindrical/annular flow, both
viscoelastic flows included)**: `geo.m0 > 1` reduces the azimuthal
domain to `θ ∈ [0, 2π/m0)` and resolves only `m = m0·j`, genuinely
cutting azimuthal cost/memory by `m0` at fixed `nz` (all array/FFT
sizes stay `nz`-driven; `m0` only scales wavenumber values). Every
`geo.lz` consumer follows automatically. Cylindrical parity
`m_is_even` tracks the *physical* `m0·j` — the correct r=0
axis-regularity per mode — and the JAX-free analysis package must
mirror exactly that selector (`analysis/_core.radial_derivative`; a
harmonic-index pick silently corrupted every even-wedge pipe
snapshot's operators). Rejected for Cartesian and triply-periodic in
`validate_parameters`. The physical-space picture (why the wedge is
fully resolved rather than decimated) and the cost argument: the
`geo.m0` field docs (`parameters.py`). Guards:
`tests/test_quasi_keplerian.py` (wedge_nonlinear),
`tests/test_transient_growth.py` (wedge-vs-full-circle equivalence),
the `test_laminar_smoke.py` wedge entries, and the `m0 = 2` pipe row
in `tests/test_snapshot_export.py`.

### Custom wall-normal grids

Grid selection precedence: (1) `params.geo.wall_grid` file (always
overrides), (2) `params.geo.grid_type` — Cartesian/annular choices
`"cgl"`/`"tanh"`, cylindrical choices `"half-cgl"`/`"rigged-cgl"`/
`"half-tanh"` (half-CGL is `iterative-cn` only; each flow's surface
narrows the Literal), (3) the flow spec's default, resolved to a
concrete `grid_type` by `update_parameters` (rationale — snapshots
embed the resolved grid: the `Geometry` docstring): full CGL for
Cartesian/annular, cylindrical half-CGL
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
- `viscoelastic_pipe.py`:
  ViscoelasticPipeFlow(ViscoelasticCylindricalFlow) -- sPTT pipe driven
  by a uniform axial body force `Pi_z = 4/Re`, 9-component **total**
  field; `Uz = 1 - r^2` at `epsilon = 0`.
- `dean.py`: DeanFlow(AnnularFlow) -- force-driven **total** field.
- `viscoelastic_dean.py`: ViscoelasticDeanFlow(ViscoelasticAnnularFlow)
  -- force-driven sPTT Dean, 9-component **total** field.

**Transient-growth hook**: each base-flow flow (all except the
total-field dean/viscoelastic-dean/viscoelastic-pipe) exports
`frozen_profile_flow(profile)`, used by
`dnsjax.analysis.transient_growth` to linearise around an arbitrary
wall-normal *total* profile via `_base.frozen_profile_flow` (operators
are profile-independent, so the jitted stepper does not retrace).
`tests/test_transient_growth.py` pins each hook against the builtin
laminar coupling.

### Tests

Relevant files: `test_cartesian.py`, `test_cylindrical.py`,
`test_annular.py`, `test_viscoelastic.py`,
`test_viscoelastic_pipe.py` (per-geometry operators),
`test_banded_solver.py`, `test_banded_solver_sharded.py`,
`test_integration.py`, `test_mean_mask.py`, `test_cnab2.py`,
`test_imm_continuity.py`, `test_energy_budget.py`,
`test_temporal_order.py`, `test_adaptive.py`,
`test_laminar_smoke.py`, `test_random_smoke.py`,
`test_rolls_smoke.py`, `test_localized_rolls.py`,
`test_quasi_keplerian.py`, `test_probes.py`, `test_forcing.py`,
`test_transient_growth.py`. One-line
descriptions in the root CLAUDE.md Tests section; detail in each
test's module docstring.
