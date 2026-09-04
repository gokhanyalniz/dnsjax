## Wall-bounded geometries

### Module layout

- `_base.py`: shared wall-bounded infrastructure (norms, integration,
  `init_state`, `apply_y_matrix`, `extract_mean_mode`, `pad_base_flow`,
  `base_flow_coupling`, `build_wall_bounded_stepper`)
- `cartesian.py`: Cartesian geometry (Fourier, CGL grid, `CartesianFlow`,
  the default v-omega_y IMM, Lk/Hk operator builders)
- `cylindrical.py`: cylindrical geometry (Fourier, radial CGL grid,
  `CylindricalFlow`, decoupled u+/u- formulation, parity-reduced FD,
  1x1 IMM on the default spin quad, `interpolate_to_axis` r=0
  evaluation)
- `annular.py`: annular geometry / concentric cylinders (Fourier, CGL
  grid on `[r1, r2]`, `AnnularFlow`, decoupled u+/u- formulation, 2x2
  IMM on the default (u_r, omega_r) pair, optional mean-mode azimuthal
  body force `force_theta`)
- `_cartesian_primitive_imm.py`, `_cylindrical_primitive_imm.py`,
  `_annular_primitive_imm.py`: each geometry's **legacy** primitive
  `(v, p)` IMM path (`res.consistent_imm = False`) — see the
  Influence-matrix section below
- `_viscoelastic_common.py`: the geometry-free sPTT half shared by
  both viscoelastic geometries (incl. the 9-component
  `to_spin_basis`/`from_spin_basis`)
- `_viscoelastic_stepping.py`: the sPTT stepping functions, written
  once against a per-geometry adapter surface its docstring tabulates.
  Imports neither geometry (that would build both families' grids on
  any viscoelastic import; guarded by `test_no_cross_geometry_import`
  in both per-geometry tests)
- `annular_viscoelastic.py` / `cylindrical_viscoelastic.py`: the two
  halves of that adapter surface (`ViscoelasticAnnularFlow` /
  `ViscoelasticCylindricalFlow` + laminar profiles). The pipe's
  axis-parity derivation and per-component table: its module docstring

**Component basis (cylindrical / annular only).** Two
representations, one boundary. The **solver basis** — decoupled
`u_± = u_r ± i u_θ` plus the conformation-spin components — is the
state's in-memory form: the carried state, the RHS, the cnab2 carry,
and the interior of every stepper. The **physical basis**
`(u_z, u_r, u_θ)` (+ the physical tensor) is what is observed or
persisted: snapshots, diagnostics, probes, forcing profiles, ICs, the
analysis package, the TG export. A given state crosses at most once,
never back (the physical form is a view, dropped after use), via
`_base.to_pm_basis`/`from_pm_basis` (aliased `to_solver_basis` /
`from_solver_basis`, re-exported by the flow modules) or the
9-component `_viscoelastic_common.to_spin_basis`/`from_spin_basis`.
`__main__` owns the field-level crossings; `extensions/probes.py` /
`extensions/forcing.py` convert their own mode columns instead.
**Anything that hands a freshly built (i.e. physical) state to a
stepper must convert first** — `__main__`'s post-IC line and
`transient_growth._linear_step` are the templates.
`_get_rhs_core`/`_l_bf` convert internally, so physical-space fields
and the CFL measurement are always physical components (why: the
`cylindrical.py` docstring).
**Cartesian carries physical `(u, v, w)` under both `res.consistent_imm`
formulations** and exports no basis pair, so every consumer skips the
conversion — `__main__` and `transient_growth` by `getattr` falling
back to the identity, the two extensions by testing
`params.phys.system in cartesian_systems`.

### Stepper factory (wall-bounded layer)

`build_wall_bounded_stepper()` in `_base.py` wraps
`timestep.make_stepper()`, binds the `fourier`/`flow` singletons, and
returns the stepping functions plus the adaptive-dt hooks
`set_dt`/`reset_ab2_kappa` (what they rebuild and why it costs no
recompile: its own docstring; the per-geometry leaves are
`_build_dt_leaves`). Each geometry provides a thin
`build_*_stepper(flow)` passing its measured RHS (CFL via the `rhs.py`
`measure_fn` hook) and `_l_bf` — the FFT-free linear base-flow coupling
(from the shared `base_flow_coupling` helper) that wall-bounded cnab2
and the opt-in split iterative-cn corrector make implicit. Why it is
stiff, and what `implicit_mean_coupling` folds in: the `TimeStepping`
docstring. Guards: `tests/test_cnab2.py`.

**Moving frame (`phys.u_grid`)**: the convective frame term is added
spectrally in each geometry's `_get_rhs_core` *and* `_l_bf`
(CN-implicit in both schemes). Do **not** shift the cross-product
velocity — `get_nonlin` keeps the lab-frame `base_flow_padded`, and
only the CFL diagnostic reads the shifted one (which field goes
where, the why, and the history: `pad_base_flow`).

### Influence-matrix method (IMM)

Every geometry has the same two-way split: `_imm_iteration` and
`Flow._derive_imm_homogeneous_data` are trace-time dispatchers over the
default reconstruction scheme (`_imm_iteration_vw` /
`_derive_vw_homogeneous_data`, in the geometry module) and the legacy
primitive one (`_imm_iteration_vp` / `derive_homogeneous_data`, in
`_<geometry>_primitive_imm.py`).
`cartesian._imm_iteration` carries the shared derivation — why there
are two schemes (the discrete-continuity residual) and the five
measured repairs, four of them retired — and the other two dispatchers
add only their geometry's amendment to that record.
`_cartesian_primitive_imm._imm_iteration_vp` documents the primitive
9-stage algorithm, its Schur-complement/Woodbury equivalence, and the
optional constant-bulk-velocity / block-mean-spanwise-velocity
corrections (shared via `cartesian._apply_bulk_corrections`).
`annular._imm_iteration_vw` carries the **cylindrical** algebra
(the `(Φ, ω_r)` pair, the mandatory conservative curl, the exact
`L_v,mod` recovery, the mean packing) and the retired-route record for
decoupling the pair; `cylindrical._imm_iteration_vw` adds only what
the axis forces (the spin quad, parity classes, the band splice).

- `params.solver.backend` selects operator storage: `"pallas"` (default
  banded sweep) or the `"dense"` reference/oracle -- see the
  `solvers.py` docstrings. Which sweep reads banded storage is a
  separate axis: `solvers._kernel_path()` takes the trace-only
  `_force_kernel_path`, then `params.solver.pallas_kernel`, then the
  live backend. The pallas build is wired in all three
  geometries: each `_build_{Lk,Hk}_band_gpu` (plus
  `_viscoelastic_stepping._build_Hc_band_gpu`) assembles directly in
  banded storage via the shared `solvers._assemble_banded_operator`,
  with the band width **measured** from the assembled operator
  (`fd.matrix_half_bandwidth`, both flag states), never assumed to be
  `fd_order`.
- Only Cartesian names `_hk_minus_matvec` (the others build `H_k^-`
  inline); the legacy `Lk` matvec is `_lk_matvec` in each
  `_<geometry>_primitive_imm.py`.
- `res.consistent_imm` (**default on**, offered by every wall-bounded
  flow) makes the discrete continuity identity hold by **one
  mechanism** in all three geometries: advance the wall-normal
  velocity + vorticity, reconstruct the tangential pair, never form a
  pressure. It is flag-independent from the outside — no geometry
  changes what it carries, so snapshots, probes/forcing, analysis and
  resume all read either path's state. Mechanism, per-geometry solve
  counts and efficacy, momentum prices, when to fall back to the
  legacy path, and the five rejected routes: the
  `Resolution.consistent_imm` docs
  (`parameters.py`); the shared scheme record:
  `cartesian._imm_iteration` (+ `_imm_iteration_vw`);
  the cylindrical algebra: `annular._imm_iteration_vw`; the pipe's
  free wall values, why its solve count and cost go the other way, and
  the instability that lagging them caused:
  `cylindrical._imm_iteration_vw`. Guards:
  `tests/test_imm_continuity.py` (continuity + the momentum ledger),
  `tests/test_random_smoke.py` (the nonlinear stability gate),
  `tests/test_temporal_order.py` (the order the formulation buys).
- `res.consistent_imm = False` is the **legacy** primitive `(v, p)`
  path: kept, tested, not recommended. Each geometry's flag-off half
  — the Neumann pressure-Poisson builders, its matvecs, the
  three-family `H_k` group (cyl/annular) and the step — lives in
  `_<geometry>_primitive_imm.py`, imported *lazily* inside the
  flag-off branches alone (`__post_init__`,
  `_derive_imm_homogeneous_data`, `_imm_iteration`, plus
  `_build_dt_leaves` on cyl/annular and `build_poisson_operator` on
  Cartesian), so the default path never imports it. That module
  imports back from its geometry module, which is why the import must
  stay deferred.

### Mean mode and padding modes

`pad_harmonics` (`operators.py`) keeps padding-slot wavenumbers
nonzero, so `Fourier.mean_mask` is a one-hot — the cross-module
invariant every pin-row/mean-mode consumer relies on. Detail: the
`Fourier` docstrings; guard: `tests/test_mean_mask.py`.

What may *write* the mean mode is a separate question, answered per
flow: only the Cartesian ones, and only through the conservation laws
in `ic/mean_mode.py` (root CLAUDE.md, "Initial conditions").

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
triad `(u_z, u_r, u_θ)` does **not** follow the `(streamwise,
wall-normal, spanwise)` order the other three geometries obey. Why,
and the per-slot table: the `annular.py` docstring.

Same decoupled `u+`/`u-` formulation as cylindrical but **two walls**,
**no `r=0` axis** (`r1 > 0`), no parity reduction, and a **2×2
influence matrix**. Two driving modes plus the viscoelastic
extension share the infrastructure (the `annular.py` "Driving"
section): shear-driven Taylor-Couette / quasi-Keplerian (perturbation
`u'`), force-driven Dean (total field, mean-mode body force
`AnnularFlow.force_theta`), and the viscoelastic total-field mode
(`annular_viscoelastic.py`).

**Azimuthal wedge (`geo.m0`, every cylindrical/annular flow, both
viscoelastic flows included)**: definition, the physical-space picture
(why the wedge is fully resolved rather than decimated) and the cost
argument: the `geo.m0` field docs (`parameters.py`). Only the
cylindrical/annular surfaces carry the field; elsewhere the CLI/TOML
reject it at parse and `validate_parameters` guards direct
assignment. Every `geo.lz` consumer
follows automatically. The cross-module rule: cylindrical parity
`m_is_even` tracks the *physical* `m0·j`, and the JAX-free analysis
package must mirror exactly that selector
(`analysis/_core.radial_derivative`; a harmonic-index pick silently
corrupted every even-wedge pipe snapshot's operators). Guards:
`tests/test_quasi_keplerian.py` (wedge_nonlinear),
`tests/test_transient_growth.py` (wedge-vs-full-circle equivalence),
the `test_laminar_smoke.py` wedge entries, and the `m0 = 2` pipe row
in `tests/test_snapshot_export.py`.

### Custom wall-normal grids

Selection precedence (`geo.wall_grid` file > `geo.grid_type` > the
flow spec's default), the per-family `grid_type` choices, and why
`update_parameters` resolves the default to a concrete value:
the `Geometry` docstring (`parameters.py`). Each flow's surface
narrows the Literal further (`specs/`). Quadrature is spectral
Clenshaw-Curtis on CGL grids, the `fd_order` composite rule on
custom/tanh grids. File format, validation, weights, and tanh-grid
properties: the `build_*_grid` and `fd.py` docstrings.

When a loaded snapshot's wall-normal grid differs,
`_interpolate_if_needed` (`__main__.py`) picks the optimal method; see
its docstring and the `fd.py` interpolation docstrings.

### Optimization patterns

- To operate on a mean-mode-derivable quantity, index the mean mode
  first (`extract_mean_mode`) then operate, when the two commute — and
  **never stack fields to feed one call**: a `shard_map` operand is
  materialised in full, so the stack is a field-sized copy read only at
  `[:, :, 0, 0]`. Two mean modes at once go through
  `extract_mean_modes(a, b)` (`_base.py`), one collective for the pair;
  the psum is latency-bound, so call count is the cost.
- `apply_y_matrix` FD matvecs batch over the leading component axis, so
  `D1`/`D2` GEMMs can be regrouped/deduplicated across IMM stages
  bit-identically (the laminar smoke `err=0.00e+00` confirms refactors).
- **Transpose-free GEMMs**: which stacks go y-leading
  (`component_axis=1`), which stay component-leading, and why: the
  `apply_y_matrix` (`_base.py`) and
  `PerModeBandedPallasOperator.solve` (`solvers.py`) docstrings.
- **Curvilinear `A_base` fusion** (cyl + annular, both schemes, plus
  both viscoelastic geometries): where a field needs `D2 x + (1/r) D1 x`
  and `D1 x` has *no other consumer*, apply the precomputed `A_base`
  (`flow.A_base`; parity-reduced `A_base_pos`/`A_base_ghost` on the
  pipe) as one matvec instead. Worth ~10 % of `_imm_iteration` on the
  velocity-only sites; on the viscoelastic `tensor_abase_matvec` it is a
  **measured wash** on CPU (−0.7 to −1.8 %, inside the spread, at
  `num_c = 0` *and* `3–4` — an sPTT step is transform-dominated), kept
  there for the FLOPs and the dropped transient. Not bit-identical, so the
  guards are `test_imm_continuity.py` + band-vs-dense parity + the two
  viscoelastic suites, **not** the laminar smoke (`u' = 0` there makes
  every stage zero either way). Where the premise fails, the split form
  stays — see the comment at `_annular_primitive_imm`'s `H_k^-` batch.

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
  ViscoelasticPipeFlow(ViscoelasticCylindricalFlow) -- axially
  force-driven sPTT pipe, 9-component **total** field.
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

Per-geometry operator suites (`test_cartesian` / `test_cylindrical` /
`test_annular` / `test_viscoelastic` / `test_viscoelastic_pipe`) plus
the solver, stepping, continuity, budget, IC and transient-growth
guards that reach this directory. One-liners in the root CLAUDE.md
Tests section; what each covers is in its own module docstring.
