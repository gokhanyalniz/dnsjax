# Wall-bounded geometries

Each module's docstring is its manual. `_base.py` is the shared layer
(`apply_y_matrix`, the basis maps, `extract_mean_mode(s)`,
`pad_base_flow`, `frozen_profile_flow`, `build_wall_bounded_stepper`);
`cartesian.py`, `cylindrical.py` and `annular.py` are the geometries.
The pipe's stepping is `_cylindrical_stepping.py`, shared by the
straight, curved and viscoelastic pipes; `cylindrical_curved.py` adds
the toroidal metric (its docstring is the worked example of a new
metric); `_*_primitive_imm.py` hold the legacy `(v, p)` path
(`res.consistent_imm = False`); `_viscoelastic_*.py` and
`*_viscoelastic.py` are the sPTT extension. The two shared stepping
modules import neither geometry (guard: `test_no_cross_geometry_import`).

## Component basis (cylindrical, annular)

- The state is carried in the solver basis `u_± = u_r ± i u_θ` (plus
  the conformation spin components), and so are the RHS, the cnab2
  carry and every stepper interior. Everything outside the stepper
  (snapshots, diagnostics, probes, forcing, ICs, `dnsjax.analysis`,
  the TG export) sees physical `(u_z, u_r, u_θ)` (+ tensor).
- A state crosses once, never back: `_base.to_pm_basis` /
  `from_pm_basis`, re-exported by each flow module as
  `to_solver_basis` / `from_solver_basis` (9 components:
  `_viscoelastic_common.to_spin_basis` / `from_spin_basis`).
  `__main__` owns the field-level crossings; `extensions/probes.py`
  and `forcing.py` convert their own mode columns.
- **Convert a freshly built (physical) state before stepping it**: an
  unconverted one steps without error, just wrong. Templates:
  `__main__`'s post-IC line, `transient_growth._linear_step`.
- The pipe family carries two more trailing slots (the spin quad's
  difference halves): its `to_solver_basis` appends them,
  `from_solver_basis` drops them, and RHS arrays and the cnab2 carry
  never have them (a carry is seeded from `state[:n_components]`).
- The outgoing map is exported already jitted; never wrap a flow export
  in a jit of your own (it bakes the flow's global arrays in, which a
  multi-process run refuses). Cartesian carries physical `(u, v, w)`
  and has no basis pair.

## Stepping and the influence matrix

- `_l_bf` is the FFT-free linear base-flow coupling that cnab2 and the
  split corrector keep implicit. The moving frame enters spectrally in
  `_get_rhs_core` and `_l_bf`; never shift the cross-product velocity
  (`pad_base_flow` says which field goes where).
- Each geometry's `_imm_iteration` and
  `Flow._derive_imm_homogeneous_data` dispatch between the default
  reconstruction (`_imm_iteration_vw`) and the legacy primitive path.
  Records: the shared derivation and retired routes in
  `cartesian._imm_iteration`, the cylindrical algebra in
  `annular._imm_iteration_vw`, the pipe's additions in
  `_cylindrical_stepping._imm_iteration_vw`; the flag in
  `Resolution.consistent_imm` (the curved pipe refuses the legacy
  path). Guards: `test_imm_continuity.py`, `test_random_smoke.py`,
  `test_temporal_order.py`.
- A legacy module is imported lazily, inside flag-off branches only: it
  imports back from its geometry module.
- Operators are assembled directly in banded storage through
  `solvers._assemble_banded_operator` (e.g. `_build_Lk_dir_band_gpu`,
  `_build_Lv_dir_band_gpu`, `_build_Hk_band_gpu`). The band width is
  measured (`fd.matrix_half_bandwidth`), never assumed to be
  `fd_order`.
- `Fourier.mean_mask` is one-hot (`operators.pad_harmonics`), which
  every mean-mode and pin-row consumer relies on. Every geometry writes
  the mean mode each step; which flows may *perturb* it:
  `.claude/rules/initial-conditions.md`.

## Cylindrical and annular specifics

- Internal slots: `nx` axial (real FFT), `nz` azimuthal, `ny` radial;
  on the CLI/TOML surface they are `nz`, `ntheta` and `nr`. So
  Taylor-Couette's streamwise (azimuthal) resolution is internal `nz`,
  not `nx` as in the Cartesian family.
- The annulus orders its components `(u_z, u_r, u_θ)` like the pipe,
  not (streamwise, wall-normal, spanwise) (the `annular.py` docstring).
- Full-disc radial quadrature is parity-specific (`y_weights` /
  `y_weights_odd`); custom and tanh grids and `dnsjax.analysis` use the
  composite rule. There is no `r = 0` point:
  `cylindrical.interpolate_to_axis`.
- `geo.m0` (every cylindrical/annular surface but the curved pipe's):
  the pipe's parity `m_is_even` follows the physical `m = m0·h`, and
  `analysis/_core.radial_derivative` must use exactly that selector
  (guards: the `m0 = 2` rows of `test_snapshot_export.py`,
  `test_quasi_keplerian.py`, `test_transient_growth.py`).

## Optimization patterns

- For a mean-mode quantity, extract the mean mode first, and never
  stack fields to feed one extraction (`_base.extract_mean_modes` takes
  two in one collective).
- `apply_y_matrix` batches over the leading axis, so FD GEMMs can be
  regrouped across IMM stages; which stacks go y-leading: its
  docstring and `PerModeBandedPallasOperator.solve`.
- The fused `A_base = D2 + (1/r) D1` replaces two matvecs where `D1 x`
  has no other consumer. It is not bit-identical: its guards are the
  ones `cylindrical._build_A_base` names, not the laminar smoke.

## Curvature (toroidal pipe)

- Carrying `w = (h u_s, u_r, u_θ)` keeps every operator the straight
  pipe's: curvature enters only the explicit RHS (the `rhs.get_nonlin`
  metric hooks) and the continuity rows (`divergence_defect`, on the
  corrector iterate).
- Zero any source added to the `L_v,mod` solve on the wall row: that
  row is a Dirichlet identity, so the RHS value there becomes `u_r` at
  the wall and the corrector stops converging.
