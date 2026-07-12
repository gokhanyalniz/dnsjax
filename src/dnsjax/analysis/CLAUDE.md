# `dnsjax.analysis` — JAX-free snapshot post-processing

External-facing API for physics/applied-maths researchers reading and
operating on dnsjax snapshots **without the solver runtime**. Not used
by the solver itself.

## Hard constraint: no JAX

Importing `dnsjax.analysis` must **never import JAX**: nothing on the
`import dnsjax.analysis` path (i.e. `__init__.py` and everything it
imports) may pull in JAX. The API depends only on NumPy, the standard
library, and dnsjax's JAX-free leaf modules: `fd.py`,
`snapshot_meta.py`, `harmonics.py`. This works because
`src/dnsjax/__init__.py` is empty (importing a submodule does not run a
package `__init__` that pulls in JAX). The guarantee is asserted in
`tests/test_snapshot_export.py` (`assert "jax" not in sys.modules`).
Do not add a JAX import (even transitively) to `__init__.py`, any
module it imports, or the three leaves.

**Exception**: `transient_growth.py` is a JAX-based (GPU-runnable) CLI
that lives in this directory but is **not** imported by `__init__.py`,
and defers every JAX / geometry import behind `configure_jax_platform`
(from `dnsjax.bootstrap`; so `import dnsjax.analysis` stays JAX-free;
only `python -m
dnsjax.analysis.transient_growth` or an explicit
`from dnsjax.analysis.transient_growth import ...` brings in JAX). Do
not import it from `__init__.py` or any JAX-free module here.

The **`response/` subpackage** extends this exception: its modules
*may* use JAX where it runs performantly on GPUs (batched
`expm`/SVD time sweeps), keeping JAX imports inside the functions
that need them and the platform selection in CLIs via
`configure_jax_platform`; SciPy (`logm`, Lyapunov solves; the
optional `dnsjax[analysis]` extra, always present in the dev group)
is imported lazily in-function with an install hint. Nothing under
`response/` is imported by `analysis/__init__.py` (its own
`__init__.py` is docstring-only), so the package-level JAX-free
guarantee is unaffected.

## Native (on-disk) layout — never transposed

Data is returned exactly as stored. A component chunk reshaped to
`(a_size, kx_global, b_size)` *is* the native layout; axis 1 is always
the real-FFT axis (`kx_global = nx // 2`). Coordinate tuples are ordered
to match these axes (no transpose to "fix" layout).

| family (systems)                                  | spectral axes (as read) | physical axes | components        |
|---------------------------------------------------|-------------------------|---------------|-------------------|
| cartesian (plane-couette/poiseuille)              | (y, k_x, k_z)           | (y, x, z)     | (u_x, u_y, u_z)   |
| cylindrical/annular (pipe, taylor-couette, dean)  | (r, k_z(axial), m)      | (r, z, θ)     | (u_z, u_r, u_θ)   |
| viscoelastic (viscoelastic-dean)                  | (r, k_z(axial), m)      | (r, z, θ)     | (u_z, u_r, u_θ, c_zz, c_rz, c_θz, c_rr, c_θθ, c_rθ) |
| triply-periodic (kolmogorov/waleffe/decaying-box) | (k_z, k_x, k_y)         | (z, x, y)     | (u_x, u_y, u_z)   |

- Cylindrical/annular: the stored basis is `(u_z, u_+, u_-)`; the reader
  converts to `(u_z, u_r, u_θ)` once at read time
  (`u_r=(u_++u_-)/2`, `u_θ=(u_+-u_-)/2i`), and the operators **expect
  `(u_z, u_r, u_θ)` in spectral space too** — `u_±` is never exposed.
  Requesting `u_r` or `u_θ` reads the `u_±` pair (chunks 1 and 2).
- Viscoelastic annular: the stored state has **9 components** — the 3
  velocity components above plus the 6 symmetric conformation-tensor
  spin projections `(c_zz, c_z+, c_z-, c_+-, c_++, c_--)`; the reader
  exposes the physical tensor `(c_zz, c_rz, c_θz, c_rr, c_θθ, c_rθ)` as
  components `3..8` (each combined from its stored spin combos, the same
  pairing idea as `u_±`; `read_state(components=...)` selects any
  subset). Tensor differential operators are out of scope in
  `snapshot_ops` (velocity operators unchanged).
- Azimuthal length is `2π` (so `m` is integer); periodic shear length is
  `LY_PERIODIC = 4`; the `r`/`y` grid comes from `meta["wall_normal_grid"]`.

## Operator convention — matches the solver's discrete operators

**Transform round-trip is machine-precision exact for every family**
(pipe / Taylor-Couette included): `to_physical`/`to_spectral` act on the
returned `(u_z, u_r, u_θ)` basis, each real and Hermitian on the real
(`k_z`) axis. The `u_±` pair is **not** individually Hermitian, so
`u_+`/`u_-` must never be `irfft`-ed directly — the reader converts
first (reality structure: `random_field._hermitian_column` and the
`_core.py` transform docstrings).

`divergence`/`curl` reproduce dnsjax's **discrete** operators
node-for-node (not just the continuous formulae) — incl. the
parity-reduced pipe radial `D1` — so `test_snapshot_export.py` checks
`curl` against the solver's `_curl_fn` at machine precision; re-run it
when changing a primitive.

- `derivative`/`gradient` of a single **pipe** component along `r` are
  parity-dependent: pass `cylindrical_parity="u_z"/"u_r"/"u_theta"`
  (raises if omitted); `divergence`/`curl` set parity internally.
- Differentiation/integration along the wall-normal axis needs the
  **full** grid — do not subset `wall_normal_points` first.
- `integrate` works on **physical** fields (e.g. `|u|²`): `L/n` along
  Fourier axes, FD quadrature (`build_integration_weights`) along the
  grid axis, with the radial Jacobian `r` for cylindrical/annular.

## System → family mapping

`_core.py` holds `CARTESIAN_SYSTEMS` / `CYLINDRICAL_SYSTEMS` /
`ANNULAR_SYSTEMS` / `VISCOELASTIC_SYSTEMS` / `PERIODIC_SYSTEMS` frozensets
that **mirror the `*_systems` lists in `parameters.py`**. Adding a flow
system there requires adding it here too (unknown system → explicit
error). This is the only place the analysis package re-encodes solver
knowledge. `geometry_info` maps a viscoelastic system to the annular
geometry family but with the 9-component schema; the per-component read
recipe (native chunks + combine function) lives in `_component_recipes`.

## Modules

- `snapshot_export.py` — `read_state` (the entry point) + `StateData`.
- `snapshot_ops.py` — `derivative`, `gradient`, `divergence`, `curl`,
  `integrate`, and `to_physical`/`to_spectral` (re-exported).
- `transient_growth.py` — the JAX-based transient-growth CLI (the
  exception above; not part of the JAX-free API). 3D linear optimal
  energy growth around an arbitrary wall-normal total profile, reusing
  the solver's linear step per Fourier mode; `--save-operator` exports
  the per-mode reduced generators consumed by `response/`, and its
  `single_mode_state`/`mode_state_energy` helpers are shared with
  `scripts/snapshot_perturb.py`. See the module docstring and the root
  CLAUDE.md "Transient-growth analysis" note.
- `response/` — input-output / response tools (JAX allowed; see the
  exception above): `probes.py` (NumPy reader for the runtime
  `probes.bin` mode streams: mean profile, `Re_tau`, TG-ready profile
  files), `operator_tools.py` (controllability Gramian/modes on the
  `--save-operator` bundles, growth/input-response curves of arbitrary
  operators, Galerkin restriction, the shared `load_modes_npz`/
  `recover_basis` basis plumbing; controllability-mode export CLI),
  `ensemble.py` (member-tree aggregation `aggregate`, the shared
  `identify_generator` multi-horizon `logm` fit, and the direct
  operator-identification CLI `identify`), `lim.py` (linear inverse
  modeling: the same identification from lagged covariances of an
  *unforced* probe stream), `ssi.py` (`forcing.bin` reader +
  kick/response cross-covariance identification for
  `[force]`-enabled runs, discrete-Lyapunov forced-variance
  prediction). The three identification routes share the fit, basis,
  and output convention (the `response/__init__.py` pipeline
  section), and the per-function JAX-vs-NumPy/SciPy split is
  deliberate (rationale: the closing paragraph of
  `response/__init__.py`). Detail: the module docstrings;
  orchestration:
  `scripts/ensemble_setup.py`.
- `_core.py` — engine: raw chunk I/O (`snapshot_meta` offsets +
  `np.frombuffer`, minimal per-component / per-slab reads), transforms
  (`norm="forward"`, Nyquist reinstated as zero), basis conversion,
  coordinate builders, the diff/quadrature primitives, `GeometryInfo`,
  and the `Namespace` object-view over embedded params/stats.

Detail (array shapes, the transform algorithm, per-function behaviour)
lives in those module/function docstrings; keep it there, not here.
