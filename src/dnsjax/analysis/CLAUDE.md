# `dnsjax.analysis` — JAX-free snapshot post-processing

External-facing API for physics/applied-maths researchers reading and
operating on dnsjax snapshots **without the solver runtime**. Not used
by the solver itself.

## Hard constraint: no JAX

Importing `dnsjax.analysis` must **never import JAX**. It depends only
on NumPy, the standard library, and dnsjax's JAX-free leaf modules:
`fd.py`, `snapshot_meta.py`, `harmonics.py`. This works because
`src/dnsjax/__init__.py` is empty (importing a submodule does not run a
package `__init__` that pulls in JAX). The guarantee is asserted in
`tests/test_snapshot_export.py` (`assert "jax" not in sys.modules`).
Do not add a JAX import (even transitively) to any module here or to the
three leaves it imports.

## Native (on-disk) layout — never transposed

Data is returned exactly as stored. A component chunk reshaped to
`(a_size, kx_global, b_size)` *is* the native layout; axis 1 is always
the real-FFT axis (`kx_global = nx // 2`). Coordinate tuples are ordered
to match these axes (no transpose to "fix" layout).

| family (systems)                                  | spectral axes (as read) | physical axes | components        |
|---------------------------------------------------|-------------------------|---------------|-------------------|
| cartesian (plane-couette/poiseuille)              | (y, k_x, k_z)           | (y, x, z)     | (u_x, u_y, u_z)   |
| cylindrical/annular (pipe, taylor-couette, dean)  | (r, k_z(axial), m)      | (r, z, θ)     | (u_z, u_r, u_θ)   |
| triply-periodic (kolmogorov/waleffe/decaying-box) | (k_z, k_x, k_y)         | (z, x, y)     | (u_x, u_y, u_z)   |

- Cylindrical/annular: the stored basis is `(u_z, u_+, u_-)`; the reader
  converts to `(u_z, u_r, u_θ)` once at read time
  (`u_r=(u_++u_-)/2`, `u_θ=(u_+-u_-)/2i`), and the operators **expect
  `(u_z, u_r, u_θ)` in spectral space too** — `u_±` is never exposed.
  Requesting `u_r` or `u_θ` reads the `u_±` pair (chunks 1 and 2).
- Azimuthal length is `2π` (so `m` is integer); periodic shear length is
  `LY_PERIODIC = 4`; the `r`/`y` grid comes from `meta["wall_normal_grid"]`.

## Operator convention — matches the solver's discrete operators

**Transform round-trip is lossy for pipe / Taylor-Couette**: `u_±` are
not individually Hermitian on the real axis (the `u_θ` axial-mean
imaginary part is dropped by `irfft` — dnsjax's own `spec_to_phys`
loses it identically), so `spec→phys→spec` / transform-invariance
checks are valid **only for cartesian & triply-periodic**.

`divergence`/`curl` reproduce dnsjax's **discrete** operators
node-for-node, not just the continuous formulae: the cylindrical/annular
forms are the expanded `∂u_r/∂r + u_r/r + (im/r)u_θ + i k_z u_z`
(not `(1/r)∂(r u_r)/∂r`), and the **pipe** radial `D1` is the
parity-reduced operator (mirrors `build_parity_reduced_matrices` in
`cylindrical.py`: `u_z` parity `(-1)^m`, `u_r`/`u_θ` parity `(-1)^{m+1}`).
The annulus has no axis, so it uses a plain `D1`. `test_snapshot_export.py`
checks `curl` against the solver's own `_curl_fn` at machine precision
(incl. the pipe). When changing a primitive, re-run that test.

- `derivative`/`gradient` of a single **pipe** component along `r` are
  parity-dependent: pass `cylindrical_parity="u_z"/"u_r"/"u_theta"`
  (raises if omitted). `divergence`/`curl` set parity internally.
- Differentiation/integration along the wall-normal axis needs the
  **full** grid — do not subset `wall_normal_points` first.
- `integrate` works on **physical** fields (form nonlinear integrands
  like `|u|²` in physical space): `L/n` along Fourier axes, FD quadrature
  (`build_integration_weights`) along the grid axis, with the radial
  Jacobian `r` for cylindrical/annular.

## System → family mapping

`_core.py` holds `CARTESIAN_SYSTEMS` / `CYLINDRICAL_SYSTEMS` /
`ANNULAR_SYSTEMS` / `PERIODIC_SYSTEMS` frozensets that **mirror the
`*_systems` lists in `parameters.py`**. Adding a flow system there
requires adding it here too (unknown system → explicit error). This is
the only place the analysis package re-encodes solver knowledge.

## Modules

- `snapshot_export.py` — `read_state` (the entry point) + `StateData`.
- `snapshot_ops.py` — `derivative`, `gradient`, `divergence`, `curl`,
  `integrate`, and `to_physical`/`to_spectral` (re-exported).
- `_core.py` — engine: raw chunk I/O (`snapshot_meta` offsets +
  `np.frombuffer`, minimal per-component / per-slab reads), transforms
  (`norm="forward"`, Nyquist reinstated as zero), basis conversion,
  coordinate builders, the diff/quadrature primitives, `GeometryInfo`,
  and the `Namespace` object-view over embedded params/stats.

Detail (array shapes, the transform algorithm, per-function behaviour)
lives in those module/function docstrings; keep it there, not here.
