# `dnsjax.analysis` — JAX-free snapshot post-processing

External-facing API for reading and operating on dnsjax snapshots
**without the solver runtime**. Not used by the solver itself.

## Hard constraint: no JAX

Importing `dnsjax.analysis` must **never import JAX**: nothing on the
`import dnsjax.analysis` path (`__init__.py` and everything it imports)
may pull in JAX. The API depends only on NumPy, the standard library,
and dnsjax's JAX-free leaf modules: `fd.py`, `snapshot_meta.py`,
`harmonics.py`. This works because `src/dnsjax/__init__.py` is empty.
The guarantee is asserted in `tests/test_snapshot_export.py`
(`assert "jax" not in sys.modules`). Do not add a JAX import (even
transitively) to `__init__.py`, any module it imports, or the three
leaves.

**Exceptions** — modules in this directory that are *not* imported by
`__init__.py`, so the package-level guarantee is unaffected. Do not
import either from `__init__.py` or any JAX-free module here.

- `transient_growth.py`: a JAX-based (GPU-runnable) CLI that defers
  every JAX / geometry import behind `configure_jax_platform`. Its
  parameter surface (the shared per-flow surface + a `[tg]` extension;
  solver-only sections parse-and-ignore) and the production-default
  metadata embedded in exported seed snapshots: the module docstring
  and `_seed_metadata_params`.
- `response/`: **may** use JAX where it runs performantly on GPUs,
  keeping JAX imports inside the functions that need them and platform
  selection in CLIs via `configure_jax_platform`. SciPy (the optional
  `dnsjax[analysis]` extra) is imported lazily in-function with an
  install hint. Its own `__init__.py` is docstring-only.

## Conventions

**Native (on-disk) layout — never transposed.** Data is returned exactly
as stored; a component chunk reshaped to `(a_size, kx_global, b_size)`
*is* the native layout, and coordinate tuples are ordered to match (no
transpose to "fix" layout). The per-family axis/component tables, the
`u_±` → `(u_z, u_r, u_θ)` basis conversion (the stored pair is **not**
individually Hermitian, so `u_±` must never be `irfft`-ed directly), and
the 9-component viscoelastic schema: the `_core.py` module docstring.

**Operators match the solver's discrete operators.** `divergence`/`curl`
reproduce dnsjax's **discrete** operators node-for-node (not just the
continuous formulae), incl. the parity-reduced pipe radial `D1` — so
`test_snapshot_export.py` checks `curl` against the solver's `_curl_fn`
at machine precision. **Re-run it when changing a primitive.** Per-
function behaviour (the pipe `cylindrical_parity` argument, needing the
full wall-normal grid, physical-field `integrate` with the radial
Jacobian): the `snapshot_ops.py` docstrings.

**System → family mapping.** `_core.py` builds its `*_SYSTEMS`
frozensets from the JAX-free `dnsjax.flows.registry` (the same source
as `parameters.py`), so a new flow spec extends them automatically;
unknown systems still raise an explicit error.

## Modules

- `snapshot_export.py` — `read_state` (the entry point) + `StateData`.
- `snapshot_ops.py` — `derivative`, `gradient`, `divergence`, `curl`,
  `integrate`, and `to_physical`/`to_spectral` (re-exported).
- `transient_growth.py` — the JAX-based transient-growth CLI (not part
  of the JAX-free API). `--tg.save_operator` exports the per-mode
  reduced generators consumed by `response/`; its `single_mode_state` /
  `mode_state_energy` helpers are shared with
  `scripts/snapshot_perturb.py`. See the module docstring and the root
  CLAUDE.md "Transient-growth analysis" note.
- `response/` — input-output / response tools: `probes.py` (reader for
  the runtime `probes.bin` streams), `operator_tools.py` (Gramians,
  controllability modes, growth curves, basis plumbing), `ensemble.py`,
  `lim.py`, and `ssi.py` (three interchangeable operator-identification
  routes sharing one fit, basis, and output convention). The full
  probe→operator pipeline, the route trade-offs, and the deliberate
  JAX-vs-NumPy/SciPy split: the `response/__init__.py` docstring.
  Orchestration: `scripts/ensemble_setup.py`.
- `_core.py` — engine: raw chunk I/O, transforms, basis conversion,
  coordinate builders, diff/quadrature primitives, `GeometryInfo`, and
  the `Namespace` object-view over embedded params/stats.

Detail (array shapes, the transform algorithm, per-function behaviour)
lives in those module/function docstrings; keep it there, not here.
