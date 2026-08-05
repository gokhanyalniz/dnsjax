# `dnsjax.analysis` — JAX-free snapshot post-processing

External-facing API for reading and operating on dnsjax snapshots
**without the solver runtime**. Not used by the solver itself.

## Hard constraint: no JAX

Importing `dnsjax.analysis` must **never import JAX**: nothing on the
`import dnsjax.analysis` path (`__init__.py` and everything it imports)
may pull in JAX. The API depends only on NumPy, the standard library,
dnsjax's JAX-free leaf modules (`fd.py`, `snapshot_meta.py`,
`harmonics.py`), and the JAX-free flow-spec registry
(`flows/registry.py`, which pulls `flow_spec.py` and every
`flows/*/specs/*` module). This works because
`src/dnsjax/__init__.py` is empty. The guarantee is asserted in
`tests/test_snapshot_export.py` (`assert "jax" not in sys.modules`).
Do not add a JAX import (even transitively) to `__init__.py`, any
module it imports, the leaves, or **any flow spec**.

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
  `dnsjax[analysis]` extra) is imported lazily in-function: the
  `expm`/`logm` routes (`ensemble.py`, `ssi.py`) require it (a bare
  `ImportError` without the extra); the Lyapunov solvers fall back to
  eigen-based routines. Its own `__init__.py` is docstring-only.

## Conventions

**Native layout — never transposed.** Data is returned exactly as
stored, and (format 6, the reader's floor) the stored bytes are the
solver's native spectral layout: a component chunk reshaped to the
`(y|r|ky, kz|m, kx)` per-component shape (`meta["native_shape"]`) *is*
the layout the solver computes in, and coordinate tuples are ordered
to match (no transpose to "fix" layout). As of format 6 the stored
components are the physical components in every family — cyl/annular
`(u_z, u_r, u_θ)`, each the transform of a real field (the solver's
`u_±`/spin working basis is converted away at the write) — so returned
components are stored components, one-to-one. The per-family
axis/component tables and the 9-component viscoelastic schema: the
`_core.py` module docstring. Chunk I/O validates each component chunk
against `meta["native_shape"]` and raises `SnapshotArchiveError` (a
`ValueError` subclass, `snapshot_meta.py`) naming the file and the
cause — catch that for damaged or mismatched archives.

**Operators match the solver's discrete operators.** `divergence`/`curl`
reproduce dnsjax's **discrete** operators node-for-node (not just the
continuous formulae), incl. the parity-reduced pipe radial `D1` — so
`test_snapshot_export.py` pins both against the solver: `curl` against
`_curl_fn`, `divergence` against the `_imm_iteration` assembly (on a
deliberately non-solenoidal probe — on a divergence-free field both
sides return zero whatever they are). **Re-run it when changing a
primitive.** Pipe radial parity must follow the **physical** azimuthal
mode `m = m0·h` exactly as `cylindrical.Fourier.m_is_even` does
(`GeometryInfo.azimuthal_m0` → `_core.radial_derivative`; the
harmonic-index pick silently corrupted every even-wedge pipe
snapshot; guard: the `m0 = 2` pipe rows in `test_snapshot_export.py`).
A viscoelastic pipe's six conformation components fall into the same
two classes, per `snapshot_ops._PARITY_CLASS` (which states the rule).
Per-function behaviour (the pipe `cylindrical_parity` argument,
needing the full wall-normal grid, physical-field `integrate` with
the radial Jacobian): the `snapshot_ops.py` docstrings.

**System → family mapping.** `_core.py` builds its `*_SYSTEMS`
frozensets from the JAX-free `dnsjax.flows.registry` (the same source
as `parameters.py`), so a new flow spec extends them automatically;
unknown systems still raise an explicit error. The geometry sets and
`VISCOELASTIC_SYSTEMS` are independent axes that deliberately overlap,
so an ordered branch mixing them must test rheology first — the rule
and its rationale: `flows/registry.py`.

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
- `twin/` — twin-run (`dnsjax-twin`) offline analysis, entirely
  JAX-free (unlike `response/`, no JAX anywhere): `series.py`
  (`twin.dat`/`twin_budget.dat`/`twin.json` readers, per-component
  budget sums), `ensemble.py` (member-tree aggregation + growth-rate
  fits, CLI), `spectra.py` (`twin_spectra.bin` reader +
  decorrelation ratio), `lengths.py` (integral length scales of the
  difference field from a snapshot pair). Not imported by
  `__init__.py`; its own `__init__` re-exports the API. Guard:
  `tests/test_twin_analysis.py`.
- `_core.py` — engine: raw chunk I/O, transforms, basis conversion
  (identity in every family since format 6; kept as the family seam),
  coordinate builders, diff/quadrature primitives, `GeometryInfo`, and
  the `Namespace` object-view over embedded params/stats.
  `geometry_info` also receives the live pydantic `params` singleton,
  so it must use plain attribute access only (its docstring).

Detail (array shapes, the transform algorithm, per-function behaviour)
lives in those module/function docstrings; keep it there, not here.
