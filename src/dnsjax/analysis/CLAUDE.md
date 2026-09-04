# `dnsjax.analysis` — JAX-free snapshot post-processing

External-facing API for reading and operating on dnsjax snapshots
**without the solver runtime**. Not used by the solver itself. The one
inbound member, `snapshot_import.py`, is an exception on both counts
(below).

## Hard constraint: no JAX

Importing `dnsjax.analysis` must **never import JAX**: nothing on the
`import dnsjax.analysis` path (`__init__.py` and everything it
imports) may pull in JAX. What the API is allowed to depend on: the
`_core.py` module docstring. This works because
`src/dnsjax/__init__.py` is empty. The guarantee is asserted in
`tests/test_snapshot_export.py` (`assert "jax" not in sys.modules`).
Do not add a JAX import (even transitively) to `__init__.py`, any
module it imports, the leaves, or **any flow spec**.

**Exceptions** — modules in this directory that are *not* imported by
`__init__.py`, so the package-level guarantee is unaffected. Do not
import either from `__init__.py` or any JAX-free module here.

- `transient_growth.py`: a JAX-based (GPU-runnable) CLI that defers
  every JAX / geometry import behind `configure_jax_platform`. Its
  parameter surface (the shared per-flow surface + a `[tg]`
  extension): the module docstring; solver-only sections
  parse-and-ignore (the `ignored` list in `_configure_parameters`);
  the seed-snapshot metadata: `_seed_metadata_params`.
- `response/`: **may** use JAX, and uses SciPy (a core dependency) for
  the `expm`/`logm` routes -- both imported lazily in-function, which
  is what keeps them off the package-level guarantee. The
  JAX-vs-NumPy/SciPy split and the fallbacks: the
  `response/__init__.py` docstring.
- `snapshot_import.py`: the **inbound** direction -- it configures the
  parameter singletons and calls `snapshot.save_snapshot`, so it needs
  the solver runtime. Every JAX import is in-function, so importing it
  is still NumPy-only. It lives here rather than in `scripts/` because
  only `src/` is packaged: a `scripts/` copy never reaches an installed
  `dnsjax`. The native input contract: its module docstring.

## Conventions

**Native layout — never transposed.** Data is returned exactly as
stored: a component chunk reshaped to `meta["native_shape"][1:]`
(`(y|r|ky, kz|m, kx)`) *is* the layout the solver computes in, and
coordinate tuples are ordered to match — never transpose to "fix"
layout. The per-family axis/component tables, the stored-are-physical
rule and the 9-component viscoelastic schema: the `_core.py` module
docstring. Chunk I/O validates each chunk against `native_shape` and
raises `SnapshotArchiveError` (a `ValueError` subclass,
`snapshot_meta.py`) naming the file and the cause — catch that for
damaged or mismatched archives.

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
- `snapshot_import.py` — native-layout velocity field -> snapshot (a
  library, not a CLI; not part of the JAX-free API). Guard:
  `tests/test_snapshot_import.py`.
- `response/` — input-output / response tools: `probes.py`,
  `operator_tools.py`, `ensemble.py`, `lim.py`, `ssi.py` (the last
  three interchangeable operator-identification routes sharing one
  fit, basis, and output convention). The module list, the full
  probe→operator pipeline and the route trade-offs: the
  `response/__init__.py` docstring. Orchestration:
  `scripts/ensemble_setup.py`. Guards: `tests/response/`.
- `twin/` — twin-run (`dnsjax-twin`) offline analysis: `series.py`,
  `ensemble.py`, `spectra.py`, `yspectra.py`, `lengths.py`. What each
  reads: the `analysis/twin/__init__.py` docstring; the streams
  themselves: `src/dnsjax/twin/CLAUDE.md`. Cross-module: it is
  entirely JAX-free (unlike `response/`, no JAX anywhere), it is not
  imported by `__init__.py` (its own `__init__` re-exports the API),
  and `yspectra`'s record layout (`stored_fields` / `record_dtype`)
  is *shared* with `scripts/twin_spectral_maps.py`'s memory map
  rather than mirrored there. Guard: `tests/test_twin_analysis.py`.
- `_core.py` — engine: raw chunk I/O, transforms, coordinate
  builders, differentiation primitives, `GeometryInfo`, and the
  `Namespace` object-view over embedded params/stats. (Quadrature is
  not here: `snapshot_ops._axis_weights` over `fd.py`.) The
  `geometry_info` attribute-access constraint: its docstring.

Detail (array shapes, the transform algorithm, per-function behaviour)
lives in those module/function docstrings; keep it there, not here.
