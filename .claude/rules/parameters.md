---
paths:
  - "src/dnsjax/{parameters,param_surface,flow_spec,bootstrap}.py"
  - "src/dnsjax/flows/registry.py"
  - "src/dnsjax/flows/**/specs/*.py"
  - "src/dnsjax/extensions/__init__.py"
  - "tests/**/*.py"
  - "scripts/*.py"
---

# Parameters

- The layer order, the per-flow surfaces and strict relevance: the
  `bootstrap.resolve_parameters` docstring. Another entry point passes
  its own `toml_path` / `extensions` / `prog`; the transient-growth CLI
  is the template. Field documentation: the `parameters.py` models,
  printed per flow by `--help <system>` / `--sample-toml <system>`.
- The CLI, TOML files and snapshot metadata use public names:
  cylindrical/annular `geo.lz`, `res.nz` (axial), `res.nr`,
  `res.ntheta` are internally `geo.lx`, `res.nx`, `res.ny`, `res.nz`.
- Per-flow `FieldSpec` defaults (`geo.grid_type`, `phys.u_grid`, the
  rheology values, ...) are re-materialized on every
  `update_parameters()` unless a layer set them. Scripts and tests set
  such a field through `update_parameters(Parameters(...))`; a direct
  `params.<section>.<field> = ...` is silently overwritten on the next
  pass (the `update_parameters` docstring).
- A deferred field (`DeferredSpec`) is refused on the CLI, in TOML and
  on direct assignment; `validate_parameters` rejects a knob the
  flow's surface does not carry.
- What a resume never inherits from a snapshot (the JAX-setup fields,
  `res.double_precision`, `[solver]`, `init.snapshot` /
  `init.force_resume`): the `parameters.py` module docstring and
  `read_snapshot_params`.
- Extension sections: `[probes]` and `[force]` are built in
  (`extensions/__init__.py`); `[twin]` belongs to `dnsjax-twin`, `[tg]`
  to `analysis.transient_growth`, `[perturb]` to
  `scripts/snapshot_perturb.py`, `[recon]` to
  `scripts/twin_postprocess.py`, `[calib]` to
  `scripts/random_ic_calibrate.py`.
- Every Fourier count is even (`validate_parameters`; `harmonics.py`
  says why). Odd *derived* sizes (`nz - 1`, padded sizes) are normal.
