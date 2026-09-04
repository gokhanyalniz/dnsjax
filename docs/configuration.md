# Parameter layering

How the configuration layers combine, what an extension section is, and
where the field-by-field documentation lives. Start at the
[README](../README.md) for the solver itself and at
[`running.md`](running.md) for a worked run.

Configuration is applied in layers, lowest priority first:

**Per-flow defaults → parameters embedded in a resumed snapshot →
`parameters.toml` → command-line flags.**

Only explicitly set fields override a lower layer, and validation runs once
after the final layer. Every layer is parsed against the **selected flow's
parameter surface**: only that flow's parameters exist (an irrelevant key is
a hard error naming the flow), fields go by their geometry-natural public
names (a pipe has `--geo.lz`/`--res.nz`/`--res.nr`/`--res.ntheta` where a
plane channel has `--geo.lx`/`--res.nx`/`--res.ny`/`--res.nz`), and per-flow
defaults (the pipe's moving frame `u_grid = 0.5`, its scheme-dependent
`grid_type`, the viscoelastic rheology values) are materialized before
printing or recording. The parameters that must be known before JAX
initializes — `dist.np0`, `dist.np1`, `dist.platform`, and
`res.double_precision` — are never inherited from a snapshot, nor are the
resume-decision fields `init.snapshot` and `init.force_resume` (recorded
for lineage only), and the entire `solver` section is execution-only.

Not every section is owned by the core parameter model. An
**extension** registers a whole section of its own — parsed as
`--<name>.<field>` and `[<name>]`, shown in `--help` and
`--sample-toml`, validated strictly per flow (a section on a flow it
does not apply to is an error like any other irrelevant key),
optionally recorded into snapshot metadata, and optionally
trajectory-defining. Two ship with the solver, `[probes]` and
`[force]`; the analysis and preprocessing entry points register their
own on the same shared surface (`[tg]` for the transient-growth CLI,
`[perturb]` for `scripts/snapshot_perturb.py`, `[twin]` for
`dnsjax-twin`). A section name colliding with a core one is rejected
at registration, so the two namespaces cannot drift into each other.

`uv run dnsjax --help` shows the global parameters and the flow list,
`--help <system>` one flow's full surface with per-field descriptions, and
`--sample-toml <system>` an annotated `parameters.toml` template with every
default commented out (all exit at the parser, before any device is
touched). The
authoritative field-by-field documentation lives in
[`src/dnsjax/parameters.py`](../src/dnsjax/parameters.py) and the per-flow
specs under `src/dnsjax/flows/*/specs/`.
