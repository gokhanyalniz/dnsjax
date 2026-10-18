# Extending

Adding a flow system, in this package or from a package of your own,
and adding parameter fields and sections. Start at the
[README](../README.md) for the solver itself and at
[`configuration.md`](configuration.md) for how the layers combine.

## A flow system

A flow system is two pieces: a **`FlowSpec`**, plain data plus
pure-Python hooks, and the flow module it names (`flow_module`), which
exports the surface the `FlowSpec` docstring lists — the steppers, the
diagnostics, the initial-condition builders `init_random_state` and
`init_localized_rolls` (where a flow composes its geometry's generators
with what it adds, such as a total-field flow's laminar state), and
what its geometry and the transient-growth driver need.
Nothing else is edited: the `phys.system` literal, the `--help` and
`parameters.toml` surfaces, `--sample-toml`, the snapshot metadata
surface, the stepping dispatch and the analysis package's geometry sets
all derive from the registry and extend themselves.

In this package, the spec goes under
`src/dnsjax/flows/<family>/specs/`, added to that package's `SPECS`
tuple. A separate package registers its specs under the `dnsjax.flows`
entry-point group instead, naming a `FlowSpec` or an iterable of them:

```toml
[project.entry-points."dnsjax.flows"]
myflows = "myflows.specs:SPECS"
```

The registry lists such flows after the core's within each geometry,
and refuses a system name registered twice.

A flow with a *metric* of its own adds a geometry module as well: on
the cylindrical
geometry it overrides `CylindricalFlow`'s metric hooks (no-ops on the
straight pipe), and its spec names the metric's analysis counterpart
(`analysis_metric`) and, where they differ, its component and axis
names (`components`, `axis_names`). The `FlowSpec` docstring states
the cylindrical contract.

A spec declares the flow's parameter surface — the shared fields that
apply, public aliases (`nr` for the internal `res.ny`), default
overrides, narrowed choices, *deferred* fields that fail with their own
message — and its derivation and validation hooks (the
`dnsjax.flow_spec` module docstring). The `FlowSpec` docstring holds
the contracts for a state of more than the three velocity components
(`n_components`, `components`, `component_parity`; the pipe's stepping
is written for three, so such a flow brings its own) and for a flow
that integrates the total field (`total_field`), from which what a
snapshot holds, the transient-growth scope and the figure labels
follow.

Specs import nothing heavier than the standard library, so `--help`
and TOML validation never configure JAX (the `dnsjax.flow_spec` module
docstring says why). A flow module, by contrast, builds on the
geometry modules' internals, as the core flows do, and before 1.0
those may change between releases.

## Parameter fields

A flow that needs a field the core sections lack adds it through the
`dnsjax.parameters` entry-point group, which names a mapping
`{section: {name: (annotation, FieldInfo)}}` in a module that imports
neither JAX nor `dnsjax.parameters`. The core composes these fields into
its section models after its own, and refuses one that clashes with a
core field or another plugin's, or that names a section the core does
not have (a section of its own is an extension, below). The flow's spec
then lists the field
like any core field: the public name, the TOML key, the CLI flag and the
snapshot metadata need nothing more.

## Parameter sections

The other extension point is a whole section of the parameter surface.
A script or analysis tool registers one of its own, as described under
[Parameter layering](configuration.md); a package registers its
sections under the `dnsjax.extensions` entry-point group, naming a
module whose import registers them (`register_extension`). A section
can take part in the solver's run: its `runtime` names a factory of
`RunHooks` (`start`, `sample`, `modify`, `flush`, `finish`), which the
solver builds once the section is configured and calls at fixed points
of every step, and its `seed_fields` are drawn, agreed across processes
and recorded like the solver's own seed (unless the section resolves
them itself, `seeds_at_start`). How a configured section is recorded
in snapshots and resumed: [Parameter layering](configuration.md).
