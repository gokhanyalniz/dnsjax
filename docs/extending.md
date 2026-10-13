# Extending

Adding a flow system, in this package or from a package of your own,
and adding parameter fields and sections. Start at the
[README](../README.md) for the solver itself and at
[`configuration.md`](configuration.md) for how the layers combine.

## A flow system

A flow system is two pieces. The first is a **`FlowSpec`**, plain data
plus pure-Python hooks. The second is the flow module it names
(`flow_module`), which exports the surface the `FlowSpec` docstring
lists: the steppers, the diagnostics, the two initial-condition
builders `init_random_state` and `init_localized_rolls`, which the
random and localized-rolls start modes call, and what its geometry and
the transient-growth driver need besides (the basis pair of the
cylindrical and annular geometries, the frozen-profile flow). A flow
composes its geometry's generators there with whatever it adds itself,
such as a total-field flow's laminar state.
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
and refuses a system name registered twice. The repository's lab
package ([`lab/`](../lab/README.md)) registers its five flows this way.

That is the count for a flow that reuses an existing geometry, and
every flow runs on one of the four geometries. A flow that brings its
own *metric* adds a geometry module as well: on the cylindrical
geometry it overrides `CylindricalFlow`'s metric hooks (no-ops on the
straight pipe), and its spec names the metric's analysis counterpart
(`analysis_metric`) and, where they differ, its component and axis
names (`components`, `axis_names`). The lab's toroidal pipe is the
worked example, and its `cylindrical_curved.py` docstring is written to
be read as one.

A spec declares which shared parameter fields apply to the flow, the
public names of any aliased ones (`nr` for the internal `res.ny`, and
so on), per-flow default overrides, narrowed choice sets, *deferred*
fields — declared but not yet implemented, so they fail with their own
message rather than looking nonsensical — and the flow's derivation and
validation hooks. A state that is not three velocity components
declares its count, and the initial-condition builders, the FFT and
sharding layers, and the steppers are all component-count-agnostic. A
flow that integrates the total field, driven by a mean-mode body force,
rather than a perturbation about a laminar profile declares that too
(`total_field`); what a snapshot holds, the transient-growth scope and
the figure labels all follow from it.

Specs import nothing heavier than the standard library: no pydantic,
no JAX, and never the parameter module itself, whose live objects the
hooks receive as arguments. That is what lets `--help` render and a
TOML validate without configuring JAX, and what keeps the import graph
acyclic. A flow module, by contrast, builds on the geometry modules'
internals, as the core flows do, and before 1.0 those may change
between releases.

## Parameter fields

A flow that needs a field the core sections lack adds it through the
`dnsjax.parameters` entry-point group, which names a mapping
`{section: {name: (annotation, FieldInfo)}}` in a module that imports
neither JAX nor `dnsjax.parameters`. The core composes these fields into
its section models after its own, and refuses one that clashes with a
core field or another plugin's. The flow's spec then lists the field
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
and recorded like the solver's own seed. A configured section is
recorded in snapshot metadata, and a snapshot recording a section that
no installed package registers refuses to resume unless
`init.force_resume` is set. The lab's `[probes]` and `[force]` sections
are the worked examples.
