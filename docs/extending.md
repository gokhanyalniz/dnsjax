# Extending

Adding a flow system, and registering a parameter section of your own.
Start at the [README](../README.md) for the solver itself and at
[`configuration.md`](configuration.md) for how the layers combine.

Adding a flow system is a two-file operation. The first is a
**`FlowSpec`** under `src/dnsjax/flows/<family>/specs/`, added to that
package's `SPECS` tuple; the second is the flow module it names, which
exports the stepping surface. Nothing else is edited: the
`phys.system` literal, the `--help` and `parameters.toml` surfaces,
`--sample-toml`, the snapshot metadata surface, the stepping dispatch
and the analysis package's geometry sets all derive from the registry
and extend themselves.

A spec is plain data plus pure-Python hooks. It declares which shared
parameter fields apply to the flow, the public names of any aliased
ones (`nr` for the internal `res.ny`, and so on), per-flow default
overrides, narrowed choice sets, *deferred* fields — declared but not
yet implemented, so they fail with their own message rather than
looking nonsensical — and the flow's derivation and validation hooks.
A state that is not three velocity components declares its count, and
the initial-condition builders, the FFT and sharding layers, and the
steppers are all component-count-agnostic.

Specs import nothing heavier than the standard library: no pydantic,
no JAX, and never the parameter module itself, whose live objects the
hooks receive as arguments. That is what lets `--help` render and a
TOML validate without configuring JAX, and what keeps the import graph
acyclic.

The other extension point is the parameter surface itself: a script or
analysis tool registers a whole section of its own, as described under
[Parameter layering](configuration.md).
