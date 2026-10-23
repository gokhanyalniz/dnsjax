# Offline tools

Standalone programs that read or write the solver's snapshots, or size
its grids; none of them changes how a simulation runs. Each script's
module docstring is its manual. A snapshot's metadata needs none of
them: the installed `dnsjax-info` command prints it ([Inspecting a
snapshot](../docs/snapshots.md#inspecting-a-snapshot)). The figure
script needs the `plots` dependency group:

```bash
uv run --group plots python scripts/snapshot_figure.py ...
```

The ensemble, post-processing, performance and diagnostic tools of the
repository's lab are listed with it ([the lab](../README.md#the-lab)).

## Figures

| Script | What it does |
|---|---|
| `snapshot_figure.py` | velocity-plane figures and animations from a snapshot, JAX-free (the figures in the README and the docs); `_figure_common.py` holds its per-flow helpers |

## Initial conditions

| Script | What it does |
|---|---|
| `snapshot_perturb.py` | injects a scaled single-mode perturbation into a snapshot (`[perturb]`) |

## Resolution

| Script | What it does |
|---|---|
| `wall_normal_resolution.py` | how many wall-normal modes a finite-difference grid resolves: sizes `res.ny`, `fd_order` and `geo.grid_type` against a Chebyshev order (Cartesian) |
