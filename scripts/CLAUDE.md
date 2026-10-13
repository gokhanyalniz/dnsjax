# Offline tools

Each script's module docstring is its manual. Only
`snapshot_figure.py` needs matplotlib: run it as
`uv run --group plots python scripts/snapshot_figure.py` (`uv sync`
alone does not install the `plots` group).

- `snapshot_perturb.py`: inject a scaled single-mode perturbation into
  a snapshot (`[perturb]`).
- `snapshot_figure.py`: velocity-plane figures and animations (the
  `docs/figures/` sources); `_figure_common.py` holds its JAX-free
  per-flow helpers.
- `wall_normal_resolution.py`: size `res.ny` / `fd_order` /
  `geo.grid_type` against a Chebyshev order (Cartesian).

The lab's scripts (analyses, performance, diagnostics, the refactor
gate, releases): `lab/CLAUDE.md`.
