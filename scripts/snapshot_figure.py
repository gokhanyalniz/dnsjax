#!/usr/bin/env python3
r"""Render a snapshot as a publication-style field figure.

The presentation counterpart of ``scripts/twin_spectral_maps.py``: it
turns one snapshot into a single PNG of a velocity plane, for a
README, a talk, or a quick look at what a run is actually doing.  It
reads the snapshot through :func:`dnsjax.analysis.read_state` alone --
NumPy and the standard library, **no JAX and no solver runtime** -- so
it runs anywhere the snapshot does, at any resolution, without a
device.

Two views, chosen from the snapshot's own geometry:

- **cylindrical / annular** -- the meridional `$(z, r)$` plane, the
  `$\theta = 0$` and `$\theta = \pi$` half-planes stacked into a full
  diameter.  This is the view a localized structure (a puff, a slug)
  reads in: the axial extent is the long axis, so the figure comes out
  as a wide strip.  ``--window`` crops it to a span centred on the
  largest `$|u|$`, which is what keeps a puff legible inside a
  100-diameter pipe.
- **Cartesian / triply-periodic** -- a wall-parallel `$(x, z)$` plane
  at ``--y``, the view streaks read in.  The default `$y = -0.7$` sits
  inside the near-wall streak region of a channel.

The quantity is the **stored** field, which is the perturbation `$u'$`
about the laminar profile for the base-flow systems and the total field
for force-driven Dean, viscoelastic Dean and viscoelastic pipe
(:data:`TOTAL_FIELD_SYSTEMS`); the colourbar label says which, and no
base flow is ever added or removed here.

The colour scale is diverging and pinned symmetric about zero
(``vmin = -vmax``), so the neutral midpoint *is* zero -- an asymmetric
range would paint zero as a signed colour and invent a mean the field
does not have.

Cells are drawn on explicit **edges** rather than on the sample points,
which matters twice in the meridional view: the wall cells then end at
the wall instead of half a cell short of it, and an annulus's core --
which holds no fluid -- is one masked cell rather than the inner wall's
no-slip value smeared across the axis.  A pipe has no such core: its two
innermost cells meet at `$r = 0$`, which is what the parity closure
across the axis means (``geometries/wall_bounded/cylindrical.py``).

matplotlib is not a solver dependency; it lives in the ``plots`` group,
so run this as::

    uv run --group plots python scripts/snapshot_figure.py \
        run/state00010.tar --out docs/figures/pipe_puff.png

``--component`` selects the stored component by index (0 is the
streamwise one in every family: `$u_x$` for the plane channels and the
periodic box, `$u_z$` for the pipe and the annulus, whose component
order is axial-first -- see ``SCALING.md``).

Producing the README figure
---------------------------
The README's opening figure is the quick-start run of its own
"Running a simulation" section, taken to `$t = 500$` on one GPU::

    mkdir -p ~/scratch/readme_puff && cd ~/scratch/readme_puff
    /path/to/dnsjax/.venv/bin/dnsjax \
      --phys.system pipe --phys.re 2300 \
      --geo.lz 200 --geo.grid_type half-cgl \
      --res.nz 512 --res.nr 48 --res.ntheta 96 --res.fd_order 8 \
      --step.scheme iterative-cn --step.dt 0.01 \
      --init.localized_rolls True \
      --init.localized_rolls_amplitude 0.2 \
      --init.localized_rolls_width 2.0 \
      --stop.max_sim_time 500 \
      --outs.it_stats 100 --outs.it_snapshot 5000 \
      --dist.platform cuda

Puff lifetimes at `$Re = 2300$` are stochastic, so
``stop.check_laminarization`` may fire first: take the last snapshot
before it did, or raise ``--phys.re`` to 2600-3000 or
``--init.localized_rolls_amplitude``.  That is a regime property, not
a solver setting.  Then::

    uv run --group plots python scripts/snapshot_figure.py \
      ~/scratch/readme_puff/state00010.tar \
      --out docs/figures/pipe_puff.png --width-px 1600

and place it in ``README.md`` immediately after the opening paragraph,
before ``## Highlights``:

.. code-block:: html

    <p align="center">
      <img src="docs/figures/pipe_puff.png" width="900"
           alt="Axial velocity perturbation of a turbulent puff in
                pipe flow at Re = 2300.">
    </p>
    <p align="center"><sub>
    A turbulent puff at <code>Re = 2300</code> in a 100-diameter pipe
    &mdash; the quick-start run below, at <code>t = 500</code>.
    Meridional plane of the axial velocity perturbation; 512 axial
    &times; 96 azimuthal Fourier modes, 48 radial points.
    </sub></p>

Update the caption's numbers if the configuration changes.  A
plane-Couette alternative is ``--phys.system plane-couette --phys.re
500`` in a box a few units on a side, rendered with the default
``--y``: the wall-parallel streaks.

Animating a sequence
--------------------
Frames must share a colour scale or the animation flickers as the
per-frame limit breathes, so pin it with ``--clim``: render one
representative frame, read the ``|u|max`` the run prints, and pass that
(or a little more, since a growing structure has not peaked yet) to
every frame.  Save the frames at a fixed cadence with
``--outs.it_snapshot``, then::

    mkdir -p frames
    for f in ~/scratch/readme_puff/state*.tar; do
      uv run --group plots python scripts/snapshot_figure.py "$f" \
        --out "frames/$(basename "$f" .tar).png" \
        --clim 0.35 --window 40 --width-px 1200
    done

    ffmpeg -framerate 12 -pattern_type glob -i 'frames/state*.png' \
      -vf "split[a][b];[a]palettegen[p];[b][p]paletteuse" \
      -loop 0 docs/figures/pipe_puff.gif

``--window`` re-centres on the largest `$|u|$` in *each* frame, so a
structure that moves stays centred but the axis labels shift under it;
pass ``--window 0`` and crop once with ffmpeg instead when a fixed
laboratory window is what the animation is about.

GitHub renders an animated GIF inline exactly like a PNG (swap the
``<img src=...>`` in the template above), but it is fetched in full on
every page view, so keep it to a few MB -- fewer frames, or a smaller
``--width-px``.  The still is the safer README opener; an animation
earns its bytes when the *evolution* is the point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dnsjax.analysis import geometry_info, read_state
from dnsjax.flows.registry import viscoelastic_systems

#: Systems whose snapshots store the **total** field rather than a
#: perturbation about a laminar profile (root ``CLAUDE.md``,
#: "Snapshots"); derived so a new viscoelastic flow follows.
TOTAL_FIELD_SYSTEMS = frozenset({"dean", *viscoelastic_systems})

#: An inner radius above this fraction of the outer one is a *wall* (an
#: annulus), not a pipe's near-axis first point.
_ANNULAR_CORE_FRACTION = 0.1


def _component_label(name: str, prime: str) -> str:
    r"""Mathtext label for a stored component name.

    ``geometry_info`` spells component names in ASCII with the tensor
    indices joined by underscores (``u_theta``, ``c_theta_theta``), so
    the subscript is rebuilt rather than sliced: ``theta`` becomes
    ``\theta`` (keeping the space mathtext needs to end the command)
    and the separators disappear.
    """
    base, _, sub = name.partition("_")
    parts = [r"\theta " if tok == "theta" else tok for tok in sub.split("_")]
    return f"${base}{prime}_{{{''.join(parts).strip()}}}$"


def _symmetric_limit(a: np.ndarray) -> float:
    """Largest ``|a|``, or 1.0 for an all-zero field.

    NaN-safe: the meridional view of an annulus masks its core cell.
    """
    m = float(np.nanmax(np.abs(a))) if a.size else 0.0
    return m if np.isfinite(m) and m > 0.0 else 1.0


def _grid_edges(c: np.ndarray) -> np.ndarray:
    """Cell edges of a non-uniform grid, pinned to its own range.

    Interior edges are the midpoints; the two outer edges sit on the
    first and last sample, so a wall-normal grid's outer cells end
    exactly at the walls.
    """
    c = np.asarray(c, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    return np.concatenate([c[:1], 0.5 * (c[1:] + c[:-1]), c[-1:]])


def _periodic_edges(c: np.ndarray) -> np.ndarray:
    """Cell edges of a uniform (Fourier) axis: half a spacing either way."""
    c = np.asarray(c, dtype=float)
    d = float(c[1] - c[0]) if c.size > 1 else 1.0
    return np.concatenate([c - 0.5 * d, [c[-1] + 0.5 * d]])


def _meridional(field: np.ndarray, r: np.ndarray) -> tuple:
    r"""Stack the `$\theta = 0, \pi$` half-planes into a diameter.

    *field* is one component in the native cylindrical layout
    ``(r, theta, z)``.  Returns ``(plane, s_edges)`` over the signed
    radius, with a masked core cell spanning `$[-r_1, +r_1]$` when the
    inner radius is a wall (:data:`_ANNULAR_CORE_FRACTION`) and the two
    innermost cells meeting at `$r = 0$` when it is a pipe axis.
    """
    r = np.asarray(r, dtype=float)
    n_theta = field.shape[1]
    upper = field[:, 0, :]  # theta = 0
    lower = field[:, n_theta // 2, :]  # theta = pi
    e = _grid_edges(r)
    if r[0] > _ANNULAR_CORE_FRACTION * r[-1]:
        core = np.full((1, field.shape[2]), np.nan)
        plane = np.concatenate([lower[::-1, :], core, upper], axis=0)
        edges = np.concatenate([-e[::-1], e])
    else:
        plane = np.concatenate([lower[::-1, :], upper], axis=0)
        edges = np.concatenate([-e[::-1][:-1], [0.0], e[1:]])
    return plane, edges


def _axial_window(plane: np.ndarray, z: np.ndarray, span: float) -> tuple:
    """Crop *plane* to *span* in ``z``, centred on the largest ``|u|``."""
    if span <= 0.0 or span >= (z[-1] - z[0]):
        return plane, z
    column_peak = np.nanmax(np.abs(plane), axis=0)
    peak = z[int(np.nanargmax(column_peak))]
    keep = (z >= peak - 0.5 * span) & (z <= peak + 0.5 * span)
    return plane[:, keep], z[keep]


def _wall_parallel(field: np.ndarray, y: np.ndarray, y0: float) -> tuple:
    """The ``(x, z)`` plane of *field* nearest wall-normal ``y0``.

    *field* is one component in the native layout ``(y, z, x)``; returns
    ``(plane, y_used)`` with ``plane`` of shape ``(n_z, n_x)``.
    """
    j = int(np.argmin(np.abs(y - y0)))
    return field[j], float(y[j])


def _render(
    plane: np.ndarray,
    horiz: np.ndarray,
    vert: np.ndarray,
    *,
    labels: tuple[str, str, str],
    out: Path,
    width_px: int,
    dpi: int,
    cmap: str,
    clim: float | None = None,
) -> None:
    """Write one panel on cell *edges*, symmetric diverging scale.

    *clim* pins that scale to ``[-clim, +clim]``; ``None`` fits it to
    this frame.
    """
    x_label, y_label, c_label = labels
    lim = clim if clim is not None else _symmetric_limit(plane)
    aspect = (vert[-1] - vert[0]) / (horiz[-1] - horiz[0])
    w_in = width_px / dpi
    # Room for the axes decoration and the colourbar below the panel;
    # constrained layout distributes it and ``bbox_inches="tight"``
    # crops whatever is left over.
    h_in = max(w_in * aspect + 1.15, 1.9)

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi, layout="constrained")
    mesh = ax.pcolormesh(
        horiz,
        vert,
        np.ma.masked_invalid(plane),
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        shading="flat",
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(horiz[0], horiz[-1])
    ax.set_ylim(vert[0], vert[-1])
    bar = fig.colorbar(
        mesh, ax=ax, location="bottom", shrink=0.85, aspect=45, pad=0.02
    )
    bar.set_label(c_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """The CLI (see the module docstring)."""
    p = argparse.ArgumentParser(
        prog="snapshot_figure.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("snapshot", type=Path, help="a dnsjax snapshot tar")
    p.add_argument("--out", type=Path, required=True, help="output PNG path")
    p.add_argument(
        "--component",
        type=int,
        default=0,
        help="stored component index (default 0, the streamwise one)",
    )
    p.add_argument(
        "--y",
        type=float,
        default=-0.7,
        help="wall-normal station of the Cartesian plane (default -0.7)",
    )
    p.add_argument(
        "--window",
        type=float,
        default=40.0,
        help=(
            "axial span kept around the largest |u| in the cylindrical / "
            "annular view; 0 keeps the whole domain (default 40)"
        ),
    )
    p.add_argument(
        "--clim",
        type=float,
        default=None,
        help=(
            "pin the colour scale to [-clim, +clim] instead of fitting "
            "it to this frame; required for an animation (see the "
            "module docstring), and read off a representative frame's "
            "printed |u|max"
        ),
    )
    p.add_argument("--width-px", type=int, default=1600)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cmap", default="RdBu_r")
    return p


def main(argv: list[str] | None = None) -> int:
    """Read one snapshot and write one figure."""
    args = build_parser().parse_args(argv)
    if args.clim is not None and args.clim <= 0.0:
        build_parser().error("--clim must be positive")

    data = read_state(args.snapshot, components=(args.component,))
    info = geometry_info(data.params)
    system = str(data.params.phys.system)
    prime = "" if system in TOTAL_FIELD_SYSTEMS else "'"
    comp = info.components[args.component]
    c_label = _component_label(comp, prime)

    field = data.physical[0]
    if info.family in ("cylindrical", "annular"):
        r, _theta, z = data.physical_coords
        plane, vert = _meridional(field, r)
        plane, z = _axial_window(plane, np.asarray(z), args.window)
        horiz = _periodic_edges(z)
        labels = ("$z$", "$r$", c_label)
    else:
        y, zc, xc = data.physical_coords
        plane, y_used = _wall_parallel(field, np.asarray(y), args.y)
        horiz, vert = _periodic_edges(xc), _periodic_edges(zc)
        labels = ("$x$", "$z$", f"{c_label}   at $y = {y_used:.3g}$")

    _render(
        plane,
        horiz,
        vert,
        labels=labels,
        out=args.out,
        width_px=args.width_px,
        dpi=args.dpi,
        cmap=args.cmap,
        clim=args.clim,
    )
    scale = (
        f"clim {args.clim:.4g} (pinned)"
        if args.clim is not None
        else f"clim {_symmetric_limit(plane):.4g} (this frame)"
    )
    print(
        f"[figure] {system}  {info.family}  component {comp}  "
        f"|u|max {_symmetric_limit(plane):.4g}  {scale}  ->  {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
