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
  at ``--y``, the view streaks read in.  Which `$y$` that is depends on
  the flow: the near-wall cycle lives at `$y^+ \approx 15$`, i.e.
  `$y = -1 + y^+/Re_\tau$`, so a channel at `$Re_\tau \approx 180$`
  wants `$-0.92$` while a low-`$Re$` Couette box wants far less.
  Recover `$Re_\tau = \sqrt{Re\,|\mathrm{d}U/\mathrm{d}y|_w}$` from
  the run's own wall shear -- the laminar value plus the ``tau'_s,*``
  columns of ``stats.dat`` (or the snapshot's embedded stats).  The
  ``--y`` default of `$-0.9$` is a near-wall plane at production
  channel Reynolds numbers, not a universal one.

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

Producing the README figures
----------------------------
Both come from one plane-Poiseuille snapshot at ``re = 4200``
(`$Re_\tau \approx 180$`).  The opener stacks three planes -- one near
each wall at `$y^+ \approx 15$` and one at the centreline::

    uv run --group plots python scripts/snapshot_figure.py \
        <run>/state00082.tar --y -0.9167 0 0.9167 --bare \
        --dpi 320 --refine 3 --palette 256 \
        --out docs/figures/channel-planes.png

``--bare`` drops the axes and the colourbar: a first-scroll image does
not need to be readable to three digits, and its caption says what it
shows.  ``--dpi 320`` puts it at ~1800 px across, so it stays sharp at
the README's 900 px on a 2x display -- and ``--refine`` has to follow,
because that is exactly the magnification at which the flat-shaded mesh
starts to show (below).  ``--palette 256`` then halves the file.

The single plane, further down, is the same near-wall station on its
own, annotated::

    uv run --group plots python scripts/snapshot_figure.py \
        <run>/state00082.tar --y -0.9167 \
        --out docs/figures/channel-streaks.png --width-px 1600

``--y`` is the only number that has to move with the snapshot: it is
`$-1 + y^+/Re_\tau$`, so recompute it from the new run's own friction
Reynolds number rather than carrying `$-0.9167$` over.  The colour
scale then fits the data, and the caption's `$Re$`, box and resolution
come from the embedded parameters -- ``dnsjax.analysis.read_meta``.

A stack shares **one** colour scale across its planes, which is the
point: the fluctuation decays away from the wall (at `$Re_\tau \approx
180$` the centreline rms is a third of the near-wall one) and
normalising each plane separately would hide that.  ``--clip`` sets the
percentile the shared scale is cut at, ``--clim`` pins it outright.

An alternative subject, when a *localized* structure is the point, is
the quick-start pipe run of the README's own "Running a simulation"
section taken to `$t = 500$` (add ``--dist.platform cuda``), rendered
with ``--window 40``: a turbulent puff inside 100 laminar diameters.
Puff lifetimes at ``re = 2300`` are stochastic, so
``stop.check_laminarization`` may fire first -- take the last snapshot
before it did, or raise ``--phys.re`` to 2600-3000.  That is a regime
property, not a solver setting.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (  # noqa: E402
    LAMINAR_WALL_SHEAR,
    friction_reynolds,
    y_plus,
)

from dnsjax.analysis import geometry_info, read_state, read_stats  # noqa: E402
from dnsjax.flows.registry import viscoelastic_systems  # noqa: E402

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


def _wall_parallel(
    field: np.ndarray, y: np.ndarray, y0: float, *, demean: bool
) -> tuple:
    r"""The ``(x, z)`` plane of *field* nearest wall-normal ``y0``.

    *field* is one component in the native layout ``(y, z, x)``; returns
    ``(plane, y_used)`` with ``plane`` of shape ``(n_z, n_x)``.

    With *demean*, the plane's own `$x$`-`$z$` average is subtracted.
    That is what makes a turbulent wall-parallel plane legible: the
    stored field is the perturbation about the **laminar** profile, and
    in a channel that carries a large offset at a fixed height -- at
    `$y^+ \approx 15$`, `$Re_\tau \approx 180$`, the plane mean is
    `$\sim\!3\times$` the fluctuation rms about it, so a scale
    symmetric about zero would paint the whole plane one colour and
    hide the streaks entirely.  What is left is the fluctuation about
    the *turbulent* mean at that height, which is the standard
    quantity; the laminar offset cancels with the mean, so the result
    carries no prime either way.
    """
    j = int(np.argmin(np.abs(y - y0)))
    plane = field[j]
    return (plane - plane.mean() if demean else plane), float(y[j])


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


def _refine(plane: np.ndarray, factor: int) -> np.ndarray:
    r"""Interpolate a wall-parallel plane onto a *factor*-finer grid.

    Both in-plane directions are Fourier, so zero-padding the spectrum
    is **exact** trigonometric interpolation -- the same operation the
    solver's own 3/2 dealiasing pad performs, not invented detail.

    It is needed because ``plot_surface`` shades each cell flat: magnify
    a `$192\times160$` plane to 1800 px and the quads are ~9 px across,
    so the mesh itself becomes visible as a diagonal weave.  Refining
    the data puts the cells back below the pixel scale.

    The Nyquist row and column are dropped rather than split, which is
    exact here: dnsjax stores no Nyquist mode on any axis, so the field
    carries none.
    """
    if factor <= 1:
        return plane
    n0, n1 = plane.shape
    k0, k1 = (n0 - 1) // 2, (n1 - 1) // 2  # keep modes -k..+k, no Nyquist
    spec = np.fft.fft2(plane)
    out = np.zeros((n0 * factor, n1 * factor), dtype=complex)
    out[: k0 + 1, : k1 + 1] = spec[: k0 + 1, : k1 + 1]
    out[: k0 + 1, -k1:] = spec[: k0 + 1, -k1:]
    out[-k0:, : k1 + 1] = spec[-k0:, : k1 + 1]
    out[-k0:, -k1:] = spec[-k0:, -k1:]
    return np.real(np.fft.ifft2(out)) * factor**2


def _render_stack(
    planes, y_used, horiz, vert, *, labels, ticks, out, dpi, cmap, opts
) -> None:
    r"""Draw several wall-parallel planes stacked in a 3D view.

    Each plane keeps its own `$x$`-`$z$` mean removed, but they share
    one symmetric colour scale, so the decay of the fluctuation away
    from the wall is visible rather than normalised away -- at
    `$Re_\tau \approx 180$` the centreline rms is a third of the
    near-wall one, and the middle plane is meant to look fainter.

    Physical `$(x, z, y)$` map onto matplotlib's `$(x, y, z)$`: its
    third axis is the vertical one on screen, and wall-normal is what
    has to be vertical for a stack to read as a stack.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    lim = opts.clim
    if lim is None:
        lim = max(float(np.percentile(np.abs(p), opts.clip)) for p in planes)
    norm, colormap = Normalize(-lim, lim), plt.get_cmap(cmap)

    fig = plt.figure(figsize=(11, 5.6), dpi=dpi)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    if opts.refine > 1:
        planes = [_refine(p, opts.refine) for p in planes]
        horiz, vert = (
            c[0]
            + np.arange(c.size * opts.refine) * (c[1] - c[0]) / opts.refine
            for c in (horiz, vert)
        )
    grid_h, grid_v = np.meshgrid(horiz, vert, indexing="ij")
    alpha = np.linspace(1.0, opts.top_opacity, len(planes))
    for rank, k in enumerate(np.argsort(y_used)):
        ax.plot_surface(
            grid_h,
            grid_v,
            np.full_like(grid_h, y_used[k]),
            facecolors=colormap(norm(planes[k].T)),
            shade=False,
            rstride=1,
            cstride=1,
            alpha=float(alpha[rank]),
            linewidth=0,
            antialiased=False,
            rasterized=True,
        )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.grid(False)
    ax.set_xlabel(labels[0], labelpad=10)
    ax.set_ylabel(labels[1], labelpad=8)
    ax.set_zlabel("$y$", labelpad=14)
    ax.set_xlim(horiz[0], horiz[-1])
    ax.set_ylim(vert[0], vert[-1])
    ax.set_zlim(-1.0, 1.0)
    ax.set_zticks(list(y_used))
    ax.set_zticklabels(ticks)
    # The wall-normal ticks sit beside the spanwise ones.
    ax.tick_params(axis="z", pad=6)
    ax.tick_params(axis="y", pad=2)
    ax.set_box_aspect(
        (
            float(horiz[-1] - horiz[0]),
            float(vert[-1] - vert[0]),
            opts.y_aspect * 2.0,
        )
    )
    ax.view_init(elev=opts.elevation, azim=opts.azimuth)
    if opts.bare:
        # A headline image, not a plot: the caption carries what the
        # axes and the colourbar would otherwise have said.
        ax.set_axis_off()
        ax.set_position([0.0, 0.0, 1.0, 1.0])
    else:
        ax.set_position([-0.05, 0.11, 1.10, 0.95])
        bar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=colormap),
            cax=fig.add_axes([0.32, 0.20, 0.37, 0.026]),
            orientation="horizontal",
        )
        bar.set_label(labels[2], labelpad=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    _trim(out, opts.palette)


def _trim(path: Path, palette: int = 0) -> None:
    """Crop the white margin a 3D axes reserves around its content.

    With *palette* set, the result is also quantised to that many
    colours.  A figure like this is a smooth colormap ramp over white,
    so 256 entries reproduce it to well under one level on average --
    measured on the README figure, mean channel error 0.4/255, with no
    banding in the pale centreline plane, for half the bytes.  It is
    opt-in because it *is* lossy, and a figure carrying fine colour
    detail rather than one ramp would show it.
    """
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    mask = (img < 250).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    pad = 10
    out = Image.fromarray(
        img[
            max(rows[0] - pad, 0) : rows[-1] + pad + 1,
            max(cols[0] - pad, 0) : cols[-1] + pad + 1,
        ]
    )
    if palette:
        out = out.quantize(
            colors=palette, method=Image.MEDIANCUT, dither=Image.NONE
        )
    out.save(path, optimize=True)


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
        nargs="+",
        default=[-0.9],
        help="wall-normal station(s) of the Cartesian plane (default "
        "-0.9; the near-wall cycle is at y = -1 + 15/Re_tau).  Give "
        "several and they are stacked in a 3D view",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=99.7,
        help="percentile of |u| setting the shared colour scale of a "
        "stack (default 99.7); --clim overrides it",
    )
    p.add_argument(
        "--y-aspect",
        type=float,
        default=2.2,
        help="wall-normal exaggeration of a stack (default 2.2)",
    )
    p.add_argument(
        "--palette",
        type=int,
        default=0,
        help="quantise the PNG to this many colours (0 = truecolour); "
        "256 roughly halves the file of a smooth colormap render",
    )
    p.add_argument(
        "--refine",
        type=int,
        default=1,
        help="interpolate each plane onto a grid this many times finer "
        "before rendering (exact: both in-plane axes are Fourier).  "
        "Above ~6 px per cell the flat-shaded mesh shows as a weave; "
        "raise this rather than lowering --dpi",
    )
    p.add_argument(
        "--bare",
        action="store_true",
        help="a stack with no axes, ticks, labels or colourbar -- for a "
        "headline image whose caption carries what they would have said",
    )
    p.add_argument(
        "--top-opacity",
        type=float,
        default=0.7,
        help="opacity of the topmost plane of a stack",
    )
    p.add_argument("--elevation", type=float, default=26.0)
    p.add_argument("--azimuth", type=float, default=-62.0)
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
        "--no-demean",
        dest="demean",
        action="store_false",
        help=(
            "keep the wall-parallel plane's own mean instead of "
            "subtracting it (the meridional view never demeans -- its "
            "plane spans the whole diameter)"
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
        if len(args.y) > 1:
            raise SystemExit(
                "snapshot_figure: a stack of wall-parallel planes is a "
                f"Cartesian view; {system!r} draws the meridional one"
            )
        r, _theta, z = data.physical_coords
        plane, vert = _meridional(field, r)
        plane, z = _axial_window(plane, np.asarray(z), args.window)
        horiz = _periodic_edges(z)
        labels = ("$z$", "$r$", c_label)
    else:
        y, zc, xc = data.physical_coords
        used = [
            _wall_parallel(field, np.asarray(y), y0, demean=args.demean)
            for y0 in args.y
        ]
        planes = [pl for pl, _ in used]
        stations = np.array([yy for _, yy in used])
        horiz, vert = _periodic_edges(xc), _periodic_edges(zc)
        if args.demean:
            base = comp.partition("_")[0]
            sub = _component_label(comp, "")[1:-1].partition("_")[2]
            c_label = rf"${base}_{sub} - \langle {base}_{sub} \rangle_{{xz}}$"
        re_tau = (
            friction_reynolds(read_stats(args.snapshot), data.params)
            if system in LAMINAR_WALL_SHEAR
            else None
        )

        def station(value: float) -> str:
            text = f"$y = {value:.3g}$"
            if re_tau is not None:
                text += rf"$,\ y^+ = {y_plus(value, re_tau):.3g}$"
            return text

        if len(planes) > 1:
            # ``argmin`` lands on the centreline at float noise, not 0.
            ticks = [f"{0.0 if abs(v) < 1e-9 else v:.3g}" for v in stations]
            print(
                f"[figure] {system}  {info.family}  component {comp}  "
                f"{len(planes)} planes at "
                + ", ".join(f"{v:.3g}" for v in stations)
                + f"  ->  {args.out}"
            )
            _render_stack(
                planes,
                stations,
                np.asarray(xc),
                np.asarray(zc),
                labels=("$x$", "$z$", c_label),
                ticks=ticks,
                out=args.out,
                dpi=args.dpi,
                cmap=args.cmap,
                opts=args,
            )
            return 0
        plane, y_used = planes[0], float(stations[0])
        labels = ("$x$", "$z$", f"{c_label}   at {station(y_used)}")

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
