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
Both come from one plane-Poiseuille run at ``re = 4200``
(`$Re_\tau \approx 180$`), and both are animations over its snapshot
series -- several snapshot arguments switch the script from a figure to
one, written as ``.webp``, ``.gif`` or ``.png`` (APNG) after the output
suffix.  The opener stacks three planes, one near each wall at
`$y^+ \approx 15$` and one at the centreline::

    uv run --group plots python scripts/snapshot_figure.py \
        <run>/state*.tar --y -0.9167 0 0.9167 --bare --lab-frame \
        --dpi 150 --refine 2 --width-px 800 --quality 42 \
        --out docs/figures/channel-planes.webp

and the second, further down, is the near-wall plane on its own::

    uv run --group plots python scripts/snapshot_figure.py \
        <run>/state*.tar --y -0.9167 --lab-frame \
        --refine 3 --width-px 800 --quality 35 \
        --out docs/figures/channel-streaks.webp

``--lab-frame`` matters for anything animated out of a run with
``phys.u_grid`` set.  That frame is a change of coordinate, so undoing
it is a pure translation, and without it the picture is confusing
rather than wrong: this run integrates in a frame moving at the bulk
velocity, which outruns the near-wall fluid but is itself outrun by the
centreline, so the planes would drift in *opposite* directions.

Drop to a single snapshot for the still versions, and raise ``--dpi``
and ``--refine`` accordingly: a still is worth rendering at 2x for a
high-DPI display (``--dpi 320 --refine 3 --palette 256`` for the
stack, ``--refine 3 --width-px 1600 --palette 256`` for the plane),
where an animation is not -- the frame count pays for the resolution
several dozen times over.

``--y`` is the only number that has to move with the run: it is
`$-1 + y^+/Re_\tau$`, so recompute it from the new run's own friction
Reynolds number rather than carrying `$-0.9167$` over.  The caption's
`$Re$`, box and resolution come from the embedded parameters --
``dnsjax.analysis.read_meta``.

Why animated WebP
=================
It is the only common format that is both small enough and sharp
enough here.  Turbulence changes everywhere between frames, so
inter-frame prediction saves little and the codec is doing near-still
compression 51 times over; GIF's 256-colour palette on top of that
costs roughly twice the bytes for a visibly worse picture, and APNG
several times that again.  Animated WebP is supported by every current
browser (Chrome, Firefox, Edge, Safari 14+) and renders inline in a
GitHub README like any other image.  ``--fps``, ``--quality`` and
``--width-px`` are the three levers; halving the frame count by
sampling every second snapshot is the fourth, and at these speeds the
motion is smooth either way.

Every frame of an animation shares one colour scale, taken from the
percentile over the whole sequence rather than from each frame, so a
colour means the same thing throughout and the picture does not pulse.

Animating a meridional view
---------------------------
Only the wall-parallel view animates: the meridional one is a
``--window`` crop that re-centres on the largest `$|u|$` in *each*
frame, so a moving structure stays centred while the axes shift under
it, which is not what an animation should do.  Render the frames one
snapshot at a time with a pinned ``--clim`` and ``--window 0``, and
stitch them outside::

    for f in <run>/state*.tar; do
      uv run --group plots python scripts/snapshot_figure.py "$f" \
        --out "frames/$(basename "$f" .tar).png" --clim 0.35 --window 0
    done
    ffmpeg -framerate 12 -pattern_type glob -i 'frames/state*.png' \
      -c:v libwebp_anim -q:v 45 -loop 0 puff.webp

An animation is fetched in full on every page view, so it earns its
bytes only when the *evolution* is the point; a still is the cheaper
opener.
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

from dnsjax.analysis import (  # noqa: E402
    geometry_info,
    read_meta,
    read_state,
    read_stats,
)
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


def _finer_centres(c: np.ndarray, factor: int) -> np.ndarray:
    """The uniform axis *c* resampled *factor* times more finely."""
    c = np.asarray(c, dtype=float)
    if factor <= 1:
        return c
    step = float(c[1] - c[0]) / factor
    return c[0] + np.arange(c.size * factor) * step


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
    planes,
    y_used,
    horiz,
    vert,
    *,
    labels,
    ticks,
    out,
    dpi,
    cmap,
    opts,
    trim_box=None,
):
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
        horiz = _finer_centres(horiz, opts.refine)
        vert = _finer_centres(vert, opts.refine)
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
    return _trim(out, opts.palette, trim_box)


def _trim(path: Path, palette: int = 0, box=None):
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
    if box is None:
        mask = (img < 250).any(axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad = 10
        box = (
            max(rows[0] - pad, 0),
            rows[-1] + pad + 1,
            max(cols[0] - pad, 0),
            cols[-1] + pad + 1,
        )
    r0, r1, c0, c1 = box
    out = Image.fromarray(img[r0:r1, c0:c1])
    if palette:
        out = out.quantize(
            colors=palette, method=Image.MEDIANCUT, dither=Image.NONE
        )
    out.save(path, optimize=True)
    return box


def _lab_frame_shift(plane, distance: float, length: float):
    r"""Translate a periodic plane *distance* downstream in ``x``.

    A run may integrate in a frame moving along the streamwise axis
    (``phys.u_grid``), which is a change of *coordinate* only -- the
    solver adds a convective term and leaves the stored field alone.
    Undoing it for a figure is therefore a pure translation: the lab
    field at time `$t$` is the stored one shifted by
    `$U_{\mathrm{grid}}\,(t - t_0)$`.

    ``x`` is periodic and the stored field carries no Nyquist mode, so
    the shift is done exactly as a phase on each Fourier mode rather
    than by rolling whole grid points -- the displacement is not a
    whole number of them.
    """
    if not distance:
        return plane
    n = plane.shape[-1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    spec = np.fft.rfft(plane, axis=-1) * np.exp(-1j * k * distance)
    return np.fft.irfft(spec, n=n, axis=-1)


def _plane_frame(plane, norm, cmap, width, aspect, refine):
    r"""One wall-parallel plane as a plain RGB image.

    An animation frame is the plane itself, with no axes and no
    colourbar: the caption carries what they would have said, and every
    frame must come out the *same size*, which ``bbox_inches="tight"``
    around a matplotlib figure cannot guarantee.  Mapping the array
    straight through the colormap also skips the flat-shaded surface
    mesh entirely, so no weave can appear.

    ``z`` is flipped because image rows run downward.
    """
    from PIL import Image

    fine = _refine(plane, refine)
    rgb = (cmap(norm(fine))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb[::-1]).resize(
        (width, max(round(width / aspect), 1)), Image.LANCZOS
    )


def _animate(args, info) -> int:
    r"""Animate one wall-parallel plane, or a stack of them, over time.

    Every frame shares one colour scale -- otherwise the animation
    pulses as each frame renormalises itself -- taken from ``--clim`` or
    from the given percentile over **all** frames, so the bounds suit
    the whole sequence rather than any one of it.  Only the requested
    component and the requested wall-normal slabs are read from each
    snapshot, so the cost is a few slabs per file rather than a field.

    A single ``--y`` maps the plane straight through the colormap: the
    frames are then pixel-identical in size by construction, which
    ``bbox_inches="tight"`` around a matplotlib figure cannot promise.
    Several ``--y`` render the 3D stack instead, and the crop box of the
    first frame is reused for the rest for the same reason.
    """
    import tempfile

    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image

    stacked = len(args.y) > 1
    frames_data, stations, times = [], None, []
    for path in args.snapshot:
        data = read_state(
            path, components=(args.component,), wall_normal_points=args.y
        )
        grid = np.asarray(data.physical_coords[0])
        used = [
            _wall_parallel(data.physical[0], grid, y0, demean=args.demean)
            for y0 in args.y
        ]
        frames_data.append([pl for pl, _ in used])
        stations = np.array([yy for _, yy in used])
        times.append(float(read_meta(path)["t"]))

    u_grid = float(getattr(data.params.phys, "u_grid", 0.0) or 0.0)
    drift = 0.0
    if args.lab_frame and u_grid:
        length = float(info.length[2])
        drift = u_grid * (times[-1] - times[0])
        frames_data = [
            [
                _lab_frame_shift(pl, u_grid * (t - times[0]), length)
                for pl in planes
            ]
            for t, planes in zip(times, frames_data, strict=True)
        ]

    lim = args.clim
    if lim is None:
        lim = max(
            float(np.percentile(np.abs(pl), args.clip))
            for planes in frames_data
            for pl in planes
        )
    args.clim = lim  # every frame renders on the sequence's own scale

    if stacked:
        coords = data.physical_coords
        frames, box = [], None
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "frame.png"
            for planes in frames_data:
                box = _render_stack(
                    planes,
                    stations,
                    np.asarray(coords[2]),
                    np.asarray(coords[1]),
                    labels=("$x$", "$z$", ""),
                    ticks=[
                        f"{0.0 if abs(v) < 1e-9 else v:.3g}" for v in stations
                    ],
                    out=shot,
                    dpi=args.dpi,
                    cmap=args.cmap,
                    opts=args,
                    trim_box=box,
                )
                frames.append(Image.open(shot).convert("RGB"))
    else:
        norm = Normalize(-lim, lim)
        cmap = plt.get_cmap(args.cmap)
        aspect = float(info.length[2]) / float(info.length[1])  # Lx / Lz
        frames = [
            _plane_frame(
                planes[0], norm, cmap, args.width_px, aspect, args.refine
            )
            for planes in frames_data
        ]

    if args.width_px and frames[0].size[0] != args.width_px:
        w = args.width_px
        h = max(round(frames[0].size[1] * w / frames[0].size[0]), 1)
        frames = [f.resize((w, h), Image.LANCZOS) for f in frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.out.suffix.lower()
    extra = {}
    if suffix == ".webp":
        extra = dict(quality=args.quality, method=6)
    elif suffix == ".gif":
        extra = dict(optimize=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / args.fps),
        loop=0,
        **extra,
    )
    print(
        f"[figure] {len(frames)} frames  t {times[0]:.5g}..{times[-1]:.5g}  "
        f"y "
        + ", ".join(f"{v:.3g}" for v in stations)
        + f"  {frames[0].size[0]}x{frames[0].size[1]} px  clim {lim:.4g}  "
        f"{args.fps:g} fps  {args.out.stat().st_size / 1024:.0f} KB  ->  "
        f"{args.out}"
    )
    if drift:
        print(
            f"[figure] lab frame: undid u_grid = {u_grid:.4g}, a "
            f"{drift:.4g} shift downstream across the sequence"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI (see the module docstring)."""
    p = argparse.ArgumentParser(
        prog="snapshot_figure.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "snapshot",
        type=Path,
        nargs="+",
        help="a dnsjax snapshot tar; several make an animation of one "
        "wall-parallel plane, written as .webp, .gif or .png (APNG) "
        "after the output suffix",
    )
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
        "--lab-frame",
        action="store_true",
        help="undo phys.u_grid, so an animation translates downstream at "
        "the true velocity rather than through the moving frame the run "
        "integrated in (a coordinate change only, applied here as an "
        "exact Fourier phase shift)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=12.5,
        help="animation frame rate (default 12.5)",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=72,
        help="animated-WebP quality, 0-100 (default 72)",
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
        help="interpolate each wall-parallel plane onto a grid this many "
        "times finer before rendering (exact: both of its axes are "
        "Fourier).  Both renderers shade cells flat, so above ~6 px per "
        "cell the mesh shows -- blockiness in the plane view, a weave in "
        "the stack.  Raise this rather than lowering --dpi.  The "
        "meridional view is unrefined: its radial axis is an FD grid",
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

    if len(args.snapshot) > 1:
        first = read_state(args.snapshot[0], components=(args.component,))
        info = geometry_info(first.params)
        if info.family in ("cylindrical", "annular"):
            raise SystemExit(
                "snapshot_figure: the animation draws a wall-parallel "
                f"plane; {first.params.phys.system!r} has none"
            )
        return _animate(args, info)

    data = read_state(args.snapshot[0], components=(args.component,))
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
        if args.refine > 1:
            planes = [_refine(pl, args.refine) for pl in planes]
            xc = _finer_centres(np.asarray(xc), args.refine)
            zc = _finer_centres(np.asarray(zc), args.refine)
        horiz, vert = _periodic_edges(xc), _periodic_edges(zc)
        if args.demean:
            base = comp.partition("_")[0]
            sub = _component_label(comp, "")[1:-1].partition("_")[2]
            c_label = rf"${base}_{sub} - \langle {base}_{sub} \rangle_{{xz}}$"
        re_tau = (
            friction_reynolds(read_stats(args.snapshot[0]), data.params)
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
