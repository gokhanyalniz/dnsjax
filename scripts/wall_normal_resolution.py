#!/usr/bin/env python3
r"""How many wall-normal modes does an FD grid actually resolve?

Offline (JAX-free, milliseconds) answer to the question that comes up
whenever a spectral-in-`$y$` setup from the literature is reproduced
in dnsjax: *the paper used N Chebyshev polynomials -- what ``res.ny``
do I need, given that dnsjax differentiates `$y$` by finite
differences?*  Guessing high wastes points; guessing low silently
changes the physics (in a marginal box it changes turbulent
lifetimes, not just the statistics).

The metric
----------
Point counts and formal orders do not compare across
discretizations, so this script measures the thing both
discretizations are *for*: the wall-normal Laplacian's spectrum.  On
`$y \in [-1, 1]$` with homogeneous Dirichlet data the exact
eigenvalues are `$\lambda_m = -(m\pi/2)^2$`; the reported
**resolving power** is the length of the leading run of numerical
eigenvalues within a relative tolerance of those, i.e. the number of
wall-normal modes the discretization represents faithfully.  It is
the honest currency for "N Chebyshev polynomials" -- a spectral
method's own resolving power is likewise well below its point count
(`$\approx 2N/\pi$` for Chebyshev).

Both operators are each method's *natural* choice: dnsjax's `$D_2$`
is a direct Fornberg fit (never `$D_1 D_1$` -- see the
``Resolution.consistent_imm`` docs for why that route was retired),
while the Chebyshev reference squares its differentiation matrix, as
a Chebyshev code would.

What it measured
----------------
At the time of writing, for the default ``fd_order = 8`` on the CGL
grid, **FD needs 1.5--1.9x the point count of a Chebyshev
expansion**: ``ny = 49`` matches 33 Chebyshev polynomials at the 1 %
criterion, ``ny = 63`` matches it at the stricter 0.1 % criterion and
exceeds it at 1 % (23 modes vs 18).  Two corollaries worth knowing
before turning knobs:

- ``fd_order`` is a weak lever, ``ny`` a strong one.  At ``ny = 63``,
  order 8 -> 12 buys 4 modes (+17 %) and widens every banded
  operator; ``ny`` 63 -> 97 at order 8 buys 13 (+57 %).
- The CGL grid is *Chebyshev*-optimal, not FD-optimal.  Its
  `$\Delta y \sim 1/N^2$` wall clustering overspends at the wall
  (`$\Delta y^+ = 0.04$` at ``ny = 63``, `$Re_\tau = 33$` -- far
  finer than anyone needs) while FD accuracy binds at the coarse
  centreline.  A mild ``geo.grid_type = "tanh"`` at
  ``grid_stretch = 1.0`` resolves ~20 % more modes at the same
  ``ny``, still with `$\Delta y^+ < 1$` at the wall.  It is not free:
  quadrature drops from spectral Clenshaw-Curtis to the ``fd_order``
  composite rule, and a grid change is trajectory-defining.

The sign of the FD error is also worth knowing: the Fornberg
`$D_2$` *under*-estimates damping at every mode (Chebyshev crosses
over to over-damping at high `$m$`), so an under-resolved FD grid
biases towards **sustaining** fluctuations, not towards decay.

Scope
-----
Two-wall grids spanning `$[-1, 1]$` -- the Cartesian family
(plane-Couette, plane-Poiseuille).  The cylindrical and annular
families are deliberately **out of scope**: their radial operators
carry metric terms and (on the pipe) parity classes, so
`$-(m\pi/2)^2$` is not their reference spectrum and the count would
be meaningless rather than merely approximate.

Wall units
----------
`$Re_\tau = \sqrt{S_w\,Re}$` with `$S_w$` the mean wall shear
relative to laminar (``--wall-shear``, default ``1.0`` = the exactly
defined laminar value).  For a turbulent estimate take `$S_w$` from a
run's ``stats.dat``: the `$\tau'$` columns are `$(\partial_y
u'_s)/Re$`, and the laminar plane-Couette shear is `$1$`, so

.. math::
    S_w = 1 + \frac{Re}{2}\,
          (\tau'_{s,b} + \tau'_{s,t}).

Every `$y^+$` in the output scales as `$\sqrt{S_w}$`.

Usage
-----
Reproduce the FD-vs-Chebyshev table, with the CN viscous-fidelity
block for two candidate time steps::

    uv run python scripts/wall_normal_resolution.py resolve \
        --chebyshev --ny 33 49 63 97 --re 400 --wall-shear 2.7 \
        --dt 0.01 0.025

Sweep the grid shape at fixed ``ny``::

    uv run python scripts/wall_normal_resolution.py resolve \
        --ny 63 --grid cgl tanh --grid-stretch 1.0 1.5 2.0

What ``res.ny`` replaces 33 Chebyshev polynomials?::

    uv run python scripts/wall_normal_resolution.py match \
        --chebyshev-ny 33 --fd-order 6 8 10

Horizontal resolution of the Hamilton-Kim-Waleffe box::

    uv run python scripts/wall_normal_resolution.py box \
        --nx 16 --nz 16 --lx 1.75pi --lz 1.2pi --re 400 \
        --wall-shear 2.7
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from numpy import ndarray

from dnsjax.fd import build_diff_matrices, tanh_two_sided_grid
from dnsjax.harmonics import complex_harmonics, real_harmonics
from dnsjax.parameters import round_up_padded_smooth

# ── Grids and operators ───────────────────────────────────────────


def cgl_grid(ny: int) -> ndarray:
    r"""Chebyshev-Gauss-Lobatto grid `$y_j = -\cos(j\pi/(N-1))$`.

    Mirrors the ``"cgl"`` branch of ``build_cartesian_grid``
    (``geometries/wall_bounded/cartesian.py``), which is the
    authority; it is one line and reproduced here so this script
    stays JAX-free (importing the geometry module would build the
    singletons and pull in JAX).
    """
    return -np.cos(np.arange(ny) * np.pi / (ny - 1))


def chebyshev_d1(ny: int) -> ndarray:
    r"""Chebyshev differentiation matrix on the CGL nodes.

    The standard barycentric construction, returned in dnsjax's
    wall-first (ascending `$y$`) node ordering so it composes with
    :func:`cgl_grid`.
    """
    x = np.cos(np.pi * np.arange(ny) / (ny - 1))  # descending
    c = np.ones(ny)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(ny)
    dx = x[:, None] - x[None, :]
    d = np.outer(c, 1.0 / c) / (dx + np.eye(ny))
    d -= np.diag(d.sum(axis=1))
    return d[::-1, ::-1]


def build_grid(kind: str, ny: int, stretch: float) -> ndarray:
    """Grid points for a named ``geo.grid_type`` on ``[-1, 1]``."""
    if kind == "cgl":
        return cgl_grid(ny)
    if kind == "tanh":
        return tanh_two_sided_grid(ny, stretch)
    raise ValueError(f"unknown grid {kind!r} (cgl, tanh)")


def build_d2(kind: str, ny: int, order: int | None, stretch: float) -> ndarray:
    r"""Second-derivative matrix; ``order = None`` selects Chebyshev.

    Chebyshev squares its `$D_1$` (what a Chebyshev code does);
    finite differences take dnsjax's direct Fornberg `$D_2$` fit.
    """
    if order is None:
        d1 = chebyshev_d1(ny)
        return d1 @ d1
    return build_diff_matrices(build_grid(kind, ny, stretch), order)[1]


# ── The resolving-power metric ────────────────────────────────────


def dirichlet_spectrum(d2: ndarray) -> tuple[ndarray, int]:
    r"""Interior (Dirichlet) spectrum of `$D_2$`, most-negative last.

    Drops the two wall rows/columns, which is homogeneous Dirichlet
    data for a second-derivative operator.  Returns the real
    eigenvalues, ordered towards `$-\infty$`, and the position at
    which the **first complex** eigenvalue sits in that ordering
    (``len(lam)`` when there is none).

    An FD `$D_2$` on a stretched grid is not symmetric and does carry
    complex pairs -- two of them (four eigenvalues) on a ``tanh``
    grid at the sizes checked so far.  They are
    grid-scale artefacts and sit far beyond the resolved band, but
    they still have to be *located*: mode `$m$` is read off position
    `$m-1$`, so a complex pair dropped from the middle of the
    sequence would silently renumber every mode above it.  Reporting
    the position lets the caller check that the two never overlap
    instead of assuming it.
    """
    lam = np.linalg.eigvals(d2[1:-1, 1:-1])
    lam = lam[np.argsort(-lam.real)]
    real = np.abs(lam.imag) < 1e-8 * np.abs(lam).max()
    first_complex = len(lam) if real.all() else int(np.argmax(~real))
    return lam[real].real, first_complex


def resolved_modes(d2: ndarray, tol: float) -> tuple[int, int]:
    r"""Leading run of eigenvalues within *tol* of `$-(m\pi/2)^2$`.

    Returns ``(count, first_complex)``; the count is trustworthy iff
    ``count <= first_complex`` (see :func:`dirichlet_spectrum`).  A
    *leading run* rather than a total count: resolution is contiguous
    from the gravest mode up, and one accidentally accurate high
    eigenvalue is not resolution.
    """
    lam, first_complex = dirichlet_spectrum(d2)
    m = np.arange(1, len(lam) + 1)
    exact = -((m * np.pi / 2.0) ** 2)
    bad = np.nonzero(np.abs(lam - exact) / np.abs(exact) > tol)[0]
    count = len(lam) if len(bad) == 0 else int(bad[0])
    return count, first_complex


def eigenvalue_error(d2: ndarray, modes: tuple[int, ...]) -> list[float]:
    r"""Signed relative error `$\lambda_{num}/\lambda_{exact} - 1$`.

    Negative = the operator under-estimates the damping of that mode.
    """
    lam, _ = dirichlet_spectrum(d2)
    out = []
    for m in modes:
        if m > len(lam):
            out.append(float("nan"))
            continue
        out.append(float(lam[m - 1] / -((m * np.pi / 2.0) ** 2) - 1.0))
    return out


# ── Crank-Nicolson viscous fidelity ───────────────────────────────


def cn_amplification(x: float) -> float:
    r"""CN amplification factor `$(1 - x/2)/(1 + x/2)$`, `$x = \nu k^2
    \Delta t$` (exact: `$e^{-x}$`)."""
    return (1.0 - x / 2.0) / (1.0 + x / 2.0)


def cn_crossover() -> float:
    r"""`$x$` where CN turns from over- to under-damping.

    The root of `$|(1-x/2)/(1+x/2)| = e^{-x}$` beyond `$x = 2$`:
    below it CN damps a mode *harder* than the exact `$e^{-x}$`,
    above it the amplification tends to `$-1$` and stiff modes are
    retained instead of killed.  Bisection -- the bracket is fixed
    and the function is smooth and monotone across it.
    """
    lo, hi = 2.0, 10.0

    def f(x: float) -> float:
        return abs(cn_amplification(x)) - math.exp(-x)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cn_faithful_wavelength(re: float, dt: float, x_star: float) -> float:
    r"""Shortest `$y$`-wavelength CN still damps faithfully.

    `$\nu k^2 \Delta t = x_*$` with `$\nu = 1/Re$`, i.e.
    `$\lambda = 2\pi\sqrt{\Delta t/(x_* Re)}$`.  Structures larger
    than this see the correct viscous decay at this `$\Delta t$`;
    smaller ones are increasingly retained.
    """
    return 2.0 * np.pi * math.sqrt(dt / (x_star * re))


# ── Subcommands ───────────────────────────────────────────────────


def _re_tau(re: float | None, wall_shear: float) -> float | None:
    return None if re is None else math.sqrt(wall_shear * re)


def cmd_resolve(args: argparse.Namespace) -> None:
    """Resolving power / spacings / CN fidelity for each config."""
    rt = _re_tau(args.re, args.wall_shear)
    configs: list[tuple[str, int, int | None, float]] = []
    for ny in args.ny:
        if args.chebyshev:
            configs.append(("cgl", ny, None, 0.0))
        for kind in args.grid:
            stretches = args.grid_stretch if kind == "tanh" else [0.0]
            for s in stretches:
                for order in args.fd_order:
                    configs.append((kind, ny, order, s))

    print(f"Resolving power  (tol {args.tol:.0%} / {args.tol_tight:.1%};")
    print("  = leading run of Dirichlet eigenvalues within tolerance)")
    if rt is not None:
        print(
            f"  wall units at Re_tau = {rt:.1f}  "
            f"(Re = {args.re:g}, S_w = {args.wall_shear:g})"
        )
    head = f"{'grid':>12} {'ny':>4} {'p':>3} | {'m@lo':>5} {'m@hi':>5} |"
    head += f" {'dy_wall':>9} {'dy_ctr':>9}"
    if rt is not None:
        head += f" | {'y+_wall':>8} {'y+_ctr':>7}"
    print()
    print(head)
    print("-" * len(head))

    rows = []
    for kind, ny, order, s in configs:
        d2 = build_d2(kind, ny, order, s)
        n_lo, first_complex = resolved_modes(d2, args.tol)
        n_hi, _ = resolved_modes(d2, args.tol_tight)
        y = cgl_grid(ny) if order is None else build_grid(kind, ny, s)
        d = np.diff(y)
        label = "chebyshev" if order is None else kind
        if order is not None and kind == "tanh":
            label = f"tanh s={s:g}"
        line = f"{label:>12} {ny:>4} {'-' if order is None else order:>3} |"
        line += f" {n_lo:>5} {n_hi:>5} | {d[0]:>9.3e} {d.max():>9.3e}"
        if rt is not None:
            line += f" | {d[0] * rt:>8.3f} {d.max() * rt:>7.3f}"
        print(line)
        if n_lo > first_complex:
            print(
                f"{'':>12} ^ a complex eigenvalue sits at mode "
                f"{first_complex + 1}, inside the counted run: the modes"
                " above it are renumbered and this count is wrong"
            )
        rows.append((label, ny, order, d2))

    if args.modes:
        print()
        print(
            "Signed relative eigenvalue error "
            "(lam_num/lam_exact - 1); negative"
        )
        print("= the operator under-estimates that mode's damping.")
        head = f"{'grid':>12} {'ny':>4} {'p':>3} | "
        head += " ".join(f"{'m=' + str(m):>9}" for m in args.modes)
        print(head)
        for label, ny, order, d2 in rows:
            errs = eigenvalue_error(d2, tuple(args.modes))
            line = f"{label:>12} {ny:>4} {'-' if order is None else order:>3}"
            print(line + " | " + " ".join(f"{e:>+9.1e}" for e in errs))

    if args.dt and args.re is not None:
        x_star = cn_crossover()
        print()
        print(
            f"Crank-Nicolson viscous fidelity (nu = 1/Re = {1 / args.re:.3e})"
        )
        print(
            f"  Faithful down to nu k^2 dt = {x_star:.2f}; beyond it CN"
            " retains a"
        )
        print("  mode instead of damping it.  Grid-independent:")
        for dt in args.dt:
            lam_min = cn_faithful_wavelength(args.re, dt, x_star)
            tail = f" = {lam_min * rt:6.2f} y+" if rt is not None else ""
            print(f"    dt = {dt:<8g} lambda_y > {lam_min:.4f}{tail}")
        print()
        print("  The stiffest mode each grid carries (CN amp -> -1 is a")
        print("  mode retained rather than damped; exact is e^-x):")
        head = f"{'grid':>12} {'ny':>4} {'p':>3} {'dt':>8} | "
        head += f"{'nu|l|max dt':>11} {'CN amp':>8} {'exact':>10}"
        print()
        print(head)
        print("-" * len(head))
        for label, ny, order, d2 in rows:
            lam, _ = dirichlet_spectrum(d2)
            nu_lmax = np.abs(lam).max() / args.re
            for dt in args.dt:
                x = nu_lmax * dt
                line = f"{label:>12} {ny:>4} "
                line += f"{'-' if order is None else order:>3} {dt:>8g} | "
                line += f"{x:>11.1f} {cn_amplification(x):>+8.3f} "
                line += f"{math.exp(-x):>10.2e}"
                print(line)
    elif args.dt:
        print()
        print("--dt given without --re: the CN viscous-fidelity block")
        print("needs a viscosity and was skipped.")


def cmd_match(args: argparse.Namespace) -> None:
    """Smallest FD ``ny`` matching a Chebyshev expansion's power."""
    target, _ = resolved_modes(
        build_d2("cgl", args.chebyshev_ny, None, 0.0), args.tol
    )
    print(
        f"{args.chebyshev_ny} Chebyshev polynomials resolve {target} "
        f"wall-normal modes to {args.tol:.1%}."
    )
    print(f"Smallest FD ny reaching that on the {args.grid} grid:")
    print()
    print(f"{'fd_order':>9} | {'ny':>5} {'modes':>6} {'ratio':>7}")
    print("-" * 33)
    for order in args.fd_order:
        found = None
        for ny in range(order + 3, args.max_ny + 1):
            got, first_complex = resolved_modes(
                build_d2(args.grid, ny, order, args.grid_stretch), args.tol
            )
            if got >= target:
                found = (ny, got, got > first_complex)
                break
        if found is None:
            print(f"{order:>9} | none at ny <= {args.max_ny}")
            continue
        ny, got, suspect = found
        line = f"{order:>9} | {ny:>5} {got:>6} "
        line += f"{ny / args.chebyshev_ny:>7.2f}"
        print(
            line
            + ("  (!) complex eigenvalue inside the run" if suspect else "")
        )
    print()
    print("ratio = FD ny / Chebyshev ny.  The scan starts at the")
    print("smallest ny the stencils fit on and stops at the first")
    print("match, so a stricter --tol raises every row.")


def cmd_box(args: argparse.Namespace) -> None:
    """Retained horizontal harmonics and their wall-unit scales."""
    rt = _re_tau(args.re, args.wall_shear)
    kx = real_harmonics(args.nx)
    kz = complex_harmonics(args.nz)
    nxp = round_up_padded_smooth(args.oversampling * args.nx // 2, 1)
    nzp = round_up_padded_smooth(args.oversampling * args.nz // 2, 1)
    print(f"Box  lx = {args.lx:.4f}  lz = {args.lz:.4f}")
    print("Modes retained (res.nx/nz are pre-dealiasing counts):")
    print(
        f"  x (real FFT) nx = {args.nx:>4} -> {len(kx):>3} modes, "
        f"k in [{int(kx.min())}, {int(kx.max())}]"
    )
    print(
        f"  z (complex)  nz = {args.nz:>4} -> {len(kz):>3} modes, "
        f"k in [{int(kz.min())}, {int(kz.max())}]"
    )
    lx_min = args.lx / max(int(kx.max()), 1)
    lz_min = args.lz / max(int(kz.max()), 1)
    print()
    print("Shortest resolved wavelength (the resolution that matters):")
    tail_x = f" = {lx_min * rt:6.1f} x+" if rt else ""
    tail_z = f" = {lz_min * rt:6.1f} z+" if rt else ""
    print(f"  lambda_x = {lx_min:.4f}{tail_x}")
    print(f"  lambda_z = {lz_min:.4f}{tail_z}")
    print()
    print(f"Collocation spacing on the {args.oversampling}/2-padded grid")
    print(f"({nxp} x {nzp} points) -- a dealiasing artefact, not the")
    print("resolution; quoted only to compare with codes that report it:")
    tail_x = f" = {args.lx / nxp * rt:6.2f} x+" if rt else ""
    tail_z = f" = {args.lz / nzp * rt:6.2f} z+" if rt else ""
    print(f"  dx = {args.lx / nxp:.4f}{tail_x}")
    print(f"  dz = {args.lz / nzp:.4f}{tail_z}")
    if rt is not None:
        print()
        print(
            f"Re_tau = {rt:.1f} (Re = {args.re:g}, S_w = "
            f"{args.wall_shear:g}); every wall unit scales as sqrt(S_w)."
        )
    print()
    print("Padded sizes assume one device per axis; a multi-device run")
    print("rounds them up further (the startup diagnostic is authority).")


# ── CLI ───────────────────────────────────────────────────────────


def _length(text: str) -> float:
    r"""Parse a domain length, accepting a ``pi`` factor.

    ``"5.4978"``, ``"1.75pi"`` and ``"1.75*pi"`` are all accepted --
    periodic box sizes are almost always quoted as multiples of
    `$\pi$`.
    """
    s = text.strip().lower().replace("*", "")
    if s.endswith("pi"):
        head = s[:-2].strip()
        return (float(head) if head else 1.0) * math.pi
    return float(text)


def _add_wall_unit_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--re",
        type=float,
        default=None,
        help="Reynolds number; enables the wall-unit columns.",
    )
    ap.add_argument(
        "--wall-shear",
        type=float,
        default=1.0,
        help="Mean wall shear relative to laminar S_w (default 1.0 = "
        "laminar); Re_tau = sqrt(S_w Re).  See the module "
        "docstring for reading it off stats.dat.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("resolve", help="resolving power of given grids")
    p.add_argument("--ny", type=int, nargs="+", default=[33, 49, 63, 97])
    p.add_argument("--fd-order", type=int, nargs="+", default=[8])
    p.add_argument(
        "--grid", nargs="+", default=["cgl"], choices=["cgl", "tanh"]
    )
    p.add_argument("--grid-stretch", type=float, nargs="+", default=[1.5])
    p.add_argument(
        "--chebyshev",
        action="store_true",
        help="Add a Chebyshev reference row at each ny.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Loose eigenvalue tolerance (default 1%%).",
    )
    p.add_argument(
        "--tol-tight",
        type=float,
        default=1e-3,
        help="Tight eigenvalue tolerance (default 0.1%%).",
    )
    p.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=[],
        help="Report the signed eigenvalue error at these mode numbers.",
    )
    p.add_argument(
        "--dt",
        type=float,
        nargs="*",
        default=[],
        help="Time steps for the CN viscous-fidelity block (needs --re).",
    )
    _add_wall_unit_args(p)
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("match", help="FD ny matching a Chebyshev ny")
    p.add_argument("--chebyshev-ny", type=int, required=True)
    p.add_argument("--fd-order", type=int, nargs="+", default=[4, 6, 8, 10])
    p.add_argument("--grid", default="cgl", choices=["cgl", "tanh"])
    p.add_argument("--grid-stretch", type=float, default=1.5)
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--max-ny", type=int, default=401)
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("box", help="horizontal resolution of a box")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument(
        "--lx",
        type=_length,
        required=True,
        help="Streamwise period (accepts '1.75pi').",
    )
    p.add_argument(
        "--lz",
        type=_length,
        required=True,
        help="Spanwise period (accepts '1.2pi').",
    )
    p.add_argument(
        "--oversampling",
        type=int,
        default=3,
        help="phys.oversampling_factor (default 3 = the 3/2 rule).",
    )
    _add_wall_unit_args(p)
    p.set_defaults(func=cmd_box)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
