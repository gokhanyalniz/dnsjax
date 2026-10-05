r"""Premultiplied `$(\lambda, y)$` maps (``twin_spectral_maps.py``).

``scripts/twin_spectral_maps.py`` turns the twin `$(y, k)$` streams
into figures, and everything between the stored entry and the drawn
contour is arithmetic this file pins.  The streams are built **in
memory** -- ``_record_dtype`` on a hand-written sidecar, then a
structured array -- so no case needs a solver run, and the three that
exercise the reader write those same bytes into a temporary directory.

matplotlib is not a solver dependency (the ``plots`` group), so the
script skips with a message rather than failing where it is absent.

What each case pins:

1. **Premultiplication.** A `$k$`-premultiplied panel is
   `$m \times \text{entry} \times V$` in the unit conversion its
   stream asks for, on both marginals and both ``kind``s; the
   `$m = 0$` column is gone, the wavelength axis ascends, and
   ``--premultiply none`` / ``--no-volume-fac`` drop exactly their own
   factor.
2. **The ``ky`` convention.** Its second factor is the wall distance
   *in the plotted units*, so a map's wall/outer ratio is `$Re_\tau$`
   under ``ky`` against `$1$` under ``k`` -- the asymmetry the module
   docstring documents ("Premultiplication"), asserted here so it
   cannot change silently in either direction.
3. **`$E^{\mathrm{ref}}$`.** The total-in-`$(y, k)$` reference energy
   less its `$(0, 0)$` mode, averaged over the **distinct** absolute
   instants of the member set: the dedup counts a pair straddling
   :data:`~twin_spectral_maps._T_ATOL` once, the summed panel takes
   the summed reference, the `$k_x = 0$` plane stays absolute, and a
   doctored second marginal is refused.
4. **The colour scale is a legend for the visible map.** A peak below
   the ordinate's floor must not set the levels -- under ``--clim
   frame`` and ``--quantile`` as much as under the frozen default --
   and ``--fill contour`` extends exactly where ``--quantile`` can put
   something above the top level.
5. **The sign family is declared.** A budget term that never goes
   negative still draws signed; the declared non-positive
   `$-\mathcal{D}_\Delta$` with a round-off positive draws signed and
   says so; ``--signs-from-data`` infers both instead.
6. **The fold.** ``mean`` averages `$j$` with `$n_y-1-j$` without
   double counting the mid-plane, ``upper`` relabels its rows with the
   opposite half's wall distances, and the grid preconditions each
   mode actually needs are the ones checked.
7. **The reader.** Members meet on relative time to a tolerance and
   the shared grid is their intersection; ``stride`` / ``first`` /
   ``last`` clip it; a member that is not the same flow, and a stream
   whose own samples are closer than the tolerance, are both refused.
   So is a set whose grids are out of *phase* -- ``--align-atol``
   accepts one, up to half a cadence -- while a member merely
   covering another stretch of the clock is not confused with it.
8. **The two decorrelations.** `$\mathcal{R}$` divides mode by mode
   and `$\mathcal{R}^k$` by the same reference summed over `$k$`;
   both take the `$(0, 0)$` mode off the reference and neither off
   the perturbation, the summed panel is one ratio of sums, the
   premultiplier reaches the second and not the first, and
   ``volume_fac`` / the unit conversion reach neither.  A saturated
   pair reads exactly 1, and an empty reference reads ``nan`` without
   touching a colour scale.
9. **The divisor is symmetrised before the fold.** On a deliberately
   asymmetric reference, ``--half mean`` gives the ratio of the
   folded halves rather than the mean of the two ratios, which is
   what makes the answer independent of the order.
10. **The `$k$`-sum.** Marginal-free on all three stream layouts, and
    what each half of a decorrelation counts: every mode of the
    perturbation, every mode but `$(0, 0)$` of the reference.
11. **Spacetime.** The `$(y, t)$` map is that `$k$`-sum in the
    plotted units, folded and never premultiplied; its colour range
    sees only the columns the box shows; the logarithmic floor is the
    higher of *decades* below the peak and the smallest positive
    value; a signed series gets no logarithmic figure and a
    one-sample selection none at all; and the ``.npz`` beside the
    pair carries the drawn arrays and every factor behind them.
12. **End to end.** ``main()`` on a two-member set draws the
    difference-spectra, shape and budget marginals, each tracked
    series with its ``_track`` directory, and nothing else;
    ``--no-budget`` drops the budget, and each of the five opt-in
    switches adds exactly its own family -- ``--reference`` and
    ``--decorr-k`` each needing ``--spacetime`` as well before their
    spacetime map.  A budget figure is three columns of the spectra's
    panels.
13. **The budget is the balance.** The panels are the balance terms
    regrouped from the stored densities, each read the same way by
    all three readers, in the rows of the 3 x 3 grid; a map's pressure
    panel leaves out the driving input its `$m = 0$` column would
    hold, a spacetime map's carries it in the same place, so those
    panels add up to the ``sum``, which stays the stored one and is
    titled `$\partial_t E_\Delta$`; a contribution's minus sign leads
    its title; and a rotational stream is refused.
14. **The budget grid.** Three columns whatever ``--ncols`` says, at
    the panel size ``--width`` fits to ``--ncols`` columns, so a
    budget figure is wider than ``--width`` rather than
    narrower-paneled; the nine panels sit row by row.
15. **``--clim ramped``.** Each frame's range is the extremes of every
    frame so far: it contains the frame's own, never shrinks, holds an
    earlier extreme through a dip, ramps the two sides of a signed
    panel separately, is the frozen range from the frame holding the
    series extreme onward, and is what the figure's levels are read
    from.
16. **Peak tracking.** The centroid of the frame's own top band --
    the cells at or above `$(1 - 1/n)$` of its peak, weighted by value
    and by trapezoidal widths in `$\ln\lambda$` and `$\ln y$`
    (`$y$` on a linear ordinate), averaged in those coordinates -- sits
    at the centre of a symmetric band, takes in the edge cell exactly
    at the threshold, and is ``nan`` with nothing positive.  Exactly
    the tracked panels carry it, it is the same under every
    ``--clim``, and its figure and ``.npz`` are written beside the
    frames.
17. **The reference's two layouts.** The same reference records, kept
    inside the difference stream (the pre-split layout) or in their
    own ``twin_yspectra_ref.bin``, give the same normalisation, maps,
    decorrelations and spacetime map, in a set of either or a mixed
    set; a reference on half the difference cadence normalises over
    its own samples and refuses a reference map only at the frames it
    lacks.
18. **Shape maps.** Each panel of each frame is the absolute spectrum
    premultiplied by `$k\,y^+$`, over its own peak on the rows the box
    shows -- whatever ``--premultiply`` says, the same in wall and
    outer units and without ``volume_fac``, and the same in every
    frame of one shape grown by decades; the peak is what the title
    reports, a zero field keeps zeros, and the figure's levels are the
    same `$[0, 1]$` bands under every ``--clim`` and ``--quantile``.
19. **History maps.** The `$(y, t)$` map is the `$k$`-summed map times
    the plotted wall distance, the `$(\lambda, t)$` map the
    wall-normal average over the doubled half-channel weights times
    `$m$` with `$m = 0$` dropped -- each premultiplied after its sum,
    ``--no-volume-fac`` reaching neither average; the two read one
    total; a shape history's rows reach 1 in any unit system; the
    budget history's production is its two parts; and the renderer
    writes what the family promises, the logarithmic figure only for
    an absolute non-negative history.
20. **Quantile lines** sit at their fractions of a uniform density in
    `$\ln x$` (in `$x$` on a linear axis), and an empty or negative
    row has none.
21. **Size and tilt.** A Gaussian planted in `$(\ln\lambda, \ln y)$`
    comes back with its centroid and covariance; a constant factor
    moves nothing; a signed map's moments are its positive part's,
    the negative share reported; the ellipse is `$d^\top C^{-1} d =
    1$`; every tracked frame carries its moments; and ``--no-frames``
    writes the tracks and moments and no frame.
22. **The moment budget closes.** On a stream whose terms add up to
    `$\partial_t e$`, the per-term rates sum to the moments' own
    rates in physical space and on both marginals, and two streams on
    different frames are refused.
23. **Front times** take the last upward crossing of the level,
    interpolated, so a dip below it is not the crossing; a cell never
    above it is ``nan``, one never below it the first time -- which
    is every cell of a saturated pair.
24. **Growth laws.** A curve built exponential-then-constant-rate is
    marked with both rates; the global curve aligns the members'
    ``twin.dat`` on whole steps, is their geometric mean, records its
    source and falls back to the spectra totals without one; and a
    band's budget rates are its own terms over its own energy.

Usage::

    uv run --group plots python tests/test_twin_spectral_maps.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

try:
    import matplotlib
except ImportError:  # pragma: no cover - the plots group is optional
    print(
        "matplotlib is not installed (the `plots` dependency group); "
        "skipping.  Run with:\n  uv run --group plots python "
        "tests/test_twin_spectral_maps.py"
    )
    raise SystemExit(0) from None

matplotlib.use("Agg")

import twin_spectral_maps as tsm  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.contour import ContourSet  # noqa: E402

from dnsjax.analysis.twin.yspectra import (  # noqa: E402
    MIN_YSPECTRA_REF_VERSION,
)

# ── Fixtures ─────────────────────────────────────────────────────────

#: A plane-Poiseuille-shaped configuration, small enough that every
#: case is a handful of arrays.  ``RE_TAU`` is the measured number of
#: the KMM4200 ensemble, so the printed inner units look like a run's.
RE, RE_TAU = 4200.0, 178.62135279727977
U_TAU = RE_TAU / RE
NY, NKZ, NKX = 33, 6, 5
LX, LZ, VOLUME_FAC = 4.0, 2.0, 2.0
#: The default (convective) stream's stored terms, which the balance
#: is built from (:func:`~dnsjax.analysis.twin.yspectra.balance_term`).
TERMS = ("P_U", "P_r", "T_ref", "T_self", "V", "eps", "Wp")


def _grid(ny: int = NY) -> np.ndarray:
    """The solver's CGL grid, ascending from the lower wall."""
    return -np.cos(np.arange(ny) * np.pi / (ny - 1))


#: The three on-disk layouts a member can have.  ``LEGACY`` is what
#: was written before ``xz00`` existed and carries no ``suffixes``
#: key at all; the other two name themselves.
LEGACY = ("x", "z", "x0")
DEFAULT = ("x", "z", "xz00")
WITH_X0 = ("x", "z", "x0", "xz00")


def _meta(stem: str, *, suffixes=DEFAULT, **over) -> dict:
    """A sidecar for *stem*; *over* replaces any key.

    *suffixes* picks the layout.  :data:`LEGACY` writes the sidecar a
    pre-``xz00`` run left -- no ``suffixes`` key, and the reader's
    floor version -- which is what the back-compatibility cases need.
    """
    ny = over.pop("ny", NY)
    y = np.asarray(over.pop("y", _grid(ny)), dtype=float)
    meta = {
        "format_version": tsm.STEMS[stem],
        "system": "plane-poiseuille",
        "ny": ny,
        "n_kz": NKZ,
        "n_kx": NKX,
        "kz_harmonics": list(range(NKZ)),
        "kx_harmonics": list(range(NKX)),
        "lx": LX,
        "lz": LZ,
        "y": [float(v) for v in y],
        "y_weights": [VOLUME_FAC / ny] * ny,
        "volume_fac": VOLUME_FAC,
        "value_dtype": "<f8",
        "twin": {"seed": 1, "e0": 1e-6, "smoothness": 4.0},
    }
    if tuple(suffixes) != LEGACY:
        meta["suffixes"] = list(suffixes)
        meta["format_version"] = tsm.STEMS[stem] + 1
    if stem == "twin_yspectra":
        meta["includes_ref"] = True
    else:
        meta["terms"] = list(over.pop("terms", TERMS))
    meta.update(over)
    return meta


def _records(
    meta: dict, stem: str, n_t: int, *, t0: float = 100.0, seed: int = 0
) -> np.ndarray:
    """*n_t* records of *stem*, one time unit apart.

    Every stored field is a marginal of a genuine `$(k_z, k_x)$`
    plane -- the budget terms as much as the spectra -- so the two
    marginals of one quantity agree on its total by construction,
    which is what a real stream does and what
    :meth:`~twin_spectral_maps.YSeries._check_marginals` and
    :func:`~twin_spectral_maps.check_k_sum` both demand.  Independent
    draws per stored field would be a stub no stream could produce,
    and would make those guards look untestable rather than tested.
    """
    rec = np.zeros(n_t, dtype=tsm._record_dtype(meta, stem))
    rec["t"] = t0 + np.arange(n_t, dtype=float)
    rng = np.random.default_rng(seed)
    stored = tsm.stored_suffixes(meta)
    if stem == "twin_yspectra":
        plane = rng.random((n_t, 3, meta["ny"], NKZ, NKX))
        fields = [("e", 0.3 * plane)]
        if meta["includes_ref"]:
            fields.append(("r", plane))
        for prefix, field in fields:
            blocks = {
                "x": field.sum(axis=4),
                "z": field.sum(axis=3),
                "x0": field[..., 0],
                "xz00": field[..., 0, 0],
            }
            for suffix in stored:
                rec[f"{prefix}_{suffix}"] = blocks[suffix]
    else:
        for term in meta["terms"]:
            plane = rng.random((n_t, meta["ny"], NKZ, NKX))
            # ``eps`` is a sum of squares in a real stream, and the
            # sign check would rightly complain about a signed one.
            if term != "eps":
                plane = plane - 0.5
            blocks = {
                "x": plane.sum(axis=3),
                "z": plane.sum(axis=2),
                "x0": plane[..., 0],
                "xz00": plane[..., 0, 0],
            }
            for suffix in stored:
                rec[f"{term}_{suffix}"] = blocks[suffix]
    return rec


def _ref_meta(meta: dict) -> dict:
    """The ``twin_yspectra_ref`` sidecar beside a split-layout *meta*."""
    ref = {k: v for k, v in meta.items() if k != "includes_ref"}
    ref["format_version"] = MIN_YSPECTRA_REF_VERSION
    return ref


def _ref_records(meta: dict, rec: np.ndarray) -> np.ndarray:
    """The ``r_*`` half of combined records *rec*, as its own stream."""
    dtype = tsm._record_dtype(meta, "twin_yspectra_ref")
    out = np.zeros(rec.size, dtype=dtype)
    for name in out.dtype.names:
        out[name] = rec[name]
    return out


def _member(
    meta: dict,
    rec: np.ndarray,
    *,
    path: str = "m",
    parent: str = "p0",
    parent_t: float | None = None,
    ref: tuple[dict, np.ndarray] | None = None,
) -> tsm._Member:
    """One opened member, without going through the filesystem.

    Resolved the way :func:`~twin_spectral_maps._open_member` resolves
    one, since these members never pass through it: the stored layout
    is written back onto the sidecar, so a legacy member has a key to
    compare, and a ``twin_yspectra`` member takes its reference from
    its own records when the sidecar ``includes_ref`` (the layout
    before the reference had a stream of its own), else from *ref*, a
    ``twin_yspectra_ref`` ``(meta, records)`` pair.
    """

    def resolve(meta: dict, rec: np.ndarray):
        meta = meta | {"suffixes": list(tsm.stored_suffixes(meta))}
        t = rec["t"].astype(np.float64)
        rows = np.sort(np.unique(t, return_index=True)[1])
        return meta, rows, t[rows]

    meta, rows, t_abs = resolve(meta, rec)
    t0 = float(t_abs[0]) if parent_t is None else float(parent_t)
    reference = None
    if "includes_ref" in meta:  # a twin_yspectra sidecar
        if meta["includes_ref"]:
            reference = tsm._Reference(meta, rec, rows, t_abs, t_abs - t0)
        elif ref is not None:
            r_meta, r_rows, r_t = resolve(*ref)
            reference = tsm._Reference(r_meta, ref[1], r_rows, r_t, r_t - t0)
        meta["has_ref"] = reference is not None
    return tsm._Member(
        Path(path), meta, rec, rows, t_abs, t_abs - t0, parent, reference
    )


def _series(stem: str, members, **over) -> tsm.YSeries:
    """A series over *members*, every record kept."""
    n = members[0].t_abs.size
    t_rel = members[0].t_rel[:n]
    over.setdefault("ref_rows", tsm.reference_rows(members, t_rel))
    return tsm.YSeries(
        stem=stem,
        members=tuple(members),
        rows=np.stack([m.rows[:n] for m in members]),
        index=np.arange(n),
        t_rel=t_rel,
        t_members=np.stack([m.t_rel[:n] for m in members]),
        matched=np.full(len(members), n),
        meta=members[0].meta,
        **over,
    )


def _write_member(
    directory: Path,
    stem: str,
    meta: dict,
    rec: np.ndarray,
    *,
    parent: str = "parent.tar",
    parent_t: float | None = None,
) -> Path:
    """Write one member's stream pair (and its ``twin.json``)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps(meta))
    (directory / f"{stem}.bin").write_bytes(rec.tobytes())
    (directory / "twin.json").write_text(
        json.dumps(
            {
                "parent": parent,
                "parent_t": float(
                    rec["t"][0] if parent_t is None else parent_t
                ),
            }
        )
    )
    return directory


def _raises(call, fragment: str) -> None:
    """*call* must refuse, naming *fragment*."""
    try:
        call()
    except (ValueError, FileNotFoundError, SystemExit) as exc:
        assert fragment in str(exc), f"want {fragment!r}, got: {exc}"
        return
    raise AssertionError(f"expected a refusal naming {fragment!r}")


def _folded(values: np.ndarray, ny: int = NY) -> np.ndarray:
    """The ``--half mean`` fold, written out independently."""
    n_half = (ny + 1) // 2
    return 0.5 * (values[:n_half] + values[::-1][:n_half])


def _symmetric(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """What a ``--half mean`` fold makes of a divisor (case 9)."""
    return 0.5 * (values + np.flip(values, axis=axis))


def _saturated(meta: dict, n_t: int = 3, seed: int = 5) -> np.ndarray:
    r"""A pair that has decorrelated completely.

    `$e = 2(r - r^{00})$`: twice the reference's own energy in every
    mode but the wall-parallel mean, of which the difference field is
    given none.  `$\mathcal{R}$` is then 1 at every plotted mode and
    the `$k$`-summed `$\mathcal{R}^k$` 1 everywhere -- exactly, and
    only because each half counts the modes it is documented to count.

    The reference is held steady in time, so its average is itself,
    and symmetric about the centreline, so the fold has nothing left
    to do and the reading does not lean on case 9's algebra.
    """
    rec = _records(meta, "twin_yspectra", n_t, seed=seed)
    stored = tsm.stored_suffixes(meta)
    for suffix in stored:
        field = rec[f"r_{suffix}"]
        field[:] = field[0]  # steady in time
        field[:] = _symmetric(field, axis=2)  # and R_y-symmetric
    name = tsm.mean_mode_name(meta, "r")
    mean_mode = tsm.mean_mode_profile(rec[name], name)
    for suffix in stored:
        rec[f"e_{suffix}"] = (
            np.zeros_like(rec[f"e_{suffix}"])
            if suffix == "xz00"
            else 2.0 * tsm.mean_free_spectrum(rec[f"r_{suffix}"], mean_mode)
        )
    return rec


# ── Cases ────────────────────────────────────────────────────────────


def test_premultiplication() -> None:
    r"""A panel is `$m \times$` entry `$\times V$`, in stream units."""
    units = tsm.Units(RE, RE_TAU)
    # ``e_x0`` is the absolute spectra panel, so it is the one that
    # exercises the un-normalised branch -- on a legacy member, which
    # is where that field now comes from.
    spectra_meta = _meta("twin_yspectra", suffixes=LEGACY)
    spectra = _records(spectra_meta, "twin_yspectra", 2)
    budget_meta = _meta("twin_ybudget")
    budget = _records(budget_meta, "twin_ybudget", 2, seed=3)
    cases = (
        # (series, field, harmonic count, box length, stored -> plotted)
        (
            _series("twin_yspectra", [_member(spectra_meta, spectra)]),
            "e_x0",
            NKZ,
            LZ,
            lambda v: units.energy(v),
        ),
        (
            _series("twin_ybudget", [_member(budget_meta, budget)]),
            "tr_visc_z",
            NKX,
            LX,
            lambda v: units.rate(v),
        ),
    )
    for series, name, n_k, length, convert in cases:
        stored = series.field(name)[0]
        if series.stem == "twin_yspectra":
            stored = stored.sum(axis=0)
        harmonics = np.arange(1, n_k, dtype=float)

        drawn = tsm.make_map(series, name, 0, options=tsm.MapOptions(units))
        want = convert(stored[:, 1:] * harmonics * VOLUME_FAC)[:, ::-1]
        assert np.allclose(drawn.values, _folded(want)), name
        # ``lambda = L / m``, ascending, with no place for ``m = 0``.
        assert drawn.values.shape[1] == n_k - 1
        assert np.allclose(drawn.lam, np.sort(length / harmonics) * RE_TAU)
        assert np.all(np.diff(drawn.lam) > 0)
        assert np.all(np.diff(drawn.y) > 0)

        # Each factor comes off on its own and takes nothing with it.
        bare = tsm.make_map(
            series, name, 0, options=tsm.MapOptions(units, premultiply="none")
        )
        assert np.allclose(bare.values * harmonics[::-1], drawn.values)
        no_v = tsm.make_map(
            series, name, 0, options=tsm.MapOptions(units, volume_fac=False)
        )
        assert np.allclose(no_v.values * VOLUME_FAC, drawn.values)

        # A single component of the spectra, against the same algebra.
        if series.stem == "twin_yspectra":
            one = tsm.make_map(
                series, name, 0, options=tsm.MapOptions(units), component=0
            )
            per = series.field(name)[0][0]
            assert np.allclose(
                one.values,
                _folded(convert(per[:, 1:] * harmonics * VOLUME_FAC)[:, ::-1]),
            )


def test_ky_is_the_plotted_y() -> None:
    r"""``ky``'s second factor is `$y^+$`, so it carries a `$Re_\tau$`.

    The `$k$` half is unit-invariant (`$k^+\Phi^+ = k\Phi$`) and the
    `$y$` half is not, which is the whole difference between a
    wall-unit ``ky`` map and an outer-unit one -- the module
    docstring's "Premultiplication".  Pinned in both directions so
    neither half can drift.
    """
    # Legacy layout: ``e_x0`` is the absolute panel the second half
    # of this case needs, and only a legacy stream carries one.
    meta = _meta("twin_yspectra", suffixes=LEGACY)
    series = _series(
        "twin_yspectra", [_member(meta, _records(meta, "twin_yspectra", 2))]
    )
    wall, outer = tsm.Units(RE, RE_TAU), tsm.Units(RE, RE_TAU, wall=False)

    def draw(name, units, premultiply):
        return tsm.make_map(
            series,
            name,
            0,
            options=tsm.MapOptions(units, premultiply=premultiply),
        )

    k, ky = draw("e_x0", wall, "k"), draw("e_x0", wall, "ky")
    assert np.allclose(ky.values, k.values * ky.y[:, None])
    assert np.allclose(ky.y, k.y * 1.0)  # the same ordinate, y^+

    # A normalised panel takes no unit conversion, so its wall/outer
    # ratio *is* the premultiplier's unit dependence.
    assert tsm.normalises(series, "e_z")
    for premultiply, factor in (("k", 1.0), ("ky", RE_TAU)):
        got = draw("e_z", wall, premultiply).values
        assert np.allclose(
            got, factor * draw("e_z", outer, premultiply).values
        )
    # An absolute panel carries the energy conversion on top of it.
    for premultiply, factor in (("k", 1.0), ("ky", RE_TAU)):
        got = draw("e_x0", wall, premultiply).values
        want = factor / U_TAU**2 * draw("e_x0", outer, premultiply).values
        assert np.allclose(got, want)


def test_reference_scale() -> None:
    r"""`$E^{\mathrm{ref}}$` and what it does and does not normalise."""
    meta = _meta("twin_yspectra", suffixes=LEGACY)
    rec = _records(meta, "twin_yspectra", 4)
    series = _series("twin_yspectra", [_member(meta, rec)])
    w = np.asarray(meta["y_weights"])
    want = np.mean(
        [
            np.einsum("j,cjk->c", w, rec["r_x"][i])
            - np.einsum("j,cj->c", w, rec["r_x0"][i][:, :, 0])
            for i in range(rec.size)
        ],
        axis=0,
    )
    assert np.allclose(series.reference_scale(), want)
    assert np.all(want > 0.0)

    # The complete marginals normalise; the k_x = 0 plane does not, and
    # a budget stream names no prefix at all.
    assert tsm.normalises(series, "e_x") and tsm.normalises(series, "r_z")
    assert not tsm.normalises(series, "e_x0")
    # The summed panel is one ratio of sums, not a sum of three
    # ratios.  Against the series' own scale, which the line above has
    # already matched to *want*: this is which number a panel picks,
    # not how it was accumulated.
    scale = series.reference_scale()
    assert tsm.reference_norm(series, "e_x", None) == float(scale.sum())
    assert tsm.reference_norm(series, "e_x", 1) == float(scale[1])
    assert tsm.reference_norm(series, "e_x0", None) is None

    # A panel really is the absolute one over that constant -- which
    # is what puts a difference map and its reference map on one
    # scale, and a saturated pair's e_* at twice its r_*.
    units = tsm.Units(RE, RE_TAU)
    options = tsm.MapOptions(units)
    harmonics = np.arange(1, NKZ, dtype=float)
    for name, component, divisor in (
        ("e_x", None, want.sum()),
        ("r_x", 2, want[2]),
    ):
        stored = series.field(name)[0]
        stored = stored.sum(axis=0) if component is None else stored[component]
        absolute = (stored[:, 1:] * harmonics * VOLUME_FAC)[:, ::-1]
        got = tsm.make_map(
            series, name, 0, options=options, component=component
        )
        assert np.allclose(got.values, _folded(absolute) / divisor)
        # The title reports E_ref in the plotted units, on its own line.
        assert "E^{\\mathrm{ref}}" in got.title
        assert got.title.count("\n") == 1

    # Both marginals must report the same total; one that does not is a
    # convention slip, not noise.
    doctored = rec.copy()
    doctored["r_z"] *= 1.5
    bad = _series("twin_yspectra", [_member(meta, doctored)])
    _raises(bad.reference_scale, "marginals disagree")

    # A stream without its reference half offers no E_ref at all.
    lean_meta = _meta("twin_yspectra", includes_ref=False)
    lean = _series(
        "twin_yspectra",
        [_member(lean_meta, _records(lean_meta, "twin_yspectra", 2))],
    )
    assert not tsm.normalises(lean, "e_x")
    _raises(lean.reference_scale, "no reference spectra")


def test_distinct_instants() -> None:
    """One reference instant is counted once, on a tolerance."""
    meta = _meta("twin_yspectra")
    first = _member(meta, _records(meta, "twin_yspectra", 3), path="a")
    second_rec = _records(meta, "twin_yspectra", 3, t0=101.0, seed=1)
    # 101 reached by different arithmetic: the same instant, off by a
    # few bits -- and a rounded key would split the pair.
    second_rec["t"] = np.array([101.0 + 2e-7, 102.0, 103.0])
    second = _member(meta, second_rec, path="b")
    series = _series("twin_yspectra", [first, second])

    picks, n_instants, n_samples = series._distinct_instants()
    assert (n_samples, n_instants) == (6, 4)
    assert picks[0].tolist() == [0, 1, 2]  # whichever sorts first owns it
    assert picks[1].tolist() == [2]
    assert sum(p.size for p in picks) == n_instants

    report = series.reference_report()
    assert "4 distinct instants of 6 samples" in report[0]
    assert "2 parent snapshot(s)" not in report[0]  # both are "p0"

    strided = _series("twin_yspectra", [first, second], ref_stride=2)
    assert "stride 2" in strided.reference_report()[0]


def test_colour_scale_is_a_legend_for_the_box() -> None:
    """A peak below the ordinate's floor sets no level, in any mode."""
    ny = 129
    meta = _meta("twin_ybudget", ny=ny, terms=["eps"])
    rec = np.zeros(1, dtype=tsm._record_dtype(meta, "twin_ybudget"))
    for suffix in tsm.stored_suffixes(meta):
        rec[f"eps_{suffix}"][:] = 1e-6
    rec["eps_x"][:, 1, :] = 1.0  # y+ ~ 0.05, four rows under the floor
    rec["eps_x"][:, -2, :] = 1.0
    series = _series("twin_ybudget", [_member(meta, rec)])
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    ylim = tsm.y_limits(series, options)
    assert np.isclose(ylim[0], tsm.Y_FLOOR_PLUS)

    panel = ("eps_x", None)
    scales, notes = tsm.scan_panels(series, [panel], options, ylim=ylim)
    visible = scales[panel].hi
    drawn = tsm.make_map(
        series, "eps_x", 0, options=options, non_negative=True
    )
    hidden = float(drawn.drawn()[1].max())
    assert hidden > 1e5 * visible  # the peak the floor hides
    assert not notes  # a stored field, drawn raw, declares no sign

    def levels(**kwargs) -> np.ndarray:
        figure = plt.figure()
        filled = tsm.draw_map(
            figure.add_subplot(),
            drawn,
            units=options.units,
            ylim=ylim,
            **kwargs,
        )
        out = np.asarray(filled.levels)
        plt.close(figure)
        return out

    frozen = levels(data_range=(scales[panel].lo, visible))
    per_frame = levels(data_range=None)
    want = tsm.contour_levels(drawn.drawn(ylim)[1], 10, non_negative=True)
    assert np.array_equal(frozen, per_frame)
    assert np.array_equal(per_frame, want)
    assert per_frame[0] <= visible
    # ... and that is not what the unrestricted rows would have given.
    assert not np.array_equal(
        want, tsm.contour_levels(drawn.drawn()[1], 10, non_negative=True)
    )

    # --quantile clips the peak, and reads it off the same rows.
    clipped = levels(data_range=(scales[panel].lo, visible), quantile=0.99)
    assert clipped[-1] <= 2.0 * visible

    # Only a clipped scale can put something above the top level, so
    # only a clipped scale extends -- the fills agree either way.
    figure = plt.figure()
    plain = tsm.draw_map(
        figure.add_subplot(), drawn, units=options.units, ylim=ylim
    )
    quantiled = tsm.draw_map(
        figure.add_subplot(),
        drawn,
        units=options.units,
        ylim=ylim,
        quantile=0.5,
    )
    assert (plain.extend, quantiled.extend) == ("neither", "max")
    plt.close(figure)


def test_sign_family_is_declared() -> None:
    """One-signed fields are declared; the data only checks it."""
    meta = _meta("twin_ybudget")
    rec = np.zeros(2, dtype=tsm._record_dtype(meta, "twin_ybudget"))
    rec["t"] = [0.0, 1.0]
    for suffix in tsm.stored_suffixes(meta):
        rec[f"P_U_{suffix}"][:] = 1.0  # signed, but never negative here
        rec[f"eps_{suffix}"][:] = 1.0
    rec["eps_x"][:, [10, NY - 1 - 10], 3] = -1e-14  # a round-off dip
    series = _series("twin_ybudget", [_member(meta, rec)])
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    ylim = tsm.y_limits(series, options)
    panels = [("prod_x", None), ("diss_x", None), ("sum_x", None)]

    scales, notes = tsm.scan_panels(series, panels, options, ylim=ylim)
    assert scales[("prod_x", None)].non_negative is False
    assert scales[("sum_x", None)].non_negative is False
    # Declared non-positive, and drawn on the signed scale all the same.
    assert scales[("diss_x", None)].non_negative is False
    assert len(notes) == 1 and "round-off" in notes[0]
    assert "diss_x" in notes[0] and "non-positive" in notes[0]

    inferred, _ = tsm.scan_panels(
        series, panels, options, declared=False, ylim=ylim
    )
    assert inferred[("prod_x", None)].non_negative is True
    assert inferred[("diss_x", None)].non_negative is False

    # Zero sits on the colour map's neutral centre however lopsided the
    # trim, which is what makes a one-sided signed term readable.
    signed = tsm.contour_levels(
        np.array([[-0.3, 1.0]]), 10, non_negative=False
    )
    assert 0.0 not in signed
    shaded, _ = tsm.band_colors(signed, "RdBu_r", non_negative=False)
    middles = 0.5 * (signed[:-1] + signed[1:])
    zero_band = int(np.argmin(np.abs(middles)))
    assert np.allclose(
        shaded(zero_band), plt.get_cmap("RdBu_r")(0.5), atol=1e-12
    )


def test_fold() -> None:
    """`$R_y$`, and the grid preconditions each mode actually needs."""
    y = np.linspace(-1.0, 1.0, 5)  # the pairing, not the CGL grid
    assert np.allclose(tsm._half_grid(y, "mean"), [0.0, 0.5, 1.0])
    values = np.arange(5.0)[:, None] * np.ones((1, 3))
    mean, distance = tsm._select_half(values, y, "mean")
    assert np.allclose(distance, [0.0, 0.5, 1.0])
    assert np.allclose(mean[:, 0], [2.0, 2.0, 2.0])  # j with n-1-j
    assert mean[-1, 0] == values[2, 0]  # the mid-plane, counted once
    assert np.allclose(
        tsm._select_half(values, y, "lower")[0][:, 0], [0, 1, 2]
    )
    assert np.allclose(
        tsm._select_half(values, y, "upper")[0][:, 0], [4, 3, 2]
    )

    # An even n_y has no mid-plane row and pairs every row.
    even = np.linspace(-1.0, 1.0, 4)
    assert np.allclose(tsm._half_grid(even, "mean"), 1.0 + even[:2])
    assert tsm._select_half(np.zeros((4, 2)), even, "mean")[0].shape == (2, 2)

    # ``lower`` needs no symmetry: 1 + y is its rows' wall distance
    # whatever the far half does.  ``mean`` and ``upper`` do need it.
    skew = np.array([-1.0, -0.2, 0.5, 1.0])
    assert np.allclose(tsm._half_grid(skew, "lower"), [0.0, 0.8])
    for mode in ("mean", "upper"):
        _raises(lambda m=mode: tsm._half_grid(skew, m), "not symmetric")
    _raises(lambda: tsm._half_grid(y[::-1], "mean"), "not ascending")
    _raises(lambda: tsm._half_grid(y, "both"), "mean/lower/upper")


def test_open_series() -> None:
    """The shared grid, its selection, and what a member set may be."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        meta = _meta("twin_yspectra")
        first = _records(meta, "twin_yspectra", 8, t0=100.0)
        second = _records(meta, "twin_yspectra", 6, t0=150.0, seed=1)
        _write_member(root / "a", "twin_yspectra", meta, first, parent_t=100.0)
        _write_member(
            root / "b", "twin_yspectra", meta, second, parent_t=150.0
        )
        pair = [root / "a", root / "b"]

        series = tsm.open_series(pair, "twin_yspectra")
        assert series.n_members == 2
        assert np.allclose(series.t_rel, np.arange(6.0))  # the intersection
        # A shorter member costs the shared grid samples, and the
        # report says so member by member -- the catch-all that keeps
        # a collapsed grid from being silent whatever collapsed it.
        report = series.grid_report()
        assert report is not None and "6 of 8" in report, report
        assert "b 6" in report, report
        assert tsm.open_series(
            [root / "a"], "twin_yspectra"
        ).grid_report() is (None)
        assert np.allclose(
            series.field("e_xz00")[0],
            0.5 * (first["e_xz00"][0] + second["e_xz00"][0]),
        )
        assert np.allclose(
            tsm.open_series(pair, "twin_yspectra", stride=2).t_rel,
            [0.0, 2.0, 4.0],
        )
        assert np.allclose(
            tsm.open_series(pair, "twin_yspectra", first=1, last=3).t_rel,
            [1.0, 2.0, 3.0],
        )
        assert tsm.open_series(pair, "twin_yspectra").index.tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]

        # A member that is not the same flow is refused: the figure
        # reads the grid and the axes off the first member alone.
        for key, value, named in (
            ("lz", 2.0 * LZ, "lz"),
            ("y", list(_grid(NY) * 0.5), "y"),
            ("volume_fac", 1.0, "volume_fac"),
            ("kz_harmonics", list(range(1, NKZ + 1)), "kz_harmonics"),
        ):
            odd = _write_member(
                root / f"odd_{key}",
                "twin_yspectra",
                _meta("twin_yspectra", **{key: value}),
                second,
                parent_t=150.0,
            )
            _raises(
                lambda d=odd: tsm.open_series(
                    [root / "a", d], "twin_yspectra"
                ),
                named,
            )

        # A different seed / e0 is exactly what an ensemble varies.
        varied = _meta("twin_yspectra")
        varied["twin"] = {"seed": 9, "e0": 1e-3, "smoothness": 2.0}
        _write_member(
            root / "c", "twin_yspectra", varied, second, parent_t=150.0
        )
        assert (
            tsm.open_series(
                [root / "a", root / "c"], "twin_yspectra"
            ).n_members
            == 2
        )

        # Samples the tolerance cannot separate would chain into one
        # instant, and an unsorted stream would pair the wrong record.
        tight = first.copy()
        tight["t"] = 100.0 + 1e-9 * np.arange(tight.size)
        _write_member(root / "tight", "twin_yspectra", meta, tight)
        _raises(
            lambda: tsm.open_series([root / "tight"], "twin_yspectra"),
            "same instant",
        )
        backwards = first.copy()
        backwards["t"] = 100.0 + np.array([0.0, 2.0, 1.0, 3, 4, 5, 6, 7])
        _write_member(root / "back", "twin_yspectra", meta, backwards)
        _raises(
            lambda: tsm.open_series([root / "back"], "twin_yspectra"),
            "not sorted ascending",
        )

        # A resume seam repeats a row; its first copy is kept.
        seam = np.concatenate([first, first[-1:]])
        _write_member(root / "seam", "twin_yspectra", meta, seam)
        assert (
            tsm.open_series([root / "seam"], "twin_yspectra").t_rel.size == 8
        )

        # A member displaced by half a cadence -- what a parent
        # snapshot off the cadence grid used to produce -- is refused
        # with the offset, not silently reduced to whatever frames
        # coincide.  ``--align-atol`` accepts it, up to half a cadence
        # and no further, and says how far apart it then pairs.
        _write_member(
            root / "off", "twin_yspectra", meta, second, parent_t=150.5
        )
        off_pair = [root / "a", root / "off"]
        _raises(
            lambda: tsm.open_series(off_pair, "twin_yspectra"),
            "out of phase",
        )
        loose = tsm.open_series(off_pair, "twin_yspectra", align_atol=0.5)
        assert loose.t_rel.size == 6, loose.t_rel
        assert loose.alignment_spread() == 0.5
        _raises(
            lambda: tsm.open_series(off_pair, "twin_yspectra", align_atol=0.6),
            "half the",
        )

        # In phase and never overlapping is a different failure: a
        # member covering another stretch of the relative clock is not
        # displaced, so it must reach the intersection to be refused
        # there.  (Measured modulo the cadence for exactly this.)
        _write_member(
            root / "late", "twin_yspectra", meta, second, parent_t=-850.0
        )
        _raises(
            lambda: tsm.open_series(
                [root / "a", root / "late"], "twin_yspectra"
            ),
            "share no relative sample time",
        )
        _raises(
            lambda: tsm.open_series(pair, "twin_yspectra", first=7),
            "select none of the 6 sample time(s)",
        )
        _raises(
            lambda: tsm.open_series([root / "a"], "twin_spectra"),
            "unknown stream",
        )

        # Members of different layouts are not one set: their records
        # do not even hold the same fields.  A legacy member has no
        # ``suffixes`` key, so the comparison is against the triple it
        # stands for, not against a missing value.
        legacy_meta = _meta("twin_yspectra", suffixes=LEGACY)
        _write_member(
            root / "legacy",
            "twin_yspectra",
            legacy_meta,
            _records(legacy_meta, "twin_yspectra", 8, t0=100.0),
            parent_t=100.0,
        )
        _raises(
            lambda: tsm.open_series(
                [root / "a", root / "legacy"], "twin_yspectra"
            ),
            "suffixes",
        )


def test_layouts_and_default_series() -> None:
    r"""What each layout offers, what is drawn, and `$E^{ref}$`.

    The three cases are the three streams that exist on disk: a
    pre-``xz00`` member, the current default, and the current default
    under ``twin.x0_planes``.  All three must open; the `$(0, 0)$`
    mode `$E^{\mathrm{ref}}$` subtracts is the same number in all
    three, whichever field it is read from; and what is *drawn* is the
    two difference-spectra marginals, their shape maps and the
    histories of both, and the budget's where there is one, unless a
    switch asks for more -- each adding its own family and nothing
    else, ``--no-history`` taking the histories away.
    """
    scales = {}
    registries = {}
    opened = {}
    for label, suffixes in (
        ("legacy", LEGACY),
        ("default", DEFAULT),
        ("x0_planes", WITH_X0),
    ):
        meta = _meta("twin_yspectra", suffixes=suffixes)
        rec = _records(meta, "twin_yspectra", 3, seed=7)
        series = _series("twin_yspectra", [_member(meta, rec)])
        assert series.suffixes == suffixes, label
        assert tsm.mean_mode_name(series.meta, "r") == (
            "r_xz00" if "xz00" in suffixes else "r_x0"
        )
        scales[label] = series.reference_scale()
        registries[label] = tsm.available_series(series, None)
        opened[label] = series

    # One plane, one E_ref -- the stored route to its (0, 0) mode is
    # a storage detail and nothing more.  The stored *values* are
    # identical; the contraction is not bit-identical between them
    # because ``einsum`` accumulates a strided ``r_x0[..., 0]`` view
    # in a different order from a contiguous ``r_xz00``.
    for label in ("default", "x0_planes"):
        assert np.allclose(scales[label], scales["legacy"], rtol=1e-14), label

    # Only a stream that carries the plane offers its tags, and they
    # are not in the default set even then.
    assert "spectra_e_x0" in registries["legacy"]
    assert "spectra_e_x0" in registries["x0_planes"]
    assert "spectra_e_x0" not in registries["default"]
    # ``xz00`` is never a tag: it has no abscissa.
    assert not any("xz00" in tag for tag in registries["x0_planes"])
    # The shape maps redraw the two true marginals, never the slice.
    assert f"spectra_{tsm.SHAPE}_x0" not in registries["x0_planes"]

    maps = [f"spectra_{p}_{m}" for p in ("e", tsm.SHAPE) for m in "xz"]
    spectra = maps + [f"history_{p}" for p in ("e", tsm.SHAPE)]
    reference = [f"spectra_r_{m}" for m in ("x", "z")]
    decorr = [f"spectra_decorr_{m}" for m in ("x", "z")]
    decorr_k = [f"spectra_decorr_k_{m}" for m in ("x", "z")]
    front = ["front_x", "front_z"]
    growth = [f"growth_{k}" for k in ("global", "x", "z", "y", "ssp")]
    everything = dict(
        x0=True,
        budget=True,
        reference=True,
        decorr=True,
        decorr_k=True,
        spacetime=True,
        front=True,
        growth=True,
        moment_budget=True,
    )
    for label, registry in registries.items():
        # The bare default of a spectra stream is the two marginals of
        # the difference spectra and of their shape maps, with the
        # histories of both, and nothing of the reference's.
        assert sorted(tsm.default_series(registry)) == sorted(spectra), label
        assert sorted(tsm.default_series(registry, history=False)) == (
            sorted(maps)
        ), label
        # Each switch adds its own family and nothing else.  The
        # reference's and R^k's spacetime maps are the composites --
        # each needs both of its switches, so neither alone brings it.
        for switch, gained in (
            ({"reference": True}, reference),
            ({"decorr": True}, decorr),
            ({"decorr_k": True}, decorr_k),
            ({"spacetime": True}, ["spacetime_e"]),
            (
                {"reference": True, "spacetime": True},
                reference + ["spacetime_e", "spacetime_r"],
            ),
            (
                {"decorr_k": True, "spacetime": True},
                decorr_k + ["spacetime_e", "spacetime_decorr_k"],
            ),
            ({"front": True}, front),
            ({"growth": True}, growth),
        ):
            got = tsm.default_series(registry, **switch)
            assert set(got) - set(spectra) == set(gained), (label, switch)
        assert set(tsm.default_series(registry, **everything)) == set(
            registry
        ), label

    # The budget is drawn unasked, its spacetime map under --spacetime,
    # and --no-budget drops both.
    budget_meta = _meta("twin_ybudget")
    budget = _series(
        "twin_ybudget",
        [_member(budget_meta, _records(budget_meta, "twin_ybudget", 2))],
    )
    both = tsm.available_series(opened["default"], budget)
    assert set(tsm.default_series(both)) == {
        *spectra,
        "budget_x",
        "budget_z",
        "history_budget",
    }
    assert set(tsm.default_series(both, budget=False)) == set(spectra)
    assert set(tsm.default_series(both, spacetime=True)) == {
        *spectra,
        "budget_x",
        "budget_z",
        "history_budget",
        "spacetime_e",
        "spacetime_budget",
    }
    assert "spacetime_budget" not in tsm.default_series(
        both, spacetime=True, budget=False
    )
    # The moment budget needs both streams and its own switch, and
    # --no-budget drops it with the rest of the budget.
    moments = {"moments_y", "moments_x", "moments_z"}
    assert moments <= set(both)
    assert not moments & set(tsm.default_series(both))
    assert moments <= set(tsm.default_series(both, moment_budget=True))
    assert not moments & set(
        tsm.default_series(both, moment_budget=True, budget=False)
    )
    assert not moments & set(tsm.available_series(opened["default"], None))
    print("layouts, their tags, and one E_ref across all three: OK")


def test_reference_layouts() -> None:
    r"""The reference spectra read alike from both layouts.

    The same records stored the pre-split way (``r_*`` inside
    ``twin_yspectra.bin``, ``includes_ref``) and the split way (their
    own ``twin_yspectra_ref.bin``) open through
    :func:`~twin_spectral_maps.open_series` -- a set of either, and a
    set mixing the two -- to the same `$E^{\mathrm{ref}}$`, reference
    maps, decorrelations and reference spacetime map.  A split member
    whose reference was sampled every other difference sample
    (``twin.it_yspectra_ref``) normalises over its own samples and
    draws every difference map and decorrelation, but a reference map
    only at the frames the two cadences share.
    """
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    split_meta = meta | {"includes_ref": False}
    ref_meta = _ref_meta(split_meta)
    lean_dtype = tsm._record_dtype(split_meta, "twin_yspectra")

    def write(directory: Path, rec: np.ndarray, layout: str, every=1):
        if layout == "legacy":
            return _write_member(directory, "twin_yspectra", meta, rec)
        lean = np.zeros(rec.size, dtype=lean_dtype)
        for name in lean.dtype.names:
            lean[name] = rec[name]
        _write_member(directory, "twin_yspectra", split_meta, lean)
        ref = _ref_records(ref_meta, rec)[::every]
        (directory / "twin_yspectra_ref.json").write_text(json.dumps(ref_meta))
        (directory / "twin_yspectra_ref.bin").write_bytes(ref.tobytes())
        return directory

    # Two members on different absolute clocks, so E_ref averages
    # eight distinct instants rather than deduplicating four.
    recs = [
        _records(meta, "twin_yspectra", 4, t0=t0, seed=seed)
        for t0, seed in ((100.0, 21), (300.0, 22))
    ]
    readings = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, layouts in (
            ("legacy", ("legacy", "legacy")),
            ("split", ("split", "split")),
            ("mixed", ("legacy", "split")),
        ):
            members = [
                write(root / label / f"m{k}", rec, layout)
                for k, (rec, layout) in enumerate(
                    zip(recs, layouts, strict=True)
                )
            ]
            series = tsm.open_series(members, "twin_yspectra")
            assert series.prefixes == ("e", "r"), label
            readings[label] = [
                series.reference_scale(),
                series.field("r_x"),
                series.field("r_z"),
                tsm.make_map(series, "r_x", 1, options=options).values,
                tsm.make_map(series, "decorr_x", 1, options=options).values,
                tsm.make_map(series, "decorr_k_x", 2, options=options).values,
                tsm.make_spacetime(series, "r", options=options).values,
            ]
        for label in ("split", "mixed"):
            for got, want in zip(
                readings[label], readings["legacy"], strict=True
            ):
                assert np.array_equal(got, want, equal_nan=True), label
        both = (recs[0]["r_x"] + recs[1]["r_x"]) / 2.0
        assert np.array_equal(readings["legacy"][1], both)

        # The reference at half the difference cadence.
        rec = recs[0]
        sparse = write(root / "sparse", rec, "split", every=2)
        alone = tsm.open_series([sparse], "twin_yspectra")
        _raises(lambda: alone.field("r_x"), "twin.it_yspectra_ref")
        _raises(
            lambda: tsm.make_map(alone, "r_x", 1, options=options),
            "twin.it_yspectra_ref",
        )
        # What divides by the time-averaged reference needs no sample
        # at the frame itself.
        for name in ("e_x", "decorr_x", "decorr_k_x"):
            drawn = tsm.make_map(alone, name, 1, options=options).values
            assert np.isfinite(drawn).any(), name
        # E_ref averages the two samples the reference has, which is
        # what a legacy member holding only those two would give.
        thinned = _series(
            "twin_yspectra", [_member(meta, rec[::2], parent_t=100.0)]
        )
        assert np.allclose(
            alone.reference_scale(), thinned.reference_scale(), rtol=1e-14
        )
        shared = tsm.open_series([sparse], "twin_yspectra", stride=2)
        assert np.array_equal(shared.field("r_x"), rec["r_x"][::2])
    print("both reference layouts, and a mix of them, read alike: OK")


def test_reference_scale_is_a_quadrature() -> None:
    r"""`$E^{\mathrm{ref}}$` averages over `$y$` with the *weights*.

    The stored entries are densities already divided by
    ``volume_fac`` and the weights sum to it, so the contraction is a
    wall-normal **average**.  Every other fixture here uses a uniform
    stand-in rule, which cannot tell that contraction from a plain
    mean scaled by ``volume_fac``; a genuine non-uniform rule can.
    """
    rng = np.random.default_rng(19)
    w = rng.random(NY) + 0.5
    w *= VOLUME_FAC / w.sum()
    meta = _meta("twin_yspectra", y_weights=[float(v) for v in w])
    rec = _records(meta, "twin_yspectra", 3, seed=5)
    series = _series("twin_yspectra", [_member(meta, rec)])

    want = np.mean(
        [
            np.einsum("j,cjk->c", w, rec["r_x"][i])
            - np.einsum("j,cj->c", w, rec["r_xz00"][i])
            for i in range(rec.size)
        ],
        axis=0,
    )
    assert np.allclose(series.reference_scale(), want)

    # A plain mean over y, scaled the same way, is a different number.
    flat = VOLUME_FAC / NY
    naive = np.mean(
        [
            flat * (rec["r_x"][i].sum(axis=(1, 2)) - rec["r_xz00"][i].sum(1))
            for i in range(rec.size)
        ],
        axis=0,
    )
    assert not np.allclose(naive, want)
    print("E_ref is the quadrature contraction, not a plain mean: OK")


def test_decorrelation() -> None:
    """R and R^k: what each divides by, and what moves them."""
    units = tsm.Units(RE, RE_TAU)
    options = tsm.MapOptions(units)
    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 5, seed=7)
    series = _series("twin_yspectra", [_member(meta, rec)])

    # One member, so every record is its own reference instant and
    # the average over them is a plain mean.
    mean_x = rec["r_x"].mean(axis=0)
    mean_00 = rec["r_xz00"].mean(axis=0)
    spectrum = mean_x.copy()
    spectrum[..., 0] -= mean_00  # the (0, 0) mode, and only it
    profile = mean_x.sum(axis=-1) - mean_00
    assert np.allclose(series.reference_spectrum("x"), spectrum)
    assert np.allclose(series.reference_profile(), profile)
    # The three divisors are one array read at three resolutions ...
    assert np.allclose(spectrum.sum(axis=-1), profile)
    assert np.allclose(
        series.reference_scale(),
        np.einsum("j,cj->c", series.y_weights, profile),
    )
    # ... and the k_x marginal is an independent reading of it.
    assert np.allclose(rec["r_z"].mean(axis=0).sum(axis=-1) - mean_00, profile)

    harmonics = np.arange(1, NKZ, dtype=float)
    for component in (0, 2, None):
        e = rec["e_x"][1]
        e = e.sum(axis=0) if component is None else e[component]
        # The divisor is symmetrised for the fold (next case), so the
        # hand computation symmetrises too.  The summed panel takes
        # the summed divisor: one ratio of sums.
        flat = profile.sum(0) if component is None else profile[component]
        resolved = (
            spectrum.sum(0) if component is None else spectrum[component]
        )
        want = (e / (2.0 * _symmetric(flat)[:, None]))[:, 1:] * harmonics
        drawn = tsm.make_map(
            series, "decorr_k_x", 1, options=options, component=component
        )
        assert np.allclose(drawn.values, _folded(want[:, ::-1])), component

        with np.errstate(divide="ignore", invalid="ignore"):
            want = (e / (2.0 * _symmetric(resolved)))[:, 1:]
        drawn = tsm.make_map(
            series, "decorr_x", 1, options=options, component=component
        )
        assert np.allclose(drawn.values, _folded(want[:, ::-1])), component

    # volume_fac and the unit conversion cancel between a ratio's two
    # halves; the premultiplier survives R^k's k-independent divisor
    # and cancels against R's.
    for name, premultiplied in (("decorr_k_x", True), ("decorr_x", False)):
        base = tsm.make_map(series, name, 1, options=options).values
        for other in (
            tsm.MapOptions(units, volume_fac=False),
            tsm.MapOptions(tsm.Units(RE, RE_TAU, wall=False)),
        ):
            same = tsm.make_map(series, name, 1, options=other).values
            assert np.allclose(base, same, equal_nan=True), name
        plain = tsm.make_map(
            series, name, 1, options=tsm.MapOptions(units, premultiply="none")
        ).values
        moved = not np.allclose(base, plain, equal_nan=True)
        assert moved is premultiplied, name

    # A pair that has decorrelated completely reads exactly 1 -- R at
    # every plotted mode, R^k once its k-sum is taken -- and it does
    # so by each half counting the modes it is documented to count.
    saturated = _series("twin_yspectra", [_member(meta, _saturated(meta))])
    for component in (1, None):
        drawn = tsm.make_map(
            saturated, "decorr_x", 0, options=options, component=component
        )
        assert np.allclose(drawn.values, 1.0), component
        summed = tsm.make_spacetime(
            saturated, tsm.DECORR_K, options=options, component=component
        )
        assert np.allclose(summed.values, 1.0), component

    # An empty reference mode is nan, and nan reaches no colour scale.
    holed = _records(meta, "twin_yspectra", 2, seed=13)
    for suffix in tsm.stored_suffixes(meta):
        # Two whole wall-normal rows, and R_y partners: the divisor is
        # symmetrised, so one alone would be filled in by the other.
        holed[f"r_{suffix}"][:, :, [5, NY - 6]] = 0.0
    empty = _series("twin_yspectra", [_member(meta, holed)])
    for name in ("decorr_x", "decorr_k_x"):
        drawn = tsm.make_map(empty, name, 0, options=options)
        assert np.isnan(drawn.values).any(), name
        panel = (name, None)
        scales, _ = tsm.scan_panels(empty, [panel], options)
        assert np.isfinite(scales[panel].lo), name
        assert np.isfinite(scales[panel].hi), name


def test_divisor_is_symmetrised_before_the_fold() -> None:
    """A y-dependent divisor must not depend on the fold's order."""
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 3, seed=11)
    # A reference that is deliberately not R_y-symmetric.  The
    # component axis is 1, so the wall-normal one is 2 in every
    # stored field, marginal and (0, 0) mode alike.
    ramp = (1.0 + np.arange(NY, dtype=float)).reshape(1, 1, NY)
    for suffix in tsm.stored_suffixes(meta):
        field = rec[f"r_{suffix}"]
        field *= ramp if field.ndim == 3 else ramp[..., None]
    series = _series("twin_yspectra", [_member(meta, rec)])

    profile = rec["r_x"].mean(0).sum(-1) - rec["r_xz00"].mean(0)
    numerator = rec["e_x"].sum(-1)[:, 0]
    drawn = tsm.make_spacetime(
        series, tsm.DECORR_K, options=options, component=0
    )
    # The ratio of the folded halves ...
    want = _folded(numerator.T).T / (2.0 * _folded(_symmetric(profile[0])))
    assert np.allclose(drawn.values, want)
    # ... which is not the mean of the two unsymmetrised ratios.
    naive = _folded((numerator / (2.0 * profile[0])).T).T
    assert not np.allclose(drawn.values, naive)

    # --half lower folds nothing and keeps each row's own divisor.
    lower = tsm.make_spacetime(
        series,
        tsm.DECORR_K,
        options=tsm.MapOptions(options.units, half="lower"),
        component=0,
    )
    n_half = (NY + 1) // 2
    assert np.allclose(
        lower.values, (numerator / (2.0 * profile[0]))[:, :n_half]
    )


def test_k_sum_counts_the_modes_each_half_counts() -> None:
    """Marginal-free, on every layout, and whose (0, 0) mode leaves."""
    for suffixes in (LEGACY, DEFAULT, WITH_X0):
        meta = _meta("twin_yspectra", suffixes=suffixes)
        rec = _records(meta, "twin_yspectra", 3, seed=17)
        series = _series("twin_yspectra", [_member(meta, rec)])
        name = tsm.mean_mode_name(meta, "r")
        mean_mode = tsm.mean_mode_profile(rec[name], name)

        # The perturbation keeps every mode it has ...
        assert np.allclose(tsm.k_summed(series, "e"), rec["e_x"].sum(-1))
        assert np.allclose(rec["e_z"].sum(-1), rec["e_x"].sum(-1))
        # ... the reference loses its (0, 0) one, off either marginal.
        want = rec["r_x"].sum(-1) - mean_mode
        assert np.allclose(tsm.k_summed(series, "r"), want)
        assert np.allclose(rec["r_z"].sum(-1) - mean_mode, want)
        tsm.check_k_sum(series, "e")  # the guard, on a real layout
        if "x0" in suffixes:
            # The slice's m = 0 *is* the (0, 0) mode, so there the two
            # readings coincide.
            assert np.allclose(
                tsm.k_summed(series, "r", "x0"),
                rec["r_x0"][..., 1:].sum(-1),
            )

    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 2, seed=19)
    rec["e_z"] *= 1.01  # one marginal is no longer a complete sum
    _raises(
        lambda: tsm.check_k_sum(
            _series("twin_yspectra", [_member(meta, rec)]), "e"
        ),
        "not a complete sum",
    )


def test_k_sum_streams() -> None:
    """The streamed k-sum is the whole-field one, chunk edges included.

    Two members of five records read two at a time, so each member's
    reads cross two chunk edges and the ensemble mean is taken of
    reduced blocks rather than of whole fields; the budget's virtual
    ``sum`` is assembled from its additive terms on the way.  Nothing
    lands in the field cache, which is what reading it this way buys.
    """
    chunk = tsm._REF_CHUNK
    tsm._REF_CHUNK = 2
    try:
        meta = _meta("twin_yspectra")
        recs = [_records(meta, "twin_yspectra", 5, seed=s) for s in (41, 43)]
        series = _series(
            "twin_yspectra",
            [_member(meta, r, path=f"m{i}") for i, r in enumerate(recs)],
        )
        name = tsm.mean_mode_name(meta, "r")
        for base in ("e", "r"):
            want = np.mean(
                [
                    r[f"{base}_x"].sum(-1)
                    - (
                        tsm.mean_mode_profile(r[name], name)
                        if base == "r"
                        else 0
                    )
                    for r in recs
                ],
                axis=0,
            )
            assert np.allclose(tsm.k_summed(series, base), want), base

        bmeta = _meta("twin_ybudget")
        brecs = [_records(bmeta, "twin_ybudget", 5, seed=s) for s in (47, 53)]
        budget = _series(
            "twin_ybudget",
            [_member(bmeta, r, path=f"b{i}") for i, r in enumerate(brecs)],
        )
        # d_t E_Delta is the stored sum however the balance regroups
        # it: every stored term but ``eps``, which ``diss`` and
        # ``tr_visc`` take away and add back.
        stored_sum = [t for t in TERMS if t != "eps"]
        want = np.mean(
            [sum(r[f"{t}_x"] for t in stored_sum).sum(-1) for r in brecs],
            axis=0,
        )
        assert np.allclose(tsm.k_summed(budget, "sum"), want)
        assert np.allclose(budget.field("sum_x").sum(-1), want)
        assert not series._cache
    finally:
        tsm._REF_CHUNK = chunk


def test_spacetime() -> None:
    """The (y, t) maps, their colour scales and their .npz."""
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 4, seed=23)
    series = _series("twin_yspectra", [_member(meta, rec)])

    # The k-sum of the panel, in the plotted units, folded, and with
    # no premultiplier whatever --premultiply says.
    scale = series.reference_scale()
    for component in (2, None):
        if component is None:
            total = rec["e_x"].sum(-1).sum(axis=1)
            over = scale.sum()
        else:
            total = rec["e_x"].sum(-1)[:, component]
            over = scale[component]
        want = _folded((total * VOLUME_FAC / over).T).T
        for premultiply in ("k", "ky", "none"):
            drawn = tsm.make_spacetime(
                series,
                "e",
                options=tsm.MapOptions(options.units, premultiply=premultiply),
                component=component,
            )
            assert np.allclose(drawn.values, want), (component, premultiply)
        assert drawn.values.shape == (rec.size, (NY + 1) // 2)
        assert np.allclose(drawn.t, options.units.time(series.t_rel))

    # The k_x = 0 slice keeps its name and stays absolute, so it
    # carries the unit conversion the normalised panels cancel.
    x0_meta = _meta("twin_yspectra", suffixes=WITH_X0)
    x0_rec = _records(x0_meta, "twin_yspectra", 3, seed=37)
    x0_series = _series("twin_yspectra", [_member(x0_meta, x0_rec)])
    slice_map = tsm.make_spacetime(
        x0_series, "r", "x0", options=options, component=0
    )
    want = options.units.energy(
        _folded((x0_rec["r_x0"][:, 0, :, 1:].sum(-1) * VOLUME_FAC).T).T
    )
    assert np.allclose(slice_map.values, want)
    assert tsm.spacetime_norm(x0_series, "r", "x0", 0) is None
    assert "x0" in slice_map.title and "\n" not in slice_map.title

    # The colour range is a legend for the columns the box shows: a
    # peak below the wall-distance floor sets no level.
    ny = 129
    wide_meta = _meta("twin_yspectra", ny=ny)
    wide = _records(wide_meta, "twin_yspectra", 2, seed=29)
    # A whole wall-normal row of the plane, so that both marginals
    # stay complete sums of one field (case 10's guard runs here).
    for suffix in tsm.stored_suffixes(wide_meta):
        wide[f"e_{suffix}"][:, :, 1] *= 1e6  # y+ ~ 0.05, under the floor
        wide[f"e_{suffix}"][:, :, -2] *= 1e6
    wide_series = _series("twin_yspectra", [_member(wide_meta, wide)])
    maps = tsm.spacetime_maps(
        wide_series,
        tsm.SeriesSpec("twin_yspectra", "e", "", tsm.SPACETIME),
        options,
    )
    ylim = tsm.y_limits(wide_series, options)
    assert np.isclose(ylim[0], tsm.Y_FLOOR_PLUS)
    scales, floors, notes = tsm.spacetime_scales(maps, ylim)
    unrestricted = tsm.spacetime_scales(maps, None)[0]
    assert not notes  # sums of squares, no sign complaint
    assert unrestricted[0].hi > 1e3 * scales[0].hi
    assert maps[0].drawn(ylim)[0].size < maps[0].drawn()[0].size

    # The logarithmic floor: decades below the peak where the data
    # spans more than that, and the smallest positive value where it
    # spans fewer, so no empty range is invented under the data.
    shown = maps[0].drawn(ylim)[1]
    positive = shown[np.isfinite(shown) & (shown > 0.0)]
    span = float(np.log10(positive.max() / positive.min()))
    assert span > 0.0
    tight = tsm.log_floor(shown, 0.5 * span)
    assert np.isclose(tight, positive.max() / 10.0 ** (0.5 * span))
    assert np.isclose(tsm.log_floor(shown, 2.0 * span), positive.min())
    levels = tsm.log_levels(tight, float(positive.max()))
    assert np.isclose(levels[0], tight) and np.isclose(
        levels[-1], positive.max()
    )
    assert floors[0] > 0.0

    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        # A signed series draws no logarithmic figure; a non-negative
        # one draws both, and both carry the .npz.
        budget_meta = _meta("twin_ybudget")
        budget = _series(
            "twin_ybudget",
            [
                _member(
                    budget_meta,
                    _records(budget_meta, "twin_ybudget", 3, seed=31),
                )
            ],
        )
        written = tsm.render_spacetime(
            budget,
            tsm.SeriesSpec("twin_ybudget", "", "", tsm.SPACETIME),
            "spacetime_budget",
            out,
            options=options,
            style=tsm.PlotStyle(dpi=50),
            quiet=True,
        )
        assert [p.name for p in written] == [
            "spacetime_budget_lin.png",
            "spacetime_budget.npz",
        ]

        written = tsm.render_spacetime(
            series,
            tsm.SeriesSpec("twin_yspectra", "decorr_k", "", tsm.SPACETIME),
            "spacetime_decorr_k",
            out,
            options=options,
            style=tsm.PlotStyle(dpi=50),
            quiet=True,
        )
        assert [p.name for p in written] == [
            "spacetime_decorr_k_lin.png",
            "spacetime_decorr_k_log.png",
            "spacetime_decorr_k.npz",
        ]
        assert all(p.stat().st_size > 0 for p in written)

        # The .npz carries the drawn arrays and every factor behind
        # them -- enough to undo the normalisation without the figure.
        stored = np.load(written[-1])
        panels = tsm.spacetime_maps(
            series,
            tsm.SeriesSpec("twin_yspectra", "decorr_k", "", tsm.SPACETIME),
            options,
        )
        assert np.allclose(stored["values"], [p.values for p in panels])
        assert list(stored["panels"]) == ["u", "v", "w", "sum"]
        assert np.allclose(stored["t"], series.t_rel)
        assert np.allclose(stored["y"], tsm._half_grid(series.y, "mean"))
        assert np.allclose(stored["y_plotted"], panels[0].y)
        assert not bool(stored["premultiplied"])
        assert float(stored["re_tau"]) == RE_TAU
        assert float(stored["volume_fac"]) == VOLUME_FAC
        assert str(stored["half"]) == "mean"
        assert np.all(np.isnan(stored["e_ref"]))  # a ratio, not an E_ref
        # Multiplying the divisor back recovers the k-summed numerator.
        numerator = _folded(rec["e_x"].sum(-1)[:, 0].T).T
        assert np.allclose(
            stored["values"][0] * stored["divisor"][0], numerator
        )
        assert json.loads(str(stored["meta_json"]))["system"] == (
            "plane-poiseuille"
        )

        # A selection of one sample time has no time axis to draw, and
        # says so rather than dying inside matplotlib; its .npz is
        # still one valid row.
        single = _series("twin_yspectra", [_member(meta, rec[:1])])
        written = tsm.render_spacetime(
            single,
            tsm.SeriesSpec("twin_yspectra", "e", "", tsm.SPACETIME),
            "one_frame",
            out,
            options=options,
            style=tsm.PlotStyle(dpi=50),
            quiet=True,
        )
        assert [p.name for p in written] == ["one_frame.npz"]
        assert np.load(written[0])["values"].shape[1] == 1


def test_budget_is_the_balance() -> None:
    r"""The budget panels are the balance terms, however they are read.

    On both layouts the `$(0, 0)$` mode lives in -- ``xz00``, and a
    legacy member's ``x0`` -- every balance term is its combination of
    stored densities, as each of the three readers reads it: the
    ensemble mean (``YSeries.field``), one member's frame
    (``_frame_mean``) and the chunked `$k$`-sum (``k_summed``).
    ``input`` is that mode alone: a map's pressure panel leaves it
    out, a spacetime map's carries it (the stored ``Wp``), and the
    ``sum``, which counts it either way, is the stored one.
    """
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    spec = tsm.SeriesSpec("twin_ybudget", "", "", tsm.SPACETIME)
    for suffixes in (DEFAULT, LEGACY):
        meta = _meta("twin_ybudget", suffixes=suffixes)
        recs = [_records(meta, "twin_ybudget", 3, seed=s) for s in (61, 67)]
        series = _series(
            "twin_ybudget",
            [_member(meta, r, path=f"m{i}") for i, r in enumerate(recs)],
        )
        assert series.terms == tsm.BALANCE_TERMS

        def mean(name: str, recs=recs) -> np.ndarray:
            return np.mean([r[name] for r in recs], axis=0)

        mode_name = tsm.mean_mode_name(meta, "Wp")
        mode = np.mean(
            [tsm.mean_mode_profile(r[mode_name], mode_name) for r in recs],
            axis=0,
        )
        for suf in ("x", "z"):

            def field(term: str, suf=suf, series=series) -> np.ndarray:
                return series.field(f"{term}_{suf}")

            def stored(term: str, suf=suf) -> np.ndarray:
                return mean(f"{term}_{suf}")

            assert np.allclose(field("prod"), stored("P_U") + stored("P_r"))
            assert np.allclose(field("prod_mean"), stored("P_U"))
            assert np.allclose(field("prod_fluct"), stored("P_r"))
            assert np.allclose(field("diss"), -stored("eps"))
            assert np.allclose(field("tr_self"), stored("T_self"))
            assert np.allclose(field("tr_ref"), stored("T_ref"))
            assert np.allclose(field("tr_visc"), stored("V") + stored("eps"))
            assert not field("input")[..., 1:].any()
            assert np.allclose(field("input")[..., 0], mode)
            assert np.allclose(
                field("tr_press") + field("input"), stored("Wp")
            )
            assert np.allclose(field("press_input"), stored("Wp"))
            assert np.allclose(
                field("sum"), sum(stored(t) for t in TERMS if t != "eps")
            )
            one = tsm._frame_mean(series, f"input_{suf}", 1)
            assert np.allclose(one[..., 0], mode[1])
            assert f"input_{suf}" in series.additive(suf)
            assert f"tr_press_{suf}" in series.additive(suf)

        assert np.allclose(tsm.k_summed(series, "input"), mode)
        assert np.allclose(
            tsm.k_summed(series, "press_input"), mean("Wp_x").sum(-1)
        )
        for base in ("input", "tr_press", "press_input", "diss"):
            tsm.check_k_sum(series, base)

        panels = tsm.budget_panels(series, "x")
        assert panels == [(f"{t}_x", None) for t in (*tsm.MAP_PANELS, "sum")]
        assert ("tr_press_x", None) in panels
        assert not {("input_x", None), ("press_input_x", None)} & set(panels)
        spacetime = [base for base, _ in tsm.spacetime_panels(series, spec)]
        assert spacetime == [*tsm.SPACETIME_PANELS, "sum"]
        assert "press_input" in spacetime and "tr_press" not in spacetime

    # The 3 x 3 grid, row by row: the production and its two parts; the
    # dissipation and the viscous and pressure transports; the two
    # advective transports and the sum budget_panels appends.  A
    # spacetime map's pressure panel sits where the map's does.
    assert tsm.MAP_PANELS == (
        "prod",
        "prod_mean",
        "prod_fluct",
        "diss",
        "tr_visc",
        "tr_press",
        "tr_ref",
        "tr_self",
    )
    assert tsm.SPACETIME_PANELS == (
        "prod",
        "prod_mean",
        "prod_fluct",
        "diss",
        "tr_visc",
        "press_input",
        "tr_ref",
        "tr_self",
    )

    # The sum is the rate of the difference energy, in the notation the
    # transport terms' titles use.
    total = tsm.make_map(series, "sum_x", 0, options=options).title
    assert total.startswith(r"$k_{z}^+\,\partial_t E_\Delta"), total
    total = tsm.make_spacetime(series, "sum", options=options).title
    assert total.startswith(r"$\partial_t E_\Delta"), total

    # A contribution's minus sign leads its title, ahead of the
    # premultiplier; a gain carries none.
    loss = tsm.make_map(series, "diss_x", 0, options=options).title
    gain = tsm.make_map(series, "prod_x", 0, options=options).title
    summed = tsm.make_spacetime(series, "tr_self", options=options).title
    assert loss.startswith("$-k_") and gain.startswith("$k_")
    assert summed.startswith(r"$-\mathcal{T}")
    # A map's pressure panel is the pressure transport alone; a
    # spacetime map's names the input it carries as well.
    press = tsm.make_map(series, "tr_press_x", 0, options=options).title
    assert press.startswith(r"$-k_{z}^+\,\mathcal{T}_{\Delta p}")
    both = tsm.make_spacetime(series, "press_input", options=options).title
    assert both.startswith(r"$(\mathcal{I}_\Delta - \mathcal{T}")

    rotational = ["P_U", "P_r", "T_vort", "T_self", "V", "eps", "Wp"]
    meta = _meta("twin_ybudget", terms=[*rotational, "P_lift"])
    series = _series(
        "twin_ybudget",
        [_member(meta, _records(meta, "twin_ybudget", 2, seed=71))],
    )
    _raises(lambda: series.terms, "rotational")
    _raises(lambda: series.field("prod_x"), "rotational")


def test_budget_grid() -> None:
    """The budget's 3 x 3 grid, at the panel size of the spectra."""
    style = tsm.PlotStyle()
    xlim, ylim = (14.0, 1122.0), (1.0, RE_TAU)
    spectra = tsm.panel_geometry(4, xlim, ylim, style, y_log=True)
    budget = tsm.panel_geometry(
        9, xlim, ylim, style, y_log=True, ncols=tsm.BUDGET_NCOLS
    )
    assert (spectra.nrows, spectra.ncols) == (2, 2)
    assert (budget.nrows, budget.ncols) == (3, 3)
    # One panel size for both, fitted so that the spectra figure is
    # exactly --width wide; the budget's third column widens it.
    assert (budget.box_w, budget.box_h) == (spectra.box_w, spectra.box_h)
    assert np.isclose(spectra.fig_w, style.width)
    assert budget.fig_w > 1.4 * style.width

    # Row by row: panel 4 (the viscous transport) under panel 1 (the
    # mean-shear production), panel 8 (the sum) in the last corner.
    lefts, bottoms = zip(
        *(budget.axes_rect(p)[:2] for p in range(9)), strict=True
    )
    assert len(set(np.round(lefts, 12))) == 3
    assert len(set(np.round(bottoms, 12))) == 3
    assert np.isclose(lefts[4], lefts[1]) and bottoms[4] < bottoms[1]
    assert lefts[8] == max(lefts) and bottoms[8] == min(bottoms)

    # Both budget figures take that rule, and nothing else does.
    meta = _meta("twin_ybudget")
    series = _series(
        "twin_ybudget", [_member(meta, _records(meta, "twin_ybudget", 1))]
    )
    assert tsm.figure_columns(series) == tsm.BUDGET_NCOLS
    meta = _meta("twin_yspectra")
    series = _series(
        "twin_yspectra", [_member(meta, _records(meta, "twin_yspectra", 1))]
    )
    assert tsm.figure_columns(series) is None


def test_ramped_clim() -> None:
    """``--clim ramped``: the extremes of every frame so far."""
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 6, seed=83)
    # One shape, growing over three decades, dipping, then saturating.
    amplitude = np.array([1e-3, 1e-2, 5e-3, 1e-1, 1.0, 0.8])
    for suffix in tsm.stored_suffixes(meta):
        field = rec[f"e_{suffix}"]
        field[:] = field[0] * amplitude.reshape((-1,) + (1,) * field[0].ndim)
    series = _series("twin_yspectra", [_member(meta, rec)])
    ylim = tsm.y_limits(series, options)
    panel = ("e_x", 0)
    scale = tsm.scan_panels(series, [panel], options, ylim=ylim)[0][panel]
    own = [
        tsm.make_map(series, "e_x", f, options=options, component=0).drawn(
            ylim
        )[1]
        for f in range(amplitude.size)
    ]
    lows = [float(v.min()) for v in own]
    highs = [float(v.max()) for v in own]
    ramped = [scale.data_range("ramped", f) for f in range(amplitude.size)]
    for f, (lo, hi) in enumerate(ramped):
        # The running extremes, which hold the frame's own range.
        assert (lo, hi) == (min(lows[: f + 1]), max(highs[: f + 1])), f
        assert lo <= lows[f] and hi >= highs[f], f
    assert all(b[1] >= a[1] for a, b in zip(ramped, ramped[1:], strict=False))
    # While the field grows it is the frame's own scale; through the dip
    # it keeps the earlier extreme; from the frame holding the series
    # extreme onward it is the frozen range.
    assert ramped[1][1] == highs[1]
    assert ramped[2][1] == highs[1] > highs[2]
    assert ramped[4] == ramped[5] == scale.data_range("series", 5)
    assert scale.data_range("series", 0) == (scale.lo, scale.hi)
    assert scale.data_range("frame", 3) is None
    _raises(lambda: scale.data_range("fixed", 0), "series/frame/ramped")

    # ... and it is what the figure's levels are read from.
    figure = tsm.panel_figure(
        series,
        2,
        [panel],
        options,
        tsm.PlotStyle(clim="ramped"),
        {panel: scale},
    )
    filled = next(
        c for c in figure.axes[0].collections if isinstance(c, ContourSet)
    )
    plt.close(figure)
    want = tsm.contour_levels(
        own[2], 10, non_negative=True, data_range=ramped[2]
    )
    assert np.allclose(filled.levels, want)

    # A signed panel ramps each side on its own: here the positive side
    # peaks at frame 1 and the negative one at frame 2, on alternate
    # rows (which the fold keeps apart, pairing j with n_y - 1 - j).
    meta = _meta("twin_ybudget")
    rec = np.zeros(4, dtype=tsm._record_dtype(meta, "twin_ybudget"))
    rec["t"] = np.arange(4.0)
    pattern = np.random.default_rng(97).random((NY, NKZ))
    even = (np.arange(NY) % 2 == 0)[:, None]
    for f, (up, down) in enumerate(((1, 1), (2, 1), (1, 3), (1, 1))):
        rec["P_U_x"][f] = up * np.where(even, pattern, 0.0)
        rec["P_r_x"][f] = -down * np.where(even, 0.0, pattern)
    budget = _series("twin_ybudget", [_member(meta, rec)])
    key = ("prod_x", None)
    scale = tsm.scan_panels(
        budget, [key], options, ylim=tsm.y_limits(budget, options)
    )[0][key]
    lo, hi = zip(
        *(scale.data_range("ramped", f) for f in range(4)), strict=True
    )
    assert hi[2] == hi[1] > hi[0] and lo[2] < lo[1] == lo[0]
    assert (lo[3], hi[3]) == (scale.lo, scale.hi)


def test_peak_track() -> None:
    r"""The top-band centroid, where it is drawn, and what is written."""
    # A band symmetric in ln(lambda) and ln(y) about one cell, on axes
    # uniform in the logarithm: its centroid is that cell exactly,
    # whatever lies outside the band.
    lam = 10.0 * 2.0 ** np.arange(7)
    y = 2.0 ** np.arange(6)
    values = np.full((y.size, lam.size), 0.5)
    values[1:4, 2:5] = 0.95
    values[2, 3] = 1.0
    values[5, 0] = 0.89  # below (1 - 1/10) of the peak: left out

    def at(v, *, y_axis=y, y_log=True) -> tsm.Map:
        return tsm.Map(
            lam=lam,
            y=y_axis,
            values=v,
            title="",
            name="e_x",
            non_negative=True,
            y_log=y_log,
        )

    centre = tsm.peak_centroid(at(values), 10)
    assert np.allclose(centre, (lam[3], y[2])), centre
    # A cell exactly at the threshold is in the band, and pulls on it.
    edged = values.copy()
    edged[5, 0] = (1.0 - 1.0 / 10) * values.max()
    moved = tsm.peak_centroid(at(edged), 10)
    assert moved[0] < centre[0] and moved[1] > centre[1], moved
    # Nothing positive, no peak.
    assert np.isnan(tsm.peak_centroid(at(-values), 10)).all()
    # Only the rows the box shows count: a larger value two rows under
    # its floor changes nothing.
    hidden = values.copy()
    hidden[0, 6] = 5.0
    box = (y[2], y[-1])
    assert tsm.peak_centroid(at(hidden), 10, box) == tsm.peak_centroid(
        at(values), 10, box
    )
    # A linear ordinate averages y itself, over dy.
    linear = np.linspace(0.0, 50.0, y.size)  # the wall included
    got = tsm.peak_centroid(at(values, y_axis=linear, y_log=False), 10)
    assert np.allclose(got, (lam[3], linear[2])), got

    # Exactly the panels with one continuous bulk are tracked: every
    # panel of the difference spectra's two marginals and of their
    # shape maps, and the production and its mean-shear part on the
    # budget's.
    for prefix in ("e", tsm.SHAPE):
        for marginal in "xz":
            difference = tsm.spectra_panels(prefix, marginal)
            assert [k for k in difference if k[0] in tsm.TRACKED] == (
                difference
            ), (prefix, marginal)
    for prefix, marginal in (
        ("r", "x"),
        (tsm.DECORR, "z"),
        (tsm.DECORR_K, "x"),
        ("e", "x0"),
    ):
        panels = tsm.spectra_panels(prefix, marginal)
        assert not any(k[0] in tsm.TRACKED for k in panels), prefix
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_ybudget")
    budget = _series(
        "twin_ybudget",
        [_member(meta, _records(meta, "twin_ybudget", 3, seed=89))],
    )
    panels = tsm.budget_panels(budget, "x")
    tracked = [k for k in panels if k[0] in tsm.TRACKED]
    assert tracked == [("prod_x", None), ("prod_mean_x", None)]

    # The track is drawn on those panels alone: the history up to the
    # frame, and the frame's point.
    ylim = tsm.y_limits(budget, options)
    scales, _ = tsm.scan_panels(budget, panels, options, ylim=ylim)
    tracks = {
        k: tsm.track_peak(budget, *k, options=options, n_levels=10, ylim=ylim)
        for k in tracked
    }
    figure = tsm.panel_figure(
        budget, 2, panels, options, tsm.PlotStyle(), scales, tracks
    )
    lines = [len(ax.get_lines()) for ax in figure.axes[0::2]]
    history = figure.axes[0].get_lines()[0].get_xdata()
    ellipse = figure.axes[0].get_lines()[2]
    plt.close(figure)
    # The history, the frame's point and the frame's one-sigma ellipse.
    assert lines == [3, 3] + [0] * 7, lines
    assert np.array_equal(history, tracks[tracked[0]].lam)
    assert ellipse.get_linestyle() == "--" and len(ellipse.get_xdata()) > 50

    # The band is the frame's own, so no colour scale moves the track,
    # and the figure and .npz beside the frames say so.
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        stored = {}
        for clim in ("series", "frame", "ramped"):
            written = tsm.render_series(
                budget,
                "budget_x",
                "",
                "x",
                out / clim,
                options=options,
                style=tsm.PlotStyle(dpi=40, clim=clim),
                quiet=True,
            )
            track_dir = out / clim / "budget_x_track"
            names = [
                "budget_x_moments.npz",
                "budget_x_moments.png",
                "budget_x_track.npz",
                "budget_x_track.png",
            ]
            assert sorted(p.name for p in track_dir.iterdir()) == names
            assert set(names) <= {p.name for p in written}
            stored[clim] = np.load(track_dir / "budget_x_track.npz")
        for clim in ("frame", "ramped"):
            for axis in ("lam", "y"):
                assert np.array_equal(
                    stored[clim][axis], stored["series"][axis]
                ), (clim, axis)
        track = stored["series"]
        assert list(track["fields"]) == ["prod_x", "prod_mean_x"]
        assert list(track["panels"]) == ["prod", "prod_mean"]
        assert track["lam"].shape == track["y"].shape == (2, 3)
        assert np.allclose(track["lam"][0], tracks[tracked[0]].lam)
        assert np.allclose(track["y"][1], tracks[tracked[1]].y)
        assert np.allclose(track["t"], budget.t_rel)
        assert np.isclose(float(track["threshold"]), 0.9)
        want = tsm.peak_centroid(
            tsm.make_map(budget, "prod_mean_x", 1, options=options),
            10,
            ylim,
        )
        assert np.allclose((track["lam"][1, 1], track["y"][1, 1]), want)


def test_shape_maps() -> None:
    r"""Each frame over its own peak, under the `$k\,y^+$` premultiplier.

    One shape grown by four decades, on a grid fine enough to put rows
    under the ordinate's floor, with a spike planted there: the peak
    is read off the rows the box shows, so the spike sets nothing, and
    every frame draws the same map on the same bands.
    """
    units = tsm.Units(RE, RE_TAU)
    options = tsm.MapOptions(units)
    ny = 129
    meta = _meta("twin_yspectra", ny=ny)
    rec = _records(meta, "twin_yspectra", 3, seed=101)
    amplitude = np.array([1e-4, 1e-2, 1.0])
    for suffix in tsm.stored_suffixes(meta):
        field = rec[f"e_{suffix}"]
        field[:] = field[0] * amplitude.reshape((-1,) + (1,) * field[0].ndim)
    # y+ ~ 0.05, under the floor, and its R_y partner for the fold.
    rec["e_x"][:, :, [1, ny - 2]] *= 1e6
    series = _series("twin_yspectra", [_member(meta, rec)])
    ylim = tsm.y_limits(series, options)
    assert np.isclose(ylim[0], tsm.Y_FLOOR_PLUS)

    harmonics = np.arange(1, NKZ, dtype=float)
    y_plus = (1.0 + _grid(ny)[: (ny + 1) // 2]) * RE_TAU
    shapes = {}
    for frame in range(amplitude.size):
        for component in (0, None):
            e = rec["e_x"][frame]
            e = e.sum(axis=0) if component is None else e[component]
            # The absolute k y+ map, written out independently: no
            # E_ref, whatever the stream carries.
            absolute = (
                _folded(
                    units.energy(e[:, 1:] * harmonics * VOLUME_FAC)[:, ::-1],
                    ny,
                )
                * y_plus[:, None]
            )
            got = tsm.make_map(
                series, "s_x", frame, options=options, component=component
            )
            assert np.allclose(got.y, y_plus)
            box = tsm.Map(
                lam=got.lam,
                y=y_plus,
                values=absolute,
                title="",
                name="e_x",
                non_negative=True,
                y_log=True,
            )
            peak = float(box.drawn(ylim)[1].max())
            assert np.isclose(got.peak, peak), (frame, component)
            assert np.allclose(got.values, absolute / peak), (frame, component)
            # 1 on the rows the box shows, and the spike under the
            # floor far above it, setting nothing.
            assert np.isclose(got.drawn(ylim)[1].max(), 1.0)
            assert got.drawn()[1].max() > 10.0
            # The title reports that peak on its own line, in wall units.
            assert got.title.count("\n") == 1
            assert tsm.latex_float(peak) in got.title
            assert "y^+" in got.title and r"/\max$" in got.title
            assert r"\max/u_\tau^2 = " in got.title
            shapes[(frame, component)] = got.values
    # The difference spectrum, as on the e_* panels.
    assert (
        r"E^{x}_{\Delta u}/"
        in tsm.make_map(series, "s_x", 0, options=options, component=0).title
    )
    # One shape draws one map, whatever its amplitude.
    for component in (0, None):
        for frame in (1, 2):
            assert np.allclose(
                shapes[(frame, component)], shapes[(0, component)]
            ), (frame, component)

    # The premultiplier is fixed, and every frame-constant factor
    # cancels from the map and survives in the peak alone.
    base = tsm.make_map(series, "s_x", 1, options=options)
    outer_units = tsm.Units(RE, RE_TAU, wall=False)
    for other, factor in (
        (tsm.MapOptions(units, premultiply="none"), 1.0),
        (tsm.MapOptions(units, premultiply="ky"), 1.0),
        (tsm.MapOptions(units, volume_fac=False), 1.0 / VOLUME_FAC),
        (tsm.MapOptions(outer_units), U_TAU**2 / RE_TAU),
    ):
        moved = tsm.make_map(series, "s_x", 1, options=other)
        assert np.allclose(moved.values, base.values)
        assert np.isclose(moved.peak, factor * base.peak)
    outer = tsm.make_map(
        series, "s_x", 1, options=tsm.MapOptions(outer_units)
    ).title
    assert r"\max = " in outer and r"u_\tau" not in outer, outer
    # Taken last: a smoothed map still reaches 1 in the box.
    smoothed = tsm.make_map(
        series, "s_x", 1, options=tsm.MapOptions(units, smooth=3)
    )
    assert np.isclose(smoothed.drawn(ylim)[1].max(), 1.0)
    # A shape-only selection never reads the reference.
    spec = tsm.SeriesSpec("twin_yspectra", tsm.SHAPE, "x", tsm.MAP)
    assert not tsm.needs_reference(series, spec)

    # The same [0, 1] bands in every frame, under every --clim, and
    # --quantile reaches none of them.
    panels = tsm.spectra_panels(tsm.SHAPE, "x")
    scales, notes = tsm.scan_panels(series, panels, options, ylim=ylim)
    assert not notes  # declared non-negative, and it is
    want = np.arange(1, 11) / 10.0
    for clim in ("series", "frame", "ramped"):
        for quantile in (None, 0.5):
            for frame in (0, 2):
                figure = tsm.panel_figure(
                    series,
                    frame,
                    panels,
                    options,
                    tsm.PlotStyle(clim=clim, quantile=quantile),
                    scales,
                )
                for ax in figure.axes[0::2]:
                    filled = next(
                        c for c in ax.collections if isinstance(c, ContourSet)
                    )
                    assert filled.filled
                    assert np.allclose(filled.levels, want), (clim, frame)
                    assert filled.extend == "neither", (clim, quantile)
                plt.close(figure)

    # An identically zero difference field has no peak to divide by,
    # and stays zero rather than going nan.
    zero = rec.copy()
    zero["e_x"][:] = 0.0
    flat = tsm.make_map(
        _series("twin_yspectra", [_member(meta, zero)]),
        "s_x",
        0,
        options=options,
    )
    assert flat.peak == 0.0 and not flat.values.any()


def test_main_renders_the_selected_series() -> None:
    """``main()`` on a two-member set: the default tags, then more."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        spectra_meta = _meta("twin_yspectra")
        budget_meta = _meta("twin_ybudget")
        for index, name in enumerate(("a", "b")):
            start = 100.0 + 0.5 * index  # interleaved reference instants
            for meta, stem, seed in (
                (spectra_meta, "twin_yspectra", index),
                (budget_meta, "twin_ybudget", 10 + index),
            ):
                _write_member(
                    root / name,
                    stem,
                    meta,
                    _records(meta, stem, 4, t0=start, seed=seed),
                    parent=f"{name}.tar",
                    parent_t=start,
                )
        out = root / "figures"
        code = tsm.main(
            [
                "--members",
                str(root / "a"),
                str(root / "b"),
                "--out",
                str(out),
                "--re",
                str(RE),
                "--re-tau",
                str(RE_TAU),
                "--stride",
                "2",
                "--usetex",
                "off",
                "--dpi",
                "50",
            ]
        )
        assert code == 0
        # The bare default set: the difference-spectra, shape and
        # budget marginals, each with the track of its tracked panels
        # beside it, and the three histories, and nothing else.
        maps = [
            f"{kind}_{m}"
            for kind in ("spectra_e", f"spectra_{tsm.SHAPE}", "budget")
            for m in "xz"
        ]
        histories = {f"history_{b}" for b in ("e", tsm.SHAPE, "budget")}
        defaults = {*maps, *(f"{tag}_track" for tag in maps), *histories}
        assert {p.name for p in out.iterdir()} == defaults
        for tag in maps:
            frames = sorted((out / tag).glob("*.png"))
            assert [f.name for f in frames] == [
                f"{tag}_0.png",
                f"{tag}_2.png",
            ], tag
            assert all(f.stat().st_size > 0 for f in frames)
            track = sorted(f.name for f in (out / f"{tag}_track").iterdir())
            assert track == [
                f"{tag}_moments.npz",
                f"{tag}_moments.png",
                f"{tag}_track.npz",
                f"{tag}_track.png",
            ], tag
        # A budget figure is three columns of the spectra's panels: the
        # geometry is independent of the axis limits, so any serve.
        style = tsm.PlotStyle(dpi=50)
        for tag, ncols, n_panels in (
            ("spectra_e_x", None, 4),
            ("budget_x", tsm.BUDGET_NCOLS, 9),
        ):
            width = plt.imread(out / tag / f"{tag}_0.png").shape[1]
            limits = (1.0, 10.0)
            want = tsm.panel_geometry(
                n_panels, limits, limits, style, y_log=True, ncols=ncols
            ).fig_w
            assert abs(width - want * style.dpi) <= 1, (tag, width)

        def run(target: Path, *extra: str) -> int:
            return tsm.main(
                [
                    "--members",
                    str(root / "a"),
                    str(root / "b"),
                    "--out",
                    str(target),
                    "--re",
                    str(RE),
                    "--re-tau",
                    str(RE_TAU),
                    "--stride",
                    "2",
                    "--usetex",
                    "off",
                    "--dpi",
                    "50",
                    *extra,
                ]
            )

        # ``--spacetime`` adds one figure per colour scale for the
        # whole run, and the ``.npz`` behind the pair.  The budget's
        # k-sum changes sign, so it draws no log figure.
        st_dir = root / "st"
        assert run(st_dir, "--spacetime") == 0
        assert {p.name for p in st_dir.iterdir()} - defaults == {
            "spacetime_e",
            "spacetime_budget",
        }
        files = sorted(f.name for f in (st_dir / "spacetime_e").iterdir())
        assert files == [
            "spacetime_e.npz",
            "spacetime_e_lin.png",
            "spacetime_e_log.png",
        ]
        assert sorted(
            f.name for f in (st_dir / "spacetime_budget").iterdir()
        ) == ["spacetime_budget.npz", "spacetime_budget_lin.png"]
        assert all(f.stat().st_size > 0 for f in st_dir.rglob("*.png"))

        # ``--no-budget`` drops the budget, and ``--budget`` -- the
        # default -- still parses, for the command lines that pass it;
        # ``--x0`` adds nothing here, these members carrying no such
        # plane.
        lean = root / "lean"
        assert run(lean, "--no-budget", "--x0") == 0
        assert {p.name for p in lean.iterdir()} == {
            f"spectra_{prefix}_{m}{track}"
            for prefix in ("e", tsm.SHAPE)
            for m in "xz"
            for track in ("", "_track")
        } | {f"history_{b}" for b in ("e", tsm.SHAPE)}
        parser = tsm.build_parser()
        required = ["--members", "m", "--out", "o", "--re", "1"]
        for extra, drawn in (
            ([], True),
            (["--budget"], True),
            (["--no-budget"], False),
        ):
            parsed = parser.parse_args([*required, "--re-tau", "1", *extra])
            assert parsed.budget is drawn, extra

        # Each switch adds exactly its own family, and the reference's
        # and R^k's spacetime maps need both of their switches.  No
        # reference map carries a track.
        for extra, gained in (
            (["--reference"], {"spectra_r_x", "spectra_r_z"}),
            (["--decorr"], {"spectra_decorr_x", "spectra_decorr_z"}),
            (
                ["--decorr-k"],
                {"spectra_decorr_k_x", "spectra_decorr_k_z"},
            ),
            (
                ["--reference", "--spacetime"],
                {
                    "spectra_r_x",
                    "spectra_r_z",
                    "spacetime_e",
                    "spacetime_r",
                    "spacetime_budget",
                },
            ),
            (
                ["--decorr-k", "--spacetime"],
                {
                    "spectra_decorr_k_x",
                    "spectra_decorr_k_z",
                    "spacetime_decorr_k",
                    "spacetime_e",
                    "spacetime_budget",
                },
            ),
            (["--front"], {"front_x", "front_z"}),
            (["--moment-budget"], {"moments_y", "moments_x", "moments_z"}),
            (
                ["--growth"],
                {f"growth_{k}" for k in ("global", "x", "z", "y", "ssp")},
            ),
        ):
            target = root / "-".join(e.strip("-") for e in extra)
            assert run(target, *extra) == 0
            got = {p.name for p in target.iterdir()} - defaults
            assert got == gained, (extra, got)

        # ``--series`` is exact, and an unknown tag is refused rather
        # than quietly dropped.
        one = root / "one"
        assert run(one, "--series", "budget_z") == 0
        assert sorted(p.name for p in one.iterdir()) == [
            "budget_z",
            "budget_z_track",
        ]
        _raises(
            lambda: run(root / "none", "--series", "spectra_e_x0"),
            "unknown series",
        )


def test_history_maps() -> None:
    r"""The premultiplied histories, each premultiplied after its sum.

    The `$(y, t)$` map is the `$k$`-summed spacetime map times the wall
    distance in the plotted units; the `$(\lambda, t)$` map is the
    wall-normal average over the doubled half-channel weights, the
    `$m = 0$` column dropped, times `$m$`, ascending in wavelength and
    over `$E^{\mathrm{ref}}$` -- untouched by ``--no-volume-fac``.  The
    two read one total; a shape history's rows each reach 1 and do not
    see the unit system; a budget history's production is its two
    parts; and the renderer writes the figures and ``.npz`` the family
    promises, the logarithmic one only where every panel is a
    non-negative absolute energy.
    """
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    rec = _records(meta, "twin_yspectra", 4, seed=41)
    series = _series("twin_yspectra", [_member(meta, rec)])
    scale = series.reference_scale()
    for component in (1, None):
        plain = tsm.make_spacetime(
            series, "e", options=options, component=component
        )
        pre = tsm.make_spacetime(
            series, "e", options=options, component=component, premultiply=True
        )
        assert np.allclose(pre.values, plain.values * plain.y[None, :])
        assert pre.provenance["premultiplier"] == "y"
        assert pre.title.startswith("$y^+")
    n_half = (NY + 1) // 2
    weights = 2.0 * np.asarray(meta["y_weights"])[:n_half]
    weights[-1] = meta["y_weights"][n_half - 1]  # the mid-plane: itself
    assert np.allclose(tsm.half_weights(series), weights)
    harmonics = np.arange(NKZ, dtype=float)
    for component in (0, None):
        e = (
            rec["e_x"].sum(axis=1)
            if component is None
            else rec["e_x"][:, component]
        )
        folded = 0.5 * (e[:, :n_half] + e[:, ::-1][:, :n_half])
        average = np.einsum("j,tjk->tk", weights, folded)
        over = scale.sum() if component is None else scale[component]
        want = (average[:, 1:] * harmonics[1:])[:, ::-1] / over
        drawn = tsm.make_scaletime(
            series, "e", "x", options=options, component=component
        )
        assert np.allclose(drawn.values, want)
        assert np.all(np.diff(drawn.lam) > 0.0)
        no_vf = tsm.MapOptions(options.units, volume_fac=False)
        again = tsm.make_scaletime(
            series, "e", "x", options=no_vf, component=component
        )
        assert np.allclose(again.values, drawn.values)
    # One total, read both ways: the lambda map's bands, unpremultiplied
    # and with the m = 0 column back, average to what the y map holds.
    bands = drawn.values[:, ::-1] / harmonics[None, 1:] * scale.sum()
    zero = tsm.y_averaged(series, "e", "x", "mean").sum(axis=1)[:, 0]
    plain = tsm.make_spacetime(series, "e", options=options)
    profile = plain.values * scale.sum() / VOLUME_FAC
    assert np.allclose(
        bands.sum(axis=1) + zero, np.einsum("j,tj->t", weights, profile)
    )
    # A shape history: each row over its own peak, in any unit system.
    outer = tsm.MapOptions(tsm.Units(RE, RE_TAU, wall=False))
    for component in (0, None):
        wall_map = tsm.make_spacetime(
            series, tsm.SHAPE, options=options, component=component
        )
        outer_map = tsm.make_spacetime(
            series, tsm.SHAPE, options=outer, component=component
        )
        ylim = tsm.y_limits(series, options)
        _, shown = wall_map.drawn(ylim)
        assert np.allclose(shown.max(axis=1), 1.0)
        assert np.allclose(wall_map.values, outer_map.values)
        assert wall_map.provenance["row_peaks"].shape == (rec.size,)
        lam_map = tsm.make_scaletime(
            series, tsm.SHAPE, "z", options=options, component=component
        )
        assert np.allclose(lam_map.values.max(axis=1), 1.0)
        assert np.allclose(
            lam_map.values,
            tsm.make_scaletime(
                series, tsm.SHAPE, "z", options=outer, component=component
            ).values,
        )
    # The budget's production is its two parts, in both histories.
    budget_meta = _meta("twin_ybudget")
    budget = _series(
        "twin_ybudget",
        [_member(budget_meta, _records(budget_meta, "twin_ybudget", 4))],
    )
    whole = tsm.make_spacetime(
        budget, "prod", options=options, premultiply=True
    )
    parts = [
        tsm.make_spacetime(budget, t, options=options, premultiply=True)
        for t in ("prod_mean", "prod_fluct")
    ]
    assert np.allclose(whole.values, parts[0].values + parts[1].values)
    whole = tsm.make_scaletime(budget, "prod", "z", options=options)
    parts = [
        tsm.make_scaletime(budget, t, "z", options=options)
        for t in ("prod_mean", "prod_fluct")
    ]
    assert np.allclose(whole.values, parts[0].values + parts[1].values)
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        style = tsm.PlotStyle(dpi=40)
        expected = {
            "e": ["lin", "log"],
            tsm.SHAPE: ["lin"],
            "": ["lin"],
        }
        for base, scales in expected.items():
            stem = "twin_ybudget" if base == "" else "twin_yspectra"
            which = budget if base == "" else series
            tag = f"history_{base or 'budget'}"
            written = tsm.render_history(
                which,
                tsm.SeriesSpec(stem, base, "", tsm.HISTORY),
                tag,
                out,
                options=options,
                style=style,
                quiet=True,
            )
            want = sorted(
                [f"{tag}_{k}_{sc}.png" for k in "yxz" for sc in scales]
                + [f"{tag}_{k}.npz" for k in "yxz"]
            )
            assert sorted(p.name for p in written) == want, (tag, written)
            stored = np.load(out / tag / f"{tag}_x.npz")
            n_panels = 3 if base == "" else 4
            assert stored["values"].shape[:2] == (n_panels, rec.size)
            assert str(stored["premultiplier"]) == "k"
            assert str(stored["abscissa"]) == "lambda_z"
            assert stored["quantiles"].shape == (n_panels, rec.size, 3)
            if base == tsm.SHAPE:
                assert np.isfinite(stored["row_peaks"]).all()
            else:
                assert np.isnan(stored["row_peaks"]).all()
            assert (
                str(np.load(out / tag / f"{tag}_y.npz")["premultiplier"])
                == "y"
            )
    print("history maps: OK")


def test_quantile_curves() -> None:
    """The quantile lines of a history: of the as-drawn density."""
    x = np.logspace(0.0, 2.0, 201)
    values = np.ones((3, x.size))
    values[1] = 0.0  # nothing drawn: no quantile
    values[2] = -1.0  # negative values carry no mass either
    q = tsm.quantile_curves(x, values, (0.1, 0.5, 0.9))
    # Uniform in ln x: the quantiles sit at those fractions of the axis.
    assert np.allclose(np.log10(q[0]), [0.2, 1.0, 1.8], atol=0.01)
    assert np.isnan(q[1:]).all()
    # A linear axis measures in x itself.
    lin = tsm.quantile_curves(
        np.linspace(0.0, 10.0, 101), np.ones((1, 101)), (0.5,), x_log=False
    )
    assert np.allclose(lin, 5.0, atol=0.06)
    print("quantile lines: OK")


def _gaussian_map(values_scale: float = 1.0, shift: float = 0.0) -> tsm.Map:
    """A planted Gaussian blob in (ln lambda, ln y), as a drawn map."""
    lam = np.logspace(1.0, 3.0, 161)
    y = np.logspace(0.0, 2.2, 151)
    dx = np.log(lam)[None, :] - np.log(150.0)
    dy = np.log(y)[:, None] - np.log(30.0)
    inv = np.linalg.inv([[0.3, 0.1], [0.1, 0.25]])
    q = inv[0, 0] * dx**2 + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2
    return tsm.Map(
        lam=lam,
        y=y,
        values=values_scale * np.exp(-0.5 * q) - shift,
        title="",
        name="s_x",
        non_negative=True,
        y_log=True,
    )


def test_size_and_tilt() -> None:
    r"""The moments of a map as drawn: planted, scaled, signed, drawn.

    A Gaussian blob planted in `$(\ln\lambda, \ln y)$` comes back with
    its centroid and covariance; a constant factor moves nothing (the
    moments are ratios over the mass); a signed field's moments are
    its positive part's, with the negative share reported; the frame's
    one-sigma ellipse is the curve `$d^\top C^{-1} d = 1$`; every
    tracked frame carries its moments; and ``--no-frames`` writes the
    tracks and moments without a single frame.
    """
    from dnsjax.analysis.twin.moments import log_moments

    sums, share = tsm.map_moment_sums(_gaussian_map())
    m = log_moments(sums)
    assert abs(float(m.mean_lam) - np.log(150.0)) < 2e-3
    assert abs(float(m.mean_y) - np.log(30.0)) < 2e-3
    assert np.allclose(
        [m.var_lam, m.var_y, m.cov], [0.3, 0.25, 0.1], atol=3e-3
    )
    assert share == 0.0
    scaled = log_moments(tsm.map_moment_sums(_gaussian_map(5.0))[0])
    for name in ("mean_lam", "mean_y", "var_lam", "var_y", "cov"):
        assert np.isclose(getattr(scaled, name), getattr(m, name))
    signed, share = tsm.map_moment_sums(_gaussian_map(shift=0.1))
    assert share > 0.0
    positive = tsm.map_moment_sums(
        tsm.Map(
            lam=_gaussian_map().lam,
            y=_gaussian_map().y,
            values=np.maximum(_gaussian_map(shift=0.1).values, 0.0),
            title="",
            name="s_x",
            non_negative=True,
            y_log=True,
        )
    )[0]
    assert np.allclose(signed, positive)
    frames = log_moments(sums[None, :])
    x, y = tsm.ellipse_points(frames, 0)
    d = np.vstack([x - frames.mean_lam[0], y - frames.mean_y[0]])
    cov = np.array(
        [[frames.var_lam[0], frames.cov[0]], [frames.cov[0], frames.var_y[0]]]
    )
    assert np.allclose(np.einsum("in,ij,jn->n", d, np.linalg.inv(cov), d), 1.0)
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_ybudget")
    budget = _series(
        "twin_ybudget",
        [_member(meta, _records(meta, "twin_ybudget", 3, seed=89))],
    )
    track = tsm.track_peak(
        budget,
        "prod_x",
        None,
        options=options,
        n_levels=10,
        ylim=tsm.y_limits(budget, options),
    )
    assert track.moments.mass.shape == (3,)
    assert track.negative.shape == (3,) and np.all(track.negative >= 0.0)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        spectra_meta = _meta("twin_yspectra")
        _write_member(
            root / "a",
            "twin_yspectra",
            spectra_meta,
            _records(spectra_meta, "twin_yspectra", 3, seed=5),
        )
        code = tsm.main(
            [
                "--members",
                str(root / "a"),
                "--out",
                str(root / "out"),
                "--re",
                str(RE),
                "--re-tau",
                str(RE_TAU),
                "--series",
                "spectra_e_x",
                "--stride",
                "1",
                "--no-frames",
                "--usetex",
                "off",
                "--dpi",
                "40",
                "--quiet",
            ]
        )
        assert code == 0
        assert sorted(p.name for p in (root / "out").iterdir()) == [
            "spectra_e_x_track"
        ]
        names = sorted(
            p.name for p in (root / "out" / "spectra_e_x_track").iterdir()
        )
        assert names == [
            "spectra_e_x_moments.npz",
            "spectra_e_x_moments.png",
            "spectra_e_x_track.npz",
            "spectra_e_x_track.png",
        ]
        stored = np.load(
            root / "out" / "spectra_e_x_track" / "spectra_e_x_moments.npz"
        )
        assert stored["rho"].shape == (4, 3)
        assert str(stored["premultiply"]) == "k"
    print("size and tilt: OK")


def _linear_pair(n_t: int = 41, dt: float = 0.01, seed: int = 3):
    r"""A spectra and a budget member whose terms add up to `$\partial_t e$`.

    The difference energy is linear in time on every plane entry, and
    the six stored densities that make up the rate are a random split
    of its slope (``eps`` is free: it cancels between the dissipation
    and the viscous transport), so the moment budget must close on the
    moments' own rates up to the differencing of the sampled clock.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((3, NY, NKZ, NKX)) + 0.5
    slope = rng.random((3, NY, NKZ, NKX)) - 0.3
    t = 100.0 + dt * np.arange(n_t)
    s_meta, b_meta = _meta("twin_yspectra"), _meta("twin_ybudget")
    spectra = np.zeros(n_t, dtype=tsm._record_dtype(s_meta, "twin_yspectra"))
    budget = np.zeros(n_t, dtype=tsm._record_dtype(b_meta, "twin_ybudget"))
    spectra["t"] = budget["t"] = t
    plane = base[None] + (t - t[0])[:, None, None, None, None] * slope[None]
    blocks = {
        "x": lambda a: a.sum(axis=-1),
        "z": lambda a: a.sum(axis=-2),
        "xz00": lambda a: a[..., 0, 0],
    }
    for suffix, block in blocks.items():
        spectra[f"e_{suffix}"] = block(plane)
        spectra[f"r_{suffix}"] = block(np.broadcast_to(base, plane.shape))
    rate = slope.sum(axis=0)  # the component-summed rate, (ny, nkz, nkx)
    shares = rng.random(6)
    shares /= shares.sum()
    noise = rng.standard_normal((6, NY, NKZ, NKX))
    noise -= noise.mean(axis=0)
    stored = ("P_U", "P_r", "T_ref", "T_self", "V", "Wp")
    for index, term in enumerate(stored):
        density = shares[index] * rate + 0.3 * noise[index]
        for suffix, block in blocks.items():
            budget[f"{term}_{suffix}"] = block(
                np.broadcast_to(density, (n_t, *density.shape))
            )
    eps = rng.random((NY, NKZ, NKX))
    for suffix, block in blocks.items():
        budget[f"eps_{suffix}"] = block(
            np.broadcast_to(eps, (n_t, *eps.shape))
        )
    return (s_meta, spectra), (b_meta, budget)


def test_moment_budget() -> None:
    """The per-term rates add up to the moments' own rates."""
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    (s_meta, s_rec), (b_meta, b_rec) = _linear_pair()
    spectra = _series("twin_yspectra", [_member(s_meta, s_rec)])
    budget = _series("twin_ybudget", [_member(b_meta, b_rec)])
    for marginal in ("", "x", "z"):
        result = tsm.moment_budget(spectra, budget, marginal, options)
        fields = (
            ("mass", "mean_y", "var_y")
            if not marginal
            else ("mass", "mean_lam", "mean_y", "var_lam", "var_y", "cov")
        )
        for name in fields:
            total = result["total"][name][1:-1]
            own = result["finite_difference"][name][1:-1]
            big = np.abs(result["contributions"][name]).max()
            assert np.allclose(total, own, atol=2e-3 * big), (marginal, name)
        assert len(result["groups"]) == 7
        assert ("press_input" in result["groups"]) == (not marginal)
    short = _series("twin_ybudget", [_member(b_meta, b_rec[:-1])])
    _raises(
        lambda: tsm.moment_budget(spectra, short, "x", options),
        "different grids",
    )
    print("moment budget closes: OK")


def test_front_times() -> None:
    """When each cell rises through the level for good."""
    t = np.arange(5.0)
    r = np.array(
        [
            [0.6, 0.0, 0.1, 0.0, 0.7],
            [0.4, 0.1, 0.2, 0.6, 0.8],
            [0.3, 0.2, 0.3, 0.7, 0.9],
            [0.6, 0.3, 0.9, 0.8, 0.9],
            [0.7, 0.4, 1.0, 0.9, 0.9],
        ]
    )
    got = tsm.front_times(t, r, 0.5)
    # A dip below the level is not the crossing; the last rise is.
    want = [2.0 + 0.2 / 0.3, np.nan, 2.0 + 0.2 / 0.6, 0.5 / 0.6, 0.0]
    assert np.allclose(got, want, equal_nan=True), got
    # A pair that has decorrelated completely is above the level from
    # the first frame, everywhere.
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    series = _series("twin_yspectra", [_member(meta, _saturated(meta))])
    for map_ in tsm.front_maps(series, "x", options):
        finite = map_.values[np.isfinite(map_.values)]
        assert finite.size and np.allclose(
            finite, options.units.plotted_time(series.t_rel[0])
        )
    print("front times: OK")


def test_growth_laws() -> None:
    r"""The growth figures' phases, global curve and band rates.

    A curve built to grow exponentially and then lose correlation at a
    constant rate is marked with both rates; the global curve aligns
    every member's ``twin.dat`` on whole steps, falls back to the
    spectra totals without one, and is their geometric mean; and a
    band's budget rates are the band's own terms over its energy.
    """
    gamma0, nu, t1 = 0.4, 0.06, 30.0
    t = np.arange(0.0, 120.0, 0.1)
    f = (nu / gamma0) * np.log1p(np.exp(gamma0 * (t - t1)))
    sat = 2.5
    phases = tsm.growth_phases(t, sat * (1.0 - np.exp(-f)), sat)
    assert abs(phases.exponential[2] - gamma0) < 0.1 * gamma0
    assert abs(phases.decorrelation[2] - nu) < 0.1 * nu
    assert phases.decorrelation[0] > phases.exponential[1]
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    meta = _meta("twin_yspectra")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        members = []
        for index, start in enumerate((100.0, 237.5)):
            rec = _records(meta, "twin_yspectra", 4, t0=start, seed=index)
            directory = _write_member(
                root / f"m{index}",
                "twin_yspectra",
                meta,
                rec,
                parent_t=start,
            )
            record = json.loads((directory / "twin.json").read_text())
            record["dt"] = 0.01
            record["format_version"] = 1
            (directory / "twin.json").write_text(json.dumps(record))
            steps = np.arange(0, 311, 10)  # every 0.1 over 3.1 units
            t_abs = start + 0.01 * steps
            energy = (1.0 + index) * np.exp(0.3 * 0.01 * steps)
            lines = ["#  t  E_d  E_ref"] + [
                f"  {ta:.9g}  {e:.9g}  {1.0:.9g}"
                for ta, e in zip(t_abs, energy, strict=True)
            ]
            (directory / "twin.dat").write_text("\n".join(lines) + "\n")
            members.append(directory)
        series = tsm.open_series(members, "twin_yspectra")
        curves = tsm.growth_global(series, options)
        assert curves.source == "twin.dat", curves.source
        assert curves.members.shape == (2, 31)
        assert np.allclose(curves.t, 0.1 * np.arange(31))
        assert np.allclose(
            curves.energy[0], np.sqrt(curves.members[0] * curves.members[1])
        )
        assert np.isclose(
            curves.saturation[0], 2.0 * series.reference_scale().sum()
        )
        (members[1] / "twin.dat").unlink()  # one member without it
        fallback = tsm.growth_global(
            tsm.open_series(members, "twin_yspectra"), options
        )
        assert fallback.source.startswith("twin_yspectra")
        assert np.allclose(fallback.t, series.t_rel)
        totals = tsm.member_energies(series)
        assert np.allclose(fallback.members, totals)
    # A band's rates are its own terms over its own energy.
    spectra = _series(
        "twin_yspectra",
        [_member(meta, _records(meta, "twin_yspectra", 4, seed=2))],
    )
    b_meta = _meta("twin_ybudget")
    budget = _series(
        "twin_ybudget",
        [_member(b_meta, _records(b_meta, "twin_ybudget", 4, seed=8))],
    )
    bands = tsm.growth_bands(spectra, "z", options, budget)
    columns = tsm._band_columns(NKX)
    want_energy = tsm.y_averaged(spectra, "e", "z", "mean").sum(axis=1)
    mean = tsm._y_averaged_many(spectra, ["e_xz00"], "mean")[0].sum(axis=1)
    want_energy[:, 0] -= mean
    assert np.allclose(bands.energy, want_energy[:, columns].T)
    prod_mean = tsm.y_averaged(budget, "prod_mean", "z", "mean")
    prod_mean[:, 0] -= tsm._y_averaged_many(
        budget, ["prod_mean_xz00"], "mean"
    )[0]
    assert np.allclose(
        bands.rates["prod_mean"], prod_mean[:, columns].T / bands.energy
    )
    print("growth laws: OK")


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for case in tests:
        case()
        print(f"  PASS  {case.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
