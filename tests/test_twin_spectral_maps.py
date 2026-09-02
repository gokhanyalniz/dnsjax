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
   negative still draws signed; a declared non-negative field with a
   round-off negative draws non-negative and says so;
   ``--signs-from-data`` infers both instead.
6. **The fold.** ``mean`` averages `$j$` with `$n_y-1-j$` without
   double counting the mid-plane, ``upper`` relabels its rows with the
   opposite half's wall distances, and the grid preconditions each
   mode actually needs are the ones checked.
7. **The reader.** Members meet on relative time to a tolerance and
   the shared grid is their intersection; ``stride`` / ``first`` /
   ``last`` clip it; a member that is not the same flow, and a stream
   whose own samples are closer than the tolerance, are both refused.
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
12. **End to end.** ``main()`` on a two-member set draws the two
    spectra marginals and nothing else, and each of the five opt-in
    switches adds exactly its own family -- ``--decorr-k`` needing
    ``--spacetime`` as well before `$\mathcal{R}^k$` gets one.

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

# ── Fixtures ─────────────────────────────────────────────────────────

#: A plane-Poiseuille-shaped configuration, small enough that every
#: case is a handful of arrays.  ``RE_TAU`` is the measured number of
#: the KMM4200 ensemble, so the printed inner units look like a run's.
RE, RE_TAU = 4200.0, 178.62135279727977
U_TAU = RE_TAU / RE
NY, NKZ, NKX = 33, 6, 5
LX, LZ, VOLUME_FAC = 4.0, 2.0, 2.0
TERMS = ("P_U", "eps", "V")


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


def _member(
    meta: dict,
    rec: np.ndarray,
    *,
    path: str = "m",
    parent: str = "p0",
    parent_t: float | None = None,
) -> tsm._Member:
    """One opened member, without going through the filesystem."""
    # ``_open_member`` writes the resolved layout back onto the
    # sidecar so a legacy member has a key to compare; do the same
    # here, since these members never pass through it.
    meta = meta | {"suffixes": list(tsm.stored_suffixes(meta))}
    t = rec["t"].astype(np.float64)
    rows = np.sort(np.unique(t, return_index=True)[1])
    t_abs = t[rows]
    t0 = float(t_abs[0]) if parent_t is None else float(parent_t)
    return tsm._Member(Path(path), meta, rec, rows, t_abs, t_abs - t0, parent)


def _series(stem: str, members, **over) -> tsm.YSeries:
    """A series over *members*, every record kept."""
    n = members[0].t_abs.size
    return tsm.YSeries(
        stem=stem,
        members=tuple(members),
        rows=np.stack([m.rows[:n] for m in members]),
        index=np.arange(n),
        t_rel=members[0].t_rel[:n],
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
    """A panel is `$m \\times$` entry `$\\times V$`, in stream units."""
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
            "P_U_z",
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
    """``ky``'s second factor is `$y^+$`, so it carries a `$Re_\\tau$`.

    The `$k$` half is unit-invariant (`$k^+\\Phi^+ = k\\Phi$`) and the
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
    """`$E^{\\mathrm{ref}}$` and what it does and does not normalise."""
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
    assert not notes  # a sum of squares, no sign complaint

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
    """Non-negativity is declared; the data only checks it."""
    meta = _meta("twin_ybudget", terms=["P_U", "eps"])
    rec = np.zeros(2, dtype=tsm._record_dtype(meta, "twin_ybudget"))
    rec["t"] = [0.0, 1.0]
    for suffix in tsm.stored_suffixes(meta):
        rec[f"P_U_{suffix}"][:] = 1.0  # signed, but never negative here
        rec[f"eps_{suffix}"][:] = 1.0
    rec["eps_x"][:, [10, NY - 1 - 10], 3] = -1e-14  # a round-off dip
    series = _series("twin_ybudget", [_member(meta, rec)])
    options = tsm.MapOptions(tsm.Units(RE, RE_TAU))
    ylim = tsm.y_limits(series, options)
    panels = [("P_U_x", None), ("eps_x", None), ("sum_x", None)]

    scales, notes = tsm.scan_panels(series, panels, options, ylim=ylim)
    assert scales[("P_U_x", None)].non_negative is False
    assert scales[("sum_x", None)].non_negative is False
    assert scales[("eps_x", None)].non_negative is True
    assert len(notes) == 1 and "round-off" in notes[0]
    assert "eps_x" in notes[0]

    inferred, _ = tsm.scan_panels(
        series, panels, options, declared=False, ylim=ylim
    )
    assert inferred[("P_U_x", None)].non_negative is True
    assert inferred[("eps_x", None)].non_negative is False  # the dip

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

        # Members whose clocks never meet, a selection that keeps
        # nothing, and an unknown stream name.
        _write_member(
            root / "off", "twin_yspectra", meta, second, parent_t=150.5
        )
        _raises(
            lambda: tsm.open_series(
                [root / "a", root / "off"], "twin_yspectra"
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
    """What each layout offers, what is drawn, and `$E^{ref}$`.

    The three cases are the three streams that exist on disk: a
    pre-``xz00`` member, the current default, and the current default
    under ``twin.x0_planes``.  All three must open; the `$(0, 0)$`
    mode `$E^{\\mathrm{ref}}$` subtracts is the same number in all
    three, whichever field it is read from; and what is *drawn* is the
    two spectra marginals unless a switch asks for more -- each of the
    five adding its own family and nothing else.
    """
    scales = {}
    registries = {}
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

    spectra = [f"spectra_{p}_{m}" for p in ("e", "r") for m in ("x", "z")]
    decorr = [f"spectra_decorr_{m}" for m in ("x", "z")]
    decorr_k = [f"spectra_decorr_k_{m}" for m in ("x", "z")]
    spacetime = ["spacetime_e", "spacetime_r"]
    everything = dict(
        x0=True, budget=True, decorr=True, decorr_k=True, spacetime=True
    )
    for label, registry in registries.items():
        # The bare default is one family: the two marginals of the
        # difference and reference spectra.
        assert sorted(tsm.default_series(registry)) == sorted(spectra), label
        # Each switch adds its own family and nothing else.  R^k's
        # spacetime map is the one composite -- it needs both of its
        # switches, so neither alone brings it.
        for switch, gained in (
            ({"decorr": True}, decorr),
            ({"decorr_k": True}, decorr_k),
            ({"spacetime": True}, spacetime),
            (
                {"decorr_k": True, "spacetime": True},
                decorr_k + spacetime + ["spacetime_decorr_k"],
            ),
        ):
            got = tsm.default_series(registry, **switch)
            assert set(got) - set(spectra) == set(gained), (label, switch)
        assert set(tsm.default_series(registry, **everything)) == set(
            registry
        ), label
    print("layouts, their tags, and one E_ref across all three: OK")


def test_reference_scale_is_a_quadrature() -> None:
    """`$E^{\\mathrm{ref}}$` averages over `$y$` with the *weights*.

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
        # The bare default set: the difference and reference spectra
        # marginals, and nothing else.
        maps = [
            f"spectra_{base}_{m}" for base in ("e", "r") for m in ("x", "z")
        ]
        assert sorted(p.name for p in out.iterdir()) == sorted(maps)
        for tag in maps:
            frames = sorted((out / tag).glob("*.png"))
            assert [f.name for f in frames] == [
                f"{tag}_0.png",
                f"{tag}_2.png",
            ], tag
            assert all(f.stat().st_size > 0 for f in frames)

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
        # whole run, and the ``.npz`` behind the pair.
        spacetime = ["spacetime_e", "spacetime_r"]
        st_dir = root / "st"
        assert run(st_dir, "--spacetime") == 0
        assert sorted(p.name for p in st_dir.iterdir()) == sorted(
            maps + spacetime
        )
        for tag in spacetime:
            files = sorted(f.name for f in (st_dir / tag).iterdir())
            assert files == [
                f"{tag}.npz",
                f"{tag}_lin.png",
                f"{tag}_log.png",
            ], tag
            assert all(f.stat().st_size > 0 for f in (st_dir / tag).iterdir())

        # ``--budget`` adds the other stream, its k-sum included under
        # ``--spacetime``; ``--x0`` adds nothing here, these members
        # carrying no such plane.  The budget changes sign, so it
        # draws no log figure.
        wider = root / "wider"
        assert run(wider, "--budget", "--x0", "--spacetime") == 0
        assert sorted(p.name for p in wider.iterdir()) == sorted(
            maps + spacetime + ["budget_x", "budget_z", "spacetime_budget"]
        )
        assert sorted(
            f.name for f in (wider / "spacetime_budget").iterdir()
        ) == ["spacetime_budget.npz", "spacetime_budget_lin.png"]

        # Each switch adds exactly its own family, and R^k's spacetime
        # map needs both of its switches.
        for extra, gained in (
            (["--decorr"], {"spectra_decorr_x", "spectra_decorr_z"}),
            (
                ["--decorr-k"],
                {"spectra_decorr_k_x", "spectra_decorr_k_z"},
            ),
            (["--spacetime"], set(spacetime)),
            (
                ["--decorr-k", "--spacetime"],
                {
                    "spectra_decorr_k_x",
                    "spectra_decorr_k_z",
                    "spacetime_decorr_k",
                    *spacetime,
                },
            ),
        ):
            target = root / "-".join(e.strip("-") for e in extra)
            assert run(target, *extra) == 0
            assert {p.name for p in target.iterdir()} - set(maps) == gained

        # ``--series`` is exact, and an unknown tag is refused rather
        # than quietly dropped.
        one = root / "one"
        assert run(one, "--series", "budget_z") == 0
        assert [p.name for p in one.iterdir()] == ["budget_z"]
        _raises(
            lambda: run(root / "none", "--series", "spectra_e_x0"),
            "unknown series",
        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for case in tests:
        case()
        print(f"  PASS  {case.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
