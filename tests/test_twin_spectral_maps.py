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
8. **End to end.** ``main()`` renders every series of both streams for
   a two-member set.

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


def _meta(stem: str, **over) -> dict:
    """A sidecar for *stem*; *over* replaces any key."""
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

    The spectra are marginals of a genuine `$(k_z, k_x)$` plane, so
    the two of them agree on the total by construction -- which is
    what :meth:`~twin_spectral_maps.YSeries._check_marginals` demands
    of a real stream.
    """
    rec = np.zeros(n_t, dtype=tsm._record_dtype(meta, stem))
    rec["t"] = t0 + np.arange(n_t, dtype=float)
    rng = np.random.default_rng(seed)
    if stem == "twin_yspectra":
        plane = rng.random((n_t, 3, meta["ny"], NKZ, NKX))
        fields = [("e", 0.3 * plane)]
        if meta["includes_ref"]:
            fields.append(("r", plane))
        for prefix, field in fields:
            rec[f"{prefix}_x"] = field.sum(axis=4)
            rec[f"{prefix}_z"] = field.sum(axis=3)
            rec[f"{prefix}_x0"] = field[..., 0]
    else:
        for name in rec.dtype.names[1:]:
            values = rng.random(rec[name].shape)
            # ``eps`` is a sum of squares in a real stream, and the
            # sign check would rightly complain about a signed one.
            rec[name] = values if name.startswith("eps") else values - 0.5
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


# ── Cases ────────────────────────────────────────────────────────────


def test_premultiplication() -> None:
    """A panel is `$m \\times$` entry `$\\times V$`, in stream units."""
    units = tsm.Units(RE, RE_TAU)
    spectra_meta = _meta("twin_yspectra")
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
    meta = _meta("twin_yspectra")
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
    meta = _meta("twin_yspectra")
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
    # The summed panel is one ratio of sums, not a sum of three ratios.
    assert tsm.reference_norm(series, "e_x", None) == float(want.sum())
    assert tsm.reference_norm(series, "e_x", 1) == float(want[1])
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
    for suffix in ("x", "z", "x0"):
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
    for suffix in ("x", "z", "x0"):
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
            series.field("e_x0")[0],
            0.5 * (first["e_x0"][0] + second["e_x0"][0]),
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


def test_main_renders_every_series() -> None:
    """``main()`` on a two-member set: both streams, every tag."""
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
        tags = sorted(p.name for p in out.iterdir())
        assert tags == sorted(
            [f"spectra_{p}_{m}" for p in ("e", "r") for m in ("x", "z", "x0")]
            + [f"budget_{m}" for m in ("x", "z", "x0")]
        )
        for tag in tags:
            frames = sorted((out / tag).glob("*.png"))
            assert [f.name for f in frames] == [
                f"{tag}_0.png",
                f"{tag}_2.png",
            ], tag
            assert all(f.stat().st_size > 0 for f in frames)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for case in tests:
        case()
        print(f"  PASS  {case.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
