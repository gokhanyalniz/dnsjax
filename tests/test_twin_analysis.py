r"""Unit tests for the JAX-free twin analysis package
(``dnsjax.analysis.twin``) and the ``build-twin`` orchestration.

Everything runs on hand-written synthetic files (no solver, no JAX --
asserted): the ``.dat``/``twin.json`` readers with resume-seam
duplicates, the per-component budget sums, member-tree aggregation
(mean/std against direct NumPy; every alignment guard tripped on a
real bad input), the growth-rate fits against planted laws, the
``twin_spectra.bin`` reader (byte-exact round trip, truncated
trailing record, duplicate-timestamp seams, version floor,
decorrelation-ratio guards), the integral-length core against an
independently evaluated two-mode reference, and
``scripts/ensemble_setup.py build-twin`` (dry run leaves no tree;
the built tree's TOMLs / ``members.json`` / ``run_commands.txt`` are
consistent and feed ``aggregate_members`` end to end via synthetic
member streams).

Usage::

    uv run python tests/test_twin_analysis.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

sys.stdout.reconfigure(line_buffering=True)

from _live import run_live  # noqa: E402

from dnsjax.analysis.twin import (  # noqa: E402
    aggregate_members,
    budget_sums,
    closure_residuals,
    fit_exponential_rate,
    fit_linear_rate,
    integral_lengths_from_modes,
    read_dat,
    read_twin,
)
from dnsjax.analysis.twin.spectra import (  # noqa: E402
    decorrelation_ratio,
    read_twin_spectra,
)

assert "jax" not in sys.modules, "the twin analysis package must be JAX-free"

_REPO = Path(__file__).resolve().parent.parent

# ── Synthetic stream writers ─────────────────────────────────────────


def _write_dat(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a ``.dat`` stream in the driver's format (17 digits)."""
    names = list(columns)
    width = max(24, max(len(n) for n in names))
    # ``#``-commented header, the ``#`` eating one space of the first
    # column's padding (``dnsjax.__main__._write_dat_header``).
    lines = [
        "#"
        + " ".join(
            n.rjust(width - 1 if i == 0 else width)
            for i, n in enumerate(names)
        )
    ]
    n_rows = len(next(iter(columns.values())))
    for i in range(n_rows):
        lines.append(
            " ".join(f"{columns[n][i]:.16e}".rjust(width) for n in names)
        )
    path.write_text("\n".join(lines) + "\n")


def _energy_columns(
    t: np.ndarray, scale: float = 1.0
) -> dict[str, np.ndarray]:
    cols = {"t": t}
    for i, name in enumerate(
        ("E_d", "E_dU", "E_du1", "E_du1_x", "E_du1_y", "E_du1_z", "E_du2")
    ):
        cols[name] = scale * (i + 1) * (1.0 + t - t[0])
    cols["E_ref"] = np.full_like(t, 5e-3)
    return cols


#: The budget stream's ``(a, b, c)`` layout, mirroring ``_PRODUCTION``
#: / ``_TRANSPORT`` in :mod:`dnsjax.twin.diagnostics` (the column
#: *names* are what the readers key on, so they must match exactly).
_TRIPLES = {
    "dU": [("dU", "rU"), ("du1", "ru1"), ("du2", "ru2")],
    "du1": [("du1", "rU"), ("dU", "ru1"), ("du1", "ru1"), ("du2", "ru2")],
    "du2": [
        ("dU", "ru2"),
        ("du1", "ru2"),
        ("du2", "rU"),
        ("du2", "ru1"),
        ("du2", "ru2"),
    ],
}
_TRANSPORTS = {
    "dU": [("ru1", "du1"), ("du1", "du1"), ("ru2", "du2"), ("du2", "du2")],
    "du1": [("ru1", "dU"), ("du1", "dU"), ("ru2", "du2"), ("du2", "du2")],
    "du2": [("ru2", "dU"), ("du2", "dU"), ("ru2", "du1"), ("du2", "du1")],
}


def _budget_columns(t: np.ndarray, scale: float = 1.0) -> dict:
    triples, transports = _TRIPLES, _TRANSPORTS
    cols = {"t": t}
    k = 0
    for a, pairs in triples.items():
        for b, c in pairs:
            k += 1
            cols[f"P_{a}({b},{c})"] = scale * k * np.ones_like(t)
    for a, pairs in transports.items():
        for b, c in pairs:
            k += 1
            cols[f"T_{a}({b},{c})"] = scale * k * np.ones_like(t)
    for x in ("dU", "du1", "du2"):
        k += 1
        cols[f"eps_{x}"] = scale * k * np.ones_like(t)
    p_names = [n for n in cols if n.startswith("P_")]
    t_names = [n for n in cols if n.startswith("T_")]
    cols["P_tot"] = sum(cols[n] for n in p_names)
    cols["T_tot"] = sum(cols[n] for n in t_names)
    cols["eps_tot"] = sum(cols[f"eps_{x}"] for x in ("dU", "du1", "du2"))
    return cols


def _write_member(
    mdir: Path,
    parent_t: float,
    n: int = 11,
    dt: float = 0.01,
    scale: float = 1.0,
    budget: bool = True,
    seam: bool = False,
) -> None:
    mdir.mkdir(parents=True, exist_ok=True)
    t = parent_t + dt * np.arange(n)
    cols = _energy_columns(t, scale)
    if seam:  # duplicate one interior sample (a resume seam)
        cols = {k: np.insert(v, 5, v[5]) for k, v in cols.items()}
    _write_dat(mdir / "twin.dat", cols)
    if budget:
        tb = parent_t + 5 * dt * np.arange((n - 1) // 5 + 1)
        _write_dat(mdir / "twin_budget.dat", _budget_columns(tb, scale))
    (mdir / "twin.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "system": "plane-couette",
                "e0": 1e-6,
                "seed": 3,
                "smoothness": 0.4,
                "it_energy": 1,
                "it_budget": 5 if budget else None,
                "it_spectra": None,
                "dt": dt,
                "double_precision": True,
                "parent": "parent.tar",
                "parent_t": parent_t,
                "parent_it": 100,
            }
        )
    )


# ── Readers ──────────────────────────────────────────────────────────


def test_readers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mdir = Path(tmp) / "m0000"
        _write_member(mdir, parent_t=2.0, seam=True)
        series = read_twin(mdir)
        # The seam duplicate is dropped; t_rel starts at 0.
        assert series.t.shape[0] == 11
        assert np.unique(series.t).shape[0] == 11
        assert_allclose(series.t_rel[0], 0.0, atol=1e-12)
        assert series.meta["seed"] == 3
        assert series.budget is not None

        sums = budget_sums(series.budget)
        for x, n_p in (("dU", 3), ("du1", 4), ("du2", 5)):
            p_cols = [n for n in series.budget if n.startswith(f"P_{x}(")]
            assert len(p_cols) == n_p
            assert_allclose(
                sums[f"P_{x}"],
                sum(series.budget[n] for n in p_cols),
                rtol=1e-12,
            )
        raw = read_dat(mdir / "twin.dat")
        assert raw["t"].shape[0] == 12  # duplicates kept by read_dat

        # Version floor.
        meta = json.loads((mdir / "twin.json").read_text())
        meta["format_version"] = 0
        (mdir / "twin.json").write_text(json.dumps(meta))
        try:
            read_twin(mdir)
        except ValueError as exc:
            assert "format_version" in str(exc)
        else:
            raise AssertionError("version floor did not trip")
    print("series readers: OK")


# ── Budget closure ───────────────────────────────────────────────────


def _closing_budget_columns(
    t: np.ndarray, slopes: dict[str, float]
) -> dict[str, np.ndarray]:
    r"""A budget stream that closes *exactly* against known slopes.

    Per component, the production terms carry the whole balance
    (all equal), the four transport terms cancel in two pairs (so
    ``T_x`` and ``T_tot`` are identically zero while the individual
    terms stay `$O(1)$` -- the normaliser must not be zero), and
    ``eps_x`` is a fixed offset.  Solving
    `$n_p a_x - e_x = \dot{E}_x$` with `$e_x = 1$` fixes `$a_x$`.
    """
    cols: dict[str, np.ndarray] = {"t": t}
    ones = np.ones_like(t)
    for x, pairs in _TRIPLES.items():
        a = (slopes[x] + 1.0) / len(pairs)
        for b, c in pairs:
            cols[f"P_{x}({b},{c})"] = a * ones
        for j, (b, c) in enumerate(_TRANSPORTS[x]):
            # +v, -v, +2v, -2v: sums to zero, terms are O(1).
            sign = 1.0 if j % 2 == 0 else -1.0
            cols[f"T_{x}({b},{c})"] = sign * (1.0 + j // 2) * ones
        cols[f"eps_{x}"] = ones.copy()
    p_names = [n for n in cols if n.startswith("P_")]
    t_names = [n for n in cols if n.startswith("T_")]
    cols["P_tot"] = sum(cols[n] for n in p_names)
    cols["T_tot"] = sum(cols[n] for n in t_names)
    cols["eps_tot"] = sum(cols[f"eps_{x}"] for x in _TRIPLES)
    return cols


def _closing_member(
    mdir: Path, n: int = 51, dt: float = 0.01, it_budget: int = 5
) -> dict[str, np.ndarray]:
    """Write a member whose budget closes; return the budget columns."""
    mdir.mkdir(parents=True, exist_ok=True)
    t = 1.0 + dt * np.arange(n)
    _write_dat(mdir / "twin.dat", _energy_columns(t))
    # ``_energy_columns`` is linear in t, so the centred difference is
    # exact; the slopes are the (i + 1) factors of its E_* columns.
    slopes = {"dU": 2.0, "du1": 3.0, "du2": 7.0}
    tb = t[::it_budget]
    cols = _closing_budget_columns(tb, slopes)
    _write_dat(mdir / "twin_budget.dat", cols)
    return cols


def test_closure_residuals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. An exactly closing stream reads back as machine zero, on
        #    the interior budget samples only (the first and last of
        #    the 11 budget rows have no energy neighbour on one side).
        mdir = root / "exact"
        _closing_member(mdir)
        res = closure_residuals(read_twin(mdir))
        assert res.n_samples == 9, res.n_samples
        assert_allclose(res.dt, 0.01, rtol=1e-12)
        for name in ("dU", "du1", "du2", "T_tot"):
            assert res[name] < 1e-12, (name, res[name])

        # 2. Breaking one component's dissipation moves *only* that
        #    component: eps_du1 1 -> 1.3 unbalances a budget whose
        #    two sides are both 3.0, so the residual is 0.3 / 3.0.
        cols = _closing_member(root / "broken")
        cols["eps_du1"] = cols["eps_du1"] + 0.3
        cols["eps_tot"] = sum(cols[f"eps_{x}"] for x in ("dU", "du1", "du2"))
        _write_dat(root / "broken" / "twin_budget.dat", cols)
        res = closure_residuals(read_twin(root / "broken"))
        assert_allclose(res["du1"], 0.1, rtol=1e-9)
        assert res["dU"] < 1e-12 and res["du2"] < 1e-12
        assert res["T_tot"] < 1e-12

        # 3. T_tot is normalised by the largest individual transport
        #    term (2.0 here), not by the balance: breaking one pair by
        #    0.5 must read 0.25.
        cols = _closing_member(root / "transport")
        key = "T_du1(ru1,dU)"
        cols[key] = cols[key] + 0.5
        t_names = [n for n in cols if n.startswith("T_") and n != "T_tot"]
        cols["T_tot"] = sum(cols[n] for n in t_names)
        _write_dat(root / "transport" / "twin_budget.dat", cols)
        res = closure_residuals(read_twin(root / "transport"))
        assert_allclose(res["T_tot"], 0.25, rtol=1e-9)

        # 4. A budget row off the energy grid is skipped, not
        #    mis-paired: the driver writes an unconditional final row
        #    that need not be cadence-aligned.
        cols = _closing_member(root / "offgrid")
        cols = {
            k: np.append(v, v[-1] + (0.003 if k == "t" else 0.0))
            for k, v in cols.items()
        }
        _write_dat(root / "offgrid" / "twin_budget.dat", cols)
        res = closure_residuals(read_twin(root / "offgrid"))
        assert res.n_samples == 9, res.n_samples
        assert res["du1"] < 1e-12

        # 5. An energy row off the cadence grid is skipped, not
        #    allowed to shift the pairing.  The driver writes an
        #    unconditional final ``twin.dat`` row whatever the cadence,
        #    and a resume seam adds an interior one when
        #    ``outs.it_snapshot`` is not a multiple of
        #    ``twin.it_energy``; at ``it_energy > 1`` both are routine.
        #    Neither may change a single residual.
        for name, extra in (
            ("tail", [1.0 + 0.01 * 50 + 0.003]),  # trailing, off-lattice
            ("seam", [1.0 + 0.01 * 20 + 0.004]),  # interior, off-lattice
            ("both", [1.0 + 0.01 * 20 + 0.004, 1.0 + 0.01 * 50 + 0.003]),
        ):
            mdir = root / f"offgrid_{name}"
            _closing_member(mdir)
            t = np.sort(np.append(1.0 + 0.01 * np.arange(51), extra))
            _write_dat(mdir / "twin.dat", _energy_columns(t))
            res = closure_residuals(read_twin(mdir))
            assert res.n_samples == 9, (name, res.n_samples)
            assert_allclose(res.dt, 0.01, rtol=1e-12)
            for term in ("dU", "du1", "du2", "T_tot"):
                assert res[term] < 1e-12, (name, term, res[term])

        # 6. Guards: no budget stream, an energy stream with no single
        #    cadence at all, a missing term column, and cadences that
        #    never overlap.
        _closing_member(root / "nobudget")
        (root / "nobudget" / "twin_budget.dat").unlink()
        _expect_value_error(
            "no twin_budget.dat",
            lambda: closure_residuals(read_twin(root / "nobudget")),
        )

        mdir = root / "jitter"
        _closing_member(mdir)
        # Every gap different: no lattice fits more than two rows, so
        # this is the case that must still refuse (a genuinely
        # non-uniform stream, as opposed to a uniform one carrying a
        # couple of off-grid rows).
        t = 1.0 + np.cumsum(0.01 * (1.0 + 0.3 * np.arange(51)))
        _write_dat(mdir / "twin.dat", _energy_columns(t))
        _expect_value_error(
            "cadence grid",
            lambda: closure_residuals(read_twin(mdir)),
        )

        cols = _closing_member(root / "short")
        del cols["P_du2(du2,ru2)"]
        cols["P_tot"] = sum(
            cols[n] for n in cols if n.startswith("P_") and n != "P_tot"
        )
        _write_dat(root / "short" / "twin_budget.dat", cols)
        _expect_value_error(
            "found 11 and 12",
            lambda: closure_residuals(read_twin(root / "short")),
        )

        mdir = root / "disjoint"
        cols = _closing_member(mdir)
        cols["t"] = cols["t"] + 100.0
        _write_dat(mdir / "twin_budget.dat", cols)
        _expect_value_error(
            "do not overlap",
            lambda: closure_residuals(read_twin(mdir)),
        )
    print("budget-closure residuals: OK")


# ── Aggregation ──────────────────────────────────────────────────────


def _make_tree(tmp: Path, n_members: int = 3) -> Path:
    tree = tmp / "tree"
    members = []
    for k in range(n_members):
        _write_member(tree / f"m{k:04d}", parent_t=10.0 + k, scale=1.0 + k)
        members.append(
            {
                "dir": f"m{k:04d}",
                "seed": k + 1,
                "parent": "parent.tar",
                "parent_t": 10.0 + k,
                "t_end": 10.0 + k + 0.1,
            }
        )
    (tree / "members.json").write_text(
        json.dumps(
            {"kind": "twin", "e0": 1e-6, "horizon": 0.1, "members": members}
        )
    )
    return tree


def test_aggregation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = _make_tree(Path(tmp))
        out = Path(tmp) / "agg.npz"
        bundle = aggregate_members(tree, out)
        assert bundle["n_members"] == 3
        # Direct NumPy reference: member k scales its columns by
        # (1 + k), so the mean E_d row is mean(1+k) * base.
        base = _energy_columns(10.0 + 0.01 * np.arange(11))["E_d"] / 1.0
        assert_allclose(
            bundle["mean_E_d"], base * np.mean([1.0, 2.0, 3.0]), rtol=1e-12
        )
        assert_allclose(
            bundle["std_E_d"], base * np.std([1.0, 2.0, 3.0]), rtol=1e-12
        )
        assert "t_rel_budget" in bundle
        assert bundle["stack_budget_P_tot"].shape[0] == 3
        # allow_pickle=False on purpose: the bundle must round-trip
        # without pickle, so no entry may be an object array.
        loaded = np.load(out, allow_pickle=False)
        assert_allclose(loaded["mean_E_d"], bundle["mean_E_d"], rtol=0)
        assert list(loaded["columns"]) == list(bundle["columns"])

        # Guards, each on a real bad input.
        (tree / "m0001" / "twin_budget.dat").unlink()
        _expect_value_error(
            "some members carry", lambda: aggregate_members(tree)
        )
        _write_member(tree / "m0001", parent_t=11.0, n=7, scale=2.0)
        _expect_value_error(
            "time grid differs", lambda: aggregate_members(tree)
        )
        _write_member(tree / "m0001", parent_t=11.0, scale=2.0)
        cols = _energy_columns(11.0 + 0.01 * np.arange(11), 2.0)
        del cols["E_du2"]
        _write_dat(tree / "m0001" / "twin.dat", cols)
        _expect_value_error(
            "column set differs", lambda: aggregate_members(tree)
        )
        spec = json.loads((tree / "members.json").read_text())
        spec["kind"] = "other"
        (tree / "members.json").write_text(json.dumps(spec))
        _expect_value_error("not a twin tree", lambda: aggregate_members(tree))
    print("aggregation (+ guards): OK")


def _expect_value_error(fragment: str, thunk) -> None:
    try:
        thunk()
    except ValueError as exc:
        assert fragment in str(exc), f"{fragment!r} not in {exc}"
        return
    raise AssertionError(f"expected ValueError({fragment!r})")


# ── Fits ─────────────────────────────────────────────────────────────


def test_fits() -> None:
    t = np.linspace(0.0, 300.0, 601)
    lam, e0 = 0.025, 1e-10
    e_exp = e0 * np.exp(2.0 * lam * t)
    lam_fit, e0_fit, rms = fit_exponential_rate(t, e_exp, 50.0, 250.0)
    assert_allclose(lam_fit, lam, rtol=1e-12)
    assert_allclose(e0_fit, e0, rtol=1e-9)
    assert rms < 1e-12

    rate, a = 1.3e-5, 0.4
    e_lin = a + rate * t
    rate_fit, a_fit, rms_lin = fit_linear_rate(t, e_lin, 20.0, 280.0)
    assert_allclose(rate_fit, rate, rtol=1e-12)
    assert_allclose(a_fit, a, rtol=1e-12)
    assert rms_lin < 1e-12

    _expect_value_error(
        "fewer than 3", lambda: fit_linear_rate(t, e_lin, 500.0, 600.0)
    )
    print("growth-rate fits: OK")


# ── Spectra reader ───────────────────────────────────────────────────


def _write_spectra(
    directory: Path,
    t: np.ndarray,
    e_delta: np.ndarray,
    e_ref: np.ndarray | None,
    truncate_bytes: int = 0,
) -> None:
    n2, n3 = e_delta.shape[1:]
    fields = [("t", "<f8"), ("e_delta", "<f8", (n2, n3))]
    if e_ref is not None:
        fields.append(("e_ref", "<f8", (n2, n3)))
    rec = np.zeros(t.shape[0], dtype=np.dtype(fields))
    rec["t"] = t
    rec["e_delta"] = e_delta
    if e_ref is not None:
        rec["e_ref"] = e_ref
    raw = rec.tobytes()
    if truncate_bytes:
        raw = raw[:-truncate_bytes]
    (directory / "twin_spectra.bin").write_bytes(raw)
    (directory / "twin_spectra.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "system": "plane-couette",
                "n2": n2,
                "n3": n3,
                "kz_harmonics": [0, 1, 2, 3, -3, -2, -1][:n2],
                "kx_harmonics": list(range(n3)),
                "lx": 5.5,
                "lz": 3.77,
                "value_dtype": "<f8",
                "includes_ref": e_ref is not None,
                "it_spectra": 1,
                "dt": 0.01,
                "double_precision": True,
            }
        )
    )


def test_spectra_reader() -> None:
    rng = np.random.default_rng(1)
    t = np.array([0.0, 0.01, 0.01, 0.02])  # one seam duplicate
    e_delta = rng.random((4, 7, 4))
    e_ref = rng.random((4, 7, 4))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_spectra(tmp, t, e_delta, e_ref, truncate_bytes=13)
        data = read_twin_spectra(tmp)
        # The truncated 4th record is dropped, then the duplicate.
        assert data.t.shape[0] == 2
        assert_allclose(data.e_delta, e_delta[:2], rtol=0)
        assert_allclose(data.e_ref, e_ref[:2], rtol=0)
        assert_allclose(
            data.kz, (2 * np.pi / 3.77) * np.array([0, 1, 2, 3, -3, -2, -1])
        )

        ratio = decorrelation_ratio(data)
        assert_allclose(ratio, data.e_delta / (2 * data.e_ref), rtol=0)
        zeroed = data.e_ref.copy()
        zeroed[0, 0, 0] = 0.0
        _write_spectra(tmp, t[:2], e_delta[:2], zeroed[:2])
        with_zero = read_twin_spectra(tmp)
        r0 = decorrelation_ratio(with_zero)
        assert np.isnan(r0[0, 0, 0]) and np.isfinite(r0[1]).all()

        _write_spectra(tmp, t[:2], e_delta[:2], None)
        no_ref = read_twin_spectra(tmp)
        assert no_ref.e_ref is None
        _expect_value_error(
            "no reference spectra", lambda: decorrelation_ratio(no_ref)
        )

        meta = json.loads((tmp / "twin_spectra.json").read_text())
        meta["format_version"] = 0
        (tmp / "twin_spectra.json").write_text(json.dumps(meta))
        _expect_value_error("format_version", lambda: read_twin_spectra(tmp))
    print("spectra reader: OK")


# ── Integral lengths (core) ──────────────────────────────────────────


def test_integral_lengths_core() -> None:
    """Two-mode spectrum against an independent dense evaluation."""
    lz = 3.7699111843077517
    ny = 33
    y = np.cos(np.linspace(0.0, np.pi, ny))[::-1]  # CGL-like, [-1, 1]
    kz = (2 * np.pi / lz) * np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    du1 = np.zeros((3, ny, 6), dtype=complex)
    # Component 0: two modes with a y envelope; conjugate pairs so the
    # physical field is real.
    g = (1.0 - y**2) * np.exp(0.3 * y)
    for idx, amp in ((0, 0.8), (1, 0.4)):
        du1[0, :, idx] = amp * g
        du1[0, :, idx + 3] = amp * g  # the -m partner
    out = integral_lengths_from_modes(du1, y, kz, lz, y0=0.0)

    # Independent reference: dense correlation on fine grids with the
    # same first-zero-crossing convention.
    j0 = int(np.argmin(np.abs(y)))
    power = np.abs(du1[0, j0]) ** 2
    r = np.linspace(0.0, lz / 2, 200001)
    f_z = (power[None, :] * np.cos(np.outer(r, kz))).sum(axis=1)
    f_z /= f_z[0]
    stop = np.nonzero(f_z <= 0)[0][0]
    l_z_ref = np.trapezoid(f_z[: stop + 1], r[: stop + 1])
    assert_allclose(out["l_z"][0], l_z_ref, rtol=2e-3)

    # l_y: the correlation of a separable field g(y0)g(y) never
    # crosses zero for this positive envelope, so the integral runs
    # to the walls: mean of the two one-sided integrals of g(y)/g(y0).
    f_y = g / g[j0]
    sides = []
    for sel in (slice(j0, None), slice(j0, None, -1)):
        rr = np.abs(y[sel] - y[j0])
        sides.append(np.trapezoid(f_y[sel], rr))
    assert_allclose(out["l_y"][0], np.mean(sides), rtol=1e-12)
    assert np.isnan(out["l_z"][1]) and np.isnan(out["l_y"][2])
    assert out["variance"][0] > 0 and out["variance"][1] == 0
    print("integral-length core: OK")


# ── build-twin orchestration ─────────────────────────────────────────


def test_build_twin() -> None:
    """Real harvest -> build-twin -> synthetic runs -> aggregation.

    The parent snapshot is built by the ``test_twin_driver.py
    --build-parent`` worker in a subprocess (this process stays
    JAX-free); ``harvest`` and ``build-twin`` run as real CLIs; the
    built tree's TOMLs and index are checked, member streams are then
    synthesised in place (as if the runs had completed), and
    ``aggregate_members`` consumes the real ``members.json`` end to
    end.
    """
    import tomllib

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = tmp / "parent.tar"
        result = run_live(
            [
                sys.executable,
                str(_REPO / "tests" / "test_twin_driver.py"),
                "--build-parent",
                "8",
                "17",
                "8",
                str(parent),
            ],
            cwd=_REPO,
        )
        assert result.returncode == 0, "parent build worker failed"

        manifest = tmp / "manifest.json"
        setup = str(_REPO / "scripts" / "ensemble_setup.py")
        result = run_live(
            [
                sys.executable,
                setup,
                "harvest",
                "--run-dir",
                str(tmp),
                "--spacing",
                "0.5",
                "--n",
                "2",
                "--out",
                str(manifest),
            ],
            cwd=_REPO,
        )
        assert result.returncode == 0

        tree = tmp / "tree"
        build_args = [
            sys.executable,
            setup,
            "build-twin",
            "--manifest",
            str(manifest),
            "--tree",
            str(tree),
            "--e0",
            "1e-6",
            "--horizon",
            "0.05",
            "--members-per-snapshot",
            "2",
            "--it-budget",
            "5",
            # The per-state stream cadences: without these a member
            # records no stats.dat / stats_twin.dat at all.  A
            # sub-10 it_corrector must drag it_error_check down with
            # it or validate_parameters refuses the member.
            "--it-stats",
            "2",
            "--it-corrector",
            "3",
            # Pinned: an unset --seed-base is drawn from the OS entropy
            # pool (:mod:`dnsjax.seeding`), and the fan-out below is
            # asserted by value.
            "--seed-base",
            "1",
        ]
        result = run_live([*build_args, "--dry-run"], cwd=_REPO)
        assert result.returncode == 0 and not tree.exists()
        assert "m0001" in result.stdout

        result = run_live(build_args, cwd=_REPO)
        assert result.returncode == 0

        spec = json.loads((tree / "members.json").read_text())
        assert spec["kind"] == "twin" and len(spec["members"]) == 2
        assert (spec["it_stats"], spec["it_corrector"], spec["it_steps"]) == (
            2,
            3,
            None,
        ), spec
        assert [m["seed"] for m in spec["members"]] == [1, 2], (
            "build-twin must fan --seed-base out as seed-base + k"
        )
        lines = (tree / "run_commands.txt").read_text().splitlines()
        assert len(lines) == 2 and all("dnsjax-twin" in ln for ln in lines)
        for record in spec["members"]:
            with open(tree / record["dir"] / "parameters.toml", "rb") as fh:
                toml = tomllib.load(fh)
            assert toml["init"]["snapshot"] == str(parent.resolve())
            assert toml["twin"]["seed"] == record["seed"]
            assert toml["twin"]["e0"] == 1e-6
            assert toml["twin"]["it_budget"] == 5
            assert toml["outs"]["it_stats"] == 2
            assert toml["outs"]["it_corrector"] == 3
            assert toml["outs"]["it_error_check"] == 3
            assert "it_steps" not in toml["outs"], toml["outs"]
            assert toml["stop"]["max_sim_time"] == spec["horizon"]
            assert toml["stop"]["check_laminarization"] is False

        # Simulate the completed runs, then aggregate through the
        # real members.json.
        for record in spec["members"]:
            _write_member(
                tree / record["dir"],
                parent_t=record["parent_t"],
                n=6,
                scale=record["seed"],
            )
        bundle = aggregate_members(tree)
        assert bundle["n_members"] == 2
        base = _energy_columns(1.0 + 0.01 * np.arange(6))["E_d"]
        assert_allclose(bundle["mean_E_d"], base * 1.5, rtol=1e-12)
    print("build-twin (harvest, dry run, tree, aggregation): OK")


# ── Wall-normal-resolved streams ─────────────────────────────────────

NY, N_KZ, N_KX = 5, 4, 3


def _y_sidecar(extra: dict) -> dict:
    """The keys both wall-normal-resolved sidecars share."""
    return {
        "system": "plane-couette",
        "ny": NY,
        "n_kz": N_KZ,
        "n_kx": N_KX,
        "kz_harmonics": list(range(N_KZ)),
        "kx_harmonics": list(range(N_KX)),
        "lx": 5.5,
        "lz": 3.77,
        "y": list(np.linspace(-1.0, 1.0, NY)),
        "y_weights": [0.1, 0.4, 0.5, 0.4, 0.6],
        "volume_fac": 2.0,
        "value_dtype": "<f8",
        "dt": 0.01,
        "double_precision": True,
        **extra,
    }


def _write_y_stream(
    directory: Path,
    stem: str,
    t: np.ndarray,
    fields: list[tuple[str, tuple[int, ...]]],
    values: dict[str, np.ndarray],
    sidecar: dict,
    truncate_bytes: int = 0,
) -> None:
    dtype = np.dtype([("t", "<f8")] + [(n, "<f8", sh) for n, sh in fields])
    rec = np.zeros(t.shape[0], dtype=dtype)
    rec["t"] = t
    for name, _ in fields:
        rec[name] = values[name]
    raw = rec.tobytes()
    if truncate_bytes:
        raw = raw[:-truncate_bytes]
    (directory / f"{stem}.bin").write_bytes(raw)
    (directory / f"{stem}.json").write_text(json.dumps(sidecar))


def test_yspectra_reader() -> None:
    """``twin_yspectra`` round trip, seam drop, truncation, floor."""
    from dnsjax.analysis.twin import (
        bin_energies,
        integrate_y,
        read_twin_yspectra,
    )

    rng = np.random.default_rng(3)
    t = np.array([0.0, 0.01, 0.01, 0.02])  # one seam duplicate
    fields = [
        (f"{p}_{suf}", (3, NY, n))
        for p in ("e", "r")
        for suf, n in (("x", N_KZ), ("z", N_KX), ("x0", N_KZ))
    ]
    values = {n: rng.random((4, *sh)) for n, sh in fields}
    # ``e_x0`` is a sub-part of ``e_x`` in the real stream; make it so
    # here too, or ``bin_energies`` would report a negative bin.
    values["e_x0"] = 0.25 * values["e_x"]
    sidecar = _y_sidecar(
        {"format_version": 1, "includes_ref": True, "it_yspectra": 1}
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_y_stream(d, "twin_yspectra", t, fields, values, sidecar)
        data = read_twin_yspectra(d)
        assert data.t.tolist() == [0.0, 0.01, 0.02]
        for name, _ in fields:
            assert_allclose(
                data[name], values[name][[0, 1, 3]], rtol=0, atol=0
            )
        w = np.asarray(sidecar["y_weights"])
        assert_allclose(
            integrate_y(data, "e_x"),
            np.einsum("j,tcjk->tck", w, values["e_x"][[0, 1, 3]]),
            rtol=0,
            atol=0,
        )
        got = bin_energies(data)
        x0 = np.einsum("j,tcjk->tk", w, values["e_x0"][[0, 1, 3]])
        x = np.einsum("j,tcjk->tk", w, values["e_x"][[0, 1, 3]])
        # Summation order differs from the reference, so eps, not bits.
        assert_allclose(got["E_dU"], x0[:, 0], rtol=1e-14)
        assert_allclose(got["E_du1"], x0[:, 1:].sum(axis=1), rtol=1e-14)
        assert_allclose(got["E_du2"], (x - x0).sum(axis=1), rtol=1e-14)

        # A partial trailing record is dropped, not misread.
        _write_y_stream(
            d, "twin_yspectra", t, fields, values, sidecar, truncate_bytes=9
        )
        assert read_twin_yspectra(d).t.tolist() == [0.0, 0.01]

        # Version floor.
        _write_y_stream(
            d,
            "twin_yspectra",
            t,
            fields,
            values,
            sidecar | {"format_version": 0},
        )
        _expect_value_error("format_version", lambda: read_twin_yspectra(d))

        # A .bin without its sidecar.
        (d / "twin_yspectra.json").unlink()
        try:
            read_twin_yspectra(d)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a sidecar-less .bin was accepted")
    print("twin_yspectra reader (round trip, seam, truncation, floor): OK")


def test_fluctuation_energy() -> None:
    """The `(0,0)`-free total, and that both marginals agree on it."""
    from dnsjax.analysis.twin import fluctuation_energy

    rng = np.random.default_rng(11)
    w = np.asarray(_y_sidecar({})["y_weights"])
    # A consistent pair of marginals: one non-negative mode energy
    # summed over the axis each of them contracts, and its k_x = 0
    # plane.  This is the structure the writer produces, so the two
    # marginals must report the same total.
    modes = rng.random((3, NY, N_KZ, N_KX))
    r_x = modes.sum(axis=-1)
    r_z = modes.sum(axis=-2)
    r_x0 = modes[..., 0]

    total = np.einsum("j,cjzx->c", w, modes)
    mean_mode = np.einsum("j,cj->c", w, modes[..., 0, 0])
    want = total - mean_mode
    assert_allclose(fluctuation_energy(r_x, r_x0, w), want, rtol=1e-14)
    assert_allclose(fluctuation_energy(r_z, r_x0, w), want, rtol=1e-14)

    # The mean mode is a real subtraction, and it is the (0,0) mode
    # alone -- not the whole k_z = 0 column, which r_x[..., 0] is.
    assert np.all(mean_mode > 0.0)
    wrong = total - np.einsum("j,cj->c", w, r_x[..., 0])
    assert not np.allclose(wrong, want)

    # Leading axes are free: one component alone gives a scalar.
    one = fluctuation_energy(r_x[0], r_x0[0], w)
    assert one.shape == ()
    assert_allclose(one, want[0], rtol=1e-14)
    print("fluctuation_energy (both marginals, (0,0) removal): OK")


def test_ybudget_reader() -> None:
    """``twin_ybudget`` round trip against the sidecar's term list."""
    from dnsjax.analysis.twin import integrate_y, read_twin_ybudget

    terms = ["P_U", "P_r", "T_ref", "T_self", "V", "eps", "Wp"]
    rng = np.random.default_rng(5)
    t = np.array([0.0, 0.02])
    fields = [
        (f"{term}_{suf}", (NY, n))
        for term in terms
        for suf, n in (("x", N_KZ), ("z", N_KX), ("x0", N_KZ))
    ]
    values = {n: rng.random((2, *sh)) for n, sh in fields}
    sidecar = _y_sidecar(
        {"format_version": 2, "terms": terms, "it_ybudget": 5}
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_y_stream(d, "twin_ybudget", t, fields, values, sidecar)
        data = read_twin_ybudget(d)
        assert len(data.fields) == 3 * len(terms)
        for name, _ in fields:
            assert_allclose(data[name], values[name], rtol=0, atol=0)
        w = np.asarray(sidecar["y_weights"])
        assert_allclose(
            integrate_y(data, "Wp_z"),
            np.einsum("j,tjk->tk", w, values["Wp_z"]),
            rtol=0,
            atol=0,
        )
        # A y grid of the wrong length is a hard error, not a reshape.
        _write_y_stream(
            d,
            "twin_ybudget",
            t,
            fields,
            values,
            sidecar | {"y_weights": [1.0, 2.0]},
        )
        _expect_value_error("y_weights", lambda: read_twin_ybudget(d))
    print("twin_ybudget reader: OK")


if __name__ == "__main__":
    test_readers()
    test_closure_residuals()
    test_aggregation()
    test_fits()
    test_spectra_reader()
    test_yspectra_reader()
    test_fluctuation_energy()
    test_ybudget_reader()
    test_integral_lengths_core()
    test_build_twin()
    print("All twin analysis tests passed.")
