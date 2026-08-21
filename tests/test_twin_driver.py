r"""Integration tests for the twin-run driver (``dnsjax-twin``).

All cases launch the real driver via ``mpirun`` (the solver launch
contract; the whole script skips when ``mpirun`` is absent) against a
tiny plane-Couette MFU parent snapshot built in-process at module
scope, in temporary member directories:

1. Fresh start: ``twin.json`` records the member (seed / e0 / parent
   clock), ``twin.dat`` has the sorted column set, a uniform time
   grid from the inherited parent clock, first-row ``E_d == e0`` to
   the float cancellation floor, per-row component partition, and the
   IC + final snapshot *pairs* with matching ``(t, it)``.
2. ``twin.e0 = 0``: the partner is an exact copy stepped by the same
   jitted stepper, so every ``E_d`` row is exactly zero -- the
   determinism guard for the whole lockstep loop.
3. Paired restart: run to ``T1``, resume from the final pair to
   ``T2``; the concatenated ``twin.dat`` matches an uninterrupted run
   to ``T2`` row-for-row **exactly** (iterative-CN is stateless
   across steps, snapshots round-trip bit-exactly, single-device CPU
   stepping is deterministic), with one duplicated seam sample; the
   resume never re-perturbs.
4. ``mpirun -np 2 --dist.np1 2``: the multi-process path produces the
   same (device-count-independent) initial ``E_d``.
5. Non-finite guard: a deliberately exploding configuration exits
   with code 3 and one ``FATAL: non-finite`` line.
6. Spectra stream: per-record ``e_delta`` sums equal ``twin.dat``'s
   ``E_d`` through the binary round trip; a paired restart appends
   on a uniform grid (seam duplicate dropped by the reader); a
   sidecar-less ``.bin`` is refused; the final pair feeds
   ``integral_lengths`` (finite, domain-bounded).
7. Budget closure: `$dE_X/dt = P_X + T_X - \epsilon_X$` per
   component on a real run at two resolutions -- absolute bounds at
   the finer one and self-convergence of every residual (the
   residual is spatial truncation; a wrong or missing term would
   not shrink).  ``--only closure`` (any test-name fragment) runs a
   subset.  ``--mean-free`` is the **control** for that closure: it
   overrides ``init.random_mean_flow`` off, so the partner carries no
   ``(0, 0)`` content and the difference field is the pure
   fluctuating one the bounds here were originally measured on.  The
   parent is unaffected either way (``_build_parents`` takes the
   generator's mean-free default), so the switch isolates the
   partner.  Use it to tell "the bounds moved because the perturbation
   energy redistributed" from "a budget term is missing": only the
   latter survives the control.
8. Every driver-level guard fires on a real input: missing
   ``twin.e0``, a snapshot recording a ``[force]`` section, the three
   bad start-mode quadrants (partner without ``twin.json``,
   ``twin.json`` without partner, stale ``twin.dat``), a ``twin.json``
   mismatch on resume, a trajectory-defining change on resume, and an
   inconsistent ``(t, it)`` snapshot pair.

Usage::

    uv run python tests/test_twin_driver.py
    uv run python tests/test_twin_driver.py --only budget --mean-free
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Single-device in-process setup: only used to *build* the parent
# snapshots the driver subprocesses start from.  The singletons are
# captured per process, so a parent at a different resolution (the
# closure self-convergence pair) is built by re-invoking this script
# as a worker: ``--build-parent NX NY NZ OUT``.
from dnsjax.bootstrap import configure_jax_platform  # noqa: E402

configure_jax_platform("cpu")

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
)

#: Control switch (see the module docstring): force the partner's
#: perturbation mean-free, the pre-``init.random_mean_flow`` behaviour.
#: Read here rather than in ``main`` because ``_twin_args`` is module
#: level; it never reaches the ``--build-parent`` worker, which is
#: correct -- the parent does not depend on it.
MEAN_FREE = "--mean-free" in sys.argv

if "--build-parent" in sys.argv:
    _i = sys.argv.index("--build-parent")
    _NX, _NY, _NZ = (int(v) for v in sys.argv[_i + 1 : _i + 4])
    _OUT = Path(sys.argv[_i + 4])
else:
    _NX, _NY, _NZ = 8, 17, 8
    _OUT = None

update_parameters(
    Parameters(
        phys={"system": "plane-couette", "re": 400.0},
        geo={"lx": 5.497787143782138, "lz": 3.7699111843077517},
        res={
            "nx": _NX,
            "ny": _NY,
            "nz": _NZ,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)
padded_res.set_padded_resolution(params)

import numpy as np  # noqa: E402
from _live import run_live  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.analysis.twin.series import read_dat  # noqa: E402
from dnsjax.extensions import force_params, reset_extensions  # noqa: E402
from dnsjax.snapshot_meta import read_snapshot_meta  # noqa: E402

PARENT_T = 1.0
PARENT_IT = 100
DT = params.step.dt
E0 = 1e-6

_SESSION = Path(tempfile.mkdtemp(prefix="twin_driver_"))
PARENT = _SESSION / "parent.tar"
PARENT_FORCED = _SESSION / "parent_forced.tar"

#: The sorted twin.dat column set (t first; the rest is the
#: JIT-canonicalised sorted key order of ``twin_energies``).
TWIN_COLS = [
    "t",
    "E_d",
    "E_dU",
    "E_du1",
    "E_du1_x",
    "E_du1_y",
    "E_du1_z",
    "E_du2",
    "E_ref",
]


def _build_parents() -> None:
    """Parent snapshots: a plain one and one recording ``[force]``."""
    from dnsjax.ic.random_field import generate_random_state
    from dnsjax.snapshot import save_snapshot

    state = generate_random_state(0.1, 0.4, 1)
    if _OUT is not None:  # --build-parent worker invocation
        save_snapshot(state, PARENT_T, PARENT_IT, _OUT, isnap=0)
        return
    save_snapshot(state, PARENT_T, PARENT_IT, PARENT, isnap=0)

    # A snapshot whose embedded params carry a configured [force]
    # section (set directly on the singleton -- the recorded dump
    # reads it): resuming it must trip the driver's force guard.
    force_params.modes = "1,0"
    force_params.profiles = "unused.npz"
    force_params.amplitude = 0.1
    force_params.it_force = 1
    save_snapshot(state, PARENT_T, PARENT_IT, PARENT_FORCED, isnap=0)
    reset_extensions()


def _run_twin(
    workdir: str | Path,
    args: list[str],
    np_count: int = 1,
    np1: int = 1,
    expect: int = 0,
):
    """Launch ``mpirun -np N python -m dnsjax.twin`` in *workdir*."""
    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        sys.executable,
        "-m",
        "dnsjax.twin",
        "--dist.platform",
        "cpu",
        "--dist.np0",
        "1",
        "--dist.np1",
        str(np1),
        "--stop.check_laminarization",
        "False",
        "--outs.stats_precision",
        "17",
        *args,
    ]
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    result = run_live(cmd, cwd=workdir, env=env)
    if result.returncode != expect:
        raise AssertionError(
            f"dnsjax-twin exited {result.returncode}, expected {expect}:\n"
            + "\n".join(result.stdout.splitlines()[-15:])
            + "\n"
            + "\n".join(result.stderr.splitlines()[-15:])
        )
    return result


def _twin_args(horizon: float, seed: int = 3, e0: float = E0) -> list[str]:
    args = [
        "--init.snapshot",
        str(PARENT),
        "--twin.e0",
        repr(e0),
        "--twin.seed",
        str(seed),
        "--stop.max_sim_time",
        repr(horizon),
    ]
    if MEAN_FREE:
        # A CLI layer beats the snapshot layer, which carries the
        # parent's resolved (Cartesian default: on) value.
        args += ["--init.random_mean_flow", "False"]
    return args


def _expect_error(result, fragment: str) -> None:
    output = result.stdout + result.stderr
    assert fragment in output, (
        f"expected {fragment!r} in the driver output; tail:\n"
        + "\n".join(output.splitlines()[-15:])
    )


# ── Fresh start ──────────────────────────────────────────────────────


def test_fresh_start_e0_exact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _run_twin(tmp, [*_twin_args(1.1), "--outs.it_stats", "5"])
        tmp = Path(tmp)

        meta = json.loads((tmp / "twin.json").read_text())
        assert meta["seed"] == 3 and meta["e0"] == E0
        assert meta["parent_t"] == PARENT_T
        assert meta["parent_it"] == PARENT_IT

        cols = read_dat(tmp / "twin.dat")
        assert list(cols) == TWIN_COLS
        n = round(0.1 / DT)
        assert cols["t"].shape[0] == n + 1
        assert_allclose(
            cols["t"], PARENT_T + np.arange(n + 1) * DT, rtol=0, atol=1e-12
        )
        # E_d(t0) == e0 up to the (state1 + delta) - state1
        # cancellation floor.
        assert_allclose(cols["E_d"][0], E0, rtol=1e-10)
        # Per-row component partition (parse-precision limited).
        assert_allclose(
            cols["E_dU"] + cols["E_du1"] + cols["E_du2"],
            cols["E_d"],
            rtol=1e-12,
        )
        assert (tmp / "stats.dat").exists()

        # IC and final snapshot pairs, each internally consistent.
        for stem in ("state00000", "state00001"):
            ref = read_snapshot_meta(tmp / f"{stem}.tar")
            twin = read_snapshot_meta(tmp / f"{stem}_twin.tar")
            assert ref["t"] == twin["t"] and ref["it"] == twin["it"]
        assert read_snapshot_meta(tmp / "state00001.tar")["it"] == (
            PARENT_IT + n
        )
    print("fresh start (e0 exactness, grids, pairs): OK")


def test_zero_perturbation_bit_identity() -> None:
    # ``--outs.it_steps 1`` is load-bearing, not incidental: on a
    # recording step the *reference* takes
    # ``predict_and_fully_correct_measured`` while the partner takes
    # the plain variant -- two separately compiled programs, the one
    # place the lockstep loop does not literally run the same jitted
    # function on both states.  Without the flag ``do_record`` is
    # always false (``outs.it_steps`` defaults to ``None``) and this
    # guard never sees that path.
    for extra in ([], ["--outs.it_steps", "1"]):
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_twin(tmp, [*_twin_args(1.05, e0=0.0), *extra])
            assert "exact copy" in result.stdout
            cols = read_dat(Path(tmp) / "twin.dat")
            for name in TWIN_COLS[1:-1]:  # every E_d* column, not E_ref
                assert (cols[name] == 0.0).all(), (
                    f"{name} nonzero with {extra}: twin stepping is "
                    "not bit-identical"
                )
            assert (cols["E_ref"] > 0).all()
            if extra:
                assert (Path(tmp) / "steps.dat").exists()
    print("e0 = 0 bit-identity (plain + measured reference): OK")


# ── Paired restart ───────────────────────────────────────────────────


def test_paired_restart_continuity() -> None:
    t_mid, t_end = 1.05, 1.1
    with (
        tempfile.TemporaryDirectory() as straight,
        tempfile.TemporaryDirectory() as split,
    ):
        _run_twin(straight, _twin_args(t_end))
        _run_twin(split, _twin_args(t_mid))
        # Resume from the final pair written at t_mid (isnap 1: the
        # IC pair is 00000).
        resume_args = list(_twin_args(t_end))
        resume_args[1] = "state00001.tar"
        result = _run_twin(split, resume_args)
        assert "Resumed twin pair" in result.stdout
        assert "random perturbation" not in result.stdout

        ref = read_dat(Path(straight) / "twin.dat")
        got = read_dat(Path(split) / "twin.dat")
        # The split stream duplicates the seam sample (the parent's
        # final row and the child's t0 row hold the same state).
        seen: dict[float, int] = {}
        keep: list[int] = []
        for i, t in enumerate(np.round(got["t"], 10)):
            if t not in seen:
                seen[t] = i
                keep.append(i)
        assert len(got["t"]) == len(ref["t"]) + 1  # one seam duplicate
        for name in TWIN_COLS:
            assert_allclose(
                got[name][keep],
                ref[name],
                rtol=0,
                atol=0,
                err_msg=f"{name}: resumed stream differs from the "
                "uninterrupted run",
            )
        # The seam duplicate itself matches the row it duplicates.
        dup = [i for i in range(len(got["t"])) if i not in keep]
        assert len(dup) == 1
        i_dup = dup[0]
        i_orig = seen[float(np.round(got["t"], 10)[i_dup])]
        for name in TWIN_COLS:
            assert got[name][i_dup] == got[name][i_orig]
    print("paired restart continuity (bit-exact): OK")


# ── Multi-process ────────────────────────────────────────────────────


def test_np2_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _run_twin(tmp, _twin_args(1.03), np_count=2, np1=2)
        cols = read_dat(Path(tmp) / "twin.dat")
        # The perturbation seed is device-count independent, so the
        # initial E_d matches the single-device runs' exactly (up to
        # the same cancellation floor).
        assert_allclose(cols["E_d"][0], E0, rtol=1e-10)
        assert (cols["E_d"] > 0).all()
    print("mpirun -np 2 (--dist.np1 2): OK")


# ── Non-finite guard ─────────────────────────────────────────────────


def test_nan_guard_exit3() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_twin(
            tmp,
            [
                *_twin_args(1.3, e0=1e12),
                # Disarm the corrector-divergence stop so the state
                # explodes to non-finite instead of stopping cleanly.
                "--step.corrector_tolerance",
                "1e300",
                "--outs.nbuffer",
                "1",
                "--outs.it_error_check",
                "5",
            ],
            expect=3,
        )
        _expect_error(result, "FATAL: non-finite")
    print("non-finite guard (exit 3): OK")


# ── Spectra stream + integral lengths ────────────────────────────────


def test_spectra_stream() -> None:
    """Write, read back, cross-check, and append the spectra stream.

    Per-record ``e_delta`` sums equal the matching ``twin.dat``
    ``E_d`` rows; a paired restart appends with one seam duplicate
    (dropped by the reader) on a uniform time grid; a ``.bin``
    without its sidecar is a hard error; the final snapshot pair,
    addressed through ``partner_of``, feeds ``integral_lengths``
    (finite, positive, bounded by the domain) while a pair from two
    *different* writes is refused.
    """
    from dnsjax.analysis.twin.lengths import integral_lengths, partner_of
    from dnsjax.analysis.twin.spectra import (
        decorrelation_ratio,
        read_twin_spectra,
    )

    t_mid, t_end = 1.05, 1.1
    with tempfile.TemporaryDirectory() as tmp:
        spectra_args = ["--twin.it_spectra", "1"]
        _run_twin(tmp, [*_twin_args(t_mid), *spectra_args])
        resume_args = list(_twin_args(t_end))
        resume_args[1] = "state00001.tar"
        _run_twin(tmp, [*resume_args, *spectra_args])

        data = read_twin_spectra(tmp)
        n = round((t_end - PARENT_T) / DT)
        assert data.t.shape[0] == n + 1  # seam duplicate dropped
        assert_allclose(
            data.t, PARENT_T + np.arange(n + 1) * DT, rtol=0, atol=1e-12
        )
        assert data.e_delta.shape == (n + 1, 7, 4)
        assert data.e_ref is not None
        assert data.kz.shape == (7,) and data.kx.shape == (4,)

        # Cross-check against twin.dat (the sum identity, end to end
        # through the binary round trip).
        cols = read_dat(Path(tmp) / "twin.dat")
        e_by_t = dict(zip(np.round(cols["t"], 10), cols["E_d"], strict=True))
        for k, t in enumerate(np.round(data.t, 10)):
            assert_allclose(data.e_delta[k].sum(), e_by_t[t], rtol=1e-10)
        ratio = decorrelation_ratio(data)
        assert np.isfinite(ratio[np.asarray(data.e_ref) > 0]).all()

        # The final pair feeds the integral-length diagnostic.
        final = Path(tmp) / "state00002.tar"
        assert partner_of(final) == Path(tmp) / "state00002_twin.tar"
        lengths = integral_lengths(final, partner_of(final))
        lz = 3.7699111843077517
        assert (lengths["variance"] > 0).all()
        assert np.isfinite(lengths["l_z"]).all()
        assert (lengths["l_z"] > 0).all() and (lengths["l_z"] <= lz / 2).all()
        assert np.isfinite(lengths["l_y"]).all()
        assert (lengths["l_y"] > 0).all() and (lengths["l_y"] <= 2).all()

        # A pair from two different writes is refused: both snapshots
        # are real and readable, and the difference of two unrelated
        # states would otherwise pass for a difference field.
        try:
            integral_lengths(final, partner_of(Path(tmp) / "state00001.tar"))
        except ValueError as exc:
            assert "not at the same time" in str(exc), exc
        else:
            raise AssertionError(
                "integral_lengths accepted a mismatched snapshot pair"
            )

        # A stream file without its sidecar is refused loudly.
        (Path(tmp) / "twin_spectra.json").unlink()
        resume2 = list(_twin_args(1.15))
        resume2[1] = "state00002.tar"
        result = _run_twin(tmp, [*resume2, *spectra_args], expect=1)
        _expect_error(result, "without its twin_spectra.json sidecar")
    print("spectra stream (round trip, append, lengths): OK")


# ── Budget closure ───────────────────────────────────────────────────


def _closure_residuals(member: Path) -> dict[str, float]:
    r"""Relative budget-closure residuals of one member directory.

    For each component ``X``: the centered-difference `$dE_X/dt$`
    from ``twin.dat`` (``it_energy = 1``) against
    `$P_X + T_X - \epsilon_X$` from ``twin_budget.dat`` at the
    interior budget sample times, normalised by the largest of the
    two magnitudes; plus ``T_tot`` relative to the largest transport
    term.  Also asserts the exact sum-consistency of the ``*_tot``
    columns and the expected per-component term counts.
    """
    tw = read_dat(member / "twin.dat")
    bg = read_dat(member / "twin_budget.dat")
    p_all = [n for n in bg if n.startswith("P_") and n != "P_tot"]
    t_all = [n for n in bg if n.startswith("T_") and n != "T_tot"]
    assert len(p_all) == 12 and len(t_all) == 12
    assert_allclose(
        bg["P_tot"], sum(bg[n] for n in p_all), rtol=1e-10, atol=1e-300
    )
    assert_allclose(
        bg["T_tot"], sum(bg[n] for n in t_all), rtol=1e-10, atol=1e-300
    )
    assert_allclose(
        bg["eps_tot"],
        bg["eps_dU"] + bg["eps_du1"] + bg["eps_du2"],
        rtol=1e-12,
    )

    t_idx = {round(t, 10): i for i, t in enumerate(tw["t"])}
    out: dict[str, float] = {}
    n_p = {"dU": 3, "du1": 4, "du2": 5}
    for x in ("dU", "du1", "du2"):
        p_cols = [n for n in bg if n.startswith(f"P_{x}(")]
        t_cols = [n for n in bg if n.startswith(f"T_{x}(")]
        assert len(p_cols) == n_p[x] and len(t_cols) == 4
        pairs = []
        for k, tb in enumerate(bg["t"]):
            i = t_idx.get(round(tb, 10))
            if i is None or i == 0 or i + 1 >= len(tw["t"]):
                continue
            dedt = (tw[f"E_{x}"][i + 1] - tw[f"E_{x}"][i - 1]) / (2 * DT)
            rhs = (
                sum(bg[n][k] for n in p_cols)
                + sum(bg[n][k] for n in t_cols)
                - bg[f"eps_{x}"][k]
            )
            pairs.append((dedt, rhs))
        arr = np.array(pairs)
        assert arr.shape[0] >= 5
        out[x] = float(np.abs(arr[:, 0] - arr[:, 1]).max() / np.abs(arr).max())
    t_scale = np.abs(np.stack([bg[n] for n in t_all])).max()
    out["T_tot"] = float(np.abs(bg["T_tot"]).max() / t_scale)
    return out


def test_budget_closure() -> None:
    r"""`$dE_X/dt = P_X + T_X - \epsilon_X$` closes on a real run.

    The residual is spatial truncation (discrete pressure work
    against the interior divergence residual + the FD
    integration-by-parts error of the wall-normal transport;
    the dissipation uses the discrete-Laplacian form so the viscous
    part closes exactly -- see the ``twin/diagnostics.py`` "Dissipation
    form" note): the mean component closes to `$O(10^{-5})$`, the
    fluctuating components to a few percent at 16x33x16, and every
    residual *decreases* from 8x17x8 to 16x33x16 (self-convergence:
    a systematically wrong or missing term would not).

    Run with ``--mean-free`` for the control (module docstring): the
    partner then carries no `$(0,0)$` content, which is the field
    composition these bounds were measured on.
    """
    parent16 = _SESSION / "parent16.tar"
    result = run_live(
        [
            sys.executable,
            __file__,
            "--build-parent",
            "16",
            "33",
            "16",
            str(parent16),
        ]
    )
    assert result.returncode == 0, "parent16 build failed"

    resids: dict[str, dict[str, float]] = {}
    for label, parent in (("coarse", PARENT), ("fine", parent16)):
        with tempfile.TemporaryDirectory() as tmp:
            args = _twin_args(1.5, e0=1e-4)
            args[1] = str(parent)
            _run_twin(tmp, [*args, "--twin.it_budget", "5"])
            resids[label] = _closure_residuals(Path(tmp))
    coarse, fine = resids["coarse"], resids["fine"]
    mode = "mean-free partner" if MEAN_FREE else "default partner"
    print(f"closure residuals [{mode}]: coarse={coarse} fine={fine}")
    # Bounds: ~2-3x margins over the measured values (deterministic
    # seeds; coarse 2e-5/6.0e-2/6.5e-2/15%, fine 3e-5/1.7e-2/3.6e-2/5%).
    assert coarse["dU"] < 1e-3 and fine["dU"] < 1e-3
    assert fine["du1"] < 0.04 and fine["du2"] < 0.08
    assert fine["T_tot"] < 0.10
    for x in ("du1", "du2", "T_tot"):
        assert fine[x] < coarse[x], (
            f"{x} closure residual did not shrink under refinement "
            f"({coarse[x]:.3e} -> {fine[x]:.3e})"
        )
    print("budget closure (+ self-convergence): OK")


# ── Driver-level guards ──────────────────────────────────────────────


def test_guards() -> None:
    # Missing twin.e0.
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_twin(tmp, ["--init.snapshot", str(PARENT)], expect=1)
        _expect_error(result, "configure the [twin] section")

    # A parent recording a configured [force] section.
    with tempfile.TemporaryDirectory() as tmp:
        args = _twin_args(1.02)
        args[1] = str(PARENT_FORCED)
        result = _run_twin(tmp, args, expect=1)
        _expect_error(result, "[force] section is not supported")

    # Partner without twin.json (e.g. a fresh start pointed at a twin
    # run's own output).
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(PARENT, Path(tmp) / "seed.tar")
        shutil.copy(PARENT, Path(tmp) / "seed_twin.tar")
        args = _twin_args(1.02)
        args[1] = "seed.tar"
        result = _run_twin(tmp, args, expect=1)
        _expect_error(result, "no twin.json member record")

    # twin.json without a partner for the pointed snapshot.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "twin.json").write_text("{}")
        result = _run_twin(tmp, _twin_args(1.02), expect=1)
        _expect_error(result, "already holds a twin trajectory")

    # Stale twin.dat without twin.json.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "twin.dat").write_text("stale\n")
        result = _run_twin(tmp, _twin_args(1.02), expect=1)
        _expect_error(result, "stale twin.dat")

    # Resume guards need a real member directory.
    with tempfile.TemporaryDirectory() as tmp:
        _run_twin(tmp, _twin_args(1.02))
        resume_args = list(_twin_args(1.04))
        resume_args[1] = "state00001.tar"

        # twin.json mismatch (a different seed).
        bad_seed = list(resume_args)
        bad_seed[bad_seed.index("3")] = "99"
        result = _run_twin(tmp, bad_seed, expect=1)
        _expect_error(result, "differs in: seed")

        # Trajectory-defining change on a paired resume.
        result = _run_twin(tmp, [*resume_args, "--phys.re", "500"], expect=1)
        _expect_error(result, "changed on a twin pair resume")

        # Inconsistent (t, it) pair: replace the partner with the IC
        # partner (earlier clock).  Last: it corrupts the directory.
        shutil.copy(
            Path(tmp) / "state00000_twin.tar",
            Path(tmp) / "state00001_twin.tar",
        )
        result = _run_twin(tmp, resume_args, expect=1)
        _expect_error(result, "the pair is inconsistent")
    print("driver-level guards: OK")


if __name__ == "__main__":
    if _OUT is not None:  # --build-parent worker invocation
        _build_parents()
        sys.exit(0)
    if shutil.which("mpirun") is None:
        print("mpirun not on PATH; skipping the twin driver tests.")
        sys.exit(0)
    _tests = [
        test_fresh_start_e0_exact,
        test_zero_perturbation_bit_identity,
        test_paired_restart_continuity,
        test_np2_run,
        test_nan_guard_exit3,
        test_spectra_stream,
        test_budget_closure,
        test_guards,
    ]
    if "--only" in sys.argv:
        frag = sys.argv[sys.argv.index("--only") + 1]
        _tests = [t for t in _tests if frag in t.__name__]
        assert _tests, f"--only {frag!r} matches no test"
    _build_parents()
    for _t in _tests:
        _t()
    shutil.rmtree(_SESSION, ignore_errors=True)
    print("All twin driver tests passed.")
