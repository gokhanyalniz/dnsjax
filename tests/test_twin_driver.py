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
   component on a real run, as a **structural** guard -- term counts,
   ``*_tot`` sum-consistency, sample count, and order-of-magnitude
   bounds.  The convergence claim lives in
   ``tests/test_twin_budget.py``, which measures it on three-rung
   ladders across both Cartesian flows and every driving mode; the
   six-run seed sweep that moved it there is tabulated in
   ``test_budget_closure``'s own docstring.  ``--only closure`` (any
   test-name fragment) runs a subset; ``--seed N`` varies the
   partner's seed, which is what that sweep used.  ``--mean-free``
   overrides ``init.random_mean_flow`` off, so the partner carries no
   ``(0, 0)`` content: it isolates the partner's mean-mode content
   from everything else (the parent is unaffected either way,
   ``_build_parents`` taking the generator's mean-free default).  It
   was added to tell "the bounds moved because the perturbation
   energy redistributed" from "a budget term is missing"; the sweep
   answered that -- **neither**, the bounds were fitted to one draw,
   and the control fails at a different seed than the default path
   does.
8. Every driver-level guard fires on a real input: missing
   ``twin.e0``, a snapshot recording a ``[force]`` section, the three
   bad start-mode quadrants (partner without ``twin.json``,
   ``twin.json`` without partner, stale ``twin.dat``), a ``twin.json``
   mismatch on resume, a trajectory-defining change on resume, and an
   inconsistent ``(t, it)`` snapshot pair.

Usage::

    uv run python tests/test_twin_driver.py
    uv run python tests/test_twin_driver.py --only budget --mean-free
    uv run python tests/test_twin_driver.py --only budget --seed 5
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

#: ``--seed N`` overrides ``_twin_args``' default partner seed, so the
#: closure bounds below can be re-measured across seeds without editing
#: the file.  Same reasoning as ``MEAN_FREE`` for reading it here.
SEED = 3
if "--seed" in sys.argv:
    SEED = int(sys.argv[sys.argv.index("--seed") + 1])

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

from dnsjax.analysis.twin.series import (  # noqa: E402
    ClosureResiduals,
    closure_residuals,
    read_dat,
    read_twin,
)
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
#: ``twin.dat`` columns without ``twin.bins`` (the default) and with
#: it.  The set is the sorted keys of the ``twin_energies`` dict, so a
#: column added there shows up here first.
TWIN_COLS = ["t", "E_d", "E_ref"]
TWIN_COLS_BINNED = [
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


def _twin_args(
    t_end: float,
    seed: int = SEED,
    e0: float = E0,
    t_start: float = PARENT_T,
) -> list[str]:
    """Driver arguments running from *t_start* to *t_end*.

    ``stop.max_sim_time`` is the horizon relative to the run's own
    initial condition, so a resume passes the resumed pair's clock as
    *t_start* (``t_start + (t_end - t_start)`` is exact in binary
    floating point for these values, so the stop time -- and with it
    the sample count every stream assertion below counts -- is the
    same whichever launch produced it).
    """
    args = [
        "--init.snapshot",
        str(PARENT),
        "--twin.e0",
        repr(e0),
        "--twin.seed",
        str(seed),
        "--stop.max_sim_time",
        repr(t_end - t_start),
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
            for name in TWIN_COLS[1:-1]:  # the difference columns
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
        # IC pair is 00000).  The horizon is relative, so the child
        # asks for what is left of it.
        resume_args = list(_twin_args(t_end, t_start=t_mid))
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
    r"""The multi-process path, binary streams included.

    The stream cadences are on here and in no other multi-process
    row, because a second *process* is the only thing that tells
    these paths apart from the single-process ones: every array a
    stream writes is assembled by a ``psum`` over both mesh axes so
    that rank 0 can pull it to the host, and every global array a
    jitted diagnostic reads must arrive as an *argument*.  Both are
    free on one process, where nothing is non-addressable --
    ``twin_ybudget`` took its difference-pressure operator as a
    ``static_argnames`` entry, so the trace closed over that
    operator's sharded banded factors, and only a run like this one
    rejects it.
    """
    from dnsjax.analysis.twin import (
        integrate_y,
        read_twin_ybudget,
        read_twin_yspectra,
    )
    from dnsjax.analysis.twin.spectra import read_twin_spectra

    with tempfile.TemporaryDirectory() as tmp:
        _run_twin(
            tmp,
            [
                *_twin_args(1.03),
                "--twin.it_spectra",
                "1",
                "--twin.it_yspectra",
                "1",
                "--twin.it_ybudget",
                "1",
            ],
            np_count=2,
            np1=2,
        )
        cols = read_dat(Path(tmp) / "twin.dat")
        # The perturbation seed is device-count independent, so the
        # initial E_d matches the single-device runs' exactly (up to
        # the same cancellation floor).
        assert_allclose(cols["E_d"][0], E0, rtol=1e-10)
        assert (cols["E_d"] > 0).all()

        # The gathered marginals carry every device's mode block, not
        # just rank 0's: both integrate to twin.dat's own E_d.
        data = read_twin_yspectra(tmp)
        by_t = {round(t, 10): i for i, t in enumerate(np.round(cols["t"], 10))}
        assert data.t.shape[0] == cols["t"].shape[0]
        for k, t in enumerate(np.round(data.t, 10)):
            for marg in ("e_x", "e_z"):
                assert_allclose(
                    integrate_y(data, marg)[k].sum(),
                    cols["E_d"][by_t[t]],
                    rtol=1e-10,
                    err_msg=f"{marg} does not integrate to E_d at t={t}",
                )
        assert read_twin_ybudget(tmp).t.shape[0] == data.t.shape[0]

        # Same identity for the (kz, kx) plane, whose blocks are
        # gathered by the sibling collective.
        spec = read_twin_spectra(tmp)
        for k, t in enumerate(np.round(spec.t, 10)):
            assert_allclose(
                spec.e_delta[k].sum(), cols["E_d"][by_t[t]], rtol=1e-10
            )
    print("mpirun -np 2 (--dist.np1 2, every stream): OK")


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
        resume_args = list(_twin_args(t_end, t_start=t_mid))
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
        resume2 = list(_twin_args(1.15, t_start=t_end))
        resume2[1] = "state00002.tar"
        result = _run_twin(tmp, [*resume2, *spectra_args], expect=1)
        _expect_error(result, "without its twin_spectra.json sidecar")
    print("spectra stream (round trip, append, lengths): OK")


# ── Budget closure ───────────────────────────────────────────────────


def _closure_residuals(member: Path) -> ClosureResiduals:
    r"""Relative budget-closure residuals of one member directory.

    Thin wrapper over
    :func:`dnsjax.analysis.twin.series.closure_residuals` (which owns
    the definition and the structural validation) that additionally
    asserts the exact sum-consistency of the ``*_tot`` columns -- a
    writer-side property of the stream, not a physics one, so it
    belongs here rather than in the reader.
    """
    series = read_twin(member)
    bg = series.budget
    assert bg is not None, "no twin_budget.dat"
    p_all = [n for n in bg if n.startswith("P_") and n != "P_tot"]
    t_all = [n for n in bg if n.startswith("T_") and n != "T_tot"]
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
    resid = closure_residuals(series)
    assert resid.n_samples >= 5, f"only {resid.n_samples} budget samples"
    assert_allclose(resid.dt, DT, rtol=1e-9)
    return resid


def test_budget_closure() -> None:
    r"""`$dE_X/dt = P_X + T_X - \epsilon_X$` closes on a real run.

    The residual is spatial truncation (discrete pressure work
    against the interior divergence residual + the FD
    integration-by-parts error of the wall-normal transport;
    the dissipation uses the discrete-Laplacian form so the viscous
    part closes exactly -- see the ``twin/diagnostics.py`` "Dissipation
    form" note): the mean component closes to `$O(10^{-5})$` and the
    fluctuating ones to a few percent at 16x33x16.

    **This case does not assert refinement, and its bounds are loose
    on purpose.**  Both were tried, and a six-run sweep (``--seed``
    3/5/7 x default/``--mean-free``) showed neither survives the seed:

    ======  =========  =======  =======  =======
    seed    mode       du1 (c)  du1 (f)  T_tot (f)
    ======  =========  =======  =======  =======
    3       default    0.038    0.063    0.014
    3       mean-free  0.109    0.020    0.035
    5       default    0.262    0.015    0.022
    5       mean-free  0.082    0.044    0.121
    7       default    0.183    0.012    0.034
    7       mean-free  0.045    0.013    0.072
    ======  =========  =======  =======  =======

    Fine ``du1`` spans 0.012 to 0.063 -- a 5.1x swing straddling the
    old 0.04 bound in **both** modes, so the control was never clean
    either; ``fine < coarse`` fails in two of the six runs (seed 3
    default on ``du1``, seed 5 mean-free on ``T_tot``), again in both
    modes.  Two rungs one seed apart cannot separate convergence from
    the draw.

    The convergence claim therefore lives where it has the evidence:
    ``tests/test_twin_budget.py`` asserts it on three-rung ladders at
    ``fd_order = 8`` off *stepped* parents, where it holds for all
    fifteen (configuration, seed) combinations.  What stays here is
    the cheap structural guard -- term counts, ``*_tot``
    sum-consistency, sample count, and an order-of-magnitude bound
    that an actually missing term would still break.

    Run with ``--mean-free`` for the control (module docstring): the
    partner then carries no `$(0,0)$` content.
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

    resids: dict[str, ClosureResiduals] = {}
    for label, parent in (("coarse", PARENT), ("fine", parent16)):
        with tempfile.TemporaryDirectory() as tmp:
            args = _twin_args(1.5, e0=1e-4)
            args[1] = str(parent)
            _run_twin(
                tmp,
                [*args, "--twin.bins", "True", "--twin.it_budget", "5"],
            )
            resids[label] = _closure_residuals(Path(tmp))
    coarse, fine = resids["coarse"], resids["fine"]
    mode = "mean-free partner" if MEAN_FREE else "default partner"
    mode += f", seed {SEED}"
    print(
        f"closure residuals [{mode}]: "
        f"coarse={coarse.components} fine={fine.components}"
    )
    # Bounds: ~3x over the worst of a six-run sweep (twin.seed 3/5/7 x
    # both modes) -- see this function's docstring for why they are not
    # the ~2x they used to be, and why nothing here asserts refinement.
    assert coarse["dU"] < 1e-3 and fine["dU"] < 1e-3
    assert fine["du1"] < 0.20 and fine["du2"] < 0.10
    assert fine["T_tot"] < 0.35
    print("budget closure: OK")


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

    # An odd res.nz with a folded-k_z stream: refused at parse, not
    # left to fail as a shape error inside ``_fold_kz`` mid-run.  At
    # odd nz the stored band is asymmetric (the highest negative mode
    # has no positive partner), so the fold is genuinely undefined --
    # and only the (y, k) streams fold, so this must *not* refuse a
    # run without them.
    with tempfile.TemporaryDirectory() as tmp:
        odd = ["--res.nz", "7", "--init.force_resume", "True"]
        for stream in ("--twin.it_yspectra", "--twin.it_ybudget"):
            result = _run_twin(
                tmp, [*_twin_args(1.02), *odd, stream, "1"], expect=1
            )
            _expect_error(result, "need an even res.nz")
        # Same odd nz, no folded stream: parses (and then runs).
        _run_twin(tmp, [*_twin_args(1.02), *odd])

    # Resume guards need a real member directory.
    with tempfile.TemporaryDirectory() as tmp:
        _run_twin(tmp, _twin_args(1.02))
        resume_args = list(_twin_args(1.04, t_start=1.02))
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


def test_yspectra_streams() -> None:
    r"""``twin_yspectra.bin`` / ``twin_ybudget.bin`` end to end.

    A fresh run plus a paired resume, then the identities that make
    these streams a strict refinement of the old diagnostics: both
    marginals integrate to ``twin.dat``'s ``E_d``; the three-bin
    energies come back out of the `$k_x = 0$` plane and partition
    ``E_d`` between them; and the budget's
    `$k$`-sums reproduce ``twin_budget.dat``'s ``P_tot`` / ``eps_tot``
    -- all through the binary round trip.  ``twin.bins`` is on here
    only so ``twin.dat`` carries the bin columns to check against; it
    is off in every other case in this file, which is the default.

    Then ``twin.spectra_ref = False``, which is a *static* flag on
    both spectra diagnostics rather than a write-time filter: the
    reference branch is not traced at all, so the run has to be
    exercised, not just the writer.
    """
    from dnsjax.analysis.twin import (
        bin_energies,
        integrate_y,
        read_twin_spectra,
        read_twin_ybudget,
        read_twin_yspectra,
    )

    t_mid, t_end = 1.05, 1.1
    with tempfile.TemporaryDirectory() as tmp:
        extra = [
            "--twin.bins",
            "True",
            "--twin.it_budget",
            "1",
            "--twin.it_yspectra",
            "1",
            "--twin.it_ybudget",
            "1",
        ]
        _run_twin(tmp, [*_twin_args(t_mid), *extra])
        resume_args = list(_twin_args(t_end, t_start=t_mid))
        resume_args[1] = "state00001.tar"
        _run_twin(tmp, [*resume_args, *extra])

        n = round((t_end - PARENT_T) / DT)
        data = read_twin_yspectra(tmp)
        assert data.t.shape[0] == n + 1  # seam duplicate dropped
        assert_allclose(
            data.t, PARENT_T + np.arange(n + 1) * DT, rtol=0, atol=1e-12
        )
        ny = params.res.ny
        assert data["e_x"].shape == (n + 1, 3, ny, 4)
        assert data["e_z"].shape == (n + 1, 3, ny, 4)
        assert "r_x" in data.fields  # twin.spectra_ref default

        cols = read_dat(Path(tmp) / "twin.dat")
        by_t = {round(t, 10): i for i, t in enumerate(np.round(cols["t"], 10))}
        bins = bin_energies(data)
        for k, t in enumerate(np.round(data.t, 10)):
            i = by_t[t]
            for marg in ("e_x", "e_z"):
                assert_allclose(
                    integrate_y(data, marg)[k].sum(),
                    cols["E_d"][i],
                    rtol=1e-10,
                    err_msg=f"{marg} does not integrate to E_d at t={t}",
                )
            for name in ("E_dU", "E_du1", "E_du2"):
                assert_allclose(
                    bins[name][k], cols[name][i], rtol=1e-9, atol=1e-30
                )
            # The three bins partition E_d (parse-precision limited).
            # Checked here rather than on a default run: twin.bins is
            # off by default, so these columns exist only in this case.
            assert_allclose(
                cols["E_dU"][i] + cols["E_du1"][i] + cols["E_du2"][i],
                cols["E_d"][i],
                rtol=1e-12,
                err_msg=f"the three bins do not partition E_d at t={t}",
            )

        budget = read_twin_ybudget(tmp)
        assert budget.t.shape[0] == n + 1
        assert budget.meta["terms"][0] == "P_U"
        bud = read_dat(Path(tmp) / "twin_budget.dat")
        b_by_t = {
            round(t, 10): i for i, t in enumerate(np.round(bud["t"], 10))
        }
        for k, t in enumerate(np.round(budget.t, 10)):
            i = b_by_t[t]
            p_sum = (
                integrate_y(budget, "P_U_x")[k].sum()
                + integrate_y(budget, "P_r_x")[k].sum()
            )
            assert_allclose(p_sum, bud["P_tot"][i], rtol=1e-9)
            assert_allclose(
                -integrate_y(budget, "V_z")[k].sum(),
                bud["eps_tot"][i],
                rtol=1e-9,
            )

        # ``twin.spectra_ref = False`` drops the reference half of
        # *both* spectra streams.  It is a static flag on the two
        # diagnostics, so this also pins that the programs still trace
        # and record with the reference branch absent -- the thing a
        # disk-only knob would not need testing for.
        with tempfile.TemporaryDirectory() as tmp2:
            _run_twin(
                tmp2,
                [
                    *_twin_args(t_mid),
                    "--twin.it_yspectra",
                    "1",
                    "--twin.it_spectra",
                    "1",
                    "--twin.spectra_ref",
                    "False",
                ],
            )
            lean = read_twin_yspectra(tmp2)
            assert lean.meta["includes_ref"] is False
            assert set(lean.fields) == {"e_x", "e_z", "e_x0"}, lean.fields
            assert read_twin_spectra(tmp2).e_ref is None
            # The kept half is unchanged by the flag.
            assert lean["e_x"].shape[1:] == (3, ny, 4)

        # A .bin without its sidecar is refused, not guessed at.
        (Path(tmp) / "twin_yspectra.json").unlink()
        resume2 = list(_twin_args(t_end + 0.05, t_start=t_end))
        resume2[1] = "state00002.tar"
        result = _run_twin(tmp, [*resume2, *extra], expect=1)
        _expect_error(result, "without its twin_yspectra.json sidecar")
    print("wall-normal-resolved streams (round trip, identities): OK")


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
        test_yspectra_streams,
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
