r"""Offline reconstruction of the twin streams (``twin_postprocess``).

``scripts/twin_postprocess.py`` rebuilds ``twin.dat``,
``twin_yspectra.bin`` and ``twin_ybudget.bin`` from a member
directory's ``state{isnap}.tar`` / ``state{isnap}_twin.tar`` pairs.
Its whole claim is that it reproduces what the live driver wrote, so
every case here runs a real ``dnsjax-twin`` member with the streams
*on* and the snapshot cadence set to the stream cadence, then rebuilds
it and compares:

1. Default driving: every stored value of both binary streams and
   every ``twin.dat`` column is **bit-identical** to the live one on a
   shared time grid.  Exact equality, not a tolerance: the same jitted
   diagnostics run on states that round-trip through the snapshot
   bit-exactly.
2. Under a driving constraint the energy columns stay bit-identical
   while ``-dPds'_d`` / ``-dPdn'_d`` carry the wall-shear *inference*
   of the driving difference: **exact** at the ``t = t0`` row, where
   the driver computes the same inference through its own code path,
   and different at every later row, where the driver has the
   corrector's applied value instead.  The size of that difference is
   a wall-normal truncation residual and is not bounded here -- at an
   unconverged ``res.ny`` it exceeds the term itself
   (``tests/test_driving.py`` owns its convergence; the script's
   module docstring tabulates a measured ladder).
3. ``twin.e0 = 0``: the partner is an exact copy, so every rebuilt
   value is **exactly** zero -- the determinism guard, and the one
   configuration that also exercises the null-seed sidecar and the
   zero-energy branch of the built-in identity check.
4. ``bin_energies`` on the rebuilt ``twin_yspectra.bin`` reproduces the
   rebuilt ``twin.dat``'s ``E_dU`` / ``E_du1`` / ``E_du2``.
5. Selection (``--recon.stride`` / ``--recon.first`` / ``--recon.last``)
   thins the sample grid and is recorded as the sidecars' cadence.
6. Every refusal fires on a real input: pre-existing output, an output
   directory aliasing the run directory, a directory with no pairs, a
   pair whose halves disagree on ``(t, it)``, and an odd ``res.nz``.
7. ``mpirun -np 2 --dist.np0 2`` reproduces the single-process streams
   to machine epsilon -- the ``psum`` in ``_marginals_replicated``
   (whose `$\pm k_z$` fold spans the ``np0`` axis) and the
   ``DifferencePressure`` pytree argument, neither of which a
   single-process run can exercise.

Usage::

    uv run python tests/test_twin_postprocess.py
    uv run python tests/test_twin_postprocess.py --unit-only
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Single-device in-process setup: used only to *write* the parent
# snapshots the driver subprocesses start from (the bootstrap
# contract -- params final and JAX configured before any geometry
# import).
from dnsjax.bootstrap import configure_jax_platform  # noqa: E402

configure_jax_platform("cpu")

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
)

NX, NY, NZ = 8, 17, 8

update_parameters(
    Parameters(
        phys={"system": "plane-poiseuille", "re": 400.0},
        geo={"lx": 3.141592653589793, "lz": 1.5707963267948966},
        res={
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)
padded_res.set_padded_resolution(params)

import numpy as np  # noqa: E402
from _live import run_live  # noqa: E402

from dnsjax.analysis.twin import (  # noqa: E402
    bin_energies,
    read_twin,
    read_twin_ybudget,
    read_twin_yspectra,
)

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = str(_REPO / "scripts" / "twin_postprocess.py")
_SESSION = Path(tempfile.mkdtemp(prefix="twin_postprocess_"))

PARENT_T = 1.0
PARENT_IT = 100
E0 = 1e-6
DT = params.step.dt

#: Written by :func:`_build_parents`; the two differ only in the
#: ``[phys]`` driving knobs their embedded parameter dump records.
PARENT = _SESSION / "parent.tar"
PARENT_DRIVEN = _SESSION / "parent_driven.tar"

#: Members built once and reused across the cases below.
_MEMBERS: dict[str, Path] = {}


def _build_parents() -> None:
    """One random state, written under two driving configurations."""
    from dnsjax.ic.random_field import generate_random_state
    from dnsjax.snapshot import save_snapshot

    state = generate_random_state(0.1, 0.4, 1, params.init.random_mean_flow)
    save_snapshot(state, PARENT_T, PARENT_IT, PARENT, isnap=0)
    # Direct assignment to the singleton, the ``test_twin_driver``
    # ``[force]`` idiom: ``recorded_params_dump`` reads the live
    # ``params``, and no further ``update_parameters`` pass follows
    # that would re-materialize the flow defaults over it.
    params.phys.driving = "constant_bulk_velocity"
    params.phys.block_mean_spanwise_velocity = True
    save_snapshot(state, PARENT_T, PARENT_IT, PARENT_DRIVEN, isnap=0)
    params.phys.driving = "constant_pressure_gradient"
    params.phys.block_mean_spanwise_velocity = False


def _member(name: str, parent: Path, n_steps: int, extra: list[str]) -> Path:
    """A twin member with snapshots and streams on the same cadence."""
    if name in _MEMBERS:
        return _MEMBERS[name]
    workdir = _SESSION / name
    workdir.mkdir()
    run_live(
        [
            sys.executable,
            "-m",
            "dnsjax.twin",
            "--init.snapshot",
            str(parent),
            "--dist.platform",
            "cpu",
            "--twin.it_energy",
            "1",
            "--twin.it_yspectra",
            "1",
            "--twin.it_ybudget",
            "1",
            "--outs.it_snapshot",
            "1",
            "--outs.snapshot_save_initial",
            "True",
            "--outs.stats_precision",
            "17",
            "--stop.max_sim_time",
            str(n_steps * DT),
            "--stop.check_laminarization",
            "False",
            *extra,
        ],
        cwd=workdir,
        check=True,
        timeout=900,
    )
    _MEMBERS[name] = workdir
    return workdir


def _recon(
    member: Path, args: list[str] | None = None, expect: int = 0, np_count=1
) -> str:
    """Run the reconstruction script; return its combined output."""
    launcher = ["mpirun", "-np", str(np_count)] if np_count > 1 else []
    result = run_live(
        [
            *launcher,
            sys.executable,
            _SCRIPT,
            "--recon.dir",
            str(member),
            "--dist.platform",
            "cpu",
            "--outs.stats_precision",
            "17",
            *(args or []),
        ],
        timeout=900,
    )
    if result.returncode != expect:
        raise AssertionError(
            f"twin_postprocess exited {result.returncode}, expected "
            f"{expect}:\n"
            + "\n".join(result.stdout.splitlines()[-10:])
            + "\n"
            + "\n".join(result.stderr.splitlines()[-10:])
        )
    return result.stdout + result.stderr


def _assert_streams_identical(live: Path, rebuilt: Path) -> None:
    """Both binary streams equal, value for value, on a shared grid."""
    for reader in (read_twin_yspectra, read_twin_ybudget):
        a, b = reader(live), reader(rebuilt)
        assert np.array_equal(a.t, b.t), (reader.__name__, a.t, b.t)
        assert set(a.fields) == set(b.fields), set(a.fields) ^ set(b.fields)
        assert np.array_equal(a.y, b.y) and np.array_equal(
            a.y_weights, b.y_weights
        )
        assert np.array_equal(a.kz, b.kz) and np.array_equal(a.kx, b.kx)
        for key in a.fields:
            assert np.array_equal(a[key], b[key]), (
                f"{reader.__name__}: {key} differs, max |d| = "
                f"{np.abs(a[key] - b[key]).max():.3e}"
            )


def test_matches_live_streams() -> None:
    """Bit-identical rebuild of a default-driving member."""
    member = _member("plain", PARENT, 4, ["--twin.e0", str(E0)])
    _recon(member)
    out = member / "recon"
    _assert_streams_identical(member, out)

    live, rebuilt = read_twin(member).energies, read_twin(out).energies
    assert list(live) == list(rebuilt) == ["t", "E_d", "E_ref"], list(rebuilt)
    for key in rebuilt:
        assert np.array_equal(live[key], rebuilt[key]), key
    # 5 pairs: the IC pair plus one per step.
    assert len(rebuilt["t"]) == 5, rebuilt["t"]
    # The rebuilt directory is a member directory in its own right.
    assert (out / "twin.json").is_file()
    assert read_twin(out).meta["parent_t"] == PARENT_T
    meta = read_twin_yspectra(out).meta
    assert meta["it_yspectra"] == 1 and meta["includes_ref"] is True


def test_driving_columns_are_inferred() -> None:
    """Energies exact; the driving columns are the wall-shear inference."""
    member = _member(
        "driven",
        PARENT_DRIVEN,
        4,
        ["--twin.e0", str(E0), "--twin.bins", "True"],
    )
    stdout = _recon(member)
    out = member / "recon"
    _assert_streams_identical(member, out)

    live, rebuilt = read_twin(member).energies, read_twin(out).energies
    assert list(live) == list(rebuilt), (list(live), list(rebuilt))
    drive = [c for c in rebuilt if c.startswith("-dP")]
    assert drive == ["-dPdn'_d", "-dPds'_d"], drive
    assert "wall-shear *inference*" in stdout
    # ``--recon.bins`` unset adopts the member's twin.json.
    for key in rebuilt:
        if key in drive:
            continue
        assert np.array_equal(live[key], rebuilt[key]), key

    for col in drive:
        a, b = live[col], rebuilt[col]
        # The t0 row is the same inference in both streams.
        assert a[0] == b[0], (col, a[0], b[0])
        assert np.isfinite(b).all()
        # Afterwards the driver reports the corrector's *applied*
        # value, so the two must differ -- a column reproducing it
        # exactly would not be the inference this claims to be.  No
        # magnitude bound belongs here: the gap is the wall-normal
        # truncation residual, which at an unconverged ny exceeds the
        # term itself (``tests/test_driving.py`` measures it, and a
        # bound fitted to one draw is what that file's own docstring
        # warns against).  The exactness at ``t0`` is the sharp check:
        # the driver computed that row through its own code path.
        assert (b[1:] != a[1:]).all(), (col, a, b)


def test_bin_energies_round_trip() -> None:
    """The rebuilt marginals recover the rebuilt three-bin energies."""
    member = _member(
        "driven",
        PARENT_DRIVEN,
        4,
        ["--twin.e0", str(E0), "--twin.bins", "True"],
    )
    out = member / "recon"
    dat = read_twin(out).energies
    assert "E_dU" in dat, list(dat)
    bins = bin_energies(read_twin_yspectra(out))
    for key, value in bins.items():
        np.testing.assert_allclose(value, dat[key], rtol=1e-12)


def test_zero_perturbation_is_exactly_zero() -> None:
    """``twin.e0 = 0``: every rebuilt value is exactly zero."""
    member = _member("zero", PARENT, 2, ["--twin.e0", "0"])
    _recon(member)
    out = member / "recon"
    _assert_streams_identical(member, out)

    dat = read_twin(out).energies
    assert (dat["E_d"] == 0).all(), dat["E_d"]
    assert (dat["E_ref"] > 0).all(), dat["E_ref"]
    ys = read_twin_yspectra(out)
    for key in ("e_x", "e_z", "e_x0"):
        assert (ys[key] == 0).all(), key
    assert (ys["r_x"] > 0).any()
    yb = read_twin_ybudget(out)
    for key in yb.fields:
        assert (yb[key] == 0).all(), key
    # ``e0 = 0`` draws no perturbation, so twin.json records no seed.
    assert ys.meta["twin"]["seed"] is None, ys.meta["twin"]


def test_selection_thins_the_grid() -> None:
    """``stride`` / ``first`` / ``last`` select, and set the cadence."""
    member = _member("plain", PARENT, 4, ["--twin.e0", str(E0)])
    full = read_twin_yspectra(member / "recon")

    _recon(member, ["--recon.out", "stride2", "--recon.stride", "2"])
    thin = read_twin_yspectra(member / "stride2")
    assert np.array_equal(thin.t, full.t[::2]), (thin.t, full.t)
    assert thin.meta["it_yspectra"] == 2, thin.meta["it_yspectra"]
    for key in thin.fields:
        assert np.array_equal(thin[key], full[key][::2]), key

    _recon(
        member,
        ["--recon.out", "mid", "--recon.first", "1", "--recon.last", "3"],
    )
    mid = read_twin_yspectra(member / "mid")
    assert np.array_equal(mid.t, full.t[1:4]), (mid.t, full.t)


def test_guards() -> None:
    """Every refusal fires on a real input."""
    member = _member("plain", PARENT, 4, ["--twin.e0", str(E0)])

    # Pre-existing output is refused rather than appended to, and
    # --recon.overwrite replaces it.
    assert "already holds twin.dat" in _recon(member, expect=1)
    assert "replacing existing output" in _recon(
        member, ["--recon.overwrite", "True"]
    )

    # The run directory itself is never a legal target.
    assert "the run directory itself" in _recon(
        member, ["--recon.out", "."], expect=1
    )

    # A directory with no complete pair.
    empty = _SESSION / "empty"
    empty.mkdir(exist_ok=True)
    result = run_live(
        [sys.executable, _SCRIPT, "--recon.dir", str(empty)], timeout=300
    )
    assert result.returncode == 1, result.returncode
    assert "no complete state*.tar" in result.stdout + result.stderr

    # A pair whose halves sit at different (t, it): a crash between the
    # driver's two writes, not something to difference across.
    broken = _SESSION / "broken"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(member, broken, ignore=shutil.ignore_patterns("recon*"))
    shutil.copy2(
        broken / "state00003_twin.tar", broken / "state00002_twin.tar"
    )
    result = run_live(
        [sys.executable, _SCRIPT, "--recon.dir", str(broken)], timeout=300
    )
    assert result.returncode == 1, result.returncode
    assert "the pair is inconsistent" in result.stdout + result.stderr

    # An odd res.nz cannot fold the k_z axis (the [twin] rule, which
    # this script's own validate hook has to repeat).
    assert "even res.nz" in _recon(
        member, ["--recon.out", "odd", "--res.nz", "7"], expect=1
    )


def test_mpi_np2_matches_single_process() -> None:
    """A (2, 1) mesh reproduces the single-process streams."""
    member = _member("plain", PARENT, 4, ["--twin.e0", str(E0)])
    _recon(
        member,
        ["--recon.out", "np2", "--dist.np0", "2"],
        np_count=2,
    )
    single, multi = member / "recon", member / "np2"
    for reader in (read_twin_yspectra, read_twin_ybudget):
        a, b = reader(single), reader(multi)
        assert np.array_equal(a.t, b.t)
        for key in a.fields:
            scale = np.abs(a[key]).max()
            if scale == 0.0:
                assert (b[key] == 0).all(), key
                continue
            assert np.abs(a[key] - b[key]).max() / scale < 1e-12, key
    one, two = read_twin(single).energies, read_twin(multi).energies
    for key in one:
        np.testing.assert_allclose(one[key], two[key], rtol=1e-12, atol=0)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    unit_only = "--unit-only" in sys.argv[1:]
    if not unit_only and shutil.which("mpirun") is None:
        print("mpirun not on PATH; running the offline subset only.")
        unit_only = True

    tests = [
        v
        for k, v in list(globals().items())
        if k.startswith("test_")
        and (not unit_only or not k.startswith("test_mpi_"))
    ]
    try:
        _build_parents()
        for tfun in tests:
            tfun()
            print(f"  PASS  {tfun.__name__}")
    finally:
        shutil.rmtree(_SESSION, ignore_errors=True)
    print(f"\nAll {len(tests)} tests passed.")
