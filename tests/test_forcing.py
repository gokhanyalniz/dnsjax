r"""Runtime stochastic-forcing tests (``dnsjax.extensions.forcing``).

Offline part (default; the ``test_probes.py`` pattern: ``params``
mutated before importing ``sharding``), on 4 forced host CPU devices
with a ``(np0, np1) = (2, 2)`` Explicit mesh so both mode axes are
genuinely sharded:

1. ``build_mode_injector`` scatter-adds exactly the given columns at
   the static global modes (owners on every mesh position; bit-exact
   against a dense host reference; adds on top of existing content).
2. On a **cylindrical/annular** system the same scatter converts each
   column into the solver's decoupled `$u_\pm$` basis first, and the
   real-FFT conjugate partner -- a plain conjugate in the physical
   components the profiles are given in -- lands there conjugated
   *and* `$u_+ \leftrightarrow u_-$` swapped; the probe gather
   inverts the whole path back to the geometry-independent physical
   reference.
3. ``StochasticForcer.kick`` places ``amplitude * sum_j w_j p_j`` and
   the conjugate partner bit-exactly (reconstructed from the same
   seeded PRNG), and streams the coefficients to ``forcing.bin``
   (read back through ``response.ssi.read_forcing``).
4. Resume semantics: a second forcer against the same directory
   skips the recorded draws, so the coefficient stream continues the
   uninterrupted sequence; a tampered sidecar / sidecar-less binary
   is rejected.
5. Profile-bundle validation (grid/system/mode/channel checks) and
   the ``force`` extension's validate (all-or-none, range, mean
   mode, kick/probe alignment, wall-bounded only), dispatched
   through ``validate_parameters``.

MPI part (skipped with ``--unit-only`` or when ``mpirun`` is absent):
solver-integration runs in temporary directories on a real
transient-growth operator + controllability bundle for the same
tiny laminar plane-Poiseuille configuration,

- **forced-laminar trajectory prediction**: with a laminar start the
  DNS response to the kicks is linear, so the probe stream must
  match the superposition of exported-propagator responses to the
  recorded kicks -- a deterministic end-to-end check of the kick
  placement, the timing convention (pre-kick samples; the ``t = 0``
  sample is exactly zero), the coefficient log, and the operator
  export, closed to ~1e-4 relative; ``identify_ssi`` runs on the
  same directory as a smoke check.
- **resume continuation**: a run split in two by a snapshot resume
  reproduces the single-shot run's ``forcing.bin`` exactly (the PRNG
  skip) and, after dropping the duplicated seam sample, its
  ``probes.bin`` bit-exactly (no kick is lost or doubled).

Usage::

    uv run python tests/test_forcing.py [--unit-only]
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Mutate global ``params`` (and the ``force``/``probes`` extension
# singletons) before importing any dnsjax module that captures values
# from them (``sharding.Sharding`` does so at class definition time).
from dnsjax.extensions import force_params, probes_params  # noqa: E402
from dnsjax.parameters import derived_params, params  # noqa: E402

NY = 6
params.phys.system = "plane-poiseuille"
params.res.nx = 8  # 4 true kx modes; nx_spec = 4 -> 2 per np1 shard
params.res.ny = NY
params.res.nz = 8  # 7 true kz modes; nz_spec = 8 -> 4 per np0 shard
params.res.fd_order = 4
params.res.double_precision = True
params.dist.np0 = 2
params.dist.np1 = 2
# Forced modes: (5,0) needs the conjugate partner at (2,0) and lives
# on np0-shard 1; (3,1) has no partner and lives on shard (0,0).
FORCE_MODES = [(5, 0), (3, 1)]
M_CHANNELS = 2
AMPLITUDE = 0.05
SEED = 7
force_params.modes = ";".join(f"{a},{b}" for a, b in FORCE_MODES)
force_params.amplitude = AMPLITUDE
force_params.it_force = 2
force_params.seed = SEED
params.outs.nbuffer = 100

import numpy as np  # noqa: E402

derived_params.wall_normal_grid = [
    float(v) for v in np.linspace(1.0, -1.0, NY)
]

import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from _live import run_live  # noqa: E402
from numpy.testing import (  # noqa: E402
    assert_allclose,
    assert_array_equal,
)

from dnsjax.analysis.response.probes import read_probes  # noqa: E402
from dnsjax.analysis.response.ssi import read_forcing  # noqa: E402
from dnsjax.extensions.forcing import (  # noqa: E402
    StochasticForcer,
    build_mode_injector,
)
from dnsjax.extensions.probes import _component_labels  # noqa: E402
from dnsjax.parameters import validate_parameters  # noqa: E402
from dnsjax.snapshot import assemble_local_shards  # noqa: E402

N2_TRUE, N3_TRUE = params.res.nz - 1, params.res.nx // 2


def _zero_state():
    return assemble_local_shards(lambda buf, *args: None)


def _write_profiles(
    path: Path, seed: int = 11, system: str = "plane-poiseuille"
) -> np.ndarray:
    """Channel-profile npz for FORCE_MODES; returns the (K, m, 3, NY).

    Carries ``component_labels`` (the basis the rows are in), so the
    loader's basis check runs on its accepting branch too."""
    rng = np.random.default_rng(seed)
    arrs = rng.standard_normal(
        (len(FORCE_MODES), M_CHANNELS, 3, NY)
    ) + 1j * rng.standard_normal((len(FORCE_MODES), M_CHANNELS, 3, NY))
    np.savez(
        path,
        system=system,
        code_grid=np.asarray(derived_params.wall_normal_grid),
        component_labels=np.asarray(_component_labels(3)),
        **{
            f"profiles_{i2}_{i3}": arrs[k]
            for k, (i2, i3) in enumerate(FORCE_MODES)
        },
    )
    return arrs


def _expected_kick_cols(
    arrs: np.ndarray, draw: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Dense reference of one kick on the (divisibility-padded)
    sharded *shape*, with host arithmetic identical to the forcer's.
    The conjugate partner mirrors about the **true** mode count
    (``nz - 1``), untouched by the padding slots appended after it."""
    dense = np.zeros(shape, dtype=complex)
    coeff = (draw[..., 0] + 1j * draw[..., 1]) / np.sqrt(2.0)
    for k, (i2, i3) in enumerate(FORCE_MODES):
        prof = AMPLITUDE * (
            coeff[k] @ arrs[k].reshape(M_CHANNELS, -1)
        ).reshape(3, NY)
        dense[:, :, i2, i3] = prof
        if i3 == 0:
            # The reality condition in **physical** components: a
            # plain conjugate in every geometry.  This reference is
            # therefore geometry-independent; on a solver-basis
            # geometry it is what the state looks like after
            # converting back (``test_kick_solver_basis``).
            dense[:, :, N2_TRUE - i2, 0] = np.conj(prof)
    return dense


# ── Offline: sharded injector ────────────────────────────────────────


def test_injector_scatter() -> None:
    """The scatter-add places exactly the given columns (owners on
    every mesh position) and adds on top of existing content."""
    pairs = [(5, 0), (2, 0), (3, 1), (0, 2)]
    rng = np.random.default_rng(1)
    cols = rng.standard_normal((len(pairs), 3, NY)) + 1j * (
        rng.standard_normal((len(pairs), 3, NY))
    )
    inject = build_mode_injector(pairs)

    out = np.asarray(inject(_zero_state(), jax.device_put(cols)))
    dense = np.zeros_like(out)
    for k, (i2, i3) in enumerate(pairs):
        dense[:, :, i2, i3] += cols[k]
    assert_array_equal(out, dense)

    # Adding on top of existing content (fresh state: donation).
    base = np.asarray(inject(_zero_state(), jax.device_put(cols)))
    out2 = np.asarray(
        inject(
            inject(_zero_state(), jax.device_put(cols)),
            jax.device_put(cols),
        )
    )
    assert_array_equal(out2, 2.0 * base)


def test_kick_solver_basis() -> None:
    r"""Cylindrical/annular: the injected columns are converted into
    the solver's decoupled `$u_\pm$` basis, and the real-FFT
    conjugate partner lands there as the reality condition demands.

    Only ``params.phys.system`` selects the conversion (the state
    layout is geometry-independent), so scoping that field exercises
    the branch on this module's ``(2, 2)`` mesh.  Three statements:

    1. the scatter places ``to_pm_basis(col)``, not ``col``;
    2. a full kick's partner column is the primary conjugated **and**
       `$u_+ \leftrightarrow u_-$` swapped -- the swap is exactly why
       the conversion must follow the partner construction rather
       than precede it;
    3. the probe gather inverts the whole path: read back in physical
       components the kick is the geometry-independent reference.
    """
    from dnsjax.extensions.probes import build_mode_extractor
    from dnsjax.geometries.wall_bounded._base import to_pm_basis

    saved = params.phys.system
    params.phys.system = "pipe"  # any cylindrical / annular flow
    try:
        assert _component_labels(3) == ["u_z", "u_r", "u_theta"]

        pairs = [(5, 0), (2, 0), (3, 1), (0, 2)]
        rng = np.random.default_rng(3)
        cols = rng.standard_normal((len(pairs), 3, NY)) + 1j * (
            rng.standard_normal((len(pairs), 3, NY))
        )
        out = np.asarray(
            build_mode_injector(pairs)(_zero_state(), jax.device_put(cols))
        )
        dense = np.zeros_like(out)
        for k, (i2, i3) in enumerate(pairs):
            dense[:, :, i2, i3] += np.asarray(to_pm_basis(cols[k]))
        assert_array_equal(out, dense)
        assert not np.array_equal(out[:, :, 5, 0], cols[0])  # not identity

        with tempfile.TemporaryDirectory() as tmp:
            arrs = _write_profiles(Path(tmp) / "prof.npz", system="pipe")
            force_params.profiles = str(Path(tmp) / "prof.npz")
            forcer = StochasticForcer(_zero_state(), tmp)
            state = forcer.kick(_zero_state(), 0.0)
            arr = np.asarray(state)

            i2, i3 = FORCE_MODES[0]  # (5, 0): on the real-FFT plane
            assert i3 == 0
            primary = arr[:, :, i2, i3]
            partner = arr[:, :, N2_TRUE - i2, 0]
            assert_array_equal(partner, np.conj(primary)[[0, 2, 1]])

            draw0 = np.random.default_rng(SEED).standard_normal(
                (len(FORCE_MODES), M_CHANNELS, 2)
            )
            ref = _expected_kick_cols(arrs, draw0, arr.shape)
            got = np.asarray(build_mode_extractor(FORCE_MODES)(state))
            for k, (j2, j3) in enumerate(FORCE_MODES):
                assert_allclose(got[k], ref[:, :, j2, j3], rtol=0, atol=1e-15)
    finally:
        params.phys.system = saved


# ── Offline: forcer kick + coefficient stream ────────────────────────


def test_kick_bit_exact_and_stream() -> None:
    """One kick lands bit-exactly (incl. the conjugate partner), and
    the coefficient records round-trip through read_forcing."""
    with tempfile.TemporaryDirectory() as tmp:
        arrs = _write_profiles(Path(tmp) / "prof.npz")
        force_params.profiles = str(Path(tmp) / "prof.npz")
        forcer = StochasticForcer(_zero_state(), tmp)

        state_dev = forcer.kick(_zero_state(), 0.0)
        state = np.asarray(state_dev)  # host copy (kick donates)
        rng = np.random.default_rng(SEED)
        draw0 = rng.standard_normal((len(FORCE_MODES), M_CHANNELS, 2))
        assert_array_equal(
            state, _expected_kick_cols(arrs, draw0, state.shape)
        )

        state2 = np.asarray(forcer.kick(state_dev, 0.02))
        draw1 = rng.standard_normal((len(FORCE_MODES), M_CHANNELS, 2))
        assert_array_equal(
            state2,
            _expected_kick_cols(arrs, draw0, state.shape)
            + _expected_kick_cols(arrs, draw1, state.shape),
        )

        forcer.flush()
        data = read_forcing(tmp)
        assert data.t.tolist() == [0.0, 0.02]
        assert data.modes.tolist() == [list(m) for m in FORCE_MODES]
        # The records hold the CN(0,1) coefficients as applied.
        for k, draw in enumerate((draw0, draw1)):
            assert_array_equal(
                data.w[k],
                (draw[..., 0] + 1j * draw[..., 1]) / np.sqrt(2.0),
            )
        assert data.meta["amplitude"] == AMPLITUDE


def test_resume_skip_and_mismatch() -> None:
    """An appending forcer continues the uninterrupted coefficient
    sequence; a tampered sidecar / sidecar-less binary is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_profiles(Path(tmp) / "prof.npz")
        force_params.profiles = str(Path(tmp) / "prof.npz")

        forcer = StochasticForcer(_zero_state(), tmp)
        state = _zero_state()
        for k in range(3):
            state = forcer.kick(state, 0.02 * k)
        forcer.flush()

        # A fresh start that finds a leftover stream is refused: the
        # kick times would restart from t0 and duplicate the records
        # already in the file.
        try:
            StochasticForcer(_zero_state(), tmp)
        except ValueError as e:
            assert "fresh start" in str(e), e
        else:
            raise AssertionError("leftover stream accepted at it0 = 0")

        # The resumed forcer's first draw is the 4th of a fresh rng.
        # ``init.it0`` is what makes it a resume: the stream must hold
        # exactly it0 // it_force records, or it belongs to another
        # point of the trajectory.
        params.init.it0 = 3 * force_params.it_force
        try:
            forcer2 = StochasticForcer(_zero_state(), tmp)
            forcer2.kick(_zero_state(), 0.06)
            forcer2.flush()

            # Resuming off an *earlier* snapshot than the stream was
            # written up to is the failure this guards.
            params.init.it0 = 2 * force_params.it_force
            try:
                StochasticForcer(_zero_state(), tmp)
            except ValueError as e:
                assert "does not belong to this state" in str(e), e
            else:
                raise AssertionError("stale resume point accepted")
        finally:
            params.init.it0 = 0
        rng = np.random.default_rng(SEED)
        for _ in range(3):
            rng.standard_normal((len(FORCE_MODES), M_CHANNELS, 2))
        draw3 = rng.standard_normal((len(FORCE_MODES), M_CHANNELS, 2))
        data = read_forcing(tmp)
        assert data.w.shape[0] == 4
        assert_array_equal(
            data.w[3],
            (draw3[..., 0] + 1j * draw3[..., 1]) / np.sqrt(2.0),
        )

        # Tampered sidecar: hard error.
        sidecar = Path(tmp) / "forcing.json"
        sidecar.write_text(
            sidecar.read_text().replace(
                f'"amplitude": {AMPLITUDE}', '"amplitude": 0.1'
            )
        )
        try:
            StochasticForcer(_zero_state(), tmp)
        except SystemExit as e:
            assert "amplitude" in str(e)
        else:
            raise AssertionError("mismatched sidecar was accepted")

        sidecar.unlink()
        try:
            StochasticForcer(_zero_state(), tmp)
        except SystemExit as e:
            assert "sidecar" in str(e)
        else:
            raise AssertionError("sidecar-less forcing.bin was accepted")


def test_profile_bundle_validation() -> None:
    """Grid / system / mode-key / channel-count checks."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good_grid = np.asarray(derived_params.wall_normal_grid)
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((M_CHANNELS, 3, NY)) + 0j

        cases = {
            "grid": dict(
                system="plane-poiseuille",
                code_grid=good_grid + 1e-3,
                profiles_5_0=arr,
                profiles_3_1=arr,
            ),
            "system": dict(
                system="plane-couette",
                code_grid=good_grid,
                profiles_5_0=arr,
                profiles_3_1=arr,
            ),
            "profiles_3_1": dict(  # missing forced-mode key
                system="plane-poiseuille",
                code_grid=good_grid,
                profiles_5_0=arr,
            ),
        }
        for fragment, payload in cases.items():
            np.savez(tmp / "bad.npz", **payload)
            force_params.profiles = str(tmp / "bad.npz")
            try:
                StochasticForcer(_zero_state(), tmp)
            except SystemExit as e:
                assert fragment in str(e), (fragment, e)
            else:
                raise AssertionError(f"{fragment}: bad bundle accepted")

        # n_channels beyond the stored count.
        _write_profiles(tmp / "prof.npz")
        force_params.profiles = str(tmp / "prof.npz")
        force_params.n_channels = M_CHANNELS + 1
        try:
            StochasticForcer(_zero_state(), tmp)
        except SystemExit as e:
            assert "n_channels" in str(e)
        else:
            raise AssertionError("oversized n_channels accepted")
        finally:
            force_params.n_channels = None


# ── Offline: parameter validation ────────────────────────────────────


def test_validate_force_params() -> None:
    saved = (
        force_params.modes,
        force_params.profiles,
        force_params.amplitude,
        force_params.it_force,
        probes_params.modes,
        probes_params.it_probes,
        params.phys.system,
    )
    try:
        force_params.profiles = "prof.npz"
        validate_parameters()  # the module configuration is valid

        force_params.it_force = None  # partial force config
        _expect_value_error("together")
        force_params.it_force = 2

        force_params.modes = "7,0"  # i2 == nz - 1 out of range
        _expect_value_error("out of range")
        force_params.modes = "0,0"
        _expect_value_error("mean mode")
        force_params.modes = "5,0"

        # Adaptive dt: the kick statistics and their readers
        # hard-code the uniform interval it_force * dt.  (dt_max set
        # too, so the step-section validation passes and the
        # rejection genuinely comes from the force hook.)
        params.step.adaptive = True
        params.step.dt_max = 0.02
        try:
            _expect_value_error("fixed time step")
        finally:
            params.step.adaptive = False
            params.step.dt_max = None

        # Kick cadence must be a whole number of probe intervals.
        probes_params.modes = "5,0"
        probes_params.it_probes = 4
        force_params.it_force = 6
        _expect_value_error("multiple")
        probes_params.modes = None
        probes_params.it_probes = None
        force_params.it_force = 2

        params.phys.system = "kolmogorov"  # periodic: rejected
        _expect_value_error("wall-bounded")
        params.phys.system = "viscoelastic-dean"  # 9 components
        _expect_value_error("wall-bounded")
    finally:
        (
            force_params.modes,
            force_params.profiles,
            force_params.amplitude,
            force_params.it_force,
            probes_params.modes,
            probes_params.it_probes,
            params.phys.system,
        ) = saved


def _expect_value_error(fragment: str) -> None:
    try:
        validate_parameters()
    except ValueError as e:
        assert fragment in str(e), e
    else:
        raise AssertionError(
            f"validate_parameters accepted force config "
            f"{force_params.modes!r}/{force_params.it_force!r}"
        )


# ── MPI integration ──────────────────────────────────────────────────

# fd_order is pinned here (not left to the model default) because
# test_mpi_forced_laminar_prediction compares the DNS against the
# exported generator's expm: the TG export and the solver children
# must step with the same discrete operator.
MPI_RES = (
    "--res.nx",
    "4",
    "--res.nz",
    "4",
    "--res.ny",
    "15",
    "--res.fd_order",
    "4",
)
MPI_RE = ("--phys.re", "500")
MPI_DT = 0.01


def _run_cmd(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    result = run_live(cmd, **kw)
    if result.returncode != 0:
        raise AssertionError(
            f"{' '.join(str(c) for c in cmd)} failed "
            f"({result.returncode}):\n"
            + "\n".join(result.stdout.splitlines()[-15:])
            + "\n"
            + "\n".join(result.stderr.splitlines()[-15:])
        )
    return result


def _run_solver(workdir: Path, args: list[str]) -> None:
    """Launch ``mpirun -np 1 python -m dnsjax`` in *workdir*."""
    cmd = [
        "mpirun",
        "-np",
        "1",
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.platform",
        "cpu",
        "--stop.check_laminarization",
        "False",
        "--phys.system",
        "plane-poiseuille",
        "--init.start_from_laminar",
        "True",
        "--step.dt",
        str(MPI_DT),
        *MPI_RE,
        *MPI_RES,
        "--outs.it_stats",
        "10",
        "--probes.modes",
        "1,0",
        "--probes.it_probes",
        "1",
        "--force.modes",
        "1,0",
        "--force.amplitude",
        "1e-3",
        "--force.it_force",
        "2",
        "--force.seed",
        "3",
        *args,
    ]
    # The forced-host-device XLA_FLAGS of the offline part must not
    # leak into the solver children.
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    _run_cmd(cmd, cwd=workdir, env=env)


def _mpi_artifacts(tmp: Path) -> tuple[Path, Path]:
    """TG operator export + controllability bundle on the MPI grid.

    ``--tg.dt`` is pinned to the solver step (the exported generator
    is the log of the dt-propagator, so the solver run must step with
    the same dt)."""
    ny = 15
    y = np.cos(np.pi * np.arange(ny) / (ny - 1))
    with open(tmp / "lam.txt", "w") as f:
        for yi, ui in zip(y, 1.0 - y**2, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
    _run_cmd(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.transient_growth",
            "--tg.profile",
            str(tmp / "lam.txt"),
            "--tg.out_dir",
            str(tmp),
            "--tg.modes",
            "1,0",
            "--tg.nt",
            "5",
            "--tg.save_operator",
            "True",
            "--phys.system",
            "plane-poiseuille",
            *MPI_RE,
            *MPI_RES,
            "--tg.dt",
            str(MPI_DT),
        ],
        cwd=tmp,
        env={k: v for k, v in os.environ.items() if k != "XLA_FLAGS"},
    )
    op_npz = tmp / "lam_tg_op.npz"
    cont_npz = tmp / "cont.npz"
    _run_cmd(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.response.operator_tools",
            "--operator",
            str(op_npz),
            "--n-modes",
            "2",
            "--out",
            str(cont_npz),
        ]
    )
    return op_npz, cont_npz


def _predict_trajectory(
    run_dir: Path, op_npz: Path, cont_npz: Path
) -> tuple[np.ndarray, np.ndarray]:
    """(measured, predicted) operator-coordinate trajectories."""
    from scipy.linalg import expm

    from dnsjax.analysis.response import operator_tools as ot
    from dnsjax.analysis.response.ensemble import project_series

    probe = read_probes(run_dir)
    forcing = read_forcing(run_dir)
    op = ot.load_operator(op_npz, 1, 0)
    m = int(forcing.meta["n_channels"])
    p = ot.recover_basis(op, ot.load_modes_npz(cont_npz, 1, 0, m))
    eps = float(forcing.meta["amplitude"])

    a_meas = project_series(probe.u[:, probe.mode_index(1, 0)], op.T_proj)
    e_step = expm(MPI_DT * np.asarray(op.A))
    kick_at = {
        int(round((t - probe.t[0]) / MPI_DT)): forcing.w[k, 0]
        for k, t in enumerate(forcing.t)
    }
    x = np.zeros(op.A.shape[0], dtype=complex)
    a_pred = np.empty_like(a_meas)
    for n in range(len(probe.t)):
        a_pred[n] = x  # pre-kick sample
        if n in kick_at:
            x = x + eps * (p @ kick_at[n])  # recorded CN(0,1) coeffs
        x = e_step @ x
    return a_meas, a_pred


def test_mpi_forced_laminar_prediction() -> None:
    """The forced-laminar DNS response equals the exported-propagator
    superposition of the recorded kicks (deterministic end-to-end)."""
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        op_npz, cont_npz = _mpi_artifacts(tmp)
        run = tmp / "run"
        run.mkdir()
        _run_solver(
            run,
            [
                "--force.profiles",
                str(cont_npz),
                "--stop.max_sim_time",
                "0.4",
            ],
        )

        forcing = read_forcing(run)
        assert np.allclose(np.diff(forcing.t), 2 * MPI_DT, atol=1e-12)
        assert forcing.t[0] == 0.0

        a_meas, a_pred = _predict_trajectory(run, op_npz, cont_npz)
        # The t = 0 sample is pre-kick on a laminar start: exactly 0.
        assert np.abs(a_meas[0]).max() < 1e-13
        scale = np.abs(a_meas).max()
        assert scale > 0
        rel = np.abs(a_meas - a_pred).max() / scale
        assert rel < 1e-4, rel

        # identify_ssi runs end-to-end on the same directory (the
        # statistical quality at 20 kicks is not asserted; the
        # estimator has offline anchors).
        from dnsjax.analysis.response.ssi import identify_ssi

        result = identify_ssi(
            [run], 1, 0, op_npz, lags=[0.1, 0.2], demean=False
        )
        assert np.isfinite(result["L"]).all()
        assert np.isfinite(result["causality"])


def test_mpi_resume_continuation() -> None:
    """A snapshot-split forced run reproduces the single-shot run's
    kick and probe streams (PRNG skip; no lost/doubled kick)."""
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        _, cont_npz = _mpi_artifacts(tmp)
        prof_args = ["--force.profiles", str(cont_npz)]

        full = tmp / "full"
        full.mkdir()
        _run_solver(full, [*prof_args, "--stop.max_sim_time", "0.4"])

        split = tmp / "split"
        split.mkdir()
        _run_solver(split, [*prof_args, "--stop.max_sim_time", "0.2"])
        _run_solver(
            split,
            [
                *prof_args,
                "--stop.max_sim_time",
                "0.4",
                "--init.snapshot",
                "state00001.tar",
            ],
        )

        f_full, f_split = read_forcing(full), read_forcing(split)
        assert_array_equal(f_split.t, f_full.t)
        assert_array_equal(f_split.w, f_full.w)

        p_full, p_split = read_probes(full), read_probes(split)
        # The seam sample (t = 0.2) is recorded by both halves with
        # identical values (snapshot round-trip); the reader drops
        # the duplicate, so the streams compare directly.
        assert_array_equal(p_split.t, p_full.t)
        assert_array_equal(p_split.u, p_full.u)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    unit_only = "--unit-only" in sys.argv
    tests = [
        v
        for k, v in list(globals().items())
        if k.startswith("test_")
        and not (unit_only and k.startswith("test_mpi"))
    ]
    if not unit_only and shutil.which("mpirun") is None:
        tests = [t for t in tests if not t.__name__.startswith("test_mpi")]
        print("mpirun not found: skipping the MPI integration tests.")
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
