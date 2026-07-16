r"""Runtime spectral-mode probe stream tests (``dnsjax.probes``).

Offline part (default; the ``test_snapshot.py`` pattern: ``params``
mutated before importing ``sharding``), on 4 forced host CPU devices
with a ``(np0, np1) = (2, 2)`` Explicit mesh so **both** mode axes are
genuinely sharded:

1. ``build_mode_extractor`` returns the exact stored columns for
   modes owned by every mesh position -- including the mean mode
   ``(0,0)`` and modes whose owner is a non-zero device on one or
   both axes (the sharded-gather class that single-device runs cannot
   exercise).  The reference state is built per device via
   ``snapshot.assemble_local_shards`` from a closed-form
   ``(c, y, i2, i3)`` formula, so the comparison is bit-exact.
2. ``ProbeStream`` writer semantics: record/auto-flush/final-flush
   byte layout (read back both raw and through the
   ``dnsjax.analysis.response.probes`` reader), append-on-matching-
   sidecar, hard rejection of a mismatched sidecar and of a
   sidecar-less ``probes.bin``, and the non-finite scan message.
3. ``harmonics.parse_mode_pairs`` syntax errors and the ``probes``
   extension's validate (pairing, range), dispatched through
   ``validate_parameters``.

MPI part (skipped with ``--unit-only`` or when ``mpirun`` is absent):
solver-integration runs in temporary directories,

- a laminar plane-Poiseuille run (``mpirun -np 1``) with per-step
  probes: uniform sample times ``t = 0, dt, ..., t_stop`` including
  the cadence-aligned final sample, probe values at the laminar
  roundoff floor, and the reader's ``mean_profile`` / ``re_tau``
  reproducing the laminar profile (`$Re_\tau = \sqrt{2 Re}$`);
- a random-IC run (``mpirun -np 2 --dist.np1 2``, real multi-process)
  probing a mode owned by the second ``np1`` shard: finite, nonzero
  probe values through the same reader.

Usage::

    uv run python tests/test_probes.py [--unit-only]
"""

from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Mutate global ``params`` (and the ``probes`` extension singleton)
# before importing any dnsjax module that captures values from them
# (``sharding.Sharding`` does so at class definition time).
from dnsjax.extensions import probes_params  # noqa: E402
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
# Probed modes cover every mesh position: (0,0) mean mode on device
# (0,0); (5,0) owned by np0-shard 1; (0,2) by np1-shard 1; (5,2) by
# the (1,1) corner; (3,1) interior of shard (0,0).
MODES = [(0, 0), (5, 0), (0, 2), (5, 2), (3, 1)]
probes_params.modes = ";".join(f"{a},{b}" for a, b in MODES)
probes_params.it_probes = 1
params.outs.nbuffer = 3
# The sidecar records the wall-normal grid; no geometry module is
# imported in the offline part, so provide it directly.
import numpy as np  # noqa: E402

derived_params.wall_normal_grid = [
    float(v) for v in np.linspace(1.0, -1.0, NY)
]

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from numpy.testing import assert_array_equal  # noqa: E402

from dnsjax.analysis.response.probes import (  # noqa: E402
    mean_profile,
    re_tau,
    read_probes,
)
from dnsjax.harmonics import parse_mode_pairs  # noqa: E402
from dnsjax.parameters import validate_parameters  # noqa: E402
from dnsjax.probes import ProbeStream, build_mode_extractor  # noqa: E402
from dnsjax.snapshot import assemble_local_shards  # noqa: E402


def _mode_column(i2: int, i3: int, scale: float = 1.0) -> np.ndarray:
    """Closed-form reference column ``(3, NY)`` for mode (i2, i3)."""
    c = np.arange(3)[:, None]
    j = np.arange(NY)[None, :]
    return scale * (
        (c + 10.0 * j + 100.0 * i2 + 1000.0 * i3) + 1j * (i2 - i3 + 0.5)
    )


def _make_state(scale: float = 1.0, nan_mode: tuple | None = None):
    """Sharded spectral state whose every true mode holds the formula."""

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            for lj in range(nkx):
                g2, g3 = kz_start + li, kx_start + lj
                col = _mode_column(g2, g3, scale)
                if nan_mode == (g2, g3):
                    col[1, 2] = np.nan
                buf[:, :, li, lj] = col

    return assemble_local_shards(fill_local)


# ── Offline: sharded extractor ───────────────────────────────────────


def test_extractor_multi_device() -> None:
    """The gather returns the exact owned columns for every mesh
    position (bit-exact vs. the construction formula)."""
    state = _make_state()
    out = np.asarray(build_mode_extractor(MODES)(state))
    assert out.shape == (len(MODES), 3, NY)
    for k, (i2, i3) in enumerate(MODES):
        assert_array_equal(
            out[k], _mode_column(i2, i3), err_msg=f"mode ({i2},{i3})"
        )


# ── Offline: ProbeStream writer ──────────────────────────────────────


def test_probe_stream_roundtrip() -> None:
    """Record/flush byte layout, auto-flush at ``nbuffer``, append on a
    matching sidecar, and reader agreement."""
    state = _make_state()
    scaled = _make_state(scale=2.0)
    with tempfile.TemporaryDirectory() as tmp:
        stream = ProbeStream(state, tmp)
        assert stream.record(state, 0.0) is None
        assert stream.record(scaled, 0.5) is None
        # Third record fills nbuffer = 3 -> checked auto-flush.
        assert stream.record(state, 1.0) is None
        assert (Path(tmp) / "probes.bin").exists()
        assert stream.record(scaled, 1.5) is None
        assert stream.flush() is None
        assert stream.flush() is None  # empty buffer: no-op

        data = read_probes(tmp)
        assert data.t.tolist() == [0.0, 0.5, 1.0, 1.5]
        assert data.modes.tolist() == [list(m) for m in MODES]
        assert data.component_labels == ["u_x", "u_y", "u_z"]
        assert data.u.shape == (4, len(MODES), 3, NY)
        for k, (i2, i3) in enumerate(MODES):
            assert_array_equal(data.u[0, k], _mode_column(i2, i3))
            assert_array_equal(data.u[1, k], _mode_column(i2, i3, 2.0))
        assert_array_equal(data.y, np.linspace(1.0, -1.0, NY))
        # complex_harmonics(8) = [0, 1, 2, 3, -3, -2, -1]: index 5 = -2.
        assert data.wavenumbers[MODES.index((5, 0))].tolist() == [-2, 0]

        # Append: a fresh stream against the matching sidecar.
        stream2 = ProbeStream(state, tmp)
        stream2.record(state, 2.0)
        assert stream2.flush() is None
        assert read_probes(tmp).t.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_probe_stream_rejects_mismatch() -> None:
    """A mismatched sidecar or a sidecar-less binary is a hard error."""
    state = _make_state()
    with tempfile.TemporaryDirectory() as tmp:
        stream = ProbeStream(state, tmp)
        stream.record(state, 0.0)
        stream.flush()

        sidecar = Path(tmp) / "probes.json"
        tampered = sidecar.read_text().replace(
            '"it_probes": 1', '"it_probes": 2'
        )
        sidecar.write_text(tampered)
        try:
            ProbeStream(state, tmp)
        except SystemExit as e:
            assert "it_probes" in str(e)
        else:
            raise AssertionError("mismatched sidecar was accepted")

        sidecar.unlink()
        try:
            ProbeStream(state, tmp)
        except SystemExit as e:
            assert "sidecar" in str(e)
        else:
            raise AssertionError("sidecar-less probes.bin was accepted")


def test_probe_stream_non_finite() -> None:
    """The flush scan names the offending mode/component."""
    state = _make_state(nan_mode=(5, 2))
    with tempfile.TemporaryDirectory() as tmp:
        stream = ProbeStream(state, tmp)
        assert stream.record(state, 0.0) is None
        bad = stream.flush()
        assert bad is not None and "non-finite probe value" in bad
        assert "(5,2)" in bad and "u_y" in bad
        # The offending record is on disk for post-mortem.
        assert not np.isfinite(read_probes(tmp).u).all()


# ── Offline: parsing and validation ──────────────────────────────────


def test_parse_mode_pairs() -> None:
    assert parse_mode_pairs(" 0,0; 5, 2 ") == [(0, 0), (5, 2)]
    for bad in ("", "1", "1,2,3", "a,0", "1,-2", "0,0;0,0", "0,0;;1,1"):
        try:
            parse_mode_pairs(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_mode_pairs accepted {bad!r}")


def test_validate_probe_params() -> None:
    """Pairing and range checks (the ``probes`` extension validate,
    dispatched through ``validate_parameters``)."""
    saved = (probes_params.modes, probes_params.it_probes)
    try:
        validate_parameters()  # the module configuration is valid

        probes_params.it_probes = None  # modes without cadence
        _expect_value_error("set together")

        probes_params.it_probes = 1
        probes_params.modes = "7,0"  # i2 == nz - 1 out of range
        _expect_value_error("out of range")
        probes_params.modes = "0,4"  # i3 == nx // 2 out of range
        _expect_value_error("out of range")
    finally:
        probes_params.modes, probes_params.it_probes = saved


def _expect_value_error(fragment: str) -> None:
    try:
        validate_parameters()
    except ValueError as e:
        assert fragment in str(e), e
    else:
        raise AssertionError(
            f"validate_parameters accepted probe config "
            f"{probes_params.modes!r}/{probes_params.it_probes!r}"
        )


# ── MPI integration ──────────────────────────────────────────────────


def _run_solver(workdir: Path, np_count: int, np1: int, args: list[str]):
    """Launch ``mpirun -np N python -m dnsjax`` in *workdir*."""
    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.platform",
        "cpu",
        "--dist.np0",
        "1",
        "--dist.np1",
        str(np1),
        "--stop.check_laminarization",
        "False",
        *args,
    ]
    # The forced-host-device XLA_FLAGS of the offline part must not
    # leak into the solver children (mpirun: one real device each).
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    result = subprocess.run(
        cmd, cwd=workdir, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(
            f"solver run failed ({result.returncode}):\n"
            + "\n".join(result.stdout.splitlines()[-15:])
            + "\n"
            + "\n".join(result.stderr.splitlines()[-15:])
        )


def test_mpi_laminar_probes() -> None:
    """Laminar plane-Poiseuille: uniform sample times incl. the final
    cadence-aligned sample; probe values at the roundoff floor; the
    reader's total mean profile and ``Re_tau`` are laminar."""
    with tempfile.TemporaryDirectory() as tmp:
        _run_solver(
            Path(tmp),
            np_count=1,
            np1=1,
            args=[
                "--phys.system",
                "plane-poiseuille",
                "--init.start_from_laminar",
                "True",
                "--stop.max_sim_time",
                "0.04",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--res.ny",
                "15",
                "--outs.it_stats",
                "1",
                "--probes.modes",
                "0,0;1,0;0,1",
                "--probes.it_probes",
                "1",
            ],
        )
        data = read_probes(tmp)
        dt = float(data.meta["dt"])
        n_expect = round(0.04 / dt) + 1
        assert data.t.shape[0] == n_expect, data.t
        assert np.allclose(
            data.t, np.arange(n_expect) * dt, rtol=0, atol=1e-12
        )
        # Laminar run: the perturbation (hence every probe) sits at
        # the roundoff floor.
        assert np.abs(data.u).max() < 1e-12

        y, u_s = mean_profile(data)
        assert np.allclose(u_s, 1.0 - y**2, atol=1e-12)
        re = float(data.meta["params"]["phys"]["re"])
        assert abs(re_tau(data) - np.sqrt(2.0 * re)) < 1e-6


def test_mpi_random_probes_np2() -> None:
    """Random-IC run on 2 MPI processes (``np1 = 2``), probing a mode
    owned by the second ``np1`` shard: finite, nonzero values."""
    with tempfile.TemporaryDirectory() as tmp:
        _run_solver(
            Path(tmp),
            np_count=2,
            np1=2,
            args=[
                "--phys.system",
                "plane-poiseuille",
                "--phys.re",
                "100",
                "--stop.max_sim_time",
                "0.02",
                "--res.nx",
                "8",
                "--res.nz",
                "8",
                "--res.ny",
                "17",
                "--outs.it_stats",
                "1",
                # (0,3): i3 = 3 lives on np1-shard 1 (nx_spec = 4).
                "--probes.modes",
                "0,0;1,0;0,3",
                "--probes.it_probes",
                "1",
            ],
        )
        data = read_probes(tmp)
        assert np.isfinite(data.u).all()
        for k, (i2, i3) in enumerate(data.modes.tolist()):
            assert np.abs(data.u[:, k]).max() > 0, f"mode ({i2},{i3})"
        assert np.isfinite(re_tau(data))


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
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
