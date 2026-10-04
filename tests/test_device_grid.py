r"""The device grid across nodes (:func:`dnsjax.sharding.device_grid`).

Under the distributed runtime every device carries a ``slice_index``,
which XLA assigns one per host, telling hosts apart by their boot id
(``/proc/sys/kernel/random/boot_id``).  ``jax.make_mesh`` refuses a
device set spanning several slices, so every multi-node launch once
died building the mesh -- a failure no run on one machine could show.

1. Unit (``--unit-only``): the grid on stub devices -- one node in id
   order; two nodes with block-numbered and with interleaved ranks,
   each node one block of rows either way; one row across both nodes;
   two rows per node; unequal nodes; and :func:`node_spans` on each.
2. Two hosts on one machine, under ``mpirun``: the ranks of the second
   host run in their own user and mount namespace, with another boot
   id bound over the real one, so XLA counts two hosts.  The ``(2, 2)``
   random-IC smoke runs on one host, on two with ranks 2 and 3 on the
   second, and on two with the odd ranks there (the order the node sort
   has to undo).  Each two-host run must report its grid -- every
   ``np1`` group on one node -- and reproduce the one-host
   ``stats.dat``.  The collectives are gloo: Open MPI's start-up fails
   in a rank that runs as root in its namespace, and gloo needs nothing
   from the launcher but the rank variables.  Where the namespace
   cannot be made (no ``unshare``, unprivileged user namespaces
   disabled) the half is skipped with a notice.

Usage::

    uv run python tests/test_device_grid.py
    uv run python tests/test_device_grid.py --unit-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

# ── unit: stub devices ───────────────────────────────────────────────


@dataclass(frozen=True)
class _Device:
    """A device of a multi-process run: an id and its node."""

    id: int
    slice_index: int


@dataclass(frozen=True)
class _LoneDevice:
    """A lone process's device: no ``slice_index`` at all."""

    id: int


_BLOCK = [_Device(i, i // 4) for i in range(8)]

# (name, devices, (np0, np1), expected grid of ids, expected spans)
_UNIT_CASES = [
    (
        "one node, id order",
        [_LoneDevice(i) for i in (2, 0, 3, 1)],
        (2, 2),
        [[0, 1], [2, 3]],
        (1, 1, 1),
    ),
    (
        "two nodes, block ranks",
        _BLOCK,
        (2, 4),
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        (2, 2, 1),
    ),
    (
        "two nodes, interleaved ranks",
        [_Device(i, i % 2) for i in range(8)],
        (2, 4),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        (2, 2, 1),
    ),
    ("one row across two nodes", _BLOCK, (1, 8), [list(range(8))], (2, 1, 2)),
    (
        "two rows per node",
        _BLOCK,
        (4, 2),
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        (2, 2, 1),
    ),
    (
        "unequal nodes",
        [_Device(i, int(i >= 3)) for i in range(4)],
        (2, 2),
        [[0, 1], [2, 3]],
        (2, 2, 2),
    ),
]


def run_unit_cases() -> tuple[int, list[tuple[str, str]]]:
    """The stub-device cases; ``(passed, failures)``."""
    from dnsjax.bootstrap import configure_jax_platform

    configure_jax_platform("cpu")
    from dnsjax.parameters import padded_res, params

    padded_res.set_padded_resolution(params)
    from dnsjax.sharding import device_grid, node_spans

    passed = 0
    failures: list[tuple[str, str]] = []
    for name, devices, (np0, np1), ids, spans in _UNIT_CASES:
        grid = device_grid(devices, np0, np1)
        got_ids = [[d.id for d in row] for row in grid]
        got_spans = node_spans(grid)
        if got_ids == ids and got_spans == spans:
            print(f"  PASS  {name}")
            passed += 1
            continue
        reason = (
            f"grid {got_ids} (want {ids}), spans {got_spans} (want {spans})"
        )
        print(f"  FAIL  {name}: {reason}")
        failures.append((name, reason))
    return passed, failures


# ── two hosts on one machine ─────────────────────────────────────────

#: ``sh -c`` wrapper around one rank's command.  A rank for which the
#: shell expression ``$DNSJAX_TEST_SECOND_HOST`` (in ``rank``) is
#: nonzero re-runs the command in a fresh user and mount namespace with
#: the file ``$DNSJAX_TEST_BOOT_ID`` bound over the real boot id.
_FAKE_HOST = r"""
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
if [ $(( $DNSJAX_TEST_SECOND_HOST )) -ne 0 ]; then
    exec unshare --user --map-root-user --mount sh -c '
        mount --bind "$DNSJAX_TEST_BOOT_ID" /proc/sys/kernel/random/boot_id &&
        exec "$@"' sh "$@"
fi
exec "$@"
"""

_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_FAKE_BOOT_ID = "00000000-0000-4000-8000-00000000d0e5\n"

#: The ``(2, 2)`` random-IC smoke: ``ny = 17`` and ``nz - 1 = 15`` also
#: engage the ``np0`` padding of the physical ``y`` and spectral
#: ``k_z`` axes.
_SMOKE = [
    "--dist.np0",
    "2",
    "--dist.np1",
    "2",
    "--phys.system",
    "plane-couette",
    "--phys.re",
    "330",
    "--geo.lx",
    "5",
    "--geo.lz",
    "5",
    "--init.random_field",
    "True",
    "--init.random_seed",
    "1",
    "--res.nx",
    "16",
    "--res.ny",
    "17",
    "--res.nz",
    "16",
    "--step.dt",
    "0.01",
    "--stop.max_sim_time",
    "0.05",
    "--outs.it_stats",
    "1",
    "--stop.check_laminarization",
    "False",
    "--outs.snapshot_save_initial",
    "False",
    "--outs.snapshot_save_final",
    "False",
]

#: What a two-host run of the smoke must print: ranks on two nodes, and
#: each node's devices whole ``np1`` groups whatever the rank order.
_GRID_LINE = (
    "Device grid on 2 nodes: each np0 group spans up to 2, each np1 group "
    "up to 1."
)

#: ``(name, ranks on the second host)``; the first is the reference.
_LAYOUTS = [
    ("one host", "0"),
    ("two hosts, block ranks", "rank >= 2"),
    ("two hosts, interleaved ranks", "rank % 2"),
]


def _fake_host_unavailable(boot_id: Path) -> str | None:
    """Why this machine cannot fake a second host, or ``None``."""
    if shutil.which("unshare") is None:
        return "no `unshare` on PATH"
    env = {
        **os.environ,
        "OMPI_COMM_WORLD_RANK": "1",
        "DNSJAX_TEST_SECOND_HOST": "rank",
        "DNSJAX_TEST_BOOT_ID": str(boot_id),
    }
    try:
        probe = subprocess.run(
            ["sh", "-c", _FAKE_HOST, "sh", "cat", _BOOT_ID_PATH],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "the namespace probe timed out"
    if probe.stdout == _FAKE_BOOT_ID:
        return None
    detail = probe.stderr.strip() or f"exit {probe.returncode}"
    return f"cannot bind a boot id in a user namespace ({detail})"


def _run_smoke(second_host: str, boot_id: Path, workdir: str) -> str:
    """Launch the smoke with *second_host* ranks on the fake host."""
    env = {
        **os.environ,
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
        "DNSJAX_TEST_SECOND_HOST": second_host,
        "DNSJAX_TEST_BOOT_ID": str(boot_id),
    }
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-np",
        "4",
        "sh",
        "-c",
        _FAKE_HOST,
        "sh",
        sys.executable,
        "-m",
        "dnsjax",
        *_SMOKE,
    ]
    result = run_live(cmd, timeout=300, env=env, cwd=workdir)
    if result.returncode != 0:
        raise AssertionError(f"exit code {result.returncode}")
    return result.stdout


def run_two_host_cases() -> tuple[int, list[tuple[str, str]]]:
    """The ``mpirun`` launches; ``(passed, failures)``."""
    with tempfile.TemporaryDirectory(prefix="device_grid_") as tmp:
        boot_id = Path(tmp) / "boot_id"
        boot_id.write_text(_FAKE_BOOT_ID)
        why = _fake_host_unavailable(boot_id)
        if why is not None:
            print(f"  SKIP  two hosts: {why}")
            return 0, []

        passed = 0
        failures: list[tuple[str, str]] = []
        reference: np.ndarray | None = None
        for name, second_host in _LAYOUTS:
            workdir = Path(tmp) / name.replace(" ", "_").replace(",", "")
            workdir.mkdir()
            try:
                stdout = _run_smoke(second_host, boot_id, str(workdir))
                stats = np.loadtxt(workdir / "stats.dat", ndmin=2)
                if reference is None:
                    if "Device grid" in stdout:
                        raise AssertionError("one host reported a grid")
                    reference = stats
                else:
                    if stdout.count(_GRID_LINE) != 1:
                        raise AssertionError(f"no {_GRID_LINE!r} line")
                    np.testing.assert_allclose(
                        stats, reference, rtol=1e-12, atol=1e-15
                    )
            except (AssertionError, subprocess.TimeoutExpired) as exc:
                reason = str(exc).strip().splitlines()[0]
                print(f"  FAIL  {name}: {reason}")
                failures.append((name, reason))
                if reference is None:
                    break
                continue
            print(f"  PASS  {name}")
            passed += 1
        return passed, failures


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The device grid across nodes.",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run the stub-device cases only (no mpirun).",
    )
    args = parser.parse_args()

    passed, failures = run_unit_cases()
    if not args.unit_only:
        two_passed, two_failures = run_two_host_cases()
        passed += two_passed
        failures += two_failures
    sys.exit(report(passed, failures))
