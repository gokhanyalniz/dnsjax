r"""Two hosts on one machine, for the ``mpirun`` launches of the tests.

XLA's distributed runtime tells hosts apart by their boot id
(``/proc/sys/kernel/random/boot_id``) and gives each host's devices a
``slice_index`` of their own, so a rank that reads another boot id is
on another node as far as JAX is concerned.  :data:`FAKE_HOST` wraps
one rank's command (``sh -c FAKE_HOST sh CMD...``): a rank for which
the shell expression ``$DNSJAX_TEST_SECOND_HOST`` is nonzero -- written
in terms of ``rank``, e.g. ``rank % 2`` -- re-runs the command in a
fresh user and mount namespace with the file ``$DNSJAX_TEST_BOOT_ID``
bound over the real boot id.  The bind needs root in that namespace,
and Open MPI's start-up fails in a rank that runs as root, so the
command itself runs one namespace further in, which maps the caller's
own uid and gid back (util-linux 2.38 or later).  gloo's start-up
needs nothing from either.  An interleaved layout (``rank % 2``) is
the one that sets the solver mesh's order, which groups each node's
devices (:func:`dnsjax.sharding.device_grid`), apart from the device
ids' rank order.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: ``sh -c`` wrapper around one rank's command (module docstring).
FAKE_HOST = r"""
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
if [ $(( $DNSJAX_TEST_SECOND_HOST )) -ne 0 ]; then
    DNSJAX_TEST_UID=$(id -u)
    DNSJAX_TEST_GID=$(id -g)
    export DNSJAX_TEST_UID DNSJAX_TEST_GID
    exec unshare --user --map-root-user --mount sh -c '
        mount --bind "$DNSJAX_TEST_BOOT_ID" /proc/sys/kernel/random/boot_id &&
        exec unshare --user --map-user="$DNSJAX_TEST_UID" \
            --map-group="$DNSJAX_TEST_GID" "$@"' sh "$@"
fi
exec "$@"
"""

BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
FAKE_BOOT_ID = "00000000-0000-4000-8000-00000000d0e5\n"


def write_boot_id(directory: Path) -> Path:
    """The second host's boot id, as a file to bind over the real one."""
    path = Path(directory) / "boot_id"
    path.write_text(FAKE_BOOT_ID)
    return path


def unavailable(boot_id: Path) -> str | None:
    """Why this machine cannot fake a second host, or ``None``.

    Runs the wrapper as a second-host rank would: it must read the fake
    boot id, as the caller's own uid.
    """
    if shutil.which("unshare") is None:
        return "no `unshare` on PATH"
    env = {
        **os.environ,
        "OMPI_COMM_WORLD_RANK": "1",
        "DNSJAX_TEST_SECOND_HOST": "rank",
        "DNSJAX_TEST_BOOT_ID": str(boot_id),
    }
    probe = ["sh", "-c", f"id -u; cat {BOOT_ID_PATH}"]
    try:
        result = subprocess.run(
            ["sh", "-c", FAKE_HOST, "sh", *probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "the namespace probe timed out"
    if result.stdout == f"{os.getuid()}\n{FAKE_BOOT_ID}":
        return None
    detail = result.stderr.strip() or f"exit {result.returncode}"
    return f"cannot bind a boot id in a user namespace ({detail})"


def wrap(command: list[str]) -> list[str]:
    """*command*, run through :data:`FAKE_HOST` (one rank's argv)."""
    return ["sh", "-c", FAKE_HOST, "sh", *command]
