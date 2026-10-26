r"""The ``dnsjax`` solver with an I/O error on one rank.

A launch helper of ``tests/test_run_status.py`` ("mpi-peer-error"),
run under ``mpirun`` in place of the ``dnsjax`` console script, with
the solver's own arguments.  The rank ``OMPI_COMM_WORLD_RANK`` names
as ``1`` raises ``OSError`` (``EIO``) where every rank reads the
layout of the snapshot being written
(``dnsjax.snapshot_meta._chunk_members``), so it leaves the run alone
while its peer waits for it in the snapshot barrier.  The teardown
bound is shortened to :data:`GRACE` seconds, keeping the case short;
the mechanism is the one :data:`dnsjax.run_status.TEARDOWN_GRACE`
bounds.
"""

import errno
import os
import sys

import dnsjax.snapshot_meta as snapshot_meta
from dnsjax import run_status

#: The teardown bound under test, in seconds.
GRACE: int = 5

_chunk_members = snapshot_meta._chunk_members


def _failing(*args, **kwargs):
    if os.environ.get("OMPI_COMM_WORLD_RANK") == "1":
        raise OSError(errno.EIO, "Input/output error (simulated)")
    return _chunk_members(*args, **kwargs)


snapshot_meta._chunk_members = _failing
run_status.TEARDOWN_GRACE = GRACE

from dnsjax.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
