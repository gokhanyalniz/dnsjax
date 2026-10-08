r"""Every MPI communicator the mesh uses is open before the solver runs.

XLA's MPI collectives open a device group's communicator at the first
collective over it, and only on the thread that initialized MPI; a
program running anywhere else then dies (``MPI: Communicator requested
from a thread that is not the one MPI was initialized from``) and the
run hangs.  Inline dispatch does not keep every collective on that
thread, so ``sharding._warm_communicators`` opens the mesh's
communicators as the mesh is built.  Its docstring has the why.

One four-rank launch (``mpirun -np 4``, a ``(2, 2)`` mesh, the MPI
collectives) runs the two patterns that put a collective off that
thread, first on the solver mesh, where each must work:

- **in-program maxima**: ``solvers._factor_checked`` with the
  cross-device maxima of its measures taken in the same program, the
  version that failed on every rank of every launch;
- **pending input**: a program compiled ahead and dispatched while its
  input is still being copied, for each kind of collective the solver
  issues -- an ``all_to_all`` and a ``ppermute`` along each axis, a
  ``psum`` over the whole mesh.

Then, as controls, the same two patterns over the mesh's diagonal
pairs, a device group the solver never uses and so nothing opened:
each must be refused, or the cases above prove nothing.

A second launch puts the odd ranks on a faked second host
(``_fake_host.py``).  The mesh then groups each node's devices, an
order of its own, while ``jax.experimental.multihost_utils`` -- the
wall-clock stop's ``any_process``, the snapshot barrier, the closing
memory line -- gathers over the devices in id order: a group of the
same devices in another order, which is another communicator.  There
a pending ``psum`` over that group must work, and ``any_process`` agree
on every rank; a pending ``psum`` over the reversed order, which
nothing opens, is the control.  Each reduces one row of its copied
block, the copy alone keeping the input pending: a 32 MiB message
stalls between the faked hosts, whose ranks Open MPI's shared memory
cannot reach by single copy across user namespaces.  Where the
namespaces cannot be made the launch is skipped with a notice.

Needs ``mpirun``, coreutils ``timeout`` and the MPIwrapper library the
MPI collectives load (``docs/cpu-collectives.md``); without the
library the run would be on gloo, so the script skips.  Run as a
script::

    uv run python tests/test_mpi_communicators.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fake_host  # noqa: E402
from _live import report, run_live  # noqa: E402

_RANKS = 4
_TIMEOUT = 300
_REFUSAL = "Communicator requested from a thread"

#: Rows of each rank's copied block: 2048 x 2048 float64 is 32 MiB, a
#: copy of milliseconds, against the microseconds between starting it
#: and dispatching the program that reads it.
_ROWS = 2048

_CASES = (
    "in-program maxima",
    "pending all_to_all np0",
    "pending all_to_all np1",
    "pending ppermute np0",
    "pending ppermute np1",
    "pending psum mesh",
)
_CONTROLS = ("control in-program maxima", "control pending psum")
_HOST_CASES = (
    "two hosts: the mesh's order is not the ids'",
    "two hosts: pending psum, id order",
    "two hosts: any_process",
)
_HOST_CONTROLS = ("two hosts: control pending psum, reversed order",)


def _child(two_hosts: bool) -> None:
    """One rank: build the mesh as the solver does, run every case."""
    from dnsjax.parameters import padded_res, params

    params.phys.system = "plane-couette"
    params.res.nx, params.res.ny, params.res.nz = 8, 33, 8
    params.dist.np0 = params.dist.np1 = 2
    padded_res.set_padded_resolution(params)
    from dnsjax.bootstrap import configure_jax_runtime

    configure_jax_runtime()
    import jax
    import numpy as np
    from jax import numpy as jnp
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from dnsjax.sharding import sharding
    from dnsjax.solvers import _factor_checked

    rank = jax.process_index()

    def emit(case: str, status: str, detail: str = "") -> None:
        print(f"RESULT {rank} {case} | {status} | {detail}", flush=True)

    if jax.config.jax_cpu_collectives_implementation != "mpi":
        cases = _HOST_CASES + _HOST_CONTROLS if two_hosts else _CASES
        for case in cases + (() if two_hosts else _CONTROLS):
            emit(case, "skipped", "the run is not on the MPI collectives")
        return

    def attempt(case: str, run) -> None:
        try:
            note = run()
        except Exception as exc:  # noqa: BLE001 -- reported to the parent
            text = " ".join(str(exc).split())
            status = "refused" if _REFUSAL in text else "error"
            emit(case, status, text[:160])
            return
        emit(case, "ok", note or "")

    def band_on(mesh: Mesh, spec: P) -> jax.Array:
        # Diagonally dominant, so the no-pivot LU is well posed.
        band = np.random.default_rng(0).uniform(-1, 1, (4, 4, 64, 5))
        band[..., 2] += 10.0
        return sharding.distribute(band, NamedSharding(mesh, spec))

    @jax.jit
    def maxima(a_band: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, _, resid, max_u, max_a = _factor_checked(a_band)
        return jnp.max(resid), jnp.max(max_u) / jnp.max(max_a)

    def in_program(a_band: jax.Array):
        def run() -> str:
            resid, growth = maxima(a_band)
            return f"resid {float(resid):.1e} growth {float(growth):.2f}"

        return run

    def pending(mesh: Mesh, body):
        """Run *body*, compiled ahead, on an input still being copied.

        A 1-D mesh gets a second, unsharded axis on its data, so that
        every device's block is the same ``(_ROWS, _ROWS)`` either way.
        """
        names = mesh.axis_names
        spec = P(*names) if len(names) == 2 else P(names[0], None)
        blocks = mesh.devices.shape if len(names) == 2 else (mesh.size, 1)
        shape = tuple(_ROWS * n for n in blocks)
        x = jax.make_array_from_callback(
            shape,
            NamedSharding(mesh, spec),
            lambda index: np.ones((_ROWS, _ROWS)),
        )
        program = (
            jax.jit(
                jax.shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec)
            )
            .lower(x)
            .compile()
        )

        def run() -> str:
            local = x.addressable_shards[0].data
            copy = jax.device_put(
                local, next(iter(local.devices())), may_alias=False
            )
            busy = not copy.is_ready()
            out = program(
                jax.make_array_from_single_device_arrays(
                    shape, x.sharding, [copy]
                )
            )
            out.block_until_ready()
            if not busy:
                raise RuntimeError("the input was ready at dispatch")
            return ""

        return run

    def a2a(axis: str):
        return lambda b: jax.lax.all_to_all(b, axis, 0, 0, tiled=True)

    def ppermute(axis: str):
        return lambda b: jax.lax.ppermute(b, axis, [(0, 1), (1, 0)])

    def one_d(devices: list) -> Mesh:
        return Mesh(np.array(devices), ("w",), axis_types=(AxisType.Explicit,))

    def psum_w(b):
        # One row of the block: a message of 32 MiB stalls between the
        # faked hosts (Open MPI's shared memory has no single-copy path
        # across user namespaces), and the copy alone keeps the input
        # pending.
        return jax.lax.psum(b[:1], "w")

    if two_hosts:
        ids = [d.id for d in jax.devices()]
        order = [d.id for d in sharding.mesh.devices.flat]
        if order != ids:
            emit(_HOST_CASES[0], "ok", f"mesh {order}, ids {ids}")
        else:
            emit(_HOST_CASES[0], "error", f"both {ids}: no second host")
        # The group multihost_utils gathers over: every device, in id
        # order (one per process).
        by_id = one_d(jax.devices())
        jax.set_mesh(by_id)
        attempt(_HOST_CASES[1], pending(by_id, psum_w))
        jax.set_mesh(sharding.mesh)

        def agree() -> str:
            some, none = (
                sharding.any_process(rank == 1),
                sharding.any_process(False),
            )
            if (some, none) != (True, False):
                raise RuntimeError(f"any_process gave {some}, {none}")
            return ""

        attempt(_HOST_CASES[2], agree)
        reversed_ = one_d(jax.devices()[::-1])
        jax.set_mesh(reversed_)
        attempt(_HOST_CONTROLS[0], pending(reversed_, psum_w))
        jax.set_mesh(sharding.mesh)
        return

    mesh = sharding.mesh
    attempt(
        "in-program maxima",
        in_program(band_on(mesh, P("np0", "np1", None, None))),
    )
    attempt("pending all_to_all np0", pending(mesh, a2a("np0")))
    attempt("pending all_to_all np1", pending(mesh, a2a("np1")))
    attempt("pending ppermute np0", pending(mesh, ppermute("np0")))
    attempt("pending ppermute np1", pending(mesh, ppermute("np1")))
    attempt(
        "pending psum mesh",
        pending(mesh, lambda b: jax.lax.psum(b, ("np0", "np1"))),
    )

    # The controls: axis "b" pairs the solver grid's diagonals, which
    # are neither an np0 group nor an np1 group.
    grid = mesh.devices
    diagonal = Mesh(
        np.array([[grid[0, 0], grid[1, 1]], [grid[0, 1], grid[1, 0]]]),
        ("a", "b"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    jax.set_mesh(diagonal)
    attempt(
        "control in-program maxima",
        in_program(band_on(diagonal, P("b", None, None, None))),
    )
    attempt(
        "control pending psum",
        pending(diagonal, lambda b: jax.lax.psum(b, "b")),
    )


def _mpi_wrapper_found() -> bool:
    from dnsjax.bootstrap import _mpiwrapper_lib

    return _mpiwrapper_lib() is not None


def _verdicts(stdout: str) -> dict[str, list[tuple[str, str]]]:
    """Each case's ``(status, detail)`` per rank, from the RESULT lines.

    Split on the ``RESULT <rank>`` token rather than on line ends:
    ``mpirun`` merges the ranks' streams, and one rank's line can land
    in the middle of another's.
    """
    seen: dict[str, list[tuple[str, str]]] = {}
    for record in re.split(r"(?=RESULT \d+ )", stdout):
        if not record.startswith("RESULT "):
            continue
        line = record.splitlines()[0]
        head, status, detail = (p.strip() for p in line.split("|", 2))
        case = head.split(" ", 2)[2]
        seen.setdefault(case, []).append((status, detail))
    return seen


def _judge(
    res: subprocess.CompletedProcess,
    cases: tuple[str, ...],
    controls: tuple[str, ...],
) -> tuple[int, list[tuple[str, str]]]:
    """``(passed, failures)`` of one launch's cases and controls."""
    seen = _verdicts(res.stdout)
    passed = 0
    failures: list[tuple[str, str]] = []
    for case in cases + controls:
        got = seen.get(case, [])
        statuses = {status for status, _ in got}
        control = case in controls
        want = "refused" if control else "ok"
        if statuses == {"skipped"}:
            print(f"  SKIP  {case}: {got[0][1]}")
            continue
        if len(got) == _RANKS and statuses == {want}:
            print(f"  PASS  {case}")
            passed += 1
            continue
        if len(got) < _RANKS:
            reason = (
                f"{len(got)} of {_RANKS} ranks reported (exit "
                f"{res.returncode}; 124 is a hang)"
            )
        elif control and "ok" in statuses:
            reason = (
                "ran on the launching thread, so the cases above prove "
                "nothing: the test went blind"
            )
        else:
            bad = next(d for s, d in got if s != want)
            reason = f"{sorted(statuses)}: {bad}"
        print(f"  FAIL  {case}: {reason}")
        failures.append((case, reason))
    if not failures and res.returncode != 0:
        failures.append(("launch", f"exit {res.returncode}"))
    return passed, failures


def main() -> int:
    if "--child" in sys.argv:
        _child(two_hosts=False)
        return 0
    if "--child-two-hosts" in sys.argv:
        _child(two_hosts=True)
        return 0
    if shutil.which("mpirun") is None or shutil.which("timeout") is None:
        print("mpirun and coreutils timeout are needed")
        return 1
    if not _mpi_wrapper_found():
        print("  SKIP  no MPIwrapper library: the run would be on gloo")
        return 0

    # The MPI collectives are the point, so nothing may pick gloo, and
    # the forced host-device count of the in-process tests must not
    # reach a production launch.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("XLA_FLAGS", "JAX_CPU_COLLECTIVES_IMPLEMENTATION")
    }
    env["NO_COLOR"] = "1"
    # coreutils ``timeout`` ends a hung run with SIGTERM, which mpirun
    # passes on to its ranks; ``-k`` kills one that ignores it.
    launch = [
        "timeout",
        "-k",
        "10",
        str(_TIMEOUT),
        "mpirun",
        "--oversubscribe",
    ]
    launch += ["-np", str(_RANKS)]
    try:
        res = run_live(
            [*launch, sys.executable, __file__, "--child"],
            env=env,
            timeout=_TIMEOUT + 60,
        )
    except subprocess.TimeoutExpired:
        return report(0, [("launch", "outlived its backstop")])
    passed, failures = _judge(res, _CASES, _CONTROLS)

    with tempfile.TemporaryDirectory(prefix="mpi_communicators_") as tmp:
        boot_id = _fake_host.write_boot_id(Path(tmp))
        why = _fake_host.unavailable(boot_id)
        if why is not None:
            print(f"  SKIP  two hosts: {why}")
            return report(passed, failures)
        child = [sys.executable, __file__, "--child-two-hosts"]
        two_env = {
            **env,
            "DNSJAX_TEST_SECOND_HOST": "rank % 2",
            "DNSJAX_TEST_BOOT_ID": str(boot_id),
        }
        try:
            res = run_live(
                [*launch, *_fake_host.wrap(child)],
                env=two_env,
                timeout=_TIMEOUT + 60,
            )
        except subprocess.TimeoutExpired:
            failures.append(("two hosts", "outlived its backstop"))
            return report(passed, failures)
    more, also = _judge(res, _HOST_CASES, _HOST_CONTROLS)
    return report(passed + more, failures + also)


if __name__ == "__main__":
    sys.exit(main())
