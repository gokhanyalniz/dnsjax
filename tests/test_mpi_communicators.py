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

Needs ``mpirun``, coreutils ``timeout`` and the MPIwrapper library the
MPI collectives load (``README.md``, "Installation"); without the
library the run would be on gloo, so the script skips.  Run as a
script::

    uv run python tests/test_mpi_communicators.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _child() -> None:
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
        for case in _CASES + _CONTROLS:
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
        """Run *body*, compiled ahead, on an input still being copied."""
        spec = P(*mesh.axis_names)
        shape = tuple(_ROWS * n for n in mesh.devices.shape)
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
    """Each case's ``(status, detail)`` per rank, from the RESULT lines."""
    seen: dict[str, list[tuple[str, str]]] = {}
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        head, status, detail = (p.strip() for p in line.split("|", 2))
        case = head.split(" ", 2)[2]
        seen.setdefault(case, []).append((status, detail))
    return seen


def main() -> int:
    if "--child" in sys.argv:
        _child()
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
    # passes on to its ranks.
    cmd = ["timeout", str(_TIMEOUT), "mpirun", "--oversubscribe"]
    cmd += ["-np", str(_RANKS), sys.executable, __file__, "--child"]
    try:
        res = run_live(cmd, env=env, timeout=_TIMEOUT + 60)
    except subprocess.TimeoutExpired:
        return report(0, [("launch", "outlived its backstop")])
    seen = _verdicts(res.stdout)

    passed = 0
    failures: list[tuple[str, str]] = []
    for case in _CASES + _CONTROLS:
        got = seen.get(case, [])
        statuses = {status for status, _ in got}
        control = case in _CONTROLS
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
    return report(passed, failures)


if __name__ == "__main__":
    sys.exit(main())
