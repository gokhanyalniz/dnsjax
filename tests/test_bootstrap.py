r"""Multi-process bootstrap helpers of ``dnsjax.bootstrap``.

Offline: the environment-driven decisions ``configure_jax_runtime``
makes before it touches a device -- how the ranks are bootstrapped
(the launcher environment, else JAX's own detection), where the
coordinator lives and on which port, the discovery of the MPIwrapper
library MPItrampoline dlopens (``MPITRAMPOLINE_LIB`` /
``LD_LIBRARY_PATH``), and the resulting CPU collectives selection
(``JAX_CPU_COLLECTIVES_IMPLEMENTATION``).

Two of those carry failures that are silent until they are fatal, and
are pinned here because no offline run reproduces them: a coordinator
port seeded per *job* rather than per *launch* makes two concurrent
runs in one allocation kill each other, and MPI collectives selected
through the environment rather than by discovery reach the same
thread-affinity abort unless dispatch is pinned inline on that path
too.

Those decisions are the only automatable part of the MPI path: the
collectives themselves need an MPIwrapper build and >= 2 ranks, and
fail *loudly* (MPItrampoline aborts on a bad ``MPITRAMPOLINE_LIB``),
whereas a discovery that wrongly returns ``None`` degrades silently
to gloo, which these cases pin down.  ``jax.distributed.initialize``
is stubbed: what is under test is which detection method dnsjax hands
it, not JAX's detection (the suite's mpirun rows cover the CPU path
for real).  The wrapper library is an empty stub file, since
discovery is a pure path test and nothing here dlopens it.

Cases mutate ``os.environ`` and restore it (``_env``), so ordering is
free; where a case sets JAX's config back (visible devices, dispatch,
the collectives), it calls ``jax.config.update`` directly rather than
through :mod:`dnsjax.bootstrap`, since it undoes what the functions
under test set.

Run as a script::

    uv run python tests/test_bootstrap.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import zlib
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dnsjax import bootstrap  # noqa: E402

FAILURES: list[str] = []

_VARS = (
    "MPITRAMPOLINE_LIB",
    "LD_LIBRARY_PATH",
    "JAX_CPU_COLLECTIVES_IMPLEMENTATION",
    "JAX_COORDINATOR_ADDRESS",
    "JAX_COORDINATOR_PORT",
    "JAX_LOCAL_DEVICE_IDS",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "MPI_LOCALRANKID",
    "MPI_LOCALNRANKS",
    # Rank markers with no layout behind them (``_rank_marker``): one
    # left set would make every later case look like a multi-rank
    # launch.
    "PMIX_RANK",
    "PMI_ID",
    "SLURM_PROCID",
    "ALPS_APP_PE",
    "FLUX_TASK_RANK",
    "JSM_NAMESPACE_RANK",
    "OMPI_MCA_orte_hnp_uri",  # noqa: SIM112
    "PRTE_MCA_prte_hnp_uri",  # noqa: SIM112
    "PMIX_NAMESPACE",
    "PBS_NODEFILE",
    "PBS_JOBID",
    "SLURM_STEP_NODELIST",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_ID",
    "LSB_DJOB_HOSTFILE",
    "LSB_HOSTS",
    "LSB_JOBID",
    "PE_HOSTFILE",
    "JOB_ID",
)

_OMPI = {
    "OMPI_COMM_WORLD_RANK": "3",
    "OMPI_COMM_WORLD_SIZE": "8",
    "OMPI_COMM_WORLD_LOCAL_RANK": "1",
}


def _in_range(port: str) -> bool:
    """Is *port* inside the ephemeral range every seed maps into?"""
    return 65535 - 2**12 < int(port) <= 65535


def check(cond: bool, label: str, detail: object = "") -> None:
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILURES.append(label)


@contextlib.contextmanager
def _env(**overrides: str | None):
    """Set/unset the bootstrap variables, restoring all of them after."""
    saved = {name: os.environ.get(name) for name in _VARS}
    for name in _VARS:
        os.environ.pop(name, None)
    for name, value in overrides.items():
        if value is not None:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _wrapper_dirs():
    """Yield ``(with_lib, without_lib)`` directories, the first stocked."""
    with tempfile.TemporaryDirectory() as tmp:
        with_lib = Path(tmp) / "have"
        without = Path(tmp) / "empty"
        with_lib.mkdir()
        without.mkdir()
        (with_lib / "libmpiwrapper.so").write_bytes(b"")
        yield with_lib, without


# ── Case A: MPIwrapper discovery ─────────────────────────────────────


def case_wrapper_discovery() -> None:
    with _wrapper_dirs() as (with_lib, without):
        lib = str(with_lib / "libmpiwrapper.so")

        with _env(MPITRAMPOLINE_LIB=lib):
            check(
                bootstrap._mpiwrapper_lib() == lib,
                "MPITRAMPOLINE_LIB honoured when it points at a file",
                bootstrap._mpiwrapper_lib(),
            )

        # A stale explicit setting must not be papered over by the
        # scan: MPItrampoline would abort on it, and silently
        # substituting another library hides the user's mistake.
        with _env(
            MPITRAMPOLINE_LIB=str(without / "libmpiwrapper.so"),
            LD_LIBRARY_PATH=str(with_lib),
        ):
            check(
                bootstrap._mpiwrapper_lib() is None,
                "MPITRAMPOLINE_LIB naming a missing file yields None",
                bootstrap._mpiwrapper_lib(),
            )

        with _env(LD_LIBRARY_PATH=str(with_lib)):
            found = bootstrap._mpiwrapper_lib()
            check(
                found == lib,
                "LD_LIBRARY_PATH scan finds libmpiwrapper.so",
                found,
            )
            check(
                os.environ.get("MPITRAMPOLINE_LIB") == lib,
                "a scanned hit is exported for MPItrampoline",
                os.environ.get("MPITRAMPOLINE_LIB"),
            )

        # Empty entries (a trailing ':' is idiomatic) must not become
        # the cwd, and the first directory that has the library wins.
        with _env(LD_LIBRARY_PATH=f":{without}::{with_lib}:{without}:"):
            check(
                bootstrap._mpiwrapper_lib() == lib,
                "LD_LIBRARY_PATH scan skips empty entries, keeps order",
                bootstrap._mpiwrapper_lib(),
            )

        with _env(LD_LIBRARY_PATH=str(without)):
            check(
                bootstrap._mpiwrapper_lib() is None,
                "no wrapper on LD_LIBRARY_PATH yields None",
                bootstrap._mpiwrapper_lib(),
            )

        with _env():
            check(
                bootstrap._mpiwrapper_lib() is None,
                "unset environment yields None",
                bootstrap._mpiwrapper_lib(),
            )


# ── Case B: reading the launcher environment ─────────────────────────


def case_launcher_params() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nodefile = Path(tmp) / "nodes"
        nodefile.write_text("head-01\nhead-01\nnode-02\n")
        pbs = {"PBS_NODEFILE": str(nodefile), "PBS_JOBID": "987654.pbs"}
        # The port rule restated: the seed's checksum mapped into the
        # ephemeral range (why a checksum: ``_coordinator_port``).
        port = zlib.crc32(b"987654.pbs") % 2**12 + (65535 - 2**12 + 1)

        with _env(**_OMPI, **pbs):
            got = bootstrap._launcher_params()
            check(
                got
                == {
                    "coordinator_address": f"head-01:{port}",
                    "num_processes": 8,
                    "process_id": 3,
                    "local_device_ids": [1],
                },
                "Open MPI ranks + PBS node list give all four values",
                got,
            )

        # Any of the four missing leaves the launch to JAX's own
        # detection (``_launcher_params``).
        for drop in ("OMPI_COMM_WORLD_RANK", "PBS_NODEFILE"):
            env = {**_OMPI, **pbs}
            env.pop(drop)
            with _env(**env):
                check(
                    bootstrap._launcher_params() is None,
                    f"without {drop} the caller has to fall back",
                    bootstrap._launcher_params(),
                )

        with _env(**_OMPI, **pbs, JAX_COORDINATOR_ADDRESS="given:1234"):
            got = bootstrap._launcher_params()
            check(
                got["coordinator_address"] == "given:1234",
                "JAX_COORDINATOR_ADDRESS wins over the node list",
                got,
            )

        # The single-process multi-device launch: one rank spanning
        # every device, which the local-rank default would undo.
        with _env(**_OMPI, **pbs, JAX_LOCAL_DEVICE_IDS="0,1,2,3"):
            got = bootstrap._launcher_params()
            check(
                got["local_device_ids"] == [0, 1, 2, 3],
                "JAX_LOCAL_DEVICE_IDS is honoured, not overwritten",
                got,
            )

        with _env(**_OMPI, PBS_NODEFILE=str(Path(tmp) / "missing")):
            check(
                bootstrap._launcher_params() is None,
                "a PBS_NODEFILE that is not there yields None",
                bootstrap._launcher_params(),
            )

    # The MPICH-derived stacks publish their own names; untested on
    # real hardware here, so pin at least that they are read.
    mv2 = {
        "MV2_COMM_WORLD_RANK": "2",
        "MV2_COMM_WORLD_SIZE": "4",
        "MV2_COMM_WORLD_LOCAL_RANK": "2",
    }
    pmi = {"PMI_RANK": "1", "PMI_SIZE": "4", "MPI_LOCALRANKID": "1"}
    for label, env in (("MVAPICH2", mv2), ("MPICH/Intel MPI", pmi)):
        with _env(**env, JAX_COORDINATOR_ADDRESS="given:1234"):
            got = bootstrap._launcher_params()
            check(
                got is not None and got["num_processes"] == 4,
                f"{label} rank variables are read too",
                got,
            )

    # A malformed value falls through to JAX's detection
    # (``_launcher_ranks``).
    with _env(
        OMPI_COMM_WORLD_RANK="not-a-rank",
        OMPI_COMM_WORLD_SIZE="8",
        OMPI_COMM_WORLD_LOCAL_RANK="1",
        JAX_COORDINATOR_ADDRESS="given:1234",
    ):
        check(
            bootstrap._launcher_params() is None,
            "an unparseable rank variable is not trusted",
            bootstrap._launcher_params(),
        )


# ── Case C: the coordinator port ─────────────────────────────────────


def case_coordinator_port() -> None:
    # Launch-scoped before job-scoped (the ``bootstrap`` Design notes,
    # "The coordinator port").
    with _env(PMIX_NAMESPACE="1240793089", PBS_JOBID="12345.pbs01"):
        first = bootstrap._coordinator_port()
    with _env(PMIX_NAMESPACE="1240662017", PBS_JOBID="12345.pbs01"):
        second = bootstrap._coordinator_port()
    with _env(PBS_JOBID="12345.pbs01"):
        job_only = bootstrap._coordinator_port()
    check(
        first != second and job_only not in (first, second),
        "the port follows the launch, not the scheduler job",
        (first, second, job_only),
    )

    # Real Open MPI namespaces: reduced modulo the range, every one
    # would give port 61441 (why the seed is checksummed: the same
    # notes).
    ports = set()
    for namespace in ("1380188161", "1240793089", "1240662017", "1240006657"):
        with _env(PMIX_NAMESPACE=namespace):
            ports.add(bootstrap._coordinator_port())
    check(
        len(ports) == 4,
        "real Open MPI namespaces spread over distinct ports",
        sorted(ports),
    )

    # Every rank of one launch has to derive the same port.
    with _env(**_OMPI, PMIX_NAMESPACE="1240793089"):
        as_rank_3 = bootstrap._coordinator_port()
    with _env(
        OMPI_COMM_WORLD_RANK="0",
        OMPI_COMM_WORLD_SIZE="8",
        OMPI_COMM_WORLD_LOCAL_RANK="0",
        PMIX_NAMESPACE="1240793089",
    ):
        as_rank_0 = bootstrap._coordinator_port()
    check(
        as_rank_3 == as_rank_0,
        "the port does not depend on which rank derives it",
        (as_rank_3, as_rank_0),
    )

    # Open MPI 5 spells the namespace "prterun-<host>-<pid>@<n>" (as
    # 5.0.10 writes it), so the rule may not assume a number, and two
    # launches on one node differ only in the pid.
    prterun = {}
    for pid in ("1528569", "1528570"):
        with _env(PMIX_NAMESPACE=f"prterun-node01-{pid}@1"):
            prterun[pid] = bootstrap._coordinator_port()
    check(
        all(_in_range(p) for p in prterun.values())
        and len(set(prterun.values())) == 2,
        "PRRTE-style namespaces give distinct ports in range",
        prterun,
    )

    seeded = {}
    for label, env in (
        ("PBS", {"PBS_JOBID": "8817.pbs01"}),
        ("SLURM", {"SLURM_JOB_ID": "774411"}),
        ("LSF", {"LSB_JOBID": "5150"}),
        ("Grid Engine", {"JOB_ID": "4242"}),
    ):
        with _env(**env):
            port = bootstrap._coordinator_port()
            seeded[label] = port
            check(
                _in_range(port),
                f"{label} job id seeds the port when no PMIx does",
                port,
            )
    check(
        len(set(seeded.values())) == len(seeded),
        "different job ids do not share a port",
        seeded,
    )

    with _env(PMIX_NAMESPACE="1240793089", JAX_COORDINATOR_PORT="12345"):
        check(
            bootstrap._coordinator_port() == "12345",
            "JAX_COORDINATOR_PORT overrides every seed",
            bootstrap._coordinator_port(),
        )

    with _env():
        port = bootstrap._coordinator_port()
        check(
            _in_range(port),
            "a seedless environment still lands in the ephemeral range",
            port,
        )


# ── Case D: the coordinator host ─────────────────────────────────────


def _ranks(size: int, local_size: int | None) -> object:
    return bootstrap._Ranks(
        rank=0,
        size=size,
        local_rank=0,
        local_size=local_size,
        family="OMPI_COMM_WORLD_RANK",
    )


def case_coordinator_host() -> None:
    # All ranks on this node: loopback (``_coordinator_host``, source 1).
    with _env():
        check(
            bootstrap._coordinator_host(_ranks(4, 4)) == "127.0.0.1",
            "all ranks on this node resolve to loopback",
            bootstrap._coordinator_host(_ranks(4, 4)),
        )
        check(
            bootstrap._coordinator_host(_ranks(8, 4)) is None,
            "a spread job does not take the loopback shortcut",
            bootstrap._coordinator_host(_ranks(8, 4)),
        )
        check(
            bootstrap._coordinator_host(_ranks(4, None)) is None,
            "a launcher publishing no local size takes no shortcut",
            bootstrap._coordinator_host(_ranks(4, None)),
        )

    # The daemon URI under ORTE's name (Open MPI 4, the one JAX's own
    # plugin reads) and the PRRTE spelling, a catch for a 5.x build that
    # exports one (5.0.10 exports none: ``_HNP_URI_VARS``).
    uris = {
        "OMPI_MCA_orte_hnp_uri": (  # noqa: SIM112
            "1531576320.0;tcp://192.0.2.1,198.51.100.1:34911"
        ),
        "PRTE_MCA_prte_hnp_uri": (  # noqa: SIM112
            "prterun-node01-32@0;tcp://192.0.2.2:1234"
        ),
    }
    for name, uri in uris.items():
        with _env(**{name: uri}):
            got = bootstrap._coordinator_host(_ranks(8, 4))
            check(
                got == ("192.0.2.1" if "orte" in name else "192.0.2.2"),
                f"{name} gives the launch node",
                got,
            )
    with _env(
        **{
            "OMPI_MCA_orte_hnp_uri": (
                "1314521088.0;tcp6://[fe80::1,2001:db8::2]:43370"
            )
        }
    ):
        got = bootstrap._coordinator_host(_ranks(8, 4))
        check(
            got == "fe80::1",
            "the tcp6 form of the daemon URI parses too",
            got,
        )

    with tempfile.TemporaryDirectory() as tmp:
        pbs = Path(tmp) / "pbs_nodes"
        pbs.write_text("node-07\nnode-07\nnode-08\n")
        lsf = Path(tmp) / "lsb_hosts"
        lsf.write_text("batch-01\nbatch-02\n")
        sge = Path(tmp) / "pe_hostfile"
        # "hostname nslots queue processors", one line per host.
        sge.write_text("uge-03 8 all.q@uge-03 UNDEFINED\nuge-04 8 all.q\n")

        for label, env, host in (
            ("PBS_NODEFILE", {"PBS_NODEFILE": str(pbs)}, "node-07"),
            (
                "LSB_DJOB_HOSTFILE",
                {"LSB_DJOB_HOSTFILE": str(lsf)},
                "batch-01",
            ),
            ("PE_HOSTFILE", {"PE_HOSTFILE": str(sge)}, "uge-03"),
            ("LSB_HOSTS", {"LSB_HOSTS": "lsf-09 lsf-09 lsf-10"}, "lsf-09"),
        ):
            with _env(**env):
                got = bootstrap._coordinator_host(_ranks(8, 4))
                check(got == host, f"{label} names the first node", got)

    # The bracket forms of a SLURM job node list (``_slurm_host``).
    for node_list in (
        "node001",
        "node001,host2",
        "node[001-0015],host2",
        "node[001,007-015],host2",
    ):
        with _env(SLURM_JOB_NODELIST=node_list):
            got = bootstrap._coordinator_host(_ranks(8, 4))
            check(
                got == "node001",
                f"SLURM node list {node_list!r} gives node001",
                got,
            )
    with _env(SLURM_JOB_NODELIST="node[005-009]", SLURM_STEP_NODELIST="s07"):
        check(
            bootstrap._coordinator_host(_ranks(8, 4)) == "s07",
            "the step node list wins over the job one",
            bootstrap._coordinator_host(_ranks(8, 4)),
        )


# ── Case E: am I the whole job? ──────────────────────────────────────


def case_solo_launch() -> None:
    # Every "which rank am I" variable vetoes solitude, layout or not
    # (why the net is wide: ``_rank_marker``).
    for marker in (
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "PMI_ID",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
        "ALPS_APP_PE",
        "FLUX_TASK_RANK",
        "JSM_NAMESPACE_RANK",
    ):
        with _env(**{marker: "1"}):
            check(
                bootstrap._rank_marker() == marker
                and not bootstrap._solo_launch(),
                f"{marker} alone vetoes the solo path",
                bootstrap._rank_marker(),
            )

    with _env():
        check(
            bootstrap._rank_marker() is None and bootstrap._solo_launch(),
            "an empty environment is a lone process",
            bootstrap._rank_marker(),
        )

    # A *complete* layout is believed over the marker veto: how
    # ``mpirun -np 1`` reaches the solo path at all.
    one = {
        "OMPI_COMM_WORLD_RANK": "0",
        "OMPI_COMM_WORLD_SIZE": "1",
        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
    }
    with _env(**one):
        check(
            bootstrap._solo_launch(),
            "a launcher reporting one process is solo",
            bootstrap._solo_launch(),
        )
    with _env(**{**one, "OMPI_COMM_WORLD_SIZE": "2"}):
        check(
            not bootstrap._solo_launch(),
            "a launcher reporting two is not",
            bootstrap._solo_launch(),
        )


def case_local_device_ids() -> None:
    """Skipping ``initialize`` must not change which devices we get."""
    import jax

    saved = jax.config.read("jax_cuda_visible_devices")
    try:
        for label, env, expect in (
            # The multi-GPU-without-MPI case (``_apply_local_device_ids``).
            ("a bare launch keeps every device", {}, saved),
            (
                "the launcher's local rank still narrows",
                {
                    "OMPI_COMM_WORLD_RANK": "0",
                    "OMPI_COMM_WORLD_SIZE": "1",
                    "OMPI_COMM_WORLD_LOCAL_RANK": "2",
                },
                "2",
            ),
            (
                "JAX_LOCAL_DEVICE_IDS wins, as in initialize()",
                {
                    "OMPI_COMM_WORLD_RANK": "0",
                    "OMPI_COMM_WORLD_SIZE": "1",
                    "OMPI_COMM_WORLD_LOCAL_RANK": "2",
                    "JAX_LOCAL_DEVICE_IDS": "0,1,2,3",
                },
                "0,1,2,3",
            ),
        ):
            jax.config.update("jax_cuda_visible_devices", saved)
            with _env(**env):
                bootstrap._apply_local_device_ids()
                got = jax.config.read("jax_cuda_visible_devices")
                check(got == expect, label, got)
    finally:
        jax.config.update("jax_cuda_visible_devices", saved)
        jax.config.update("jax_rocm_visible_devices", saved)


# ── Case F: launcher bootstrap ───────────────────────────────────────


def _exits(fn) -> str | None:
    """Run *fn*, returning the ``SystemExit`` message it raised."""
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    return None


@contextlib.contextmanager
def _stub_initialize(explicit_ok: bool = True):
    """Stand in for ``jax.distributed.initialize``; record the methods.

    *explicit_ok* false makes it fail the way JAX does when nothing
    identifies the launcher and no parameters were supplied.
    """
    import jax

    calls: list[str | None] = []

    def fake(cluster_detection_method: str | None = None, **kwargs):
        calls.append(cluster_detection_method)
        if not explicit_ok and not kwargs:
            raise ValueError("coordinator_address should be defined.")

    real = jax.distributed.initialize
    jax.distributed.initialize = fake
    try:
        yield calls
    finally:
        jax.distributed.initialize = real


def case_bootstrap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nodefile = Path(tmp) / "nodes"
        nodefile.write_text("head-01\n")
        described = {
            **_OMPI,
            "PBS_NODEFILE": str(nodefile),
            "PBS_JOBID": "12.pbs",
        }

        with _env(**described), _stub_initialize() as calls:
            bootstrap._bootstrap_distributed()
            check(
                calls == [None],
                "a self-describing launcher skips JAX's detection",
                calls,
            )

        # A marker with no usable layout is still several processes, so
        # JAX's own detection gets its turn.
        with _env(SLURM_PROCID="2"), _stub_initialize() as calls:
            bootstrap._bootstrap_distributed()
            check(
                calls == [None],
                "otherwise JAX's own detection runs",
                calls,
            )

        # One process starts no runtime, whether the launcher said "one"
        # or nothing said anything (the ``bootstrap`` Design notes, "No
        # distributed runtime for one process").
        single = {**described, "OMPI_COMM_WORLD_SIZE": "1"}
        single["OMPI_COMM_WORLD_RANK"] = "0"
        single["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"
        for label, env in (
            ("a one-process launcher", single),
            ("no launcher at all", {}),
            (
                "one process narrowing its devices",
                {
                    **single,
                    "JAX_LOCAL_DEVICE_IDS": "0,1",
                },
            ),
        ):
            with _env(**env), _stub_initialize() as calls:
                bootstrap._bootstrap_distributed()
                check(
                    calls == [],
                    f"{label} skips the distributed runtime",
                    calls,
                )

    # An unrecognized launcher names the way out, not JAX's bare
    # ``coordinator_address should be defined``, and which way depends
    # on what was missing (``_undetectable_launcher``).
    with _env(**_OMPI), _stub_initialize(explicit_ok=False):
        msg = _exits(bootstrap._bootstrap_distributed) or ""
        check(
            "JAX_COORDINATOR_ADDRESS" in msg and "rank 3 of 8" in msg,
            "ranks without an address name JAX_COORDINATOR_ADDRESS",
            msg,
        )

    with _env(PMIX_RANK="1"), _stub_initialize(explicit_ok=False):
        msg = _exits(bootstrap._bootstrap_distributed) or ""
        check(
            "PMIX_RANK" in msg and "mpirun" in msg,
            "a bare marker names itself as the thing to unset",
            msg,
        )


# ── Case G: CPU collectives selection ────────────────────────────────


def case_cpu_collectives() -> None:
    import jax

    default = jax.config.jax_cpu_collectives_implementation
    async_default = jax.config.read("jax_cpu_enable_async_dispatch")

    with _wrapper_dirs() as (with_lib, without):
        lib = str(with_lib / "libmpiwrapper.so")

        with _env(JAX_CPU_COLLECTIVES_IMPLEMENTATION="mpi"):
            note = bootstrap._select_cpu_collectives(lib)
            check(
                "mpi" in note and "JAX_CPU_COLLECTIVES_IMPLEMENTATION" in note,
                "an explicit JAX choice wins over discovery",
                note,
            )
            check(
                jax.config.jax_cpu_collectives_implementation == default,
                "an explicit JAX choice is left to JAX to apply",
                jax.config.jax_cpu_collectives_implementation,
            )
            # JAX applies the variable but not the dispatch pin
            # (``_select_cpu_collectives``).
            check(
                jax.config.read("jax_cpu_enable_async_dispatch") is False,
                "an explicit mpi choice still pins dispatch inline",
                jax.config.read("jax_cpu_enable_async_dispatch"),
            )
            jax.config.update("jax_cpu_enable_async_dispatch", async_default)

        with _env(JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo"):
            bootstrap._select_cpu_collectives(lib)
            check(
                jax.config.read("jax_cpu_enable_async_dispatch")
                == async_default,
                "an explicit gloo choice leaves dispatch alone",
                jax.config.read("jax_cpu_enable_async_dispatch"),
            )

        with _env(MPITRAMPOLINE_LIB=str(without / "libmpiwrapper.so")):
            note = bootstrap._select_cpu_collectives(None)
            check(
                "gloo" in note and str(without) in note,
                "a stale MPITRAMPOLINE_LIB is named, not re-advised",
                note,
            )

        with _env():
            note = bootstrap._select_cpu_collectives(None)
            check(
                "gloo" in note and "docs/cpu-collectives.md" in note,
                "no wrapper reports gloo and points at the CPU-collectives "
                "doc",
                note,
            )
            check(
                jax.config.jax_cpu_collectives_implementation == default
                and jax.config.read("jax_cpu_enable_async_dispatch")
                == async_default,
                "no wrapper leaves JAX's config untouched",
                jax.config.jax_cpu_collectives_implementation,
            )

        with _env():
            note = bootstrap._select_cpu_collectives(lib)
            check(
                "MPItrampoline" in note and lib in note,
                "a found wrapper reports MPI and its path",
                note,
            )
            check(
                jax.config.jax_cpu_collectives_implementation == "mpi",
                "a found wrapper switches JAX to MPI",
                jax.config.jax_cpu_collectives_implementation,
            )
            check(
                jax.config.read("jax_cpu_enable_async_dispatch") is False,
                "choosing MPI also pins dispatch inline",
                jax.config.read("jax_cpu_enable_async_dispatch"),
            )
            jax.config.update("jax_cpu_collectives_implementation", default)
            jax.config.update("jax_cpu_enable_async_dispatch", async_default)

        # ``test_laminar_smoke`` asserts that no unexpected padded-size
        # rounding note appears in a run's stdout by scanning for the
        # word; a startup line of ours must not trip it.
        with _env():
            note = bootstrap._select_cpu_collectives(None)
            check(
                "rounded" not in note,
                "the diagnostic does not read as a rounding note",
                note,
            )


# ── runner ───────────────────────────────────────────────────────────


def main() -> int:
    case_wrapper_discovery()
    case_launcher_params()
    case_coordinator_port()
    case_coordinator_host()
    case_solo_launch()
    case_local_device_ids()
    case_bootstrap()
    case_cpu_collectives()

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll bootstrap tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
