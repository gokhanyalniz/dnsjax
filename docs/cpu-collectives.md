# Faster CPU collectives

Across processes on CPU, JAX exchanges data over TCP (`gloo`) unless it
can route the collectives through MPI instead — which is faster, and
which costs a multi-process run nothing to arrange, being under `mpirun`
by definition. This page is only about that choice. GPU runs are
unaffected: their collectives go through NCCL.

Start at the [README](../README.md) for the solver itself, and at
[`SCALING.md`](../SCALING.md) for how the work is split across devices in
the first place.

## Building the wrapper

JAX embeds [MPItrampoline](https://github.com/eschnett/MPItrampoline) for
this but ships no MPI of its own, so it needs a thin wrapper built
against the machine's MPI:

```bash
git clone https://github.com/eschnett/MPIwrapper.git
cd MPIwrapper
cmake -S . -B build -DMPIEXEC_EXECUTABLE=mpiexec \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$HOME/mpiwrapper
cmake --build build
cmake --install build
export MPITRAMPOLINE_LIB=$HOME/mpiwrapper/lib/libmpiwrapper.so
```

A multi-device CPU run picks MPI up by itself once `MPITRAMPOLINE_LIB` is
set, or once `libmpiwrapper.so` sits on `LD_LIBRARY_PATH`, and prints
which backend it ended up with. Without the wrapper it stays on `gloo`
and says so; `JAX_CPU_COLLECTIVES_IMPLEMENTATION` overrides the choice
either way.

## On a cluster

Every rank looks for the wrapper on its own filesystem, so export the
variable in the job script rather than relying on a path some nodes may
not mount: a node that cannot see the library falls back to `gloo` while
its peers take MPI, and the run then hangs.

On macOS the search cannot fire at all — it scans `LD_LIBRARY_PATH` for a
`.so`, where macOS has `DYLD_LIBRARY_PATH`, a `.dylib` convention, and
SIP stripping that variable from spawned processes — so set
`MPITRAMPOLINE_LIB` explicitly there, and expect to find out whether the
macOS wheel carries the MPI collectives at all, which is untested.

## Why the ordering matters

All of this works because nothing in a `dnsjax` run touches MPI before
XLA does — XLA initializes it without checking whether it is already up,
so anything that gets there first breaks the run. Worth knowing only if
you add something that might. The rank bootstrap, the single-process
skip and the collectives auto-selection live in
`configure_jax_runtime` (`src/dnsjax/bootstrap.py`); the environment
variables it reads (`JAX_COORDINATOR_ADDRESS`, `JAX_COORDINATOR_PORT`,
`MPITRAMPOLINE_LIB`, `JAX_CPU_COLLECTIVES_IMPLEMENTATION`) are none of
them dnsjax parameters.
