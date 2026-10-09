#!/bin/bash --login
#
# dnsjax: a production run on CPU nodes under SLURM.
#
# Copy this file into the run folder (the folder that holds the run's
# parameters.toml, and where the run writes its streams, its snapshots
# and this job's log), fill in the fields marked <...>, and submit it
# from there:
#
#     cp /path/to/dnsjax/examples/slurm/cpu.slurm /path/to/run/
#     cd /path/to/run && sbatch cpu.slurm
#
# What to fill in, in order: the #SBATCH lines marked <...> just below;
# section 1, for each run; section 2, once per machine.  Sections 3 and
# 4 need no edits.  sbatch refuses a placeholder left in an #SBATCH
# line, and the script one left in sections 1 and 2.
#
# Every simulation parameter comes from parameters.toml: the command
# line carries only the device grid, the platform and the wall-clock
# budget.  To continue from a snapshot, name it there:
#
#     [init]
#     snapshot = "state00012.tar"
#
# To stop the run early, with a final snapshot: `rm RUNNING` in the
# run folder.
#
# Tested on ARCHER2, an HPE Cray EX system (Cray MPICH, Spindle,
# Lustre).  The machine-specific parts (Spindle, Lustre striping, the
# Cray library path) switch themselves off where their tools are
# missing.

# ---- Resources: per run ----------------------------------------------
#SBATCH --job-name=dnsjax
#SBATCH --nodes=<NODES>
# Ranks per node T, and cores per rank c, from the node's physical
# core count C:
#   - a full node: T = C and c = 1 (the default, and the cheapest);
#   - fewer ranks per node, for more memory per rank: choose T to
#     divide C and set c = C / T (on a 128-core node: T = 64, c = 2;
#     T = 32, c = 4).  The ranks then spread over the whole node and
#     all of its memory channels, instead of packing onto its first T
#     cores.  Billed as the whole node either way: T x c = C;
#   - part of a node, shared with other jobs (under --mem): c = 1 and
#     T = the cores you ask for.
# More cores do not make a rank faster: each runs one thread.  Section
# 1's np0 x np1 counts nodes x T.
#SBATCH --ntasks-per-node=<TASKS_PER_NODE>
#SBATCH --cpus-per-task=1
# Memory: choose ONE of the next two lines, and leave the other
# disabled ('##').  --mem asks for this much memory per node (0: all
# of it); --exclusive asks for whole nodes, whose cores, memory and
# network no other job shares.
#SBATCH --mem=<MEMORY_PER_NODE>
##SBATCH --exclusive
#SBATCH --time=<HH:MM:SS>

# ---- Resources: per machine and account ------------------------------
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
# Where the machine needs a QoS, enable this line by deleting one '#'.
##SBATCH --qos=<QOS>

# ---- Defaults ----------------------------------------------------------
# SIGUSR1 to every task 5 minutes before the time limit: dnsjax's stop
# request (a final snapshot, then STOPPED).  A backstop for
# MAX_WALL_TIME below, which normally ends the run first.
#SBATCH --signal=USR1@300
# The job's log, in the run folder: <job name>-<job id>.out.
#SBATCH --output=%x-%j.out


# ======================================================================
# 1. Per run
# ======================================================================

# The device grid, one device per rank on CPU: np0 x np1 must equal the
# nodes times the tasks per node.  How to choose it: docs/scaling.md
# ("Choosing the device grid") and the Distribution docstring in
# src/dnsjax/parameters.py.
NP0="<NP0>"
NP1="<NP1>"

# The solver's wall-clock budget, an ISO 8601 duration (PT23H30M for a
# 24-hour job): some 20-30 minutes below --time, so that the run stops
# in time to write its final snapshot.
MAX_WALL_TIME="<PThhHmmM>"


# ======================================================================
# 2. Once per machine
# ======================================================================

# The dnsjax checkout, and the Python that runs it: by default the
# checkout's own environment (made by `uv sync`).  Point PYTHON
# elsewhere if dnsjax is installed into another interpreter.  The ranks
# write no bytecode (section 3), so compile it once after every update
# of the checkout:
#     uv sync --compile-bytecode && .venv/bin/python -m compileall -q src
DNSJAX_DIR="<PATH_TO_DNSJAX>"
PYTHON="$DNSJAX_DIR/.venv/bin/python"

# The MPIwrapper library that routes JAX's CPU collectives through the
# machine's MPI, faster than its TCP fallback, gloo
# (docs/cpu-collectives.md).  It must sit where every node can read
# it.  Empty: not set, and the solver looks for libmpiwrapper.so on
# LD_LIBRARY_PATH, else runs on gloo; it prints which it took.
MPITRAMPOLINE_LIB=""

# Environment modules the run needs, loaded in this order: at least the
# MPI that MPIwrapper was built against.  Empty: none.
MODULES=""

# Spindle: one process per node reads Python's ~800 modules for all of
# that node's ranks, instead of every rank reading them from the shared
# filesystem.  On many nodes the start-up is much faster, and kinder to
# the filesystem, for some memory per node.  1: use it where it can be
# found (on PATH, or as the module `spindle`), and run without it
# otherwise; 0: never.
USE_SPINDLE=1

# Extra srun flags the machine needs, e.g. --mpi=pmix for Open MPI.
SRUN_EXTRA=""


# ======================================================================
# 3. Defaults (no edits needed)
# ======================================================================

# This script runs on CPUs.
PLATFORM=cpu

# Each node's memory, sampled every MEMWATCH_INTERVAL seconds into
# memwatch/<job id>/ (scripts/memory_watch.py), summarized at the end:
# peak and typical use per node.  The record survives an out-of-memory
# kill, which the solver's own closing memory lines do not.  A peak
# between two samples still shows in the summary's cgroup column (the
# kernel's high-water mark, where it keeps one).  0: off.
MEMWATCH=1
MEMWATCH_INTERVAL=10

# On Lustre, stripe the run folder over every storage target: each
# snapshot is one file that many ranks write into at once.  Applies to
# the files created from now on; skipped on any other filesystem.
# 0: leave the striping alone.
LUSTRE_STRIPE=1

# The cores' clock for srun --cpu-freq, in kHz (e.g. 2250000).
# Empty: the machine's default.
CPU_FREQ=""

# One thread per rank for the host-side numerical libraries.  (The
# solver pins XLA's own thread pool to one thread per rank itself.)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Python on many ranks.  No rank writes bytecode: thousands of
# processes would race to write the same files on a shared filesystem.
# No rank looks for modules in the run folder, which `python -m` would
# otherwise search first on every import, and where a stray file could
# shadow a module.  A PYTHONPATH from the login environment goes for
# the same reasons.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH

# Every srun step inherits this script's environment, even where jobs
# are submitted with --export=NONE: the ranks would otherwise run
# without the settings above, and without MPITRAMPOLINE_LIB (on gloo).
export SLURM_EXPORT_ENV=ALL

# No core files: a crash on N full nodes would write one per rank, each
# the size of that rank's memory.  srun --propagate=CORE (below)
# carries the limit to the ranks.  `ulimit -c unlimited` to debug.
ulimit -c 0


# ======================================================================
# 4. Checks and launch
# ======================================================================

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1

fail() {
    echo "cpu.slurm: $*" >&2
    exit 1
}

# Refuse a field left unfilled or inconsistent, before anything runs.
for name in NP0 NP1 MAX_WALL_TIME DNSJAX_DIR; do
    case "${!name}" in
        *"<"*">"*) fail "fill in $name (section 1 or 2)" ;;
    esac
done
for name in NP0 NP1; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] ||
        fail "$name must be a positive integer, not '${!name}'"
done
NTASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
[ $((NP0 * NP1)) -eq "$NTASKS" ] ||
    fail "np0 x np1 = $NP0 x $NP1 is not the job's $NTASKS tasks" \
        "($SLURM_NNODES nodes x $SLURM_NTASKS_PER_NODE)"
[[ "$MAX_WALL_TIME" == P* ]] ||
    fail "MAX_WALL_TIME must be an ISO 8601 duration such as PT23H30M"
[ -d "$DNSJAX_DIR" ] || fail "DNSJAX_DIR: no directory $DNSJAX_DIR"
[ -x "$PYTHON" ] || fail "PYTHON: $PYTHON is not executable"
[ -f parameters.toml ] ||
    fail "no parameters.toml in $PWD: submit from the run folder"

if [ -n "$MODULES" ]; then
    # Split into module names on purpose.
    # shellcheck disable=SC2086
    module load $MODULES || fail "module load $MODULES failed"
fi

# A rank that cannot read the wrapper falls back to gloo while its peers
# take MPI, and the run hangs: refuse a path that does not exist here.
if [ -n "$MPITRAMPOLINE_LIB" ]; then
    [ -f "$MPITRAMPOLINE_LIB" ] ||
        fail "MPITRAMPOLINE_LIB: no file $MPITRAMPOLINE_LIB"
    export MPITRAMPOLINE_LIB
    # On a Cray system, the loaded modules' libraries come first, so
    # that the wrapper reaches the MPI they select rather than the one
    # it was linked against.  No empty entry: the loader would read one
    # as the current directory.
    if [ -n "${CRAY_LD_LIBRARY_PATH:-}" ]; then
        p=${LD_LIBRARY_PATH:-}
        export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH${p:+:$p}
    fi
else
    unset MPITRAMPOLINE_LIB
fi

# Spindle wraps the solver's srun; its own processes take a job step,
# so the solver's step must overlap it.
launch=()
if [ "$USE_SPINDLE" = 1 ]; then
    command -v spindle > /dev/null || module load spindle > /dev/null 2>&1
    if command -v spindle > /dev/null; then
        # Spindle serves the files under these directories: every one
        # Python imports from, each both as Python reaches it and as
        # its real path, since Spindle matches paths as they are opened
        # and a filesystem may be reached through a symlink.
        prefixes=$("$PYTHON" - << 'EOF'
import os
import sys

dirs = {sys.base_prefix, sys.prefix, *filter(os.path.isdir, sys.path)}
print(":".join(sorted({p for d in dirs for p in (d, os.path.realpath(d))})))
EOF
        )
        # The interpreter runs from where it is installed: copied into
        # Spindle's cache, a relocatable Python build (uv's) looks for
        # its standard library beside the copy, and fails.
        launch=(spindle --slurm --reloc-aout=no --reloc-exec=no
            --python-prefix="$prefixes")
    else
        echo "note: Spindle not found; running without it"
    fi
fi

if [ "$LUSTRE_STRIPE" = 1 ] && [ "$(stat -f -c %T .)" = lustre ]; then
    lfs setstripe -c -1 -S 4M . ||
        echo "note: lfs setstripe failed; striping left alone"
fi

memdir=memwatch/$SLURM_JOB_ID
start=$("$PYTHON" -c 'import tomllib
print(tomllib.load(open("parameters.toml", "rb")).get("init", {})
      .get("snapshot", ""))')
newest=$(ls -1 state*.tar 2> /dev/null | sort | tail -n 1)
echo "== $(date): job $SLURM_JOB_ID, $SLURM_NNODES node(s) x" \
    "$SLURM_NTASKS_PER_NODE tasks ($SLURM_JOB_NODELIST)"
echo "   (np0, np1) = ($NP0, $NP1), wall-clock budget $MAX_WALL_TIME"
echo "   python: $PYTHON"
echo "   MPI wrapper: ${MPITRAMPOLINE_LIB:-none set}"
echo "   launch prefix: ${launch[*]:-none}"
echo "   memory record: $([ "$MEMWATCH" = 1 ] && echo "$memdir" || echo off)"
echo "   starts from: ${start:-the initial condition in parameters.toml}"
echo "   newest snapshot in this folder: ${newest:-none}"
if [ -z "$start" ] && [ -n "$newest" ]; then
    echo "   WARNING: a fresh start numbers its snapshots from" \
        "init.isnap0 again, overwriting those here; to continue," \
        "set [init] snapshot in parameters.toml"
fi

# The memory sampler: one per node, sharing the nodes with the solver
# (--overlap).  Wait, at most 30 s, for every node's first sample, the
# baseline before the solver starts.
if [ "$MEMWATCH" = 1 ]; then
    srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
        --ntasks-per-node=1 --cpus-per-task=1 \
        "$PYTHON" "$DNSJAX_DIR/scripts/memory_watch.py" sample \
        --out "$memdir" --interval "$MEMWATCH_INTERVAL" &
    memwatch_pid=$!
    for _ in $(seq 30); do
        n=$(ls "$memdir"/memwatch_*.csv 2> /dev/null | wc -l)
        [ "$n" -ge "$SLURM_NNODES" ] && break
        sleep 1
    done
fi

# One rank per physical core (--hint=nomultithread), consecutive ranks,
# which form the np1 groups, on adjacent cores (block:block); one
# failed rank ends the step rather than leaving the others to hang.
srun_args=(
    --nodes="$SLURM_NNODES"
    --ntasks="$NTASKS"
    --ntasks-per-node="$SLURM_NTASKS_PER_NODE"
    --cpus-per-task="$SLURM_CPUS_PER_TASK"
    --hint=nomultithread
    --distribution=block:block
    --kill-on-bad-exit=1
    --propagate=CORE
)
[ -n "$CPU_FREQ" ] && srun_args+=(--cpu-freq="$CPU_FREQ")
[ ${#launch[@]} -gt 0 ] && srun_args+=(--overlap)
# Split into flags on purpose.
# shellcheck disable=SC2206
srun_args+=($SRUN_EXTRA)

echo "== $(date): launching"
"${launch[@]}" srun "${srun_args[@]}" "$PYTHON" -m dnsjax \
    --dist.np0 "$NP0" --dist.np1 "$NP1" --dist.platform "$PLATFORM" \
    --stop.max_wall_time "$MAX_WALL_TIME"
rc=$?

# The memory summary.  The solver's own closing lines, above, add each
# rank's peak and what the run still held at the end ("Peak host
# memory", "Resident host memory at the end").
if [ "$MEMWATCH" = 1 ]; then
    touch "$memdir/stop"
    wait "$memwatch_pid"
    echo "== memory, from $memdir:"
    "$PYTHON" "$DNSJAX_DIR/scripts/memory_watch.py" summary "$memdir" |
        tee "$memdir/summary.txt"
fi

# How the run ended (a RUNNING left behind: it was killed outright).
for f in FINISHED STOPPED TERMINATED RUNNING; do
    [ -f "$f" ] && { echo "== $(date): $f"; cat "$f"; }
done
exit "$rc"
