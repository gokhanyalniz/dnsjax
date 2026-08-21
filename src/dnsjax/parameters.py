"""Simulation parameter management via Pydantic models and TOML files.

Configuration is layered, lowest priority first: hard-coded defaults ->
parameters embedded in a resumed snapshot (:func:`read_snapshot_params`)
-> ``parameters.toml`` (if present) -> command-line arguments.  The
snapshot layer is skipped for the parameters that must be known to
configure JAX *before* the snapshot is read (``dist.np0``, ``dist.np1``,
``dist.platform``, ``res.double_precision``) and for the
resume-decision fields ``init.snapshot`` / ``init.force_resume``
(recorded for lineage, never inherited); those come only from
defaults / TOML / CLI.  The global singletons ``params``,
``derived_params``, and ``padded_res`` are mutated in-place by
:func:`update_parameters` so that every module sees the same state.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from math import cos, pi, sin
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .flow_spec import UNSET

# The flow registry aggregates the per-flow parameter specs
# (``flows/*/specs/``) into the system list and family groupings
# (consumers import the groupings from ``flows.registry`` directly).
# ``viscoelastic_systems`` is the *rheology* axis and cuts across the
# geometry lists, each of which contains its own viscoelastic member
# (so a consumer picking machinery by geometry gets it right, and one
# asking "does this carry a conformation tensor?" reads the other
# list) -- the rationale, and the ordering rule for a chain that mixes
# the two, are in ``flows.registry``.  Import direction is strictly
# ``parameters -> flows.registry -> flows.*.specs -> flow_spec`` (the
# spec hooks receive ``params``/``derived_params`` as arguments), so
# no cycle exists.
from .flows.registry import (
    periodic_systems,
    spec_for,
    viscoelastic_systems,
    walled_systems,
)

ns_to_s: float = 10 ** (-9)  # nanoseconds to seconds


class Distribution(BaseModel):
    r"""Device distribution and backend platform.

    The total device count is ``np0 * np1``.  Both default
    to 1 (single device).

    Double parallelisation
    ----------------------
    ``np0`` splits the wall-normal axis (`$y$` / `$r$`) in
    physical space and the spanwise-wavenumber axis
    (`$k_z$` / `$m$`) in spectral space.  ``np1`` splits
    the spanwise axis (`$z$`) in physical space and the
    streamwise-wavenumber axis (`$k_x$`) in spectral space.
    Each device holds the full wall-normal extent in spectral
    space, so FD solves are unchanged.

    When ``np0 == 1`` (default), the decomposition collapses
    to a 1D scheme (only `$k_x$` / `$z$` distributed).

    Divisibility
    ------------
    No divisibility requirements: every axis a mesh
    direction splits is auto-padded, and each adjustment is
    reported by a startup diagnostic (padding costs FFT and
    solve work, so it is never silent).  The spectral `$k_z$`
    / `$k_x$` axes are zero-padded to the next multiple of
    ``np0`` / ``np1``; for wall-bounded flows the physical
    `$y$` axis is likewise zero-padded (stripped after the
    `$y \leftrightarrow k_z$` reshard); the padded physical
    sizes -- ``nz_padded`` (split by ``np1``) and, for
    periodic flows, ``ny_padded`` (split by ``np0``) -- are
    rounded up to the next FFT-friendly (7-smooth) multiple
    (:func:`round_up_padded_smooth`; marginally more
    oversampling, physically neutral).  Divisible,
    smooth-padding sizes avoid the padding overhead entirely.
    Note that CGL grids traditionally
    use ``ny = 2^k + 1`` (``N + 1`` collocation points for
    ``N`` Chebyshev polynomials), but any ``ny >= 2`` is
    valid: the code uses finite differences, not spectral
    Chebyshev transforms, and the Clenshaw--Curtis
    quadrature handles both even and odd ``ny``.

    Choosing the grid
    -----------------
    The two exchanges are not equivalent.  The ``np1`` one
    (`$z \leftrightarrow k_x$`) runs while the array still carries the
    **oversampled** spanwise extent, the ``np0`` one
    (`$y \leftrightarrow k_z$`) after the truncation to stored modes,
    so at the default oversampling ``np1`` moves `$3/2$` as many bytes
    (2.654 against 1.769 MB per device per forward+inverse pair at
    ``64 x 144 x 144`` on four devices).  And a second grid axis does
    not divide the first exchange more finely, it **adds** a second
    one: one all-to-all per transform on a 1D grid, two on a 2D one,
    each a synchronisation point.  Both the count and the volume are
    readable off the compiled program.

    Whatever the device type:

    1. **On one node, stay one-dimensional**, splitting on ``np0`` by
       default -- its exchange carries `$2/3$` of the bytes and its
       mode axis tiles far more coarsely on GPU.  Split on ``np1``
       instead when ``ny`` (``nr``) will not divide the device count
       or is too small for it.
    2. **Across nodes, align the grid with them**: ``np1`` = devices
       per node, ``np0`` = number of nodes.  ``jax.make_mesh`` lays
       the grid out row-major over the sorted devices and device ids
       group by process, so with one task per node the ``np1`` groups
       fall inside a node and the ``np0`` groups hold one device each.
       That confines the heavier exchange to the intra-node
       interconnect, and the network carries ``np0 - 1`` large
       messages per device instead of the many small ones a grid-wide
       exchange sends -- at equal volume (`$(N-g)/N^2 = (n-1)/(nN)$`
       per device, for `$N$` devices in `$g$`-device groups on `$n$`
       nodes).  Splitting on ``np1`` alone across nodes is the one
       arrangement to avoid: it puts the `$3/2$`-sized exchange on the
       network.
    3. **Snapshots** add only the same 1D preference -- a
       one-dimensional grid reshards once per save instead of twice.
       Write granularity does not enter the choice: the reshard trims
       the divisibility padding, so every grid writes one contiguous
       range per component per device (:mod:`dnsjax.snapshot`).

    **On CPU** the mode plane carries no tile round-up (the Pallas
    kernel never runs), so ``np1`` may be taken as far as the mode
    count allows, and one device per process makes ``np0 * np1`` the
    rank count.  Measured at four and eight ranks, the per-exchange
    cost dominates its volume: a 2D grid costs 9 to 19 % against the
    best 1D one, where the `$3/2$` volume difference between the two
    1D grids is worth some 18 % of the transform pair itself but only
    a few percent of the step around it (the transforms being roughly
    half of it).  That is why the 1D rule is the firm one and the axis
    a lesser trade.  Routing the collectives through MPI rather than
    gloo (below) speeds up every exchange, shifting weight from the
    per-exchange cost back toward volume.

    **On GPU** the mode plane is tiled, which makes ``np1`` the
    granular axis: keep ``(nx // 2) / np1`` a multiple of
    ``solver.pallas_block_m1`` (32), where ``(nz - 1) / np0`` need
    only clear ``pallas_block_m0`` (2).  A minimal-box ``nx = 32``
    split four ways leaves four streamwise modes per device, padded to
    32 -- lower the block size, or move the split to ``np0``.  With a
    fast intra-node interconnect and production-sized arrays the
    exchange is likelier limited by volume than by its per-exchange
    cost, and that is the regime where ``np0`` moving `$2/3$` of the
    bytes should tell; comparing the two 1D grids on the target
    machine is then worth one pair of runs.

    Process topology
    ----------------
    Only a **multi-process** run needs a launcher.  One
    process starts no distributed runtime at all
    (``bootstrap._bootstrap_distributed``), so a run that
    fits in one is launched directly -- no ``mpirun``, no
    coordinator, no MPI on the machine, ``uv run dnsjax ...``
    included.  On CPU that means exactly one device: several
    CPU devices in one process is oversubscription, and
    asking for it is refused with the ``mpirun -np N`` that
    works.

    ``np0 * np1`` counts *devices*, not processes: the mesh
    only requires ``jax.device_count() == np0 * np1``, so a
    multi-GPU run may be one process per device (the usual
    ``mpirun``/``srun -n N`` launch) **or a single process
    addressing all devices** -- which needs no launcher
    either, since a lone process takes every visible GPU.
    Under a launcher, that process is instead narrowed to its
    local rank's device unless
    ``JAX_LOCAL_DEVICE_IDS=0,1,...`` spans the GPUs
    (overrides the SLURM one-device-per-task heuristic),
    e.g. ``srun -n 1 --overlap`` inside a 4-GPU allocation
    with ``JAX_LOCAL_DEVICE_IDS=0,1,2,3 --dist.np0 2
    --dist.np1 2``.  Both topologies produce identical
    global meshes, trajectories, and snapshots (resume is
    np-agnostic), and both are validated on real multi-GPU
    hardware (``scripts/solver_benchmark.py``, 2026-07).
    The single-process form avoids cross-process NCCL
    entirely -- the reliable choice on single-node
    allocations whose multi-process collective stack is
    broken (observed: JAX 0.10.2 + NCCL 2.30 H100 nodes
    hang in the first execution of a large multi-collective
    program while every small-program collective works).
    Multi-node runs necessarily remain multi-process; use
    one task per node spanning that node's GPUs.

    In multi-process launches, never narrow per-task GPU
    visibility (SLURM ``--gpus-per-task`` and the like):
    NCCL's cuMem cross-process P2P import requires the peer
    device to be visible to the importing process and fails
    hard otherwise (``ncclP2pImportShareableBuffer ...
    Cuda failure 101 'invalid device ordinal'``).  Leave
    every job GPU visible to every task and select the
    per-task device explicitly (``JAX_LOCAL_DEVICE_IDS``;
    JAX's SLURM detection uses ``[SLURM_LOCALID]`` by
    default, which is correct under full visibility).

    The ranks find each other from the launcher environment
    when it describes itself fully enough, and otherwise from
    JAX's own cluster detection (SLURM under ``srun``, the
    cloud environments, Open MPI 4).  The layout comes from
    the MPI implementation (``OMPI_COMM_WORLD_*``, or the
    MPICH/MVAPICH2 equivalents) and the coordinator from
    ``JAX_COORDINATOR_ADDRESS``, else -- in order -- loopback
    when every rank is on this node, the launcher's own
    daemon URI, or the queueing system's node list
    (``PBS_NODEFILE``, the SLURM node list,
    ``LSB_DJOB_HOSTFILE`` / ``LSB_HOSTS``, ``PE_HOSTFILE``).
    That covers Open MPI 5, whose PRRTE launcher dropped the
    ``OMPI_MCA_orte_hnp_uri`` JAX's plugin keys on, and every
    scheduler JAX has no plugin for; a machine matching
    nothing is one ``JAX_COORDINATOR_ADDRESS`` export away,
    and says so.  The port is derived from a per-*launch*
    identifier so that concurrent runs inside one allocation
    cannot collide on it (``JAX_COORDINATOR_PORT``
    overrides).  Rank *discovery* only: GPU collectives
    remain NCCL, and ``JAX_LOCAL_DEVICE_IDS`` is honoured
    either way, so the single-process launch above is
    unaffected.

    A launch the environment reports as **one process** skips
    the distributed runtime altogether -- there is nothing to
    coordinate -- which is why a single-rank run needs no
    coordinator, and no site knowledge, anywhere.  Setting
    ``JAX_LOCAL_DEVICE_IDS`` opts back in, since narrowing a
    process to a subset of its devices is JAX's to apply.

    CPU runs: threads per rank
    -------------------------
    **A CPU run takes one XLA thread per rank.**  Parallelism on CPU
    comes from MPI ranks and from nothing else; an intra-op thread pool
    is not a second axis to tune, and this holds however many devices
    the run has -- a lone process is pinned exactly like a rank of
    sixteen, and is not special-cased.  ``bootstrap.
    configure_jax_runtime`` applies it: ``NPROC`` sizes the pool and is
    set there with ``setdefault``, so ``export NPROC=<n>`` before
    launching overrides the pin for a deliberate experiment;
    ``--xla_cpu_multi_thread_eigen=false`` rides along as a small extra
    serialisation and is applied only while the pin is 1.

    Nothing measured here argues against the rule, which is the only
    role measurement has in this section: plane-Couette
    ``64 x 48 x 64``, 30 steps, gives 2 ranks 5.2 s at 1 thread against
    5.1 s at 8 and 4 ranks 3.2 s at 1 thread against 3.7 s at 4, and
    the same case at 1 rank 17.3 / 18.1 s/t at 1 thread against 17.6 /
    18.0 at 16 (interleaved, 16-core box) -- i.e. threads buy nothing
    at a realistic per-rank block size and can cost.  Do not re-open
    the question with another timing: a faster threaded arm would not
    change the rule, so measuring one is waste.  If a CPU run is
    device-starved, the answer is more ranks.

    CPU runs: cross-process collectives
    -----------------------------------
    JAX's CPU backend defaults to **gloo** (TCP) for cross-process
    collectives; routing them through MPI instead is faster, and a
    multi-device CPU run is launched under ``mpirun`` by definition --
    it is one process per device.  Such a run therefore selects MPI by
    itself whenever it can
    (``bootstrap.configure_jax_runtime``), and prints which backend it
    got.  What it needs is the MPItrampoline path JAX's MPI
    collectives dlopen: ``MPITRAMPOLINE_LIB`` pointing at an
    MPIwrapper-built ``libmpiwrapper.so``, or that library on
    ``LD_LIBRARY_PATH``.  Without one the run stays on gloo and says
    so (building the wrapper: ``README.md``, "Installation"); with
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION`` set, that choice wins
    outright.  Worth having: measured on a 16-core box at 4 ranks,
    plane-Couette ``32^3``, MPI runs at 0.80 s/t against gloo's 1.14
    (interleaved), on top of gloo's own strong scaling there (1.39x on
    2 ranks, 2.28x on 4).  By how much is a property of the target
    machine's interconnect, so it is worth timing again there.

    Selecting MPI also turns CPU async dispatch off -- XLA's MPI
    backend cannot take a communicator request from a thread pool, and
    the failure is load-dependent rather than obvious
    (``bootstrap._select_cpu_collectives``).  The numbers above already
    include that cost.  It applies to
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` as much as to the
    discovered choice: the two reach the same backend.

    The wrapper is looked for per rank, on that rank's own filesystem,
    so it has to be visible identically on every node -- export
    ``MPITRAMPOLINE_LIB`` in the job script rather than relying on a
    path that some nodes may not mount, or a node that cannot see it
    picks gloo while its peers pick MPI and the run hangs.  On macOS
    the discovery cannot fire at all (it scans ``LD_LIBRARY_PATH`` for
    ``libmpiwrapper.so``, where macOS has ``DYLD_LIBRARY_PATH``, a
    ``.dylib`` convention, and SIP stripping that variable from
    spawned processes), so an explicit ``MPITRAMPOLINE_LIB`` is the
    only route there -- and whether the macOS wheel carries the MPI
    collectives at all is untested.

    This holds only while XLA is the first thing in the process to
    initialize MPI, which is why the rank bootstrap above reads the
    environment rather than asking an MPI library -- the failure modes
    are ugly and late, see ``bootstrap._select_cpu_collectives``.
    """

    np0: int = Field(
        ge=1,
        default=1,
        description=(
            "Devices splitting the wall-normal physical axis and the "
            "spanwise/azimuthal spectral axis (np0 * np1 devices total)."
        ),
    )
    np1: int = Field(
        ge=1,
        default=1,
        description=(
            "Devices splitting the spanwise/azimuthal physical axis and "
            "the streamwise/axial spectral axis (np0 * np1 devices "
            "total)."
        ),
    )
    platform: Literal["cpu", "cuda", "rocm", "tpu"] = Field(
        default="cpu", description="JAX backend platform to run on."
    )


class Physics(BaseModel):
    """Physical parameters: Reynolds number, flow system, dealiasing."""

    re: float = Field(
        gt=0, default=1000, description="Reynolds number of the flow."
    )
    # Taylor-Couette control parameters (system == "taylor-couette").
    # Inner / outer cylinder Reynolds numbers, with gap d = r2 - r1:
    #   re1 = Omega1 * r1_dim * d / nu,  re2 = Omega2 * r2_dim * d / nu.
    # Sign convention: re1 >= 0; re2 may be negative (counter-rotation).
    # Case 1 (inner-driven): re1 > 0  -> Re_ref = re1.
    # Case 2 (outer-driven): re1 == 0, re2 > 0 -> Re_ref = re2.
    # ``update_parameters`` validates these, derives the circular-Couette
    # coefficients A0, B0 and radii (onto ``derived_params``), and sets
    # ``re = Re_ref`` so every downstream 1/re viscous/IMM/stats path is
    # reused unchanged.  See the annular branch of ``update_parameters``
    # and ``flows.wall_bounded.taylor_couette``.
    #
    # The quasi-Keplerian system (``system == "quasi-keplerian"``) reuses
    # ``re1`` as the inner Reynolds number Re_i and takes ``r_omega``
    # (below) instead of ``re2``; the annular branch derives and stores
    # ``re2`` so every downstream circular-Couette path is shared with
    # Taylor-Couette.  See ``flows.wall_bounded.quasi_keplerian``.
    re1: float | None = Field(
        default=None,
        description=(
            "Inner-cylinder Reynolds number Re_1 = Omega_1 r_1 d / nu "
            "(>= 0 by sign convention)."
        ),
    )
    re2: float | None = Field(
        default=None,
        description=(
            "Outer-cylinder Reynolds number Re_2 = Omega_2 r_2 d / nu "
            "(negative = counter-rotating)."
        ),
    )
    # Rotation number R_Omega (system == "quasi-keplerian"), following
    # Dubrulle et al. 2005:
    #   R_Omega = (1 - eta) (Re_i + Re_o) / (eta Re_o - Re_i),
    # with Re_i = re1 and Re_o the derived outer Reynolds number.  It is
    # constant along half-lines through the origin of (Re_o, Re_i) space.
    # The quasi-Keplerian regime (co-rotating, angular momentum
    # increasing / angular velocity decreasing outward -- linearly stable
    # by Rayleigh's criterion) is the open half-line -inf < R_Omega < -1,
    # bounded by the Rayleigh line R_Omega = -1 (Re_o = eta Re_i) and the
    # solid-body limit R_Omega -> -inf (Re_o = Re_i / eta).  The annular
    # branch of ``update_parameters`` requires re1 > 0 and r_omega < -1,
    # then derives re2 = re1 (1 - eta + R_Omega) / (eta R_Omega -
    # (1 - eta)).
    r_omega: float | None = Field(
        default=None,
        description=(
            "Rotation number R_Omega < -1 selecting a quasi-Keplerian "
            "regime; the outer Reynolds number is derived from "
            "(re1, r_omega, eta)."
        ),
    )
    # Viscoelastic (sPTT) control parameters, shared by both
    # viscoelastic systems ("viscoelastic-pipe" / "viscoelastic-dean";
    # see ``flows.wall_bounded.viscoelastic_{pipe,dean}`` and
    # ``geometries.wall_bounded.{cylindrical,annular}_viscoelastic``).
    # All ``None`` for other systems; unset values fall back to each
    # flow's own FieldSpec defaults.  beta 0.8 / epsilon 0.001 /
    # kappa 5e-5 are shared, but el and wi are not: Re := Wi/El is
    # derived, so those two *are* the Reynolds number, and each flow
    # picks the regime it is about -- el 80, wi 105 (Re ~ 1.3,
    # inertialess and strongly elastic) for the annulus; el 0.02,
    # wi 20 (Re = 1000, elasto-inertial) for the pipe.
    el: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Elasticity number El; the Reynolds number is derived as "
            "Re = Wi / El."
        ),
    )
    wi: float | None = Field(
        default=None, gt=0, description="Weissenberg number Wi."
    )
    beta: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Solvent-to-total viscosity ratio in (0, 1]: the solvent "
            "carries nu = beta/Re, the polymer stress (1-beta)/(Re Wi)."
        ),
    )
    # Deliberately unbounded above: the relaxation factor
    # f = 1 + epsilon (tr c - 3) is positive for every admissible c only
    # while epsilon <= 1/3 (above it, f flips sign once tr c < 3 - 1/eps,
    # turning relaxation into growth), but the literature range extends
    # past that and no realizable state reaches the threshold at the
    # shipped defaults.  The cnab2 split stays exact either way -- its
    # implicit half carries (1 - 3 epsilon)/Wi, which merely turns
    # anti-dissipative there and costs corrector contraction, not
    # correctness.
    epsilon: float | None = Field(
        default=None,
        ge=0,
        description=(
            "sPTT extensibility parameter (>= 0); f = 1 + epsilon "
            "(tr c - 3) is sign-definite only up to 1/3."
        ),
    )
    kappa: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Artificial conformation stress diffusivity; 0 makes the "
            "conformation transport purely hyperbolic (no wall BC on "
            "c)."
        ),
    )
    # Default "plane-couette": a wall-bounded flow that integrates
    # cleanly from the default random IC at the default dt (Kolmogorov +
    # random needs a smaller dt; see the corrector-contraction note in
    # the ``TimeStepping`` docstring).
    system: Literal[*periodic_systems, *walled_systems] = Field(
        default="plane-couette",
        description=(
            "Flow system to integrate; `dnsjax --help <system>` lists "
            "the parameters that apply to it."
        ),
    )
    oversampling_factor: int = Field(
        ge=2,
        default=3,
        description=(
            "Physical-space oversampling for dealiasing: (n+1)/2 grid "
            "points per mode dealias an n-th order nonlinearity "
            "(default 3 = the 3/2 rule)."
        ),
    )
    driving: Literal[
        "constant_pressure_gradient", "constant_bulk_velocity"
    ] = Field(
        default="constant_pressure_gradient",
        description=(
            "Hold either the mean streamwise/axial pressure gradient "
            "or the bulk velocity constant."
        ),
    )
    # Independent of ``driving``: Cartesian blocks the spanwise (z)
    # mean, annular the axial (z) mean; the azimuthal mean evolves
    # freely.
    block_mean_spanwise_velocity: bool = Field(
        default=False,
        description=(
            "Zero the mean velocity in the undriven homogeneous "
            "direction (Cartesian: spanwise z; annular: axial z)."
        ),
    )
    # Speed U_grid of the moving frame of reference, translating along
    # the homogeneous "grid" direction: streamwise x (Cartesian) or
    # axial z (cylindrical / annular).  The time derivative becomes
    # d/dt - U_grid d/dx_0, i.e. the *convective-form* frame term
    # +U_grid d/dx_0 u' = i k_0 U_grid u' is added to the RHS -- a
    # mode-diagonal, non-stiff, divergence-free (projection-neutral)
    # term, integrated implicitly (inside the iterative-CN corrector;
    # via ``_l_bf`` for CN/AB2).  It de-advects snapshots, improves
    # temporal accuracy, and relaxes the corrector-contraction dt limit
    # (the advecting velocity drops to ``U - U_grid``) -- though not
    # cnab2's explicit self-advection CFL, which is set by the
    # frame-invariant ``u' x omega'``.  NOT the
    # rotational-form splitting `omega' x c + grad(c . u')` of the
    # removed first implementation, whose explicit `c d/dy u'` piece
    # was wall-stiff and blew up.  When ``None`` (default) it resolves
    # to the laminar bulk velocity in the grid direction (1/2 for both
    # pipes, 2/3 plane-Poiseuille, 0 otherwise); see
    # ``update_parameters`` and ``derived_params.u_grid``.  Only
    # meaningful for wall-bounded systems (periodic flows reject it).
    # A changed ``u_grid`` on resume is trajectory-defining (the stored
    # fields drift between frames); pre-feature snapshots resume into
    # the new default.
    u_grid: float | None = Field(
        default=None,
        description=(
            "Speed of the moving frame of reference along the "
            "streamwise/axial grid direction; the default is the "
            "flow's laminar bulk velocity."
        ),
    )


class Geometry(BaseModel):
    r"""Domain size and optional tilt angle for the forcing direction.

    Wall-normal grid selection (precedence order):

    1. ``wall_grid`` (file path): load a custom grid from file.
       A custom grid always overrides dnsjax's grid generation.
    2. ``grid_type``: generate a named grid at startup.
    3. Default (``grid_type`` unset): ``update_parameters`` resolves
       it to a concrete value from the flow spec
       (``FlowSpec.grid_type_default``) -- full CGL (``"cgl"``) for
       the Cartesian / annular families, and for the cylindrical
       family ``"half-cgl"`` under the default ``iterative-cn``
       scheme or ``"rigged-cgl"`` under ``cnab2`` (see below).
       Because the resolved value is concrete, snapshots embed the
       grid they actually ran and a resume pins it -- the
       scheme-dependent default never silently re-grids an old
       trajectory.

    Setting both ``wall_grid`` and ``grid_type`` is an error, and each
    family accepts only its own grid names (the per-flow surface and
    ``validate_parameters`` enforce it):

    - Cartesian / annular: ``"cgl"`` (plain Chebyshev-Gauss-Lobatto)
      or ``"tanh"`` (two-sided tanh-stretched, both walls clustered,
      strength ``grid_stretch``).
    - Cylindrical: ``"half-cgl"``, ``"rigged-cgl"``, or
      ``"half-tanh"`` (a one-sided tanh over `$(0, 1]$` clustering
      only at the outer wall -- there is no inner wall, so the
      two-sided ``"tanh"`` name does not apply).

    **Cylindrical radial CGL grids.**  Both keep the ``ny`` outermost
    *positive* points of an auxiliary CGL grid on `$[-1, 1]$`, so the
    near-axis spacing is `$\Delta r \approx \pi/(2 n_y)$` and no
    degree of freedom lives in `$[0, r_0)$` (parity ghosts close the
    FD stencils across the axis; the quadrature covers the segment):

    - ``"rigged-cgl"`` (the ``cnab2`` default): the positive half of
      a `$(2 n_y + 1)$`-point grid.  The odd total's centre point
      falls exactly on the coordinate-singular axis and is dropped,
      so the innermost point sits at `$r_0 \approx \Delta r$`
      (`$= \sin(\pi/(2 n_y))$`).
    - ``"half-cgl"`` (the ``iterative-cn`` default): the positive
      half of a `$2 n_y$`-point grid, staggered so
      `$r_0 \approx \Delta r/2$`
      (`$= \sin(\pi/(2\,(2 n_y - 1)))$`) -- half the rigged value.

    Rationale: the near-axis *azimuthal* advection CFL
    `$\propto 1/r_0$` is a stability artifact of explicit stepping
    evaluated at grid points only, so it relaxes `$\propto r_0$` at a
    truncation-level accuracy cost.  The rigged grid's `$2\times$`
    larger `$r_0$` doubles the admissible explicit-``cnab2`` ``dt``
    (measured), which is why it is the ``cnab2`` default; the tighter
    half-CGL axis makes ``cnab2`` blow up at low ``dt`` (near-axis
    explicit instability), so half-CGL is restricted to the
    implicitly-iterated ``iterative-cn`` scheme, which integrates it
    cleanly, gains its finer near-axis resolution, and defaults to
    it.  See ``build_radial_cgl_grid`` in ``cylindrical.py``.
    """

    lx: float = Field(
        gt=0,
        default=4.0,
        description="Streamwise period of the domain.",
    )
    lz: float = Field(
        gt=0,
        default=4.0,
        description="Spanwise period of the domain.",
    )
    tilt_degree: float = Field(
        gt=-180,
        le=180,
        default=0,
        description=(
            "Tilt angle (degrees) of the driving direction within the "
            "homogeneous plane."
        ),
    )
    # Required for every annular system (Taylor-Couette,
    # quasi-Keplerian, Dean); the viscoelastic annulus uses ``delta``
    # instead.
    eta: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description=(
            "Radius ratio eta = r1/r2 of the annulus (unit gap: "
            "r1 = eta/(1-eta), r2 = 1/(1-eta))."
        ),
    )
    # Azimuthal wedge fundamental wavenumber m0 (annular and cylindrical
    # families only; rejected elsewhere).  The azimuthal domain is the
    # reduced wedge theta in [0, 2*pi/m0), so ``update_parameters``
    # derives lz = 2*pi/m0 and the resolved azimuthal wavenumbers are the
    # multiples m = m0 * {0, 1, ..., nz/2-1, -(nz/2-1), ..., -1}.  m0 > 1
    # restricts the simulation to the m0-periodic subspace (invariant
    # under the dynamics), cutting azimuthal cost/memory by a factor m0
    # at fixed nz: the same physical azimuthal resolution as a full
    # circle with m0*nz modes.  Default 1 (full circle).  A changed m0 on
    # resume is trajectory-defining (geo section).
    #
    # In *physical* space the wedge is fully resolved, not decimated: the
    # FFT is purely index-based and never sees theta, so it maps mode
    # index j to grid index p and returns one period of the field it was
    # handed.  Every retained harmonic being a multiple of m0, that
    # period *is* the wedge -- the nz (dealiased nz_padded) points span
    # [0, 2*pi/m0) at spacing dtheta = lz/nz, i.e. m0-times *finer* than
    # the full circle at the same nz, exactly what resolving m0-times
    # higher wavenumbers requires.  Equivalently the code solves in
    # phi = m0*theta in [0, 2*pi), with m0 entering only where a physical
    # wavenumber (``Fourier.m``) or length (``lz``) is needed; every
    # ``geo.lz`` consumer then follows automatically (the CFL azimuthal
    # spacing nz/lz, the random_field / localized_rolls generators,
    # ``analysis/_core`` lengths).  Pinned end-to-end by the
    # ``wedge_nonlinear`` case in ``tests/test_quasi_keplerian.py``: a
    # wedge decimated over the full azimuth would evaluate the
    # pseudo-spectral product on an m0-times too coarse grid and fail it
    # outright.
    m0: int = Field(
        ge=1,
        default=1,
        description=(
            "Azimuthal wedge fundamental wavenumber: simulate the "
            "m0-periodic wedge theta in [0, 2*pi/m0) (1 = full "
            "circle), cutting azimuthal cost/memory by m0 at fixed "
            "resolution."
        ),
    )
    # Default 11 (applied in the viscoelastic ``derive`` hook).
    delta: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Inner radius of the viscoelastic annulus in half-gap "
            "units (gap fixed at 2: r1 = delta, r2 = delta + 2)."
        ),
    )
    wall_grid: Path | None = Field(
        default=None,
        description=(
            "File with a custom wall-normal grid (one point per "
            "line, wall first); overrides grid_type."
        ),
    )
    grid_type: (
        Literal["cgl", "tanh", "half-cgl", "rigged-cgl", "half-tanh"] | None
    ) = Field(
        default=None,
        description=(
            "Named wall-normal grid; unset resolves to the flow's "
            "default.  Cartesian/annular: 'cgl', 'tanh'; cylindrical: "
            "'half-cgl', 'rigged-cgl', 'half-tanh'."
        ),
    )
    grid_stretch: float = Field(
        gt=0,
        default=1.5,
        description=(
            "Stretching of the tanh grids (larger = stronger wall "
            "clustering); tanh grids only."
        ),
    )


class Resolution(BaseModel):
    """Grid resolution (number of Fourier modes before dealiasing)."""

    nx: int = Field(
        ge=1,
        default=128,
        description=(
            "Streamwise Fourier modes (= physical grid points before "
            "dealiasing)."
        ),
    )
    ny: int = Field(
        ge=1,
        default=128,
        description=(
            "Wall-normal grid points (wall-bounded) or shear-direction "
            "Fourier modes (periodic box)."
        ),
    )
    nz: int = Field(
        ge=1,
        default=128,
        description=(
            "Spanwise Fourier modes (= physical grid points before "
            "dealiasing)."
        ),
    )
    fd_order: int = Field(
        ge=2,
        default=8,
        description=(
            "Wall-normal finite-difference accuracy order p; "
            "stencils span p+1 points (D1) / p+2 points (D2), "
            "one-sided at the walls, so order p holds everywhere. "
            "openpipeflow's 9-point stencils correspond to "
            "fd_order = 2*i_KL = 8 (its i_KL = 4 is a stencil "
            "half-width, not an accuracy order)."
        ),
    )
    # Wall-bounded only (all three families), **on by default**.  It
    # buys a discretely exact projection with a **reformulation** of
    # the implicit step; the price is a truncation-level residual moved
    # into a momentum equation nothing solves.  Every measurement below
    # says that trade is worth making everywhere, which is why it is
    # the default rather than an opt-in.
    #
    # *Setting it to ``False``* selects the **legacy** primitive
    # Kleiser-Schumann `$(v, p)$` scheme (each geometry's
    # ``_<geometry>_primitive_imm.py``).  It is kept, tested and
    # supported, but not recommended: a state it steps carries the
    # `$O(1)$` *relative* discrete divergence described next.  Two
    # reasons remain to select it -- reproducing a trajectory computed
    # before the default moved, and the one corner where the default
    # costs corrector iterations: a **deep annulus** (small ``geo.eta``)
    # at tight ``step.corrector_tolerance``, where the `$(u_r,
    # \omega_r)$` pair's Picard-lagged spin partners contract slowly
    # (the measured table, and why that degradation is loud rather than
    # silent: ``annular._imm_iteration_vw``).
    #
    # *What it enforces.*  The primitive influence-matrix method's
    # continuity argument (Kleiser-Schumann; Canuto, Hussaini,
    # Quarteroni & Zang 1988, sec. 7.3) is derived for *continuous*
    # differentiation operators.  Two discrete identities have to hold
    # for the stepped state's divergence to vanish:
    # `$\nabla\cdot\nabla = L_k$` (i.e. `$D_1 D_1 = D_2$`) and
    # `$[D_1, D_2] = 0$`.  Independent Fornberg fits satisfy neither,
    # and -- separately -- replacing the momentum wall rows by
    # Dirichlet rows leaves an unaccounted residual that the
    # divergence's own `$D_1$` spreads into the interior.  So a state
    # the legacy path steps carries a discrete divergence that is O(1)
    # *relative*: a convergent truncation error, physically inert for
    # resolved fields, but not zero.
    #
    # *The mechanism* (one, in all three geometries, since
    # 2026-07-26).  Advance the **wall-normal velocity and vorticity**
    # instead of the three velocity components, and *reconstruct* the
    # tangential pair from them.  Continuity is then an algebraic
    # identity -- exact at every row including the walls, for any
    # operator, grid or axis fit -- and the pressure is eliminated
    # discretely, never formed.  `$D_1$` and `$D_2$` stay individually
    # Fornberg-fit and the band stays at ``fd_order``.  The
    # wall-normal-velocity equation is the pressure-eliminated
    # fourth-order one, integrated as two second-order banded solves
    # that commute exactly (Tuckerman 1989; Luchini & Quadrio 2006 is
    # the FD-in-`$y$` precedent) -- no fourth-order operator is
    # assembled anywhere.  Tangential no-slip is not imposed but
    # *emerges* from the reconstruction, so the tangential wall values
    # become a live diagnostic of influence-matrix health.
    #
    # Per geometry: **Cartesian** advances `$(v, \omega_y)$`, four
    # per-mode banded solves down to three.  **Annular** advances the
    # `$(u_r, \omega_r)$` pair, whose two slots share one Helmholtz
    # operator (`$m_{\mathrm{eff}}^2 = m^2+1$`, the spin-block
    # diagonal): four solves down to three, four band families down to
    # three, `$u_\pm$` unchanged.  **Pipe** the same, except that the
    # pair's `$-2im/r^2$` spin coupling cannot be lagged near the axis
    # (it diverges: measured contraction 1.13 on the plain-`$r$` fit,
    # 19.1 on the retired `$x = r^2$` axis fit), so the **spin quad**
    # `$(\Phi_\pm, \omega_\pm)$` is advanced through the *existing*
    # `$H_{k,\pm}$` families, which diagonalise that coupling exactly
    # -- five solves over three band families, with only the quad's two
    # free wall differences taken from the corrector iterate (four wall
    # values against two conditions is what the exact diagonalisation
    # costs).  **No geometry changes what it carries**:
    # the evolved scalars are re-derived from the carried state at the
    # top of each corrector pass and reconstructed away at its exit, so
    # snapshots, probes, forcing, diagnostics, the analysis package and
    # resume are identical under both formulations.  The price is two
    # wall rows per
    # mode (the influence coefficients cannot be carried), a bounded
    # truncation-level substitute -- ``cartesian._imm_iteration_vw``
    # carries the argument and the measurement.  Construction, boundary
    # conditions and the retired routes: the
    # ``cartesian._imm_iteration`` (shared record),
    # ``annular._imm_iteration_vw`` (cylindrical algebra) and
    # ``cylindrical._imm_iteration_vw`` (the quad) docstrings.
    #
    # *Efficacy* (measured, ``fd_order = 8``, ``ny = 25`` / ``ny = 97``,
    # one step from a random IC, seed 7 -- ten steps from an
    # axis-regular rolls IC on the pipe; ``tests/test_imm_continuity``).
    # Stepped-state relative divergence, ``legacy -> default``:
    #
    #   plane-couette   4.5e-2 -> 2.9e-16   1.1e-3 -> 1.6e-15
    #   taylor-couette  6.4e-2 -> 5.6e-16   5.7e-4 -> 1.9e-15
    #   pipe            2.8e-2 -> 1.1e-15   1.5e-5 -> 8.2e-15
    #
    # -- round-off everywhere, and following no `$h^p$` law at all (the
    # mild growth with `$N_y$` is the longer `$D_1$` dot product, not
    # truncation), because continuity here is an identity rather than
    # something a solve delivers; which is why every default-formulation
    # bound is asserted at every ``--ny``.  These replace the
    # operator-identity route's floors (4.2e-14 Cartesian, 8.0e-6
    # annular, 5.6e-5 pipe), each set by a commutator that route could
    # not remove.  The wall-bounded *temporal* error improves too:
    # plane-Couette iterative-CN self-convergence goes from ``1.3e-2``
    # at order ~0.5 on the legacy path to ``3.6e-5`` at order ~1.2 on
    # the default, and Taylor-Couette likewise (the divergence residual
    # **was** the dominant projection-splitting error) -- pinned by
    # ``tests/test_temporal_order.py``.
    #
    # *Price.*  Exact continuity is bought by *not* imposing the
    # tangential momentum combination, so what continuity gains, that
    # equation loses.  Measured end to end on the Cartesian pair --
    # the same random IC stepped once by each scheme, differenced in
    # the Helmholtz norm `$\max|\tilde H \delta|/\max|\tilde H u|$` --
    # the two answers differ by ``2.4e-3`` (``ny = 25``) / ``3.2e-5``
    # (``ny = 97``) in the tangential pair and ``8.9e-3`` / ``1.8e-5``
    # in `$v$`: truncation-level, refining at roughly third to fourth
    # order, with **no plateau** (the signature that would mean a
    # formulation error rather than a truncation one).  The
    # ``CHI-MOM`` figure ``tests/test_imm_continuity.py`` prints
    # (``4.5e-2`` / ``1.6e-3``) is a cruder *upper bound* on the same
    # quantity.  Nothing reads the residual back -- the difference
    # between this and the rejected projection below -- so it neither
    # accumulates nor re-excites; the stepped energy budget in fact
    # closes *tighter* on the default (``2.8e-3`` vs the legacy path's
    # ``5.1e-3``, ``tests/test_energy_budget.py``), as it must when
    # pressure does
    # no work on an exactly solenoidal field.  There is no operator
    # price at all (same `$D_1$`, same direct-fit `$D_2$`, same band)
    # and operator storage *drops*; against that, `$L(Lv)$` is applied
    # in the explicit half, and the reconstruction's `$1/k^2$` (
    # `$1/(k_z^2 + m^2/r^2)$` in the cylindrical geometries) amplifies
    # the gravest mode by `$1/k_{\min}$` (`$O(1-10)$` for sane boxes).
    #
    # *Rejected alternatives.*  (1) **Operator-side identities**
    # (shipped on annular/pipe until 2026-07-26): `$D_2 := D_1 D_1$`
    # plus the CHQZ (7.3.51)-(7.3.58) boundary closure.  It works, but
    # cannot reach round-off in a cylindrical geometry (the metric
    # commutator `$[D_1, 1/r] \ne -1/r^2$` survives; on the pipe a
    # parity invariant forbids both parities' commutators vanishing at
    # once), it widens every banded operator, it costs an order in the
    # `$D_2$` truncation constant, and -- because a composed `$D_2$`
    # is not grid-scale-dissipative -- it made the pipe unstable from a
    # grid-white random IC.  All four drawbacks are gone.
    # (2) **Commutator cancellation**: feeding the commutator back into
    # the Poisson RHS reaches machine-zero but contracts like
    # `$N_y^{-2}$`.  (3) **State-side tangential projection**
    # (2026-07-24): back-solving the tangential pair from continuity at
    # the primitive `$v$` zeroes the interior divergence and passes
    # every *linear* gate, but is violently unstable nonlinearly
    # (x5-10 per step at the gravest modes, worse per unit time at
    # smaller ``dt``, not cured by the boundary closure) -- Kleiser's
    # tau-method instability (CHQZ p. 219) in FD form.  (4) Merely
    # solving an `$\omega_y$` Helmholtz beside the existing `$(v, p)$`
    # IMM and reconstructing is the *same state map* as (3), so it
    # inherits the instability: only advancing the wall-normal velocity
    # by the pressure-eliminated dynamics escapes it.  (5) Decoupling
    # the annular `$(u_r, \omega_r)$` pair the way `$u_\pm$` decouples
    # `$(u_r, u_\theta)$` is impossible -- it mixes two vector fields;
    # the exactly-decoupled candidates are enumerated and dismissed in
    # the ``annular._imm_iteration_vw`` docstring.
    #
    # *Measured step cost*, ``legacy -> default``.  The per-mode banded
    # solve count goes 4 -> 3 on the Cartesian and annular families and
    # 4 -> **5** on the pipe, which is the one place the default costs
    # throughput (~+6 % per step; its axis forces the exact spin-quad
    # diagonalisation, doubling the evolved scalars against only two
    # wall conditions -- ``cylindrical._imm_iteration_vw``).  Against
    # that, the corrector contracts in fewer iterations: measured as a
    # *paired* run (one configuration, one backend, the formulation the
    # only difference), the pipe's ``c/it`` drops ``1.00 -> 0.10`` and
    # Taylor-Couette's ``0.09 -> 0.00``, because the reconstruction
    # removes the projection error the corrector was working against
    # (consistent with Kleiser's report, via CHQZ p. 220, of *lower*
    # time-step stability limits when the boundary correction is
    # omitted).  Do **not** expect ``test_random_smoke.py`` to reprint
    # those: its ``*-legacy-imm`` entries deliberately run different
    # ``Re``/box/resolution from their default counterparts, so the
    # ``c/it`` values it prints side by side are not a controlled pair.
    # One configuration on one backend either way: treat the net
    # speedup as a bonus, not a guarantee.
    consistent_imm: bool = Field(
        default=True,
        description=(
            "Make the influence-matrix projection discretely "
            "consistent (**the default**) by advancing the "
            "wall-normal velocity and vorticity and reconstructing "
            "the tangential components: a stepped state's discrete "
            "divergence is then round-off at any resolution, on the "
            "same operators and with less operator storage -- a "
            "solve fewer in the plane and annular geometries, one "
            "more in the cylindrical -- at the cost of a "
            "truncation-level tangential-momentum residual no solve "
            "reads back.  False selects the legacy primitive (v, p) "
            "Kleiser-Schumann scheme instead, whose stepped state "
            "carries an O(1) relative discrete divergence; it is "
            "kept for reference and is not recommended.  "
            "Trajectory-defining, and inherited from a resumed "
            "snapshot like every other [res] field -- so a run "
            "continued from a snapshot written before this became "
            "the default stays on the legacy scheme (a clean "
            "continuation, reported in the startup printout) until "
            "the resuming run overrides it explicitly."
        ),
    )
    double_precision: bool = Field(
        default=True,
        description=(
            "Use double precision (float64/complex128); False runs "
            "single precision."
        ),
    )


class Initiation(BaseModel):
    """Initial condition: from a snapshot, a random field (default), or
    laminar.

    Start-mode precedence (resolved in ``__main__.py``): a provided
    ``snapshot`` file (a single-file tar snapshot; see
    :mod:`dnsjax.snapshot`) wins over every in-process mode -- and
    *only* a real snapshot satisfies it: a ``snapshot`` path that is
    not one (a typo, an unrelated file) aborts the run rather than
    falling through to a mode the user did not ask for;
    otherwise ``start_from_laminar`` (the laminar / closed-form base
    state); otherwise ``localized_rolls`` (an in-process deterministic
    localized-spot perturbation, wall-bounded only);
    otherwise ``random_field`` -- an in-process random divergence-free
    perturbation, which is **the default**: a run with no snapshot and
    no explicit mode selected starts from a random IC.  The ``random_*``
    knobs feed :func:`dnsjax.ic.random_field.generate_random_state`; the
    ``localized_rolls_*`` knobs feed
    :func:`dnsjax.ic.localized_rolls.generate_localized_rolls`.

    Resume policy: when ``snapshot`` is a dnsjax snapshot, ``it``/``t``/
    ``isnap`` are inherited only when none of the Physics/Geometry/
    Resolution parameters were overridden to a value different from the
    snapshot's (a *continuation*).  Any such change starts a NEW
    trajectory by default (``it = t = isnap = 0``); ``force_resume``
    keeps the run continuous instead.  See
    :func:`trajectory_defining_changes`.
    """

    start_from_laminar: bool = Field(
        default=False,
        description=(
            "Start from the laminar base state (zero perturbation; "
            "total-field flows start on the analytical laminar "
            "profile)."
        ),
    )
    snapshot: Path | None = Field(
        default=None,
        description=(
            "Snapshot file to start/resume from; takes precedence "
            "over every other start mode."
        ),
    )
    t0: float = Field(
        default=0,
        description="Initial simulation time of a fresh start.",
    )
    it0: int = Field(
        default=0,
        description="Initial time-step counter of a fresh start.",
    )
    # Mirrors ``it0``/``t0``: the fresh-start value; a *continuation*
    # resume inherits the resumed file's index + 1 instead.
    isnap0: int = Field(
        ge=0,
        default=0,
        description=(
            "Initial snapshot counter (snapshots are named state{isnap}.tar)."
        ),
    )
    force_resume: bool = Field(
        default=False,
        description=(
            "Continue the resumed trajectory (inherit t/it/isnap) even "
            "when trajectory-defining parameters changed; hard "
            "resolution/system mismatches still reject."
        ),
    )
    random_field: bool = Field(
        default=True,
        description=(
            "Start from a random divergence-free perturbation -- the "
            "default start mode when no other mode is selected "
            "(total-field flows add the analytical laminar profile)."
        ),
    )
    random_amplitude: float = Field(
        default=0.1,
        description=("Target L2 norm of the random initial perturbation."),
    )
    random_smoothness: float = Field(
        gt=0,
        lt=1,
        default=0.4,
        description=(
            "Spectral decay rate of the random perturbation "
            "(0 < s < 1; larger = smoother)."
        ),
    )
    random_seed: int = Field(
        default=1,
        description=(
            "Seed of the random-IC generator (device-count independent)."
        ),
    )
    # Cartesian-only, and defaulted **on** there by the flow spec: only
    # the Cartesian flows have their (kx, kz) = (0, 0) conservation
    # laws established, so every other flow defers this field.  The
    # model default stays False -- that is the inert value the deferred
    # check in ``validate_parameters`` compares a direct assignment
    # against.  Also read by ``dnsjax-twin`` for its partner field; the
    # localized-rolls perturbation stays mean-free whatever it is set to
    # (its (0, 0) content is a cubic in y, which the compatibility
    # conditions annihilate -- ``ic/localized_rolls.py``), and the
    # runtime ``[force]`` kicks still reject the mode (``extensions``).
    random_mean_flow: bool = Field(
        default=False,
        description=(
            "Also perturb the mean (kx = kz = 0) streamwise/spanwise "
            "profile, conditioned on its conservation laws: "
            "compatibility with no-slip at both walls, and an "
            "unchanged bulk velocity in each direction whose mean the "
            "driving holds."
        ),
    )
    # Radially windowed to zero at both walls (the reference restart
    # recipe); shares ``random_smoothness`` for the spectral envelope.
    random_conformation_amplitude: float = Field(
        default=700.0,
        description=(
            "Amplitude of the random symmetric-tensor perturbation "
            "added to the laminar conformation in the viscoelastic "
            "random IC."
        ),
    )
    # A compact fixed-physical structure localized in every homogeneous
    # direction (growing a box length adds laminar around the spot);
    # precedence between ``start_from_laminar`` and ``random_field``.
    localized_rolls: bool = Field(
        default=False,
        description=(
            "Start from a deterministic localized-rolls ('turbulent "
            "spot') perturbation."
        ),
    )
    localized_rolls_amplitude: float = Field(
        default=0.1,
        description="Peak |u'| of the localized-rolls perturbation.",
    )
    localized_rolls_width: float = Field(
        default=2.0,
        description=(
            "Physical localization half-width of the rolls (flow units)."
        ),
    )
    localized_rolls_wavelength: float = Field(
        default=4.0,
        description=(
            "Cross-roll spanwise wavelength (flow units; ignored by "
            "the pipe, whose cross-section is the fixed m = +-1 "
            "mode)."
        ),
    )


class Outputs(BaseModel):
    """Output frequency controls (in time-step counts)."""

    # All cadences count time steps taken.
    it_stats: int | None = Field(
        default=None,
        description=(
            "Steps between stats.dat records; unset disables the stream."
        ),
    )
    # Measured from the current state at the step's first
    # nonlinear-term evaluation (no extra Fourier transforms).
    it_steps: int | None = Field(
        default=None,
        description=(
            "Steps between CFL time-step diagnostics in steps.dat; "
            "unset disables the stream."
        ),
    )
    it_snapshot: int | None = Field(
        default=None,
        description=("Steps between periodic snapshots; unset disables them."),
    )
    # Same on-device buffering and file format as ``stats.dat``.
    it_corrector: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between corrector diagnostics (iteration count and "
            "error) in corrector.dat; unset disables the stream; "
            "requires it_error_check <= it_corrector.  All-zero rows "
            "under triply-periodic cnab2 (no corrector there)."
        ),
    )
    # Between checks the host enqueues steps ahead of the device (JAX
    # async dispatch); corrector divergence is therefore detected up
    # to ``it_error_check`` steps late, each late step bounded by
    # ``max_corrector_iterations``.  1 restores a per-step check (and
    # a per-step host-device sync).  Also the cadence of the
    # non-finite (NaN/inf) guard on the synced corrector error and
    # perturbation energy (a hit aborts the run with exit code 3; the
    # ``__main__`` module docstring documents the full guard).
    it_error_check: int = Field(
        ge=1,
        default=10,
        description=(
            "Steps between host syncs of the corrector error and "
            "perturbation energy (convergence, laminarization and "
            "non-finite guards); larger = deeper async dispatch but "
            "later detection."
        ),
    )
    nbuffer: int = Field(
        ge=1,
        default=100,
        description=(
            "Rows buffered on device before each diagnostics flush "
            "(stats/steps/corrector/probes) to disk."
        ),
    )
    stats_precision: int = Field(
        ge=1,
        le=17,
        default=9,
        description=("Significant digits of the .dat diagnostic streams."),
    )
    # A *minimum* width; a larger ``isnap`` is not truncated.
    snapshot_pad_width: int = Field(
        ge=1,
        default=5,
        description=(
            "Minimum zero-padded width of the snapshot counter in "
            "state{isnap}.tar filenames."
        ),
    )
    # The periodic-snapshot path reuses the ``it_stats`` computation
    # when the iterations coincide, else it computes the stats once.
    snapshot_embed_stats: bool = Field(
        default=True,
        description=(
            "Embed the state's stats into every snapshot "
            "(_dnsjax_stats.json member)."
        ),
    )
    snapshot_save_initial: bool = Field(
        default=True,
        description=(
            "Save the initial condition as a snapshot on a fresh "
            "(non-continuation) start; independent of it_snapshot."
        ),
    )
    snapshot_save_final: bool = Field(
        default=True,
        description=(
            "Save the final state as a snapshot when the run "
            "terminates (skipped when it was just written)."
        ),
    )
    snapshot_write_mode: Literal["concurrent", "serial"] = Field(
        default="concurrent",
        description=(
            "How processes write the shared snapshot tar: "
            "'concurrent' disjoint-range writes (POSIX/parallel "
            "filesystems) or 'serial' rank-ordered writes (NFS-safe)."
        ),
    )


class TimeStepping(BaseModel):
    r"""Time integration parameters.

    Two schemes (``scheme``), both semi-implicit (implicit viscous,
    IMM pressure) and both second-order:

    - ``"iterative-cn"`` (default): Euler predictor + iterative
      Crank-Nicolson corrector (Willis 2017).  Implicitness *c* is
      applied to **both** the viscous *and* the nonlinear term; the
      nonlinear ``c N^{n+1}`` is resolved by the corrector fixed-point
      iteration (the RHS is re-evaluated until the correction converges).
      Stable well past the advective CFL; costs ``2 + num_corrector``
      RHS/FFT evaluations per step.

      Wall-bounded split corrector (``split_corrector``).  In the
      rotational perturbation form the iterated RHS contains the
      *linear* coupling -- ``L_bf``, the frame term, and (per
      ``implicit_mean_coupling``) ``L_mf`` -- i.e. exactly the
      FFT-free ``_l_bf`` the CN/AB2 scheme makes implicit.  The split
      corrector iterates that part FFT-free: each outer (FFT)
      iteration freezes the pure self-advection ``N_nl = get_rhs -
      l_bf`` at the last full evaluation, converges the coupling by
      re-evaluating only ``l_bf`` (matvecs / a mean-mode ``psum``, no
      transform), then refreshes the full RHS once and corrects.  The
      composite fixed point is the same CN equation and the loop
      always exits on a fresh-RHS correction, so trajectories agree
      within ``corrector_tolerance`` and the reported ``num_c`` /
      ``error`` keep their meaning (extra FFT evaluations / last
      fresh-RHS correction norm) -- ``corrector.dat`` is comparable
      across the setting.  It is an **opt-in** (``split_corrector``,
      default **off**): at realistic ``dt`` the corrector converges in
      ~1--2 iterations for *every* flow (measured, including the
      total-field Dean and high-Wi viscoelastic-dean -- unsplit ``c``
      stays 2--3, far from the cap, at ``dt = 0.01``), so the unsplit
      corrector is both correct and faster (the split is measured a
      few % slower at production sizes on GPU, up to tens of % on small
      problems).  The split only pays off once ``dt`` is pushed far
      enough that the unsplit corrector approaches
      ``max_corrector_iterations`` -- e.g. Dean at ``dt = 0.15`` (an
      unrealistically large step), where the unsplit corrector hits the
      cap (``c = 10``) and fails while the split converges it FFT-free.
      When enabled: the tail launches an implicit solve only while the
      coupling estimate still moves the state
      (``c dt ||l_bf(u_j) - l_bf(u_{j-1})|| > tol``, a cheap test), so a
      fluctuation-driven iteration adds one ``l_bf`` evaluation and a
      norm, not a solve -- and a step whose first correction already
      meets tolerance costs 2 FFT evaluations either way.  A split
      corrector that fails to reach tolerance automatically redoes the
      step with the unsplit corrector (``lax.cond``, stdout
      diagnostic), pinning the worst case to the unsplit path.
      Triply-periodic flows have no coupling (``l_bf_fn = None``) and
      always run unsplit.
    - ``"cnab2"``: Crank-Nicolson viscous (implicitness *c*) + 2nd-order
      Adams-Bashforth nonlinear (explicit ``1.5 N^n - 0.5 N^{n-1}``).
      **One** expensive nonlinear/FFT evaluation per step; the previous
      nonlinear RHS is carried by the main loop, seeded by a discarded
      priming ``step_cnab2(state, zeros)`` call while ``iterative-cn``
      takes the very first integration step (see ``step_cnab2`` in
      ``timestep.py``).  Explicit
      *self-advection*, so ``dt`` is advective-CFL-limited -- a net win
      (~3x fewer FFTs) on CFL-limited (turbulent) runs.

      Wall-bounded caveat and coupling corrector.  In the rotational
      perturbation form the nonlinear term includes the *linear*
      base-flow coupling ``L_bf = u' x curl(U) + U x omega'``, whose
      ``U d(u')/dy`` piece is a wall-normal derivative.  On the
      wall-clustered CGL grid (``dy ~ 1/N^2`` near the wall) treating
      ``L_bf`` explicitly gives it a Chebyshev-type CFL ``dt <~ 1/N^2``
      -- far below the advective limit and *amplitude-independent* -- so
      a naive explicit-AB2 cnab2 blows up at CFL << 1 for moving-wall
      flows (plane-Couette, Taylor-Couette, where ``U ~ O(1)`` at the
      wall).  To restore the advective limit, wall-bounded cnab2 makes
      only the *self-advection* ``u' x omega'`` explicit (AB2) and treats
      ``L_bf`` implicitly (Crank-Nicolson) via an **FFT-free** fixed-point
      corrector (it re-evaluates only the matrix-free ``L_bf``, no FFT),
      so ``corrector_tolerance`` / ``max_corrector_iterations`` **do**
      apply here.  The first step self-starts with ``iterative-cn``,
      and if the coupling corrector fails to converge on a step (its
      Picard rate reaches 1 -- only at ``dt`` well past the advective
      limit, e.g. plane-Couette ``dt >~ 0.2``) that step automatically
      falls back to a full ``iterative-cn`` step (a stdout diagnostic
      is printed).  The
      residual ``dt`` bound is then the ordinary explicit self-advection
      CFL of the *fluctuations* on the clustered grid (stationary-wall
      flows -- Poiseuille, pipe, Dean -- are bounded only by this, their
      ``L_bf`` being mild as ``U -> 0`` at the wall).  Where it binds is
      geometry-specific: for the **pipe** it is the *near-axis
      azimuthal* advection (the innermost radial node
      ``r_0 ~ pi/(2 ny)`` on the default rigged-CGL grid makes
      ``CFL_th = dt |u_th(r_0)| nz/(2 pi r_0)`` the dominant
      column -- linear in ``nz`` and in the fluctuation amplitude, and
      a *weak* AB2 imaginary-axis instability, so it needs sustained
      ``CFL_th >~ 0.5`` and pass/fail is trajectory-marginal near the
      boundary; the **rigged-CGL** radial grid sits at
      ``r_0 ~ Delta r`` -- twice the half-CGL ``Delta r/2`` --
      which raises the admissible cnab2 ``dt`` (measured ``dt* ~
      0.0125 -> 0.0175`` at the 32^3 / Re = 1800 reference config;
      ``iterative-cn`` rides ``CFL_th ~ 1.5--2`` there at growing
      corrector cost) and is why it is the ``cnab2``-default radial
      grid, whereas the tighter half-CGL grid
      (``geo.grid_type = 'half-cgl'``, the ``iterative-cn`` default)
      destabilises cnab2 and is
      restricted to ``iterative-cn``); Cartesian flows feel the
      near-wall ``dy ~ 1/N^2`` spacing instead.  A strongly
      non-normal base flow
      (counter-rotating Taylor-Couette) amplifies the explicit
      self-advection error further into a delayed blow-up needing
      ~8x smaller ``dt``; the coupling corrector converges throughout
      (it is *not* a corrector failure), so the fallback typically
      does not fire -- at coarser resolution the induced corrector
      stress can trip it into rescuing single steps (``ny = 32``
      completes via fallbacks; ``ny = 48`` diverges with the
      corrector still converged).  These are inherent
      explicit-nonlinear
      limits, not the coupling bug: such regimes want ``iterative-cn``
      or a smaller ``dt``.  Triply-periodic cnab2 has none of this (uniform
      Fourier grid, no coupling stiffness): it is the plain one-FFT
      no-corrector explicit-AB2 step.

      ``implicit_mean_coupling`` (wall-bounded cnab2 only, default on)
      additionally folds the coupling with the *instantaneous mean
      flow* -- ``L_mf = u' x curl(mean u') + (mean u') x omega'``, the
      same cross-product structure as ``L_bf`` with the time-varying
      mean profile ``extract_mean_mode(u')`` in place of ``U`` -- into
      the implicit coupling term, still FFT-free (the mean mode is a
      ``psum``; each geometry ``_l_bf`` adds the mean profiles onto the
      base-flow profiles, the coupling being linear in the profile
      pair).  The explicit AB2 remainder is then the pure
      fluctuation-fluctuation advection: the mean-flow *distortion*
      (streaks; for total-field Dean the entire evolving mean profile,
      whose ``L_bf`` is otherwise zero) no longer rides the explicit
      term, removing its advective-CFL contribution.  Decisive for
      Dean: at ``nz = 64, dt = 0.15`` (mean-flow ``CFL_th ~ 0.5``)
      coupling-off NaNs by ``t ~ 4`` while coupling-on runs clean,
      matching ``iterative-cn`` at ~4 FFT-free Picard iterations per
      step vs its ~4 FFT evaluations; neutral where the limit is
      fluctuation-driven (the pipe near-axis) and only mildly slowing
      the counter-rotating-TC blow-up.  The double-counted
      mean-mean product is a purely wall-normal (radial) profile at the
      mean mode, absorbed by the mean pressure in the projection -- and
      the AB2/CN split is second-order consistent for *any* choice of
      the implicit functional, since the explicit part is always the
      exact remainder ``get_rhs - l_bf``.

    ``implicitness`` *c* is the Crank-Nicolson split weight
    (``c = 0.5`` = second-order trapezoidal): in ``"iterative-cn"`` it
    weights both the viscous *and* the nonlinear term (see the geometry
    ``_imm_iteration``); in ``"cnab2"`` it weights the viscous term (and,
    wall-bounded, the implicit base-flow coupling), while the explicit
    AB2 self-advection is independent of *c*.

    Corrector convergence is ``dt``-limited, not CFL-limited
    ------------------------------------------------------------
    The corrector is a fixed-point iteration whose contraction rate
    scales with ``dt``, so a ``corrector failed to converge`` at *low*
    CFL -- the final error only just above ``corrector_tolerance`` --
    means the step is too large to contract within
    ``max_corrector_iterations``, **not** a blow-up: reduce ``dt`` (or
    raise the cap).  The limit is per-flow and unrelated to the
    advective CFL bounding ``cnab2``: random-IC Kolmogorov needs
    ``dt = 0.005`` (capped in ``tests/test_random_smoke.py``) while the
    wall-bounded flows contract fine at the default ``dt = 0.01``.
    ``phys.u_grid`` relaxes it (the advecting velocity drops to
    ``U - U_grid``) -- this contraction limit only, not the advective
    CFL bounding ``cnab2``, whose ``u' x omega'`` term is
    frame-invariant (see the ``u_grid`` field docs).

    Adaptive CFL time stepping (``adaptive``)
    -----------------------------------------
    With ``adaptive = True`` the main loop re-selects ``dt`` from the
    measured total CFL every ``cfl_cadence`` steps (a one-scalar host
    sync, the same stall class as ``outs.it_error_check``).  The
    controller (:mod:`dnsjax.adaptive`) proposes
    `$\Delta t \, \mathrm{CFL}_{\mathrm{target}} / \mathrm{CFL}$`,
    caps it by ``dt_max`` and the per-evaluation growth ratio
    ``dt_max_change``, floors it by ``dt_min`` and the shrink ratio
    ``dt_min_change`` (``0`` = uncapped shrink, the safe default),
    and accepts only when the result moves ``dt`` by more than the
    relative deadband ``dt_threshold`` (suppressing rebuild churn
    from CFL noise).  An accepted change rebuilds the
    ``dt``-dependent implicit-operator / IMM pytree leaves on device
    (the flow module's ``set_dt``: a jitted rebuild costing a few
    implicit solves -- no FFTs, no stepper recompilation) and, under
    ``cnab2``, weights the next AB2 step with
    `$\kappa = \Delta t_n / \Delta t_{n-1}$` (variable-step AB2; see
    ``make_stepper`` in ``timestep.py``).  ``params.step.dt`` tracks
    the live value: every snapshot embeds it, so a resume continues
    at the adapted ``dt`` unless an explicit TOML/CLI ``step.dt``
    overrides it, and ``steps.dat`` always carries a ``dt`` column.

    ``dt_max`` is required when adaptive: besides bounding the step
    it anchors the setup-time no-pivot stability check -- the
    Helmholtz diagonal `$1/\Delta t + c\,\nu\,k^2$` is least
    dominant at ``dt_max``, so one checked factorization there
    covers every ``dt <= dt_max`` and the runtime rebuilds skip the
    check.  Pick ``cfl_target`` for the scheme: ``cnab2``'s explicit
    self-advection is CFL-limited (target well below 1), while
    ``iterative-cn`` tolerates CFL above 1 but stays bound by the
    corrector-contraction ``dt`` limit above -- enforce that through
    ``dt_max``.  The ``[probes]`` / ``[force]`` extensions reject
    adaptive runs: their streams and readers assume a uniform
    sample/kick interval ``it_* x dt``.
    """

    scheme: Literal["iterative-cn", "cnab2"] = Field(
        default="iterative-cn",
        description=(
            "'iterative-cn': predictor + iterative CN corrector, "
            "stable past the advective CFL.  'cnab2': explicit AB2 "
            "nonlinear term, one FFT evaluation per step, "
            "advective-CFL-limited."
        ),
    )
    dt: float = Field(gt=0, default=0.01, description="Time-step size.")
    implicitness: float = Field(
        ge=0,
        le=1,
        default=0.5,
        description=(
            "Crank-Nicolson implicit weight c (0.5 = second-order "
            "trapezoidal)."
        ),
    )
    corrector_tolerance: float = Field(
        gt=0,
        default=1e-5,
        description=(
            "Convergence tolerance of the corrector fixed point: the "
            "iterative-cn corrector everywhere, and the FFT-free "
            "wall-coupling corrector of wall-bounded cnab2."
        ),
    )
    max_corrector_iterations: int = Field(
        ge=1,
        default=10,
        description=(
            "Iteration cap of the same correctors; failure to "
            "converge at low CFL means dt is too large to contract "
            "-- reduce dt."
        ),
    )
    # Folds ``L_mf`` into the FFT-free coupling term ``_l_bf`` shared
    # by the wall-bounded CN/AB2 scheme and the split ``iterative-cn``
    # corrector.  See the class docstring.
    implicit_mean_coupling: bool = Field(
        default=True,
        description=(
            "Fold the instantaneous mean-flow coupling into the "
            "implicit FFT-free coupling term (wall-bounded cnab2 and "
            "the split iterative-cn corrector)."
        ),
    )
    # Same CN fixed point; coupling-driven corrector iterations stop
    # costing one FFT evaluation each.  At realistic ``dt`` the
    # unsplit corrector converges in ~1--2 iterations for every flow
    # (measured), so the split only pays off once ``dt`` pushes the
    # unsplit corrector towards its iteration cap.  See the class
    # docstring.
    split_corrector: bool = Field(
        default=False,
        description=(
            "Opt-in, wall-bounded iterative-cn only: iterate the "
            "linear coupling FFT-free between full-RHS corrector "
            "refreshes; no effect on cnab2."
        ),
    )
    # Adaptive CFL controller knobs; semantics in the class docstring
    # and :mod:`dnsjax.adaptive`.
    adaptive: bool = Field(
        default=False,
        description=(
            "Re-select dt at runtime from the measured total CFL "
            "(see the TimeStepping docstring); requires dt_max."
        ),
    )
    cfl_target: float = Field(
        gt=0,
        default=0.5,
        description=(
            "Total-CFL setpoint of the adaptive controller (safety "
            "folded in; cnab2 needs a target well below 1)."
        ),
    )
    dt_min: float = Field(
        gt=0,
        default=1e-6,
        description=(
            "Adaptive floor on dt; the run continues at the floor "
            "(the non-finite diagnostics still abort a blow-up)."
        ),
    )
    dt_max: float | None = Field(
        gt=0,
        default=None,
        description=(
            "Adaptive cap on dt; required when adaptive (also "
            "anchors the setup-time no-pivot stability check)."
        ),
    )
    dt_min_change: float = Field(
        ge=0,
        default=0.0,
        description=(
            "Adaptive floor on the per-evaluation ratio "
            "dt_new/dt_old (0 = shrink uncapped, the safe default)."
        ),
    )
    dt_max_change: float = Field(
        gt=0,
        default=1.2,
        description=(
            "Adaptive cap on the per-evaluation ratio "
            "dt_new/dt_old (growth-rate limiter)."
        ),
    )
    dt_threshold: float = Field(
        ge=0,
        default=0.05,
        description=(
            "Relative deadband: keep dt unless the restricted "
            "proposal moves it by more than dt_threshold * dt."
        ),
    )
    cfl_cadence: int = Field(
        ge=1,
        default=10,
        description=(
            "Steps between CFL reads / adaptive-controller "
            "evaluations (each read is a host sync)."
        ),
    )


class Termination(BaseModel):
    """Stopping criteria for the simulation."""

    max_sim_time: float | None = Field(
        default=None,
        description=("Stop once the simulation time reaches this value."),
    )
    max_wall_time: timedelta | None = Field(
        default=None,
        description=(
            "Stop after this much wall-clock time (ISO 8601 "
            "duration, e.g. 'PT11H30M')."
        ),
    )
    # ``E'`` is read on the host every ``outs.it_error_check`` steps
    # (the corrector-error sync point), so detection lags by up to
    # that many steps.  For the total-field flows ``E'`` is the
    # kinetic energy of the deviation from the analytical laminar
    # profile.  Disabled in all tests.
    check_laminarization: bool = Field(
        default=True,
        description=(
            "Stop once the perturbation energy E' drops below "
            "laminarization_threshold (the flow relaminarized)."
        ),
    )
    laminarization_threshold: float = Field(
        gt=0,
        default=1e-9,
        description=(
            "Perturbation-energy threshold of the laminarization check."
        ),
    )


class Solver(BaseModel):
    """Numerical-kernel execution configuration.

    Linear-solver backends (pallas / dense) and pseudo-spectral
    transform batching.  These knobs select *how* the numerics are
    executed (speed / memory trade-offs), never the results.
    """

    # ``"pallas"`` (the default for all wall-bounded systems): the
    # production backend -- a one-program-per-mode sequential banded
    # sweep via a Pallas/Triton kernel on GPU (single- and multi-GPU
    # validated 2026-07), the same banded math as a sequential
    # pure-JAX sweep on CPU, and the smallest operator storage
    # (``O(N_y p)`` no-pivot banded factors per mode vs the dense
    # ``O(N_y^2)``).  Operators are assembled directly in banded
    # storage by each geometry's ``_build_{Lk,Hk}_band_gpu`` via the
    # shared ``solvers._assemble_banded_operator`` helper, then
    # factored and stability-checked once at setup by
    # ``solvers._build_pallas_operator`` (an unstable no-pivot LU is
    # a hard error; a merely ill-conditioned operator prints a notice
    # -- see ``pallas_stability_tol``).  The solve is a
    # shard_map-local region (each device runs the kernel on its
    # local mode-plane block; no communication).
    # ``"dense"``: full ``Ny x Ny`` pivoted LU factors per Fourier
    # mode.  The *reference* backend: the mathematically readable
    # formulation of the operators and the regression oracle the
    # Pallas path is tested against -- not a production backend (a
    # wall-bounded run selecting it prints a warning).
    # Triply-periodic systems have no wall-normal matrix solves (the
    # implicit step is diagonal in spectral space), so no ``solver``
    # backend applies there: the field is wall-bounded-only (absent
    # from the periodic parameter surfaces) and the periodic
    # geometry never reads it.
    backend: Literal["pallas", "dense"] = Field(
        default="pallas",
        description=(
            "'pallas': the production banded solver (smallest "
            "storage, fastest).  'dense': the reference Ny x Ny LU "
            "backend kept for readability and regression checks."
        ),
    )
    # ``"pallas"`` backend only: one Pallas program solves a
    # ``bm0 x bm1`` tile of Fourier modes, vectorising the banded
    # sweep across the tile.  ``1`` is one program per mode; ``> 1``
    # coalesces mode loads and fills more SIMD lanes (default 2: the
    # H100 tuning, 4 warps/program).  The mode plane is padded up to
    # whole tiles (a masked partial-tile band load miscompiles on real
    # Triton -- see ``solvers._pallas_banded_solve``): factors once at
    # construction, the RHS per solve.  The padded modes cost memory
    # and solve work proportional to the roundup fraction --
    # negligible at DNS mode counts, but worth shrinking the tile for
    # when the plane is small relative to it (e.g. ``nx/2 < 32``).
    pallas_block_m0: int = Field(
        ge=1,
        default=2,
        description=(
            "Pallas mode-tile size along the spanwise/azimuthal mode "
            "axis (power of two; default tuned on H100)."
        ),
    )
    # ``"pallas"`` backend only: the innermost, coalesced mode axis.
    # The default ``32`` is the H100 tuning -- one warp wide, so a
    # warp's band load fully coalesces.  Same internal padding to full
    # tiles as ``pallas_block_m0``.
    pallas_block_m1: int = Field(
        ge=1,
        default=32,
        description=(
            "Pallas mode-tile size along the contiguous "
            "streamwise/axial mode axis (power of two; default tuned "
            "on H100)."
        ),
    )
    # ``"pallas"`` backend only: max relative residual
    # ``||A x - b|| / ||b||`` over modes, measured once per operator
    # group at setup.  Above it, ``solvers._build_pallas_operator``
    # prints an ill-conditioning notice (benign LU element growth) or
    # raises on genuine no-pivot instability (see
    # ``solvers._NO_PIVOT_GROWTH_TOL``).
    pallas_stability_tol: float = Field(
        gt=0,
        default=1e-6,
        description=(
            "Residual threshold of the setup-time no-pivot banded-LU "
            "stability check."
        ),
    )
    # Applies to the 6-field velocity+vorticity batch of
    # ``rhs.get_nonlin`` and the ~36-field fused viscoelastic batch
    # (``_get_rhs_core`` in
    # ``geometries/wall_bounded/_viscoelastic_stepping.py``), both via
    # ``fft.chunked_transform``.  ``k > 1`` splits the batch into
    # ``k`` balanced groups, cutting the transform-stage transient
    # (the padded intermediate buffers; see the ``fft.py`` memory
    # note) by ~``k`` at the cost of ``k``x the FFT dispatches (and
    # ``k`` smaller reshard rounds per stage on multi-device runs).
    # Raise it only when that transient sets the device-memory peak:
    # chiefly the viscoelastic batch; the Newtonian 6-field transient
    # is ~6x smaller.  Forward transforms stay fused.
    rhs_transform_chunks: int = Field(
        ge=1,
        default=1,
        description=(
            "Split the batched inverse FFT of the nonlinear term "
            "into k chunks: ~k times smaller transform transient at "
            "k times the FFT dispatches; results identical."
        ),
    )


class Parameters(BaseModel):
    """Top-level parameter container aggregating all categories."""

    dist: Distribution = Distribution()
    phys: Physics = Physics()
    geo: Geometry = Geometry()
    res: Resolution = Resolution()
    init: Initiation = Initiation()
    outs: Outputs = Outputs()
    step: TimeStepping = TimeStepping()
    stop: Termination = Termination()
    solver: Solver = Solver()


@dataclass
class DerivedParameters:
    """Parameters derived from the user-facing configuration.

    ``volume_fac`` is fixed by the geometry
        (1 for periodic, 2 for Cartesian, 0.5 for cylindrical, and
        ``(r2^2 - r1^2)/2`` for the annulus).
    ``ccf_A``, ``ccf_B`` are the circular-Couette base-flow coefficients
    ``U_theta = ccf_A * r + ccf_B / r`` and ``r_inner``, ``r_outer`` the
    non-dim annular radii (set for the circular-Couette systems
    "taylor-couette" and "quasi-keplerian").
    ``u_grid`` is the resolved moving-frame speed (always a concrete
    float; see ``params.phys.u_grid`` and ``update_parameters``).
    ``nu`` is the (solvent) kinematic viscosity multiplying the velocity
    Laplacian: ``1/re`` for the Newtonian systems, ``beta/re`` for the
    viscoelastic system (whose polymer stress carries the remaining
    ``(1-beta)/(re wi)``).
    """

    volume_fac: float = 1
    cos_tilt: float = 0
    sin_tilt: float = 0
    wall_normal_grid: list[float] | None = None
    ccf_A: float = 0
    ccf_B: float = 0
    r_inner: float = 0
    r_outer: float = 0
    u_grid: float = 0
    nu: float = 0


params: Parameters = Parameters()
derived_params: DerivedParameters = DerivedParameters()


# Parameters that must be known to configure JAX *before* a snapshot is
# read (precision, platform, device mesh); they are never inherited from
# a snapshot's embedded parameters.  Resume is device-agnostic;
# precision is chosen per run, and a mismatch with the snapshot is
# rejected by ``snapshot.validate_snapshot_params``.
_SNAPSHOT_SKIP_FIELDS: tuple[tuple[str, str], ...] = (
    ("dist", "np0"),
    ("dist", "np1"),
    ("dist", "platform"),
    ("res", "double_precision"),
)

# Recorded for lineage but never inherited as a layer: the resume
# *decision* fields.  A stored ``force_resume: true`` (the writing run
# was itself force-resumed) must not make a later resume silently
# continue across trajectory-defining changes, and the stored
# ``snapshot`` path (what the writing run started from) must not leak
# into the resuming run -- the resume source always comes from the
# TOML/CLI layers.
_SNAPSHOT_LINEAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("init", "snapshot"),
    ("init", "force_resume"),
)


def read_snapshot_params(
    snapshot_path: Path,
) -> tuple[Parameters, dict[str, dict]] | None:
    """Read a snapshot's embedded parameters (core + extensions).

    Reads the ``_dnsjax_meta.json`` member of the snapshot tar (via the
    standard-library :mod:`dnsjax.snapshot_meta`; no JAX import, so it is
    safe to call before the distributed backend is configured) and
    returns ``(core, extension_overlays)``: the embedded core sections
    as a :class:`Parameters` -- with the JAX-setup fields in
    :data:`_SNAPSHOT_SKIP_FIELDS`, the resume-decision fields in
    :data:`_SNAPSHOT_LINEAGE_FIELDS`, and the whole ``solver`` section
    removed so they are not inherited -- plus the stored extension
    sections (e.g. ``force``, ``probes``) keyed by section name, for
    :func:`dnsjax.extensions.apply_extension_layer`.

    Returns ``None`` when *snapshot_path* is not a dnsjax snapshot file
    (a laminar start, or a missing path), so the caller simply skips
    the snapshot layer.  Stored metadata
    records the flow-relevant **public** names; they are mapped back
    to internal names via
    :func:`dnsjax.flows.registry.internalize_stored`, which **raises**
    on a stored core-section key this version does not define
    (``solver`` excepted, see below).
    Snapshots embed the *resolved* configuration (concrete
    ``geo.grid_type``, materialized per-flow defaults), so the
    snapshot layer pins the trajectory's actual setup on resume; the
    hidden-derived internal fields (e.g. the annular azimuthal
    ``geo.lz``) are *not* rehydrated here -- ``update_parameters``
    re-derives them from the stored public fields, and rehydrating
    would mark them explicitly set.
    """
    from .extensions import EXTENSIONS
    from .flows.registry import internalize_stored
    from .snapshot_meta import is_snapshot_file, read_snapshot_meta

    if not is_snapshot_file(snapshot_path):
        return None
    meta = read_snapshot_meta(snapshot_path)
    stored = meta.get("params")
    if not stored:
        return None
    system = meta.get("system") or stored.get("phys", {}).get("system")
    snap = internalize_stored(stored, system)
    for section, key in _SNAPSHOT_SKIP_FIELDS + _SNAPSHOT_LINEAGE_FIELDS:
        if section in snap and key in snap[section]:
            snap[section].pop(key)
    # Solver knobs are execution-only (they select *how* the numerics
    # run, never the results), so they are never inherited from a
    # snapshot.  That is also why ``internalize_stored`` exempts this
    # one section from its unknown-key error: the drop happens here,
    # *after* internalization, so a snapshot naming a solver knob this
    # version no longer defines must survive the pass above to reach
    # it.
    snap.pop("solver", None)
    overlays = {
        name: snap.pop(name)
        for name in tuple(snap)
        if name in EXTENSIONS and isinstance(snap[name], dict)
    }
    return Parameters.model_validate(snap), overlays


# Core parameter sections that *define* the trajectory: on resume, a
# change to any of their fields (other than the JAX-setup skip fields)
# marks a new trajectory (reset ``it``/``t``/``isnap``) rather than a
# continuation.  Trajectory-defining *extensions* (``force``:
# stochastic kicks alter the dynamics exactly like a ``phys`` change)
# are compared alongside in :func:`trajectory_defining_changes`.
_TRAJECTORY_SECTIONS: tuple[str, ...] = ("phys", "geo", "res")


def trajectory_defining_changes(snapshot_params: dict) -> list[str]:
    """List the trajectory-defining params the run overrides on resume.

    Compares the snapshot's embedded ``phys``/``geo``/``res`` parameters
    (``snapshot_params`` is the ``params`` sub-dict of
    ``_dnsjax_meta.json``, in the stored **public**-named
    representation) against the live global :data:`params`, and the
    stored trajectory-defining extension sections (``force``) against
    their live singletons, skipping the JAX-setup fields in
    :data:`_SNAPSHOT_SKIP_FIELDS` (of which only
    ``res.double_precision`` lies in these sections).  Returns a
    human-readable ``"section.key: old -> new"`` description per
    differing field (**internal** names -- the code-level identity of
    the field); an empty list means the resume is a pure
    *continuation* (inherit ``it``/``t``/``isnap``).

    The stored dump is internalized with ``rehydrate=True``
    (:func:`dnsjax.flows.registry.internalize_stored`): the
    hidden-derived internal fields (the annular azimuthal ``geo.lz``,
    the derived ``phys.re``/``re2``) are recomputed from the stored
    public fields with the same formulas ``update_parameters`` uses,
    so on a clean continuation they compare bit-equal instead of
    flagging spuriously.  Snapshots embed the resolved configuration
    (a concrete ``geo.grid_type``, materialized per-flow defaults), so
    values compare directly; a genuine grid switch (e.g. rigged-cgl
    <-> half-cgl) is flagged.  A stored core-section key the current
    surface no longer defines makes the internalization raise
    (``ValueError``) rather than resume against a setup nothing
    reports; fields *added* since the snapshot compare against the
    model default (the old run effectively ran it).
    """
    from .extensions import EXTENSIONS
    from .flows.registry import internalize_stored

    internal = internalize_stored(
        snapshot_params, params.phys.system, rehydrate=True
    )
    skip = set(_SNAPSHOT_SKIP_FIELDS)
    changes: list[str] = []
    for section in _TRAJECTORY_SECTIONS:
        snap = internal.get(section, {})
        model = getattr(params, section)
        cur = model.model_dump(mode="json")
        defaults = type(model)().model_dump(mode="json")
        for key in sorted(set(snap) | set(cur)):
            if (section, key) in skip:
                continue
            if key not in cur:
                continue
            old = snap[key] if key in snap else defaults.get(key)
            if old != cur.get(key):
                changes.append(f"{section}.{key}: {old!r} -> {cur.get(key)!r}")
    for name, ext in EXTENSIONS.items():
        if not (ext.trajectory_defining and ext.relevant(params.phys.system)):
            continue
        snap = internal.get(name) or {}
        cur = ext.values.model_dump(mode="json")
        defaults = ext.model().model_dump(mode="json")
        for key in sorted(set(snap) | set(cur)):
            if key not in cur:
                continue
            old = snap[key] if key in snap else defaults.get(key)
            if old != cur.get(key):
                changes.append(f"{name}.{key}: {old!r} -> {cur.get(key)!r}")
    return changes


# ``(section, key)`` of every field explicitly provided through any
# configuration layer (snapshot / TOML / CLI), accumulated across all
# :func:`update_parameters` calls.  Read wherever a per-flow default
# must distinguish "left at the class default" from "explicitly set to
# the class default" (the ``FieldSpec.default`` materialization, the
# ``grid_type`` resolution, and consumers such as the
# transient-growth CLI).
_user_set_fields: set[tuple[str, str]] = set()

# ``(section, key)`` of the per-flow default overrides
# (``FieldSpec.default``) the *previous* ``update_parameters`` pass
# materialized onto ``params``.  Restored to the model default before
# the next pass materializes the (possibly different) current flow's
# overrides, so a ``system`` switch across layers never leaks another
# flow's default -- and ``_user_set_fields`` stays clean (a
# materialized default is not a user choice).
_materialized_defaults: set[tuple[str, str]] = set()

# ``(section, key)`` of the fields the *previous* pass's derive hook
# wrote onto ``params`` (the hidden-derived values: the annular /
# cylindrical azimuthal ``geo.lz``, the circular-Couette ``phys.re``,
# the quasi-Keplerian ``phys.re2``, the viscoelastic ``phys.re``).
# Restored alongside :data:`_materialized_defaults` for the same
# reason: a ``system`` switch across layers must not leak one flow's
# derived value into another flow's derivation (e.g. a leaked
# quasi-Keplerian ``re2`` silently parameterizing Taylor-Couette).
_derive_written: set[tuple[str, str]] = set()


def update_parameters(params_new: Parameters) -> None:
    """Merge *params_new* into the global ``params``
    and recompute derived values.

    Only fields that were explicitly set in *params_new* are applied, so
    unset fields retain their previous values.  The ``(section, key)`` of
    every explicitly-set field is recorded (across all layers) in the
    module-level :data:`_user_set_fields`, so a per-flow default (a
    ``FieldSpec.default`` override, e.g. the viscoelastic axial period
    ``geo.lx``) is materialized only when the user / snapshot / TOML
    never set the field.  Corollary: a **direct** assignment
    (``params.geo.grid_type = ...``) never enters
    :data:`_user_set_fields` and is silently restored /
    re-materialized on the next call -- scripts and tests must set
    spec-defaulted fields through ``update_parameters(Parameters(...))``.
    The flow-specific parameter math itself lives
    in the flow specs (``flows/*/specs/``, dispatched via
    :func:`dnsjax.flows.registry.spec_for`).
    """
    for category, dict in params_new.model_dump(exclude_unset=True).items():
        if dict is not None:
            for key, value in dict.items():
                if value is not None:
                    _user_set_fields.add((category, key))
                    setattr(getattr(params, category), key, value)

    # Per-flow parameter resolution.  Re-run after every layer, so a
    # later ``system``/``scheme`` override can never inherit a stale
    # per-flow default.  First restore any field a previous layer's
    # spec materialized or its derive hook wrote (unless some layer
    # has set the field since), then materialize the current spec's
    # default overrides and run its derive hook -- the flow-specific
    # parameter math (required-field checks, derived control
    # parameters, geometry-forced fields, ``derived_params`` entries)
    # declared in ``flows/*/specs/``.
    system = params.phys.system
    spec = spec_for(system)
    for section, key in tuple(_materialized_defaults | _derive_written):
        if (section, key) not in _user_set_fields:
            model = getattr(params, section)
            setattr(
                model,
                key,
                type(model)
                .model_fields[key]
                .get_default(call_default_factory=True),
            )
        _materialized_defaults.discard((section, key))
        _derive_written.discard((section, key))
    for fs in spec.fields:
        if fs.default is not UNSET and fs.key not in _user_set_fields:
            # Direct assignment (not the merge loop): a materialized
            # default never enters ``_user_set_fields``.
            setattr(getattr(params, fs.section), fs.name, fs.default)
            _materialized_defaults.add(fs.key)
    if spec.derive is not None:
        # Track what the hook writes onto ``params`` (dump diff) so
        # the next pass can restore it; like a materialized default,
        # a derive-written value is not a user choice.  The diff is
        # taken even when the hook raises (``finally``): a partial
        # write before a validation error (e.g. the annular ``lz``
        # ahead of a rejected ``re1``/``re2`` pair) must not escape
        # the tracking.
        before = params.model_dump()
        try:
            spec.derive(params, derived_params, _user_set_fields)
        finally:
            after = params.model_dump()
            for section, sec_after in after.items():
                sec_before = before[section]
                for key, value in sec_after.items():
                    if value != sec_before[key]:
                        _derive_written.add((section, key))

    # Solvent viscosity multiplying the velocity Laplacian: 1/re for the
    # Newtonian systems; beta/re for the viscoelastic ones (the polymer
    # stress carries the remaining (1-beta)/(re wi)).  Read by the
    # cylindrical / annular operator builders and IMM -- which is why
    # they take it from here rather than forming 1/re themselves.
    if system in viscoelastic_systems:
        derived_params.nu = params.phys.beta / params.phys.re
    else:
        derived_params.nu = 1.0 / params.phys.re

    # Resolve the moving-frame speed U_grid.  The per-flow default (the
    # laminar bulk velocity in the grid direction) is a materialized
    # FieldSpec override; a value on a flow without the field is the
    # deferred moving-frame feature (the CLI/TOML surfaces reject it at
    # parse time -- this guards direct assignment and layered dicts).
    if (
        params.phys.u_grid is not None
        and ("phys", "u_grid") not in spec.field_map
    ):
        deferred = spec.deferred_map.get(("phys", "u_grid"))
        raise ValueError(
            deferred.message
            if deferred is not None
            else f"phys.u_grid is not supported for system {system!r}"
        )
    derived_params.u_grid = (
        params.phys.u_grid if params.phys.u_grid is not None else 0.0
    )

    # Resolve the wall-normal grid's per-flow default when the user /
    # snapshot / TOML never set ``geo.grid_type``.  Resolving to a
    # *concrete* value here (rather than interpreting ``None`` at grid
    # construction) makes every snapshot embed the grid it actually
    # ran, so a resume pins the trajectory's grid independently of
    # later defaults.  Re-resolved on every layer application (a later
    # ``system`` / ``scheme`` override cannot inherit a stale
    # default); assigned directly -- not via the merge loop -- so the
    # field never enters ``_user_set_fields``.  A custom
    # ``geo.wall_grid`` file forces ``grid_type = None`` (setting both
    # is an error); flows without a wall-normal grid stay ``None``.
    if ("geo", "grid_type") not in _user_set_fields:
        default = spec.grid_type_default
        if params.geo.wall_grid is not None or default is None:
            params.geo.grid_type = None
        elif callable(default):
            params.geo.grid_type = default(params.step.scheme)
        else:
            params.geo.grid_type = default

    # Select tilting parameters to exact precision for special angles
    if abs(params.geo.tilt_degree) == 0:
        derived_params.cos_tilt = 1
        derived_params.sin_tilt = 0
    elif abs(params.geo.tilt_degree - 180) == 0:
        derived_params.cos_tilt = -1
        derived_params.sin_tilt = 0
    elif abs(params.geo.tilt_degree - 90) == 0:
        derived_params.cos_tilt = 0
        derived_params.sin_tilt = 1
    elif abs(params.geo.tilt_degree + 90) == 0:
        derived_params.cos_tilt = 0
        derived_params.sin_tilt = -1
    else:
        tilt_rad = pi * params.geo.tilt_degree / 180
        derived_params.cos_tilt = cos(tilt_rad)
        derived_params.sin_tilt = sin(tilt_rad)

    if (
        params.geo.wall_grid is not None
        and not Path(params.geo.wall_grid).is_file()
    ):
        raise FileNotFoundError(
            f"Wall grid file not found: {params.geo.wall_grid}"
        )

    if params.geo.wall_grid is not None and params.geo.grid_type is not None:
        raise ValueError(
            "Cannot set both wall_grid and grid_type"
            " (wall_grid takes precedence; remove one)"
        )


def validate_parameters() -> None:
    """Validate cross-field constraints on the merged global ``params``.

    Run once, after every configuration layer (snapshot, TOML, CLI) has
    been applied -- not as a Pydantic validator, which would fire on each
    partial layer and reject, e.g., a ``it_corrector`` set in TOML while
    ``it_error_check`` is still at its default.
    """
    o = params.outs
    if o.it_corrector is not None and o.it_error_check > o.it_corrector:
        raise ValueError(
            f"outs.it_error_check ({o.it_error_check}) must be <= "
            f"outs.it_corrector ({o.it_corrector}) so the corrector "
            "convergence is checked at least as often as it is logged."
        )

    s = params.step
    if s.adaptive:
        if s.dt_max is None:
            raise ValueError(
                "step.adaptive requires step.dt_max: it bounds the "
                "adapted dt and anchors the setup-time no-pivot "
                "stability check (see the TimeStepping docstring)."
            )
        if not s.dt_min < s.dt_max:
            raise ValueError(
                f"step.dt_min ({s.dt_min}) must be < step.dt_max ({s.dt_max})."
            )
        if not s.dt_min <= s.dt <= s.dt_max:
            raise ValueError(
                f"step.dt ({s.dt}) must lie in [step.dt_min, "
                f"step.dt_max] = [{s.dt_min}, {s.dt_max}] when "
                "step.adaptive is enabled."
            )
        if s.dt_min_change > s.dt_max_change:
            raise ValueError(
                f"step.dt_min_change ({s.dt_min_change}) must be "
                f"<= step.dt_max_change ({s.dt_max_change})."
            )

    spec = spec_for(params.phys.system)

    # Azimuthal wedge (geo.m0): only flows whose surface carries the
    # field (the cylindrical/annular geometries, both viscoelastic
    # members included -- the u_+/u_- and tensor-spin
    # integer-harmonic formulations); rejected elsewhere.  Guards
    # direct assignment; the CLI/TOML surfaces reject it at parse.
    if params.geo.m0 != 1 and ("geo", "m0") not in spec.field_map:
        raise ValueError(
            f"geo.m0 = {params.geo.m0}: the azimuthal wedge is "
            "supported only for the annular and cylindrical geometries "
            f"(system {params.phys.system!r})."
        )

    # The wall-normal grid name must belong to this flow's family
    # (Cartesian/annular: cgl, tanh; cylindrical: half-cgl,
    # rigged-cgl, half-tanh).  Guards direct assignment.
    if params.geo.grid_type is not None:
        choices = spec.choices_for("geo", "grid_type") or ()
        if params.geo.grid_type not in choices:
            raise ValueError(
                f"geo.grid_type={params.geo.grid_type!r} is not valid "
                f"for system {params.phys.system!r}"
                + (
                    f"; choose one of: {', '.join(choices)}."
                    if choices
                    else " (no wall-normal grid there)."
                )
            )

    # Deferred fields must reject a *direct* assignment too.  The
    # CLI/TOML surfaces reject them at parse time
    # (``param_surface.internalize``), but scripts and tests set
    # ``params`` directly; a field left at its inert model default
    # passes, exactly as in the surface path.
    for (section, name), deferred in spec.deferred_map.items():
        model = getattr(params, section, None)
        if model is None or name not in type(model).model_fields:
            continue
        default = (
            type(model)
            .model_fields[name]
            .get_default(call_default_factory=True)
        )
        if getattr(model, name) != default:
            raise ValueError(deferred.message)

    # The moving frame is a deferred feature for flows without the
    # field (triply-periodic).  Guards direct assignment when the flow
    # declares no ``DeferredSpec`` for it either.
    if params.phys.u_grid is not None and (
        ("phys", "u_grid") not in spec.field_map
    ):
        deferred = spec.deferred_map.get(("phys", "u_grid"))
        raise ValueError(
            deferred.message
            if deferred is not None
            else "phys.u_grid is not supported for system "
            f"{params.phys.system!r}"
        )

    # Extension sections (the probe stream, stochastic forcing, and
    # any script-registered section): each relevant extension's
    # validate hook runs here, with the merged global ``params``.
    from .extensions import validate_extensions

    validate_extensions(params)

    # Flow-specific cross-field checks and startup summaries (e.g.
    # the pipe's half-cgl/iterative-cn restriction, the plane-Couette
    # driving restriction, the quasi-Keplerian derived summary) live
    # in the flow specs.
    if spec.validate is not None:
        spec.validate(params, derived_params)

    # The two mean-mode driving knobs, same guard for the same reason:
    # a geometry reads them only when its flows offer them (Cartesian
    # and cylindrical read ``driving``, Cartesian and annular read
    # ``block_mean_spanwise_velocity``), so a direct assignment on a
    # flow whose surface omits one is either silently inert -- e.g.
    # ``block_mean_spanwise_velocity`` on the pipe, which nothing reads
    # -- or, worse, half-applied: ``phys.driving`` on the viscoelastic
    # pipe would make the cylindrical corrector emit a ``-dPdz'``
    # diagnostic for a flow module that exports no ``get_driving``, so
    # ``__main__`` would size ``stats.dat`` one column short and fail
    # mid-run on the first stats row.  The CLI/TOML reject both at
    # parse; this is the scripts-and-tests path.  It runs after
    # ``spec.validate`` so a flow that refuses one of these itself
    # (plane-couette's constant-bulk refusal) keeps its own, more
    # specific message.
    for _drive_field in ("driving", "block_mean_spanwise_velocity"):
        _default = Physics.model_fields[_drive_field].get_default(
            call_default_factory=True
        )
        if getattr(params.phys, _drive_field) != _default and (
            ("phys", _drive_field) not in spec.field_map
        ):
            raise ValueError(
                f"phys.{_drive_field}="
                f"{getattr(params.phys, _drive_field)!r} is not a "
                f"parameter of system {params.phys.system!r} "
                f"(see `dnsjax --help {params.phys.system}`); it would "
                "not be applied."
            )

    # The Pallas kernel tiles the mode plane in ``bm0 x bm1`` blocks;
    # Triton block loads require power-of-two tile dims.
    for name in ("pallas_block_m0", "pallas_block_m1"):
        v = getattr(params.solver, name)
        if v & (v - 1) != 0:
            raise ValueError(
                f"solver.{name} ({v}) must be a power of two "
                "(Triton block-load constraint)."
            )

    # The dense backend is a readability/regression reference, not a
    # production path; nudge wall-bounded production runs back to the
    # default.  Plain ``print``: this runs before JAX is configured.
    if (
        params.solver.backend == "dense"
        and params.phys.system in walled_systems
    ):
        print(
            "[solver] backend='dense' selected: full Ny x Ny per-mode "
            "LU factors -- a reference backend kept for readability "
            "and regression checks; prefer the default 'pallas' "
            "backend for production runs."
        )


def round_up_padded(n_padded: int, divisor: int) -> int:
    r"""Round a padded FFT size up to a multiple of *divisor*.

    Returns the smallest `$m \ge n_\mathrm{padded}$` with
    `$m \equiv 0 \pmod d$` (``d = max(divisor, 1)``), so the padded
    physical axis splits evenly across *divisor* devices.  The padded
    region carries only zero (dealiased) modes, so rounding up is
    physically neutral (it costs marginally more FFT work).  The pad
    parity is unconstrained: the :func:`dnsjax.fft.zeropad_fft` /
    ``truncate_fft`` wrap-order mode placement is exact for even and
    odd pads alike, so no combination of grid size and mesh axis is
    rejected.
    """
    d = max(divisor, 1)
    return -(-n_padded // d) * d


#: Primes with specialised FFT radix kernels (cuFFT, pocketfft/XLA).
#: A transform length with only these factors takes the fast kernels;
#: any larger prime factor falls back to a markedly slower generic
#: (Bluestein-type) algorithm.
_FFT_SMOOTH_PRIMES = (2, 3, 5, 7)


def is_fft_smooth(n: int) -> bool:
    r"""True when *n* has no prime factor beyond `$\{2, 3, 5, 7\}$`.

    Such 7-smooth transform lengths take the specialised FFT radix
    kernels; padded FFT sizes are therefore rounded up to them
    (:func:`round_up_padded_smooth`).
    """
    if n <= 0:
        return False
    for prime in _FFT_SMOOTH_PRIMES:
        while n % prime == 0:
            n //= prime
    return n == 1


def round_up_padded_smooth(n_padded: int, divisor: int) -> int:
    r"""Round a padded FFT size up to a 7-smooth multiple of *divisor*.

    The smallest `$m \ge n_\mathrm{padded}$` that is a multiple of
    ``max(divisor, 1)`` **and** 7-smooth (:func:`is_fft_smooth`), so
    the FFT at the padded size takes the fast radix-2/3/5/7 kernels
    while the axis still splits evenly across *divisor* devices.
    Physically neutral for the same reason as
    :func:`round_up_padded` -- the extra slots carry only zero
    (dealiased) modes -- and 7-smooth numbers are dense at practical
    sizes, so the bump beyond the divisibility rounding is a few
    percent at most.  When *divisor* itself has a prime factor beyond
    7 no multiple of it can be smooth; the plain divisibility rounding
    is returned instead.
    """
    d = max(divisor, 1)
    m = round_up_padded(n_padded, d)
    if not is_fft_smooth(d):
        return m
    while not is_fft_smooth(m):
        m += d
    return m


def _rounding_note(
    name: str, old: int, new: int, divisor: int, axis: str
) -> str:
    """Compose the startup-diagnostic line for one padded-size bump."""
    reasons = []
    if old % max(divisor, 1) != 0:
        reasons.append(f"{axis} divisibility")
    if not is_fft_smooth(old) and is_fft_smooth(new):
        reasons.append("FFT-friendly size")
    return f"{name} rounded from {old} to {new} ({', '.join(reasons)})."


@dataclass
class PaddedResolution:
    """Grid sizes after 3/2-rule oversampling for dealiasing.

    The oversampled (padded) grid is used when evaluating nonlinear terms
    in physical space.  Each direction is expanded by a factor of
    ``oversampling_factor / 2`` (typically 3/2); every FFT axis is
    then rounded up to a mesh-divisible, FFT-friendly 7-smooth length
    (:meth:`apply_rounding`), with every adjustment recorded in
    :attr:`notes` for the startup diagnostics printed by
    :mod:`dnsjax.sharding`.
    """

    nx_padded: int = params.phys.oversampling_factor * params.res.nx // 2
    ny_padded: int | None = None
    if params.phys.system in periodic_systems:
        ny_padded = params.phys.oversampling_factor * params.res.ny // 2
    nz_padded: int = params.phys.oversampling_factor * params.res.nz // 2
    notes: list[str] = field(default_factory=list)

    def apply_rounding(self, parameters: Parameters) -> None:
        r"""Round padded sizes for mesh divisibility and FFT speed.

        Rounds every FFT axis up with :func:`round_up_padded_smooth`:
        ``nz_padded`` and the periodic ``ny_padded`` to 7-smooth
        multiples of the mesh direction that shards them (``np1`` for
        `$z$`, ``np0`` for `$y$`), and ``nx_padded`` (the real-FFT
        axis, never sharded in physical space -- divisor 1) to the
        next 7-smooth length.  Both roundings insert only zero
        (dealiased) modes, so they are physically neutral; smoothness
        keeps every transform on the fast radix-2/3/5/7 kernels
        regardless of the base resolution.  Idempotent; re-applied at
        :mod:`dnsjax.sharding` import for entry points that set
        ``params.dist`` after (or without)
        :meth:`set_padded_resolution`.  Each adjustment appends a
        diagnostic line to :attr:`notes`, which
        :mod:`dnsjax.sharding` prints once on the main process.
        """
        np0 = parameters.dist.np0
        np1 = parameters.dist.np1

        nx_new = round_up_padded_smooth(self.nx_padded, 1)
        if nx_new != self.nx_padded:
            self.notes.append(
                _rounding_note("nx_padded", self.nx_padded, nx_new, 1, "")
            )
            self.nx_padded = nx_new
        if self.ny_padded is not None:
            ny_new = round_up_padded_smooth(self.ny_padded, np0)
            if ny_new != self.ny_padded:
                self.notes.append(
                    _rounding_note(
                        "ny_padded", self.ny_padded, ny_new, np0, "np0"
                    )
                )
                self.ny_padded = ny_new
        nz_new = round_up_padded_smooth(self.nz_padded, np1)
        if nz_new != self.nz_padded:
            self.notes.append(
                _rounding_note("nz_padded", self.nz_padded, nz_new, np1, "np1")
            )
            self.nz_padded = nz_new

    def set_padded_resolution(self, parameters: Parameters) -> None:
        r"""Recompute padded sizes from *parameters*.

        The natural sizes are ``oversampling_factor * n // 2`` (`$y$`
        oversampled for periodic flows, never for wall-bounded ones --
        their wall-normal direction is FD, not Fourier);
        :meth:`apply_rounding` then rounds every
        FFT axis up to a mesh-divisible, FFT-friendly 7-smooth length
        (``nx_padded`` is exempt from the divisibility part only:
        real-FFT axis, never sharded in physical space).
        """
        self.notes = []

        self.nx_padded = (
            parameters.phys.oversampling_factor * parameters.res.nx // 2
        )
        if parameters.phys.system in periodic_systems:
            self.ny_padded = (
                parameters.phys.oversampling_factor * parameters.res.ny // 2
            )
        else:
            self.ny_padded = None
        self.nz_padded = (
            parameters.phys.oversampling_factor * parameters.res.nz // 2
        )
        self.apply_rounding(parameters)


padded_res: PaddedResolution = PaddedResolution()
