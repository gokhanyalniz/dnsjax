# Memory, layout and parallelization

How `dnsjax` lays its data out, what a configuration costs in memory,
and how the work is split across devices. Everything here is
independent of the flow — the geometry sets the meaning of each axis,
and the device grid is chosen the same way for all nine systems.

Start at the [README](README.md) for the solver itself.

## Memory footprint

Every contribution below scales linearly with the point count
$n_x n_y n_z$ — nothing grows faster under the default backends — and the
total divides by $n_{p0} \cdot n_{p1}$ across devices. At the default double
precision one real number is 8 bytes, so a *field* of $n_x n_y n_z$ reals
occupies $n_x n_y n_z / 2^{27}$ GiB; that is the unit used below. Single
precision (`res.double_precision = false`) halves everything and roughly
doubles the throughput of the bandwidth-bound FFT stages on GPUs
(considerably more on consumer GPUs, which throttle double-precision
arithmetic), at reduced accuracy. Assuming the default 3/2 dealiasing and
the default backends:

- **Spectral state** — exactly $n_c$ fields, with $n_c = 3$ velocity
  components (9 for the viscoelastic flows): one component is
  $(n_x/2) \cdot n_y \cdot (n_z - 1)$ complex numbers ($n_y - 1$ in place
  of $n_y$ for the periodic box), i.e.
  $\approx n_x n_y n_z$ reals. The time stepper holds about three further
  state-sized arrays within a step, and `cnab2` carries one across steps
  (for the wall-bounded systems its allocated peak still matches the
  default scheme's, whose corrector branch XLA keeps reserved); the
  total-field systems — Dean, viscoelastic Dean, viscoelastic pipe —
  keep one extra state-sized laminar reference.
- **Nonlinear term, every step** — the rotational form inverse-transforms a
  6-field batch (velocity + vorticity) to the oversampled grid, multiplies
  pointwise, and forward-transforms the 3 product fields. Counting the held
  fields, the products, and the one to two batch-sized intermediates inside
  the transforms, the working set is $W \approx 15\text{–}21$ oversampled
  fields; each oversampled field is $(3/2)^2 = 2.25$ fields for
  wall-bounded systems (the wall-normal direction is never oversampled) and
  $(3/2)^3 = 3.375$ fields for triply-periodic ones. How much of this
  coexists is decided by XLA's buffer reuse, so treat the upper end as the
  sizing estimate. The viscoelastic right-hand side instead transforms a
  36-field batch with 9 outputs, and `solver.rhs_transform_chunks = k` —
  the knob applies to every flow's batch, but bites here — cuts its
  transform-stage share $k$-fold at identical results. Both viscoelastic
  flows share that right-hand side.
- **Wall-normal operators** — the Pallas backend stores no-pivot banded LU
  factors: $(2p + 1) \cdot n_y$ reals per matrix per Fourier mode, with the
  half-bandwidth $p$ equal to `fd_order`, over the $(n_z - 1)(n_x/2)$ mode
  plane — that is $m (2p + 1)/2$ fields for $m$ banded matrices, the one
  term that grows with `fd_order`. Here $m = 2$ for
  plane-Couette/Poiseuille, $4$ for pipe, Taylor–Couette,
  quasi-Keplerian, and Dean, and $10$ for the viscoelastic flows (the
  same $4$ plus the six conformation Helmholtz matrices), plus
  $v = 3\text{–}6$ field-sized boundary-response vectors ($v/2$
  fields). Switching to
  `solver.backend = "dense"` replaces $(2p + 1)$ by $n_y$ per matrix — the
  one super-linear option, and the reason Pallas is the wall-bounded
  default. Triply-periodic systems store no matrices at all (their implicit
  solve is diagonal in spectral space), only four real coefficient arrays
  — wavenumber and inverse-Laplacian factors, $\approx 2$ fields.

Summing these, the leading-order total per device is

```math
\text{wall-bounded:} \qquad
  \Bigl[\, 4 n_c + \tfrac{9}{4} W +
    \tfrac{1}{2} \bigl( m (2p + 1) + v \bigr) \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

```math
\text{triply-periodic:} \qquad
  \Bigl[\, 4 n_c + \tfrac{27}{8} W + 2 \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

with $W \approx 15\text{–}21$ as above (for the viscoelastic flows,
$W \approx 45 + 72/k$ with $k$ = `rhs_transform_chunks`) and
$(n_c, m, v) = (3, 2, 4)$ for the plane flows, $(3, 4, 3)$ for the pipe,
$(3, 4, 6)$ for Taylor–Couette, quasi-Keplerian, and Dean,
$(9, 10, 3)$ for the viscoelastic pipe, and
$(9, 10, 6)$ for viscoelastic Dean. The sum is an upper estimate —
XLA's buffer reuse typically realizes less — and halves at single
precision. Off the stepping path, a snapshot write reshards the state
onto an I/O layout before moving each device's bytes directly to disk
(staging through host memory only when GPUDirect Storage is
unavailable) — a transient second state-sized copy on multi-device
runs, nothing extra on a single device — and the on-device diagnostic
buffers are resolution-independent.

## Array layout by geometry

The solver keeps one internal axis order for every flow — physical
`[axis0, axis1, axis2]` and spectral `[axis0, axis1, axis2]` — and the
physical meaning of each axis is set by the geometry (a row per
geometry, not per flow). The leading axis is device-local; the two
sharded axes are split by `np0` and `np1` (elaborated under
[Parallelization](#parallelization)). Role abbreviations: **sw**
streamwise, **wn** wall-normal, **sh** shearwise, **sp** spanwise.

| Geometry | Velocity components `(0, 1, 2)` | Physical `[0, 1, 2]` | Spectral `[0, 1, 2]` | `np0` splits | `np1` splits |
|---|---|---|---|---|---|
| Triply-periodic (Kolmogorov) | $(u_x, u_y, u_z)$ = (sw, sh, sp) | $[y, z, x]$ | $[k_y, k_z, k_x]$ | $y$ / $k_z$ | $z$ / $k_x$ |
| Cartesian (plane-Poiseuille/Couette) | $(u_x, u_y, u_z)$ = (sw, wn, sp) | $[y, z, x]$ | $[y, k_z, k_x]$ | $y$ / $k_z$ | $z$ / $k_x$ |
| Cylindrical (pipe, viscoelastic pipe) | $(u_z, u_r, u_\theta)$ = (sw, wn, sp) | $[r, \theta, z]$ | $[r, k_\theta, k_z]$ | $r$ / $k_\theta$ | $\theta$ / $k_z$ |
| Annular (Taylor–Couette, quasi-Keplerian, Dean, viscoelastic Dean) | $(u_z, u_r, u_\theta)$ = (**sp**, wn, **sw**) | $[r, \theta, z]$ | $[r, k_\theta, k_z]$ | $r$ / $k_\theta$ | $\theta$ / $k_z$ |

Each `np0` / `np1` cell reads *physical axis* / *spectral axis*.
Velocity components are stored in `(streamwise, wall-normal, spanwise)`
order for every geometry **except the annulus**, which reuses the
pipe's axial-first $(u_z, u_r, u_\theta)$ order so the solver's shared,
right-handed curl / cross / finite-difference operators apply
unchanged. Because the annular main flow is azimuthal, its streamwise
velocity is component 2 ($u_\theta$) and its spanwise velocity is
component 0 ($u_z$) — the sole departure from the component-order
convention.

## Parallelization

The device grid is $(n_{p0}, n_{p1})$, and the two axes distribute the data
differently:

- **`np0`** splits the wall-normal axis ($y$ / $r$) in physical space and the
  spanwise / azimuthal wavenumber axis ($k_z$ / $m$) in spectral space. The
  split is padding-free when `np0` divides both the wall-normal point
  count (`ny`, or `nr`) and the stored mode count ($n_z - 1$, or
  $n_\theta - 1$); otherwise the layer zero-pads to the next multiple
  and strips the padding around the reshard (the stored mode count is
  odd, so a one-mode pad is the norm — and harmless).
- **`np1`** splits the spanwise / azimuthal axis ($z$ / $\theta$) in
  physical space and the streamwise / axial wavenumber axis ($k_x$) in
  spectral space. The spectral side is auto-padded the same way
  (padding-free when `np1` divides the streamwise / axial mode count,
  $n_x/2$ or $n_z/2$); on the physical side the oversampled size
  ($3/2 \times$ the base resolution of that axis at the default
  oversampling) is rounded up to the next FFT-friendly multiple of
  `np1` when needed (see
  [Spatial discretization](NUMERICS.md#spatial-discretization)), which
  amounts to a sliver of extra oversampling.
- Independently of the device grid, the **Pallas banded solver** tiles each
  device's $(k_z, k_x)$ mode plane in blocks of
  (`solver.pallas_block_m0`, `solver.pallas_block_m1`) $= (2, 32)$ and pads
  up to whole tiles, so the padded modes cost memory and solve work in
  proportion to the round-up (what to do about it: *Choosing the device
  grid* below).

No divisibility choice is rejected, and none of the padding — for the
device grid or for FFT-friendly sizes — is silent: every adjustment is
reported by a one-line startup diagnostic, so its (usually marginal) cost
stays visible.

Crucially, **every device holds the full wall-normal extent in spectral
space**, so the per-mode banded solves need no communication. The forward and
inverse FFTs move data between layouts with two reshards implemented as a
`shard_map` with explicit `reshard` calls; with either grid axis at 1 the
decomposition collapses to a one-dimensional split and only the other
reshard remains. `jax.device_count()` must equal $n_{p0} \cdot n_{p1}$.

### Choosing the device grid

The two exchanges are not equivalent, which is what makes the choice
matter. The `np1` exchange ($z \leftrightarrow k_x$) runs while the array
still carries the **oversampled** spanwise extent, whereas the `np0`
exchange ($y \leftrightarrow k_z$) runs after the truncation to stored
modes — so at the default oversampling `np1` moves $3/2$ as many bytes.
And a second grid axis does not divide the first exchange more finely, it
**adds** a second one: a one-dimensional grid performs one exchange per
transform, a two-dimensional grid two, each a synchronization point. Both
the exchange count and its byte volume are visible in the compiled
program.

**Independently of the device type:**

1. **On one node, stay one-dimensional.** Split on `np0` by default:
   its exchange carries $2/3$ of the bytes, and its mode axis tiles far
   more coarsely on GPU. Split on `np1` instead when `ny` (`nr`) will
   not divide the device count or is too small for it.
2. **Across nodes, align the grid with them** — `np1` = devices per node,
   `np0` = number of nodes. The grid is laid out row-major over the
   sorted devices, so `np1` groups fall within a node and `np0` groups
   hold one device per node. That confines the heavier exchange to the
   intra-node interconnect and leaves the network $n_{p0} - 1$ large
   messages per device in place of the many small ones a grid-wide
   exchange sends, at equal network volume. Splitting on `np1` alone
   across nodes is the worst choice: it puts the $3/2$-sized exchange
   on the network.
3. **Snapshots follow the same pattern**, but only for the first
   reason: a one-dimensional grid reshards once per save instead of
   twice. Write granularity does not enter the choice — the reshard
   trims the divisibility padding as it goes, so every grid writes each
   component as one contiguous range per device.

**On CPU** the mode plane carries no tile round-up — the Pallas kernel
never runs — so `np1` may be taken as far as the mode count allows, and
one device per process makes $n_{p0} \cdot n_{p1}$ the rank count.
Measured at four and eight ranks, the per-exchange cost dominates its
volume: a two-dimensional grid costs 9 to 19% against the best
one-dimensional one, where the $3/2$ volume difference between the two
one-dimensional grids is worth some 18% of the transform itself but only
a few percent of the step around it. Routing the collectives through MPI
rather than `gloo` (see
[CPU collectives](docs/cpu-collectives.md)) speeds up
every exchange, shifting weight from the per-exchange cost back toward
volume.

**On GPU** the mode plane is tiled, which makes `np1` the granular axis:
keep $(n_x/2)/n_{p1}$ a multiple of `solver.pallas_block_m1` $= 32$,
where $(n_z-1)/n_{p0}$ need only clear `pallas_block_m0` $= 2$. A
minimal-box `nx = 32` split four ways leaves four streamwise modes per
device, padded to 32 — lower the block size, or move the split to `np0`.
With a fast intra-node interconnect and production-sized arrays the
exchange is likelier to be limited by volume than by its per-exchange
cost, and that is the regime where `np0` moving $2/3$ of the bytes should
tell; comparing the two one-dimensional grids on the target machine is
then worth one pair of runs.

The README's [pipe example](README.md#quick-start) on four
devices of one node, one-dimensionally:
`np0 = 4` splits the 48 radial points into 12 per device and the 95
stored azimuthal modes into 24, one padding mode included, leaving the
whole $n_z/2 = 256$ axial mode axis local (eight whole Pallas tiles):

```bash
# CPU: one device per process
mpirun -np 4 .venv/bin/dnsjax \
  --dist.np0 4 --dist.platform cpu \
  --phys.system pipe --phys.re 2300 --geo.lz 200 \
  --res.nz 512 --res.nr 48 --res.ntheta 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

```bash
# GPU: a single process addressing all four GPUs on the node, no MPI
.venv/bin/dnsjax \
  --dist.np0 4 --dist.platform cuda \
  --phys.system pipe --phys.re 2300 --geo.lz 200 \
  --res.nz 512 --res.nr 48 --res.ntheta 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

Because `np0 * np1` counts *devices* rather than processes, a single-node
multi-GPU run is most reliably launched as one process that addresses every
visible GPU; multi-node runs use one process per node spanning that node's
GPUs. The `Distribution` docstring in `parameters.py` covers the SLURM
launch details. The ranks discover each other from the launcher environment
where it says enough — the MPI implementation's rank variables plus a
coordinator address, taken from `JAX_COORDINATOR_ADDRESS`, else from
loopback when the whole job is on one node, the launcher's own daemon URI,
or the queueing system's node list (PBS, SLURM, LSF, Grid Engine) — and
otherwise from JAX's own cluster detection. That covers Open MPI 5, whose
PRRTE launcher drops the variable JAX's own Open MPI plugin looks for, and
the schedulers JAX has no plugin for; a site matching nothing is one
`JAX_COORDINATOR_ADDRESS` export away, and says so rather than failing
obscurely. A single-process launch coordinates nothing, so it starts no
distributed runtime and needs none of this — not even a launcher to be
detected in. On CPU, though, one process means one device: several CPU
devices in one process is oversubscription, and asking for it is refused
with the `mpirun -np N` that works.

A **CPU** run is pinned to one XLA thread per rank — a lone process
exactly like a rank of sixteen: the pool follows `NPROC`, which the run
sets only if unset, so `export NPROC=<n>` raises it. It also routes its
cross-process collectives through MPI when it finds the MPItrampoline
wrapper library (see [CPU collectives](docs/cpu-collectives.md)), falling
back to `gloo` otherwise. The same docstring covers when raising
`NPROC` is worth doing.
