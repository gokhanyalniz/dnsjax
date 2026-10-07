---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "scripts/**/*.py"
---

# JAX conventions (any Python in this repo)

- Sharding is Explicit mode everywhere; never call
  `jax.lax.with_sharding_constraint`.
- Allocate a sharded array on its devices (`out_sharding` on
  `jnp.zeros`, `.at[...].get/set`, ...). Where that is impossible,
  place a host array with `sharding.distribute`, never `jnp.asarray`
  and never `jax.device_put` onto the mesh: across processes that
  gathers every process's copy onto each first (the why:
  `sharding.Sharding.distribute`). A `jax.device_put` to one device,
  or of an array already on the mesh, is fine.
- Reshard an existing multi-device array inside `jax.jit` with
  `jax.sharding.reshard`, one mesh axis per step: moving both axes at
  once replicates the array on every device, which shows only when
  `np0 > 1` and `np1 > 1`, so check on a `(2, 2)` mesh. Jit only the
  reshards that repeat (each jit is a compile). Pattern and failure
  signature: `snapshot._via_mid` / `snapshot._to_io_layout_core`.
- A global (multi-device) array reaches a jitted function as an
  argument, never through a closure or `static_argnames`: a baked-in
  global is legal in one process and fails at trace time in two. So
  every array-carrying object (flows, `Fourier`, solver operators,
  `DifferencePressure`) is a `sharding.register_dataclass_pytree`
  pytree passed in, and an outer jit around a function that hands
  module globals to an inner jit bakes them in all the same. Only a
  real multi-process run catches a slip: `test_twin_driver.py`'s
  `test_np2_run`, the `*-mpi-pad` rows of `test_random_smoke.py`.
- Nothing may initialize MPI before XLA does: an earlier `MPI_Init`
  aborts the run. Rank bootstrap and the CPU collectives:
  `bootstrap.configure_jax_runtime`. Its environment knobs
  (`JAX_COORDINATOR_ADDRESS`, `JAX_COORDINATOR_PORT`,
  `MPITRAMPOLINE_LIB`, `JAX_CPU_COLLECTIVES_IMPLEMENTATION`) are not
  parameters; the user-facing contract is the `Distribution` docstring
  and `docs/cpu-collectives.md`.
- Under the MPI collectives a device group's communicator opens only
  on the thread that initialized MPI, and a collective can run off it
  even with async dispatch off. `sharding._warm_communicators` opens
  the solver mesh's groups (whole mesh, each np0 and np1 group) as the
  mesh is built. A collective over any other group (another `Mesh`, a
  regrouped or sub-mesh) must be opened the same way first, or a run
  dies at random with `MPI: Communicator requested from a thread...`
  and hangs. Guard: `tests/test_mpi_communicators.py`.
- Code on a solver rank starts no process through CPython's `vfork`
  path (a `subprocess` call naming a bare executable, or keeping
  `close_fds=True`): a launcher's exec wrapper (Spindle's) runs in the
  child and corrupts the rank. Spawn as `snapshot_meta.git_hash` does,
  through `posix_spawn` (the why: its docstring).
- Scripts and in-process tests take the platform from
  `--dist.platform` (default cpu) through
  `bootstrap.configure_jax_platform` / `platform_from_argv`, before
  importing `sharding` or a geometry module.
- JAX has no zero-copy complex<->real bitcast: a real operator on a
  complex field splits re/im (never promotes the operator). Reuse
  `geometries/wall_bounded/_base.apply_y_matrix` (either path) or the
  `solvers.py` pattern.
- FFTs use `norm="forward"`.
- A dict returned from a jitted function comes back in sorted key
  order (pytree flattening); never rely on insertion order.
- `fourier`'s wavenumber arrays are global multi-device arrays: host
  code recomputes them from `harmonics.real_harmonics` /
  `complex_harmonics` times `2π/L`, never `np.asarray`.
- Memory and throughput levers: `phys.oversampling_factor` and
  `res.double_precision` dominate (`parameters.PaddedResolution`);
  `solver.rhs_transform_chunks` trades the RHS transform batch for
  peak memory (`fft.chunked_transform`); `solver.wall_normal_matvec`
  picks GEMM or stencil derivatives (`_base.apply_y_matrix`); cnab2
  buys throughput, not memory (`timestep.py`). The human memory model:
  `docs/scaling.md`.
