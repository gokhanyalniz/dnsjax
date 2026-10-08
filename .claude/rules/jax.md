---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "scripts/**/*.py"
---

# Python conventions (JAX, sharding)

- Sharding is Explicit mode everywhere; never call
  `jax.lax.with_sharding_constraint`.
- Allocate a sharded array on its devices (`out_sharding` on
  `jnp.zeros`, `.at[...].get/set`, ...). Where that is impossible,
  place a host array with `sharding.distribute`, never `jnp.asarray`
  and never `jax.device_put` onto the mesh: across processes that
  gathers every process's copy onto each first (the why:
  `sharding.Sharding.distribute`). A `jax.device_put` to one device,
  or of an array already on the mesh, is fine.
- `jnp.broadcast_to` keeps its source's sharding, not the target's:
  pass `out_sharding` wherever the result meets a fully sharded array
  (a `jnp.where` against `fourier.mean_mask`, then a `jnp.stack`), or
  the operands mismatch once `np1 > 1` (precedent:
  `_cylindrical_stepping`'s `psv_b`).
- Slicing a sharded axis to a length its mesh axis does not divide
  raises `ShardingTypeError`. Crop or pad per device inside a
  `shard_map` on local shards, as `PerModeBandedPallasOperator.solve`
  does.
- Reshard an existing multi-device array inside `jax.jit` with
  `jax.sharding.reshard`, one mesh axis per step: moving both axes at
  once replicates the array on every device, which shows only when
  `np0 > 1` and `np1 > 1`, so check on a `(2, 2)` mesh. Jit only the
  reshards that repeat (each jit is a compile). Pattern and failure
  signature: `snapshot._via_mid` / `snapshot._to_io_layout_core`.
- A global (multi-device) array reaches a jitted function as an
  argument, never through a closure or `static_argnames`, so every
  array-carrying object is a `sharding.register_dataclass_pytree`
  pytree passed in. The why and the multi-process guards that catch a
  slip: that function's docstring.
- Nothing may initialize MPI before XLA does: an earlier `MPI_Init`
  aborts the run. Rank bootstrap, the CPU collectives and their
  environment knobs: `bootstrap.configure_jax_runtime`; the user-facing
  contract: the `Distribution` docstring and `docs/cpu-collectives.md`.
- A collective over a device group that `sharding._warm_communicators`
  does not open (another `Mesh`; a regrouped, reordered or sub-mesh,
  since the same devices in another order are another group) needs its
  MPI communicator opened the same way first, or a multi-process CPU
  run dies at random (`MPI: Communicator requested from a thread...`)
  and hangs. Guard: `tests/test_mpi_communicators.py`.
- Code on a solver rank starts no process through CPython's `vfork`
  path (a `subprocess` call naming a bare executable, or keeping
  `close_fds=True`): a launcher's exec wrapper (Spindle's) runs in the
  child and corrupts the rank. Spawn through `posix_spawn`, as
  `snapshot_meta.git_hash` does (the why: its docstring).
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
- Memory and throughput levers and the per-rank memory model:
  `docs/scaling.md` and the `parameters.PaddedResolution` docstring;
  measure a layout offline with `scripts/memory_budget.py`.
