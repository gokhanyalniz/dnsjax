---
paths:
  - "src/dnsjax/solvers.py"
  - "tests/test_banded_solver*.py"
  - "scripts/{pallas_solve_profile,pallas_tiling_diagnostic,solver_benchmark,grad_probe}.py"
---

# The banded solve and its Pallas/Triton kernel

- The banded path exists to keep each per-mode wall-normal solve at
  O(Ny·p) memory. Never propose an `(Nkz, Nkx, Ny, Ny)` array, dense or
  triangular, as a fix; `solver.backend = "dense"` is the reference,
  not an option.
- Interpret mode (CPU) checks a kernel's numerics, not Triton's
  lowering. After a kernel edit run the cuda-lowering guards
  `test_pallas_cuda_lowering{,_sharded_solve}` in
  `tests/test_banded_solver.py`, which lower inside an abstract GPU
  mesh (`_abstract_gpu_mesh`) on a machine with no GPU.
- Pad tiled arrays to whole tiles: Triton miscompiles the masked
  partial-tile path on a real GPU, invisibly to interpret mode and to
  lowering (the `_pallas_banded_solve` docstring).
- A `pallas_call` inside a `shard_map` needs `check_vma=False`
  (`PerModeBandedPallasOperator.solve`).
- `solvers._kernel_path()` picks both the solve body and the factor
  storage: from `_force_kernel_path`, then `solver.pallas_kernel`,
  then the live backend. A test that flips `_force_kernel_path` does
  so before building the operator.
- The kernel's `custom_vjp` is checked against the portable
  `_banded_solve_batched`, which differentiates through its own
  `lax.scan`. Never route that oracle through the rule.
