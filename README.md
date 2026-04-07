# dnsjax

**Work in progress!**

A GPU-accelerated & parallelized 3D pseudo-spectral solver for direct numerical simulations of Navier–Stokes equations in [JAX](https://github.com/jax-ml/jax).

Through JAX, it can run on (multiple) CPUs, GPUs, and TPUs.

Currently, only a fully-periodic geometry with monochromatic forcing is implemented. In particular, Kolmogorov flow is supported. The time stepping scheme is a second-order predictor-corrector (Euler + Crank–Nicolson), adapted from [openpipeflow](https://openpipeflow.org). **[Periodic orbits of Kolmogorov flow](https://doi.org/10.1103/PhysRevLett.126.244502) integrate with dnsjax!**

Intent is to eventually implement wall-bounded flows, such as plane-Couette flow, plane-Poiseuille flow, circular- and Taylor-Couette flow, and pipe flow.

**Currently implementing plane-Couette flow!**

## To-do

### Features
- [ ] Finite-differences + the influence-matrix method for wall-bounded flows
- [ ] Initialize from random states
- [ ] Save to disk stats with a cache
- [ ] Save to disk states
- [ ] Script to plot parameters (with laminar normalization) and a state reader / writer
- [ ] Include the [periodic orbits for Kolmogorov flow](https://github.com/gokhanyalniz/dnsbox/wiki/Periodic-orbits)
- [x] Solve for perturbations around the laminar state
- [x] Save diagnostics (parameters, sharding, benchmarks, runtime) to a log file
- [x] Read parameters from a TOML configuration file
- [x] Command-line interface

#### Future
- [ ] Newton solver (with auto-differentiation)
- [ ] Flow symmetries

### Cleanup and refactoring
- [ ] Add docstrings and type hints
- [ ] Add asserts
