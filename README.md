# dnsjax

**Work in progress!**

A GPU-accelerated & doubly-parallelized pseudo-spectral solver for direct numerical simulations of Navier–Stokes equations in [JAX](https://github.com/jax-ml/jax).

Through JAX, it can run on (multiple) CPUs, GPUs, and TPUs.

Currently, only a three-dimensional box with periodic boundaries and monochromatic forcing is implemented. In particular, Kolmogorov flow is supported. (Waleffe flow will be supported at a later time.) The parallelization for this geometry uses [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp). The time stepper is a second-order predictor-corrector (Euler + Crank–Nicolson), adapted from [openpipeflow](https://openpipeflow.org). *This is an ongoing port of [dnsbox](https://github.com/gokhanyalniz/dnsbox), which was written in Fortran and parallelized with MPI. [Periodic orbits of Kolmogorov flow found with it](https://doi.org/10.1103/PhysRevLett.126.244502) integrate correctly with dnsjax!*

Intent is to eventually implement wall-bounded flows, such as plane-Couette flow, plane-Poiseuille flow, and pipe flow.

## To-do

### Features
- [ ] Read parameters from a TOML configuration file
- [ ] Save diagnostics (parameters, sharding, benchmarks, runtime) to a log file
- [ ] CLI (read PDIMS etc.)
- [ ] Solve for perturbations around the laminar state (and recompute the energy balances)
- [ ] Initialize from random states
- [ ] Save to disk stats with a cache
- [ ] Save to disk states
- [ ] Script to plot parameters (with laminar normalization) and a state reader / writer
- [ ] Include the format-converted [dnsbox periodic orbits for Kolmogorov flow](https://github.com/gokhanyalniz/dnsbox/wiki/Periodic-orbits) in the repository, along with the conversion script

#### Future
- [ ] Newton solver (with auto-differentiation)
- [ ] Flow symmetries
- [ ] Finite-differences + the influence-matrix method for wall-bounded flows

### Cleanup
- [ ] Add docstrings and type hints
- [ ] Add asserts

### Optimizations
- [ ] Check whether lower-precision inner products save any time
- [ ] Check timestepper parameters
- [ ] Try autotune for CUDA
- [ ] Complex <-> typecasting vs. zeroing of imaginary parts (memory and speed)