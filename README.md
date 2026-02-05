# dnsjax

**Work in progress!**

A pseudo-spectral solver for direct numerical simulations of Navier–Stokes equations, written in [JAX](https://github.com/jax-ml/jax).

Through JAX, it can run on (multiple) CPUs, GPUs, and TPUs.

Currently, only a three-dimensional box with periodic boundaries and monochromatic forcing is implemented. In particular, Kolmogorov flow is supported. (Waleffe flow will be supported at a later time.) Parallelization for this geometry uses [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp). This is an ongoing port of [dnsbox](https://github.com/gokhanyalniz/dnsbox), which was written in Fortran and parallelized with MPI.

Intent is to eventually implement wall-bounded flows, such as plane-Couette flow, plane-Poiseuille flow, and pipe flow.
