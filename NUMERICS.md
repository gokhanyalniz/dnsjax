# Governing equations and numerics

What `dnsjax` integrates, and how it is discretized: the equations, the
pseudo-spectral / finite-difference spatial discretization, the two
time-stepping schemes, the wall-normal grids, and the influence-matrix
method that reconciles incompressibility with the wall boundary
conditions. The closing section collects the parameter conventions each
flow follows.

Start at the [README](README.md) for the solver itself, and
[`SCALING.md`](SCALING.md) for array layout, memory and the device grid.

## Equations

The solver advances the non-dimensional incompressible Navier–Stokes
equations,

```math
\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u}
  = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u},
\qquad \nabla \cdot \mathbf{u} = 0 .
```

Most systems evolve the **perturbation** $\mathbf{u}'$ about an analytical
laminar profile $\mathbf{U}$, using the **rotational form** of the nonlinear
term,

```math
\mathrm{NL} = \mathbf{u}' \times \boldsymbol{\omega}' +
  \mathbf{u}' \times (\nabla \times \mathbf{U}) +
  \mathbf{U} \times \boldsymbol{\omega}',
\qquad \boldsymbol{\omega}' = \nabla \times \mathbf{u}',
```

in which the base-flow self-interaction is a pure gradient absorbed by the
pressure. The force-driven systems instead integrate the **total** field
with a mean-mode body force: azimuthal for Dean and viscoelastic Dean,
axial for the viscoelastic pipe.

The viscoelastic flows couple a symmetric **conformation tensor** $\mathbf{c}$
through a simplified Phan-Thien–Tanner constitutive law,

```math
\partial_t \mathbf{c} + (\mathbf{u}\cdot\nabla)\mathbf{c} -
  (\nabla\mathbf{u})^\top \mathbf{c} - \mathbf{c}\,(\nabla\mathbf{u})
  = \kappa \nabla^2 \mathbf{c} -
    \frac{1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}}{Wi}
    (\mathbf{c} - \mathbf{I}),
```

with the polymer stress feeding momentum as
$\tfrac{1-\beta}{Re\,Wi}\nabla\cdot\mathbf{c}$ and solvent viscosity
$\nu = \beta/Re$. This grows the state to nine components on the same solver
machinery.

## Spatial discretization

Periodic directions are treated **pseudo-spectrally** with Fourier
transforms; the single wall-bounded direction uses **banded finite
differences** with Fornberg weights of order `fd_order` (half-bandwidth $p$
equal to `fd_order`). The quadratic nonlinearity is dealiased with the
**3/2 rule** — physical fields are evaluated on a
$\tfrac{3}{2}$-oversampled grid and the product is truncated back — and the
Nyquist mode is dropped on every stored spectral axis (FFTs use
`norm="forward"`). The dealiasing pad carries no parity constraint — the
omitted Nyquist mode re-enters as a zero in its exact wrap-order slot for
even and odd pads alike — so any spanwise / azimuthal `nz` is accepted
(likewise `ny` for the triply-periodic box). The oversampled sizes are
then rounded up, with a startup note, to **7-smooth** lengths (no prime
factor beyond 7, so every transform takes the fast FFT radix kernels
whatever the base resolution) that also divide evenly across the device
grid; the streamwise real-FFT axis, never sharded, gets the smoothness
rounding only. The extra slots carry only zero modes, so the rounding is
physically neutral.

## Temporal discretization

Two second-order, semi-implicit schemes share the same predictor and the
same influence-matrix implicit solve:

- **`iterative-cn`** (default) — a semi-implicit Euler predictor followed
  by an iterative Crank–Nicolson corrector that makes the nonlinear term
  implicit through its fixed point. This costs $2 + n_c \approx 3$ FFT
  evaluations per step ($n_c$ the corrector's iteration count) and is
  stable well past the advective CFL limit.
- **`cnab2`** — Crank–Nicolson viscous term with an explicit second-order
  Adams–Bashforth nonlinear term, costing a **single** FFT evaluation per
  step (roughly a threefold saving on CFL-limited turbulent runs), at the
  price of an advective-CFL step restriction. The wall-bounded systems
  keep the wall-stiff coupling implicit through an FFT-free corrector; a
  step whose corrector fails to contract falls back to one full
  `iterative-cn` step.

The `implicitness` knob $c$ is the Crank–Nicolson weight ($c = 0.5$ is the
trapezoidal rule), with `corrector_tolerance` and `max_corrector_iterations`
governing the fixed point. An **opt-in split corrector**
(`split_corrector`, off by default) iterates the wall-stiff linear coupling
FFT-free between full right-hand-side refreshes; it only helps when the step
is pushed near the corrector iteration cap and is otherwise slower, hence the
default. A related `implicit_mean_coupling` (on by default) folds the
instantaneous mean-flow coupling into the implicit term.

Both schemes can run at a fixed `step.dt` or under **adaptive CFL time
stepping** (`step.adaptive`): the main loop re-reads the measured total
CFL every `cfl_cadence` steps and rescales the step toward the
`cfl_target` setpoint, bounded by `dt_min`/`dt_max`, per-change growth
and shrink limiters, and a relative deadband that suppresses churn from
CFL noise. An accepted change rebuilds the $\Delta t$-dependent
implicit operators on the device — a few implicit solves, no
recompilation — and under `cnab2` the following Adams–Bashforth step
is ratio-weighted (variable-step AB2), so both schemes remain
second-order accurate.
Snapshots embed the live `dt`, so a resume continues at the adapted
step.

A moving frame of reference (`u_grid`) translates the domain along the
streamwise / axial direction and is integrated implicitly by both schemes —
convenient for following traveling structures. By default the frame moves at
the laminar bulk velocity: $1/2$ for both pipes, $2/3$ for plane-Poiseuille,
and zero for the others (Dean's driving is azimuthal, so its axial bulk
vanishes). The pipes and plane-Poiseuille therefore integrate, and
store snapshots, in the moving frame unless `u_grid` is set to `0` (see also
item 9 in [What's in the box](README.md#whats-in-the-box)).

## Grids

The default wall-normal grid is a **Chebyshev–Gauss–Lobatto (CGL)** grid for
the Cartesian and annular geometries. The pipe's radial grid is half of a
CGL grid, in one of two variants: the **rigged-CGL** grid is the positive
half of a CGL grid with an *odd* number of points — whose middle node falls
exactly on the axis and is dropped — placing its innermost node about one
grid spacing from the axis, while the **half-CGL** grid is the positive half
of a CGL grid with an *even* number of points (no node ever sits on the
axis), placing it about half as far. `cnab2` defaults to the rigged grid:
the admissible step of its explicit azimuthal advection grows with the
innermost node's radius. `iterative-cn` integrates the near-axis
coupling implicitly and defaults to the finer-resolving half-CGL grid.
Optional tanh stretching (`grid_type = "tanh"`, or `"half-tanh"` for
the pipe's one-sided variant, with the `grid_stretch` factor) and fully
**custom grids** (via a `geo.wall_grid` file) are supported; quadrature is
spectral Clenshaw–Curtis on CGL grids (a weighted parity variant on the
pipe's half grids) and an order-`fd_order` composite rule otherwise.

## The influence-matrix method

Enforcing incompressibility together with the wall boundary conditions is the
central difficulty of a wall-bounded spectral discretization: the wall-normal
momentum equation supplies only the *interior* pressure Poisson problem,
while the correct wall pressure boundary condition is fixed *indirectly* by
requiring $\nabla \cdot \mathbf{u} = 0$ at the walls. The classical
**Kleiser–Schumann influence-matrix method** resolves this by precomputing a
small set of homogeneous responses once, so that each time step recovers the
boundary condition with a tiny per-mode solve, after which the velocity is
corrected by linearity. It enforces the wall conditions exactly — but the
*interior* divergence of a stepped state is left with a residual that is
`O(1)` relative to the individual terms of the divergence sum, a convergent
truncation error rather than zero.

The solver therefore ships a reformulation that removes it by construction,
and it is **the default** (`res.consistent_imm`): advance the wall-normal
velocity and vorticity instead of the three velocity components, reconstruct
the tangential pair from them, and no discrete pressure appears anywhere.
Continuity becomes an algebraic identity — exact at every row including the
walls, for any operator, grid or axis fit — so a stepped state's divergence
sits at round-off *at any resolution*, and tangential no-slip is never
imposed but *emerges* from the reconstruction. It costs nothing to buy:
the same banded operators at the same bandwidth, less operator storage
(fewer boundary-response vectors everywhere, and one banded operator family
fewer in the pipe and annular geometries), and a solve fewer per mode in the
plane and annular ones. The pipe is the exception — its axis forces an exact
diagonalization that doubles the scalars it evolves, so it pays one solve
more. The wall boundary condition is still recovered by the same tiny
per-mode capacitance solve as before — $1 \times 1$ for the pipe's single
wall, $2 \times 2$ for the two-walled Cartesian and annular geometries.

What the reformulation gives up is the tangential momentum combination,
which it no longer imposes: a truncation-level residual that refines with
resolution and that nothing feeds back into a solve. Setting
`res.consistent_imm` to `false` selects the primitive $(\mathbf{u}, p)$
scheme instead; it is kept for reference and for reproducing older
trajectories, lives in its own modules, and is not recommended. The
discrete-divergence, energy-budget and temporal-order tests pin both
formulations against each other.

## Conventions

A few conventions worth knowing across the flow surfaces:

- **Reynolds number.** `re` sets the viscosity $\nu = 1/Re$. For the pipe it
  is simultaneously the centerline–radius and the bulk-velocity–diameter
  Reynolds number (the factors of two cancel in the chosen normalization).
  The viscoelastic flows are the exception: they expose no `re` (it is
  derived as $Re = Wi/El$), and $\nu = \beta/Re$ is the *solvent*
  viscosity, the polymer stress carrying the rest.
- **Driving.** The pressure-driven flows (pipe, plane-Poiseuille) accept
  `phys.driving = "constant_bulk_velocity"` to hold the bulk velocity
  fixed instead of the mean pressure gradient, and every wall-bounded
  flow but the two pipes can pin the mean velocity of its undriven
  homogeneous direction to zero (`phys.block_mean_spanwise_velocity`) —
  the spanwise mean in the channels, the axial mean in the annulus.
- **Taylor–Couette rotation.** `re1` and `re2` are the inner and outer
  cylinder Reynolds numbers on a unit gap, with `re1 >= 0` and `re2` free to
  be negative. The sign pattern selects the configuration: inner-driven
  (`re1 > 0, re2 = 0`), outer-driven (`re1 = 0, re2 > 0`), co-rotating (same
  signs), or counter-rotating (`re2 < 0`); `eta = r_1/r_2` is the radius
  ratio. The quasi-Keplerian flow is the same annulus parameterized by
  `re1`, the rotation number `r_omega` on the quasi-Keplerian half-line
  $R_\Omega < -1$, and `eta`, with the outer Reynolds number `re2`
  derived from them.
- **Viscoelastic controls.** `el` is the elasticity number and sets
  $Re = Wi/El$; `wi` is the Weissenberg number; `beta` the solvent-to-total
  viscosity ratio; `epsilon` the sPTT extensibility; `kappa` an artificial
  stress diffusivity. Both viscoelastic flows share these, and both take
  the axial length `lz` like their Newtonian counterparts; the annular one
  adds `delta`, the inner radius (the gap is fixed at 2), where the pipe
  needs no radius parameter — its radius is 1.
- **Viscoelastic memory.** With 9 state components, the viscoelastic
  right-hand side inverse-transforms a 36-field batch every step, and that
  batch dominates the step's peak memory — the one configuration where
  the `solver.rhs_transform_chunks` memory–throughput dial is worth
  reaching for (see
  [Memory footprint](SCALING.md#memory-footprint)).
- **Grid axes.** Each flow's parameters use the names natural to its
  geometry: the cylindrical and annular flows expose `lz` (axial
  length), `nz` (axial modes), `nr` (radial points), and `ntheta`
  (azimuthal modes); their azimuthal extent is not a free length — it
  is the full circle, or the $2\pi/m_0$ wedge under `--geo.m0`. The
  wall-normal extent is fixed by the geometry (the channel spans
  $[-1, 1]$, the pipe radius is 1, the annulus $[r_1, r_2]$, and the
  periodic box uses $L_y = 4$).
- **Azimuthal wedge.** `--geo.m0` restricts a cylindrical or annular
  flow to the $m_0$-periodic subspace, which the dynamics preserve.
  It is a cost lever, not a coarsening: at fixed `ntheta` the wedge
  costs $m_0$ times less azimuthal work and memory while resolving
  the same physical azimuthal scales a full circle would need
  $m_0 \cdot n_\theta$ modes for, and physical space is fully
  resolved over the wedge rather than decimated. Changing `m0` on
  resume is trajectory-defining.
- **Tilted domains.** `--geo.tilt_degree` rotates the driving
  direction by $\theta$ within the homogeneous plane, so the base
  flow becomes $\mathbf{U} = U_s(\cos\theta, 0, \sin\theta)$ with the
  matching curl — the setup for structures oblique to the mean flow.
  Available for plane-Couette, plane-Poiseuille and Kolmogorov;
  $0^\circ$, $\pm 90^\circ$ and $180^\circ$ take exact values for
  $\cos\theta$ and $\sin\theta$ rather than going through the
  trigonometric functions.
