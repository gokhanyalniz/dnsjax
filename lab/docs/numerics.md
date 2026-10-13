# The lab's flows: equations and conventions

The lab's five flows — the curved pipe, quasi-Keplerian, Dean, and the
viscoelastic pipe and Dean — on top of the solver's own
[`docs/numerics.md`](../../docs/numerics.md), whose discretization,
time stepping, grids and influence-matrix method they all share. The
lab itself: [`lab/README.md`](../README.md).

## Equations

The force-driven systems integrate the **total** field with a mean-mode
body force (`FlowSpec.total_field`): azimuthal for Dean and viscoelastic
Dean, streamwise for the viscoelastic pipe and the curved pipe. The
quasi-Keplerian flow evolves a perturbation about Taylor–Couette's
circular-Couette profile, which it shares.

The **curved pipe** is the one system whose metric is not the identity: a
pipe bent onto a circle of radius $R_c$, with curvature
$\kappa = a/R_c$ (`geo.curvature`) and zero torsion, whose scale factors
are $(1, r, h)$ with $h = 1 + \kappa r\cos\theta$. It solves the same
equations in that metric, and it does so on the *straight* pipe's
operators: carrying $h u_s$ in place of $u_s$ leaves the curl, the
pressure projection, the influence matrix and every Helmholtz solve
untouched, so no operator couples azimuthal modes. The four identities
that make that work, and the two places curvature does survive, are
derived in the `dnsjax_lab/geometries/wall_bounded/cylindrical_curved.py`
module docstring. Curvature also reaches the dealiasing: the nonlinear
term carries $1/h^2$, whose harmonics never end, so the 3/2 rule alone
leaves it aliased. `phys.curvature_padding`, on by default, adds a fixed
number of azimuthal points, set by $\kappa$ and the working precision,
that puts that aliasing below roundoff at any resolution. Both initial
conditions, the random field and the localized rolls, are mapped onto
the toroidal continuity constraint.

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
$\tfrac{1-\beta}{Re\ Wi}\nabla\cdot\mathbf{c}$ and solvent viscosity
$\nu = \beta/Re$. This grows the state to nine components on the same solver
machinery. The system is that of Lellep, Linkmann & Morozov
([2024](#references), eqs. 1–3), carried over from their channel to the
pipe and the annulus, with their stress diffusivity $\kappa$ and its wall
condition: at a wall the conformation obeys the $\kappa = 0$ equation,
which dnsjax imposes in the equivalent form $\nabla^2 \mathbf{c} = 0$.
The defaults $\beta = 0.8$, $\epsilon = 10^{-3}$ and
$\kappa = 5 \cdot 10^{-5}$ are theirs too; the Weissenberg and
elasticity numbers are each flow's own.

## Conventions

- **Reynolds number.** The viscoelastic flows expose no `re` (it is
  derived as $Re = Wi/El$), and $\nu = \beta/Re$ is the *solvent*
  viscosity, the polymer stress carrying the rest. The curved pipe keeps
  the pipe's normalization, so under a held flux its `re` is the
  bulk-velocity–diameter Reynolds number of the toroidal-pipe
  literature, with Dean number $De = Re\sqrt{\kappa}$; under the default
  constant pressure gradient it is the *nominal* value of the same
  driving, and the realized bulk is lower by the Dean friction
  increase.
- **Driving.** The curved pipe, like the pipe, accepts
  `phys.driving = "constant_bulk_velocity"`; the annular flows, like
  Taylor–Couette, can pin their axial mean velocity to zero
  (`phys.block_mean_spanwise_velocity`). The moving frame's default
  speed is the laminar bulk velocity, $1/2$ for the curved and
  viscoelastic pipes, and zero for the annular flows (Dean's driving is
  azimuthal, so its axial bulk vanishes).
- **Quasi-Keplerian rotation.** The quasi-Keplerian flow is
  Taylor–Couette's annulus parameterized by `re1`, the rotation number
  `r_omega` on the quasi-Keplerian half-line $R_\Omega \lt -1$, and
  `eta`, with the outer Reynolds number `re2` derived from them.
- **Viscoelastic controls.** `el` is the elasticity number and sets
  $Re = Wi/El$; `wi` is the Weissenberg number; `beta` the solvent-to-total
  viscosity ratio; `epsilon` the sPTT extensibility; `kappa` an artificial
  stress diffusivity. Both viscoelastic flows share these, and both take
  the axial length `lz` like their Newtonian counterparts; the annular one
  adds `delta`, the inner radius (the gap is fixed at 2), where the pipe
  needs no radius parameter — its radius is 1.
- **Memory.** In the solver's memory model
  ([Memory footprint](../../docs/scaling.md#memory-footprint)),
  $(n_c, m, v) = (5, 3, 3)$ for the curved pipe, $(3, 3, 2)$ for
  quasi-Keplerian and Dean, $(11, 9, 3)$ for the viscoelastic pipe and
  $(9, 9, 2)$ for viscoelastic Dean (the viscoelastic $m$ adds the six
  conformation Helmholtz matrices). The curved pipe forward-transforms
  8 product fields, its curvature terms riding the same batch rather
  than paying for transforms of their own. With 9 state components, the
  viscoelastic right-hand side inverse-transforms a 36-field batch with
  9 outputs every step, so $W \approx 45 + 72/k$ with $k$ =
  `solver.rhs_transform_chunks`: that batch dominates the step's peak
  memory, the one configuration where the memory–throughput dial is
  worth reaching for.
- **Azimuthal wedge.** `--geo.m0` is on every lab surface but the curved
  pipe's, whose metric breaks the $m_0$-periodic symmetry.

## References

- M. Lellep, M. Linkmann and A. Morozov, *Purely elastic turbulence in
  pressure-driven channel flows*, Proc. Natl. Acad. Sci. USA **121**(9),
  e2318851121 (2024). The sPTT system with stress diffusivity, its wall
  condition and the default $\beta$, $\epsilon$ and $\kappa$.
