# Validation

What `dnsjax` is checked against, and which test pins each claim. Every
row here is executed by the suite — nothing on this page is a number
someone typed once. Start at the [README](../README.md) for the solver
itself and at [`numerics.md`](numerics.md) for the formulation
being checked.

## Non-modal optimal growth against published values

`dnsjax.analysis.transient_growth` computes the 3D linear optimal energy
growth $G(t)$ about a wall-normal total profile by reusing the solver's
own linear step, one Fourier mode at a time — so an anchor here tests the
production stepper, not a separate linear code. Each case below is a
single mode, matched on the solver's finite-difference-in-$y$
discretisation to about **2 %** or better:

| Flow | Control parameters | Mode | Published $G_{\max}$ | at $t$ | Source |
|---|---|---|---|---|---|
| Plane-Poiseuille | $Re = 1000$ | $(\alpha, \beta) = (0,\ 2.044)$ | $\approx 196$ | $\approx 76$ | Reddy & Henningson 1993; Butler & Farrell 1992 |
| Plane-Couette | $Re = 1000$ | $(\alpha, \beta) = (0.035,\ 1.60)$ | $\approx 1185$ | $\approx 117$ | Butler & Farrell 1992 |
| Pipe | $Re = 3000$ | $m = 1$, $\alpha = 0$ | $649$ | $147$ | Schmid & Henningson 1994, p. 217 |
| Taylor–Couette | $\eta = 0.881$, $Re_1 = 591$, $Re_2 = -2588$ | $n = 10$, $k = 1.994$ | $71.58$ | — | Maretzke, Hof & Avila 2014, table 3 |
| Quasi-Keplerian | $\eta = 0.71$, $R_\Omega = -1.2$, $Re_1 = 10^4$ | $m = 4$, $k_z = 0$ | $13.04$ | $27\,\tau_d$ | Shi et al., Phys. Fluids **29**, 044107 (2017), table III case I |

Two further checks ride along: the centrifugally unstable
Taylor–Couette case ($Re_1 = 100$, $Re_2 = 0$, $\eta = 1/2$) must come
out linearly **unstable** — a positive spectral abscissa, the
Taylor-vortex onset — and, under `--slow`, the extracted generator's
leading eigenvalue for plane-Poiseuille at $Re = 10^4$, $\alpha = 1$ is
matched against the Orszag (1971) value.

Reynolds-number and length-scale conventions are chosen so the numbers
are directly comparable without rescaling; where they are not obvious
the reasoning is recorded next to the check. Run them with

```bash
uv run python tests/test_transient_growth.py                 # all anchors
uv run python tests/test_transient_growth.py --system pipe   # one flow
```

`tests/test_transient_growth.py` is the authoritative record: it carries
the exact tolerances, the full conventions argument for each case, and
the `--legacy-imm` variant that repeats every anchor on the retired
primitive formulation.

## What else the suite pins

The full list of test scripts and what each covers is in
[`../tests/README.md`](../tests/README.md). The claims made in the README
map to them as follows.

| Claim | Pinned by |
|---|---|
| A stepped state's discrete divergence is round-off at any resolution | `tests/test_imm_continuity.py` |
| The default formulation beats the primitive one in absolute error *and* decay rate, in all three wall-bounded geometries | `tests/test_temporal_order.py` |
| Second-order temporal convergence, fixed and variable step | `tests/test_temporal_order.py` |
| $dE/dt = I - D$ closes to truncation order, pressure-gradient work included | `tests/test_energy_budget.py` |
| Laminar states step at machine precision, every wall-bounded flow | `tests/test_laminar_smoke.py` |
| Random initial conditions integrate through the full nonlinear path, every distinct stepping machinery | `tests/test_random_smoke.py` |
| The Pallas banded kernel agrees with the dense reference solver | `tests/test_banded_solver.py`, `tests/test_banded_solver_sharded.py` |
| Reverse-mode gradients of a step match a central difference, and the default corrector still refuses | `tests/test_autodiff.py` |
| The banded kernel's adjoint matches the portable sweep's own autodiff, and composes inside the sharded solve | `tests/test_banded_solver.py` |
| Triton lowering does not regress on GPU-less machines | `tests/test_banded_solver.py` (CUDA-lowering rows) |
| Per-geometry operators and matvecs match independent NumPy constructions | `tests/test_cartesian.py`, `test_cylindrical.py`, `test_annular.py`, `test_viscoelastic.py`, `test_viscoelastic_pipe.py` |
| Snapshots round-trip, resume across any device count, and carry lineage | `tests/test_snapshot.py`, `tests/test_resume.py` |
| The JAX-free analysis API reproduces the solver's own discrete operators | `tests/test_snapshot_export.py` |
| The $(k_x, k_z) = (0, 0)$ perturbation respects its conservation laws | `tests/test_mean_mode.py` |
| The applied mean-mode driving column agrees with the wall-shear inference at converged resolution | `tests/test_driving.py` |
| A zero-energy twin perturbation reproduces the reference bit-for-bit, every stream included | `tests/test_twin_driver.py` |
| Quadrature weights and interpolation matrices | `tests/test_integration.py` |
