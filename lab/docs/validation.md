# The lab's validation

What the lab's flows and tools are checked against, and which test in
`lab/tests/` pins each claim — the lab's counterpart of the solver's
[`docs/validation.md`](../../docs/validation.md), whose checks of the
shared stepping core cover the lab's flows too. The lab itself:
[`lab/README.md`](../README.md).

## Non-modal optimal growth against a published value

The solver's optimal-growth module runs on the quasi-Keplerian flow as
on Taylor–Couette, and matches its published value to about **2 %** or
better on the solver's own discretization:

| Flow | Control parameters | Mode | Published $G_{\max}$ | at $t$ | Source |
|---|---|---|---|---|---|
| Quasi-Keplerian | $\eta = 0.71$, $R_\Omega = -1.2$, $Re_1 = 10^4$ | $m = 4$, $k_z = 0$ | $13.04$ | $27 \tau_d$ | Shi et al. 2017, table III case I |

```bash
uv run python lab/tests/test_transient_growth.py
```

- L. Shi, B. Hof, M. Rampp and M. Avila, *Hydrodynamic turbulence in
  quasi-Keplerian rotating flows*, Phys. Fluids **29**, 044107 (2017).

## What else the lab's suite pins

| Claim | Pinned by |
|---|---|
| On the toroidal pipe, where the metric makes the constraint iterative, the discrete divergence tracks the corrector tolerance | `lab/tests/test_curved_pipe.py` |
| The toroidal equation set reproduces that of Webster & Humphrey (*Phys. Fluids* **9**, 407, 1997), and $\kappa = 0$ reproduces the straight pipe | `lab/tests/test_curved_pipe.py` |
| Curvature produces Dean vortices: the streamwise maximum moves to the outer wall and the flux drops under a fixed pressure gradient | `lab/tests/test_curved_pipe.py` |
| On the toroidal pipe the applied mean-mode driving column agrees with the whole mean-momentum balance | `lab/tests/test_curved_pipe.py` |
| Laminar states step at machine precision, every lab flow with a closed-form laminar state (all but the curved pipe; the sPTT rows at $\epsilon = \kappa = 0$, where that state is exact) | `lab/tests/test_laminar_smoke.py` |
| The viscoelastic operators and matvecs match independent NumPy constructions | `lab/tests/test_viscoelastic.py`, `lab/tests/test_viscoelastic_pipe.py` |
| Host-built arrays reach the devices without JAX's cross-process gather, at every forcing kick and in the twin driver | `lab/tests/test_host_placement.py` |
| A zero-energy twin perturbation reproduces the reference bit-for-bit, every stream included | `lab/tests/test_twin_driver.py` |
