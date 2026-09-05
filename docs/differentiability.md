# Differentiability

What differentiates through a `dnsjax` time step, what does not, and the
two opt-in knobs that decide it. Start at the [README](../README.md) for
the solver itself and at [`numerics.md`](numerics.md) for the
formulation.

The interesting direction is **reverse** mode: `jax.grad` of a scalar
diagnostic with respect to an initial state is the shape an optimal
perturbation, a sensitivity map or a control problem starts from.
Forward mode (`jax.jvp`) works on every configuration and needs nothing
from this page.

## What blocks reverse mode, and what does not

The 3/2-dealiased nonlinear term, the influence-matrix pass and the
banded LU sweep all differentiate as they stand. The one construct that
does not is the corrector's **dynamic trip count**: a `lax.while_loop`
has no transpose rule, so `jax.grad` of a step refuses outright.

| step | reverse mode |
|---|---|
| `iterative-cn`, any flow, default corrector | refused (`lax.while_loop`) |
| `cnab2`, wall-bounded, default corrector | refused (the coupling corrector is the same loop) |
| `cnab2`, triply-periodic | **works** — the explicit-AB2 step runs no corrector at all |
| either scheme, `step.corrector_iterations > 0` | **works** |

## The two knobs

**`step.corrector_iterations`** (default `0`, meaning iterate to
`corrector_tolerance`) runs exactly *n* corrections per step in a
static-trip-count loop, which lowers to a scan and differentiates. It is
a *different integrator* unless *n* covers what the dynamic corrector
actually used — `corrector.dat` is where that number comes from, and the
`TimeStepping` docstring in `src/dnsjax/parameters.py` carries the rest
of the trade-off, including the two things that change with it (the
iteration cap goes inert, and the reported corrector error becomes a
diagnostic rather than a stop condition).

**`solver.pallas_kernel`** (default unset) pins which sweep reads the
banded factors. It is not needed for differentiability — the Triton
kernel carries its own adjoint, below — but `false` selects the portable
pure-JAX sweep on a GPU, which differentiates through its own `lax.scan`
with no hand-written rule. That is what makes it the oracle the kernel's
adjoint is checked against.

## The adjoint of the banded solve

A Pallas kernel is opaque to reverse mode, so the wall-normal solve
carries an explicit `jax.custom_vjp`. Its backward pass is the *same
sweep mirrored*: the factorisation $A = LU$ has no pivoting, so
$A^{\mathsf{T}} = U^{\mathsf{T}} L^{\mathsf{T}}$, and the transposed
solve reads the stored factors in place — forward-substituting with
$U^{\mathsf{T}}$, whose diagonal is the same reciprocated slot the
forward sweep multiplies by, then back-substituting with unit-diagonal
$L^{\mathsf{T}}$. No un-inversion, no second factorisation, no extra
storage. The rule is complete: it returns cotangents for the factors as
well as the right-hand side, all $O(N_y p)$. The derivation, including
why the reciprocated diagonal slot carries an extra factor that is easy
to lose, is in `_pallas_banded_solve_t` in `src/dnsjax/solvers.py`.

## Running the probe

```bash
uv run python scripts/grad_probe.py                 # 12 configurations
uv run python scripts/grad_probe.py --full          # the cross product
uv run python scripts/grad_probe.py --dist.platform cuda
```

One subprocess per configuration — the parameter singletons and the
jitted steppers capture their configuration at import and trace time —
printing a Markdown table of forward mode, reverse mode, and a central
difference against every gradient.

## What is verified where

`tests/test_autodiff.py` pins the gradients themselves against finite
differences, and pins that the *default* configuration still refuses, so
the fixed-count rows cannot pass vacuously.
`tests/test_banded_solver.py` pins the adjoint: the transposed sweep
against a dense oracle, the `custom_vjp` against the portable sweep's
own automatic differentiation, and finite differences on the operator
cotangents including the reciprocated diagonal.

All of that runs on CPU, including the composition: the adjoint is
executed *through* `.solve`'s `shard_map` in interpret mode with the
kernel branch forced, and finite-differenced there, so the rule is known
to compose and not merely to be right in isolation.

Two things stay unverified until the probe runs on a GPU. Triton's real
lowering of the transposed kernel — the forward kernel shipped on the
same footing, and the partial-tile miscompile it once hit is invisible to
both interpret mode and a lowering check. And the *differentiated*
sharded region's cuda lowering, which cannot be checked on a machine with
no GPU at all: `shard_map`'s transpose compares cotangent shardings, and
against the abstract GPU mesh the other lowering guards use they do not
compare equal. The forward region and the transposed kernel are each
lowered for cuda separately.

If you would rather not depend on any of that, `solver.pallas_kernel =
false` takes the portable sweep on a GPU as well, and that path
differentiates through its own `lax.scan` with no hand-written rule at
all.
