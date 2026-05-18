"""Predictor-corrector time integration factory.

Provides :func:`make_stepper`, which builds JIT-compiled
``predict_and_correct`` and ``iterate_correction`` functions from
flow-specific callables.  The overall iteration structure (Euler
predictor + iterative Crank-Nicolson corrector, Willis 2017) is shared
across all flow types; only the RHS evaluation, Helmholtz solve, and
norm computation differ.

For triply-periodic flows the Helmholtz solve is algebraic (pointwise
multiply by ``ldt_1``, ``ildt_2``).  For wall-bounded flows it is a
matrix solve per Fourier mode, with a different ordering for the velocity
components (v first, then pressure via IMM, then u, v, w all updated).
"""

from collections.abc import Callable

import jax.lax
from jax import Array, jit
from jax import numpy as jnp

from .bench import timer
from .parameters import params


def make_stepper(
    get_rhs_fn: Callable[..., Array],
    predict_fn: Callable[..., Array],
    correct_fn: Callable[..., tuple[Array, Array]],
    norm_fn: Callable[..., Array],
) -> tuple[
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
]:
    """Build JIT-compiled predict-and-correct and iterate-correction functions.

    The returned functions close over the flow-specific callables, so
    precomputed data (wavenumbers, time-stepping coefficients, base flow)
    is captured at construction time rather than passed on every call.

    Parameters
    ----------
    get_rhs_fn:
        ``state -> rhs_no_lapl``.  Computes the divergence-free
        RHS (nonlinear term minus pressure gradient, without the
        Laplacian / viscous term).
    predict_fn:
        ``(state, rhs_no_lapl) -> prediction_state``.  Euler predictor
        step (flow-specific Helmholtz solve).
    correct_fn:
        ``(state_prev, prediction_state, rhs_prev, rhs_next) ->
        (prediction_state_new, correction)``.
        Crank-Nicolson corrector step.
    norm_fn:
        ``correction -> error``.  Convergence norm (L2 norm of the
        correction vector).

    Returns
    -------
    predict_and_correct:
        Full predictor-corrector step.  Signature:
        ``state -> (prediction_state, rhs_next, error)``.
        No buffers are donated.
    iterate_correction:
        One additional corrector iteration.  Signature:
        ``(state_prev, prediction_state, rhs_prev) ->
        (prediction_state_next, rhs_next, error)``.
        Only *prediction_state* is donated.
    predict_and_fully_correct:
        Fused predict + corrector loop in a single JIT scope
        via ``lax.while_loop``.  Signature:
        ``state -> (prediction_state, rhs_next, error, num_c)``.
    """

    @timer("timestep/predict_and_correct")
    @jit
    def predict_and_correct(state: Array, *args) -> tuple[Array, Array, Array]:
        """Full predictor-corrector time step (Euler predict + one CN correct).

        Computes the RHS at the current velocity, applies the Euler
        predictor, recomputes the RHS at the predicted velocity, and
        applies one Crank-Nicolson corrector.  Additional corrector
        iterations (if the error exceeds tolerance) are handled by
        ``iterate_correction``.

        No buffers are donated because *state* (aliased as *state_prev*
        in the caller) is reused across corrector iterations that follow.
        """
        rhs_prev = get_rhs_fn(state, *args)
        prediction_state = predict_fn(state, rhs_prev, *args)

        rhs_next = get_rhs_fn(prediction_state, *args)
        prediction_state, correction = correct_fn(
            state, prediction_state, rhs_prev, rhs_next, *args
        )

        error = norm_fn(correction, *args)

        return prediction_state, rhs_next, error

    @timer("timestep/iterate_correction")
    @jit(donate_argnums=1)
    def iterate_correction(
        state_prev: Array,
        prediction_state: Array,
        rhs_prev: Array,
        *args,
    ) -> tuple[Array, Array, Array]:
        """One corrector iteration: recompute RHS, apply CN correction.

        **Functional Purity Exception:** The input buffer
        *prediction_state* is donated (via
        `donate_argnums=1`), meaning its memory is safely destroyed
        and reused for the outputs within XLA. Its reference outside this
        function call becomes invalidated. *state_prev* is NOT donated
        because it is reused across multiple corrector iterations.
        """
        rhs_next = get_rhs_fn(prediction_state, *args)
        prediction_state, correction = correct_fn(
            state_prev, prediction_state, rhs_prev, rhs_next, *args
        )

        error = norm_fn(correction, *args)

        return prediction_state, rhs_next, error

    @timer("timestep/predict_and_fully_correct")
    @jit
    def predict_and_fully_correct(
        state: Array, *args
    ) -> tuple[Array, Array, Array]:
        """Predict + all corrector iterations in one JIT scope.

        Uses ``lax.while_loop`` so that the corrector convergence
        check stays on-device, eliminating per-iteration
        GPU-to-CPU synchronisation.
        """
        rhs_prev = get_rhs_fn(state, *args)
        prediction = predict_fn(state, rhs_prev, *args)

        rhs_next = get_rhs_fn(prediction, *args)
        prediction, correction = correct_fn(
            state, prediction, rhs_prev, rhs_next, *args
        )
        error = norm_fn(correction, *args)

        tol = params.step.corrector_tolerance
        max_c = params.step.max_corrector_iterations

        def cond_fn(carry):
            _, _, err, c = carry
            return jnp.logical_and(err > tol, c < max_c)

        def body_fn(carry):
            pred, rhs_p, _, c = carry
            rhs_n = get_rhs_fn(pred, *args)
            pred, corr = correct_fn(state, pred, rhs_p, rhs_n, *args)
            return pred, rhs_n, norm_fn(corr, *args), c + 1

        init = (prediction, rhs_next, error, jnp.int32(0))
        prediction, rhs_next, error, num_c = jax.lax.while_loop(
            cond_fn, body_fn, init
        )
        return prediction, error, num_c

    return predict_and_correct, iterate_correction, predict_and_fully_correct
