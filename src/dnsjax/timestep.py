"""Predictor-corrector time integration factory.

Provides :func:`make_stepper`, which builds JIT-compiled
``predict_and_correct`` and ``iterate_correction`` functions from
flow-specific callables.  The overall iteration structure (Euler
predictor + iterative Crank-Nicolson corrector, Willis 2017) is shared
across all flow types; only the RHS evaluation, Helmholtz solve, and
norm computation differ.

For triply-periodic flows the Helmholtz solve is algebraic (pointwise
multiply by ``ldt_1``, ``ildt_2``).  For wall-bounded flows it will be a
matrix solve per Fourier mode, with a different ordering for the velocity
components (v first, then pressure via IMM, then u and w).
"""

from collections.abc import Callable

from jax import Array, jit

from .bench import timer


def make_stepper(
    get_rhs_fn: Callable[..., Array],
    predict_fn: Callable[..., Array],
    correct_fn: Callable[..., tuple[Array, Array]],
    norm_fn: Callable[..., Array],
) -> tuple[
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
        The input buffer is donated.
    iterate_correction:
        One additional corrector iteration.  Signature:
        ``(state_prev, prediction_state, rhs_prev) ->
        (prediction_state_next, rhs_next, error)``.
        Both input buffers are donated.
    """

    @timer("timestep/predict_and_correct")
    @jit(donate_argnums=0)
    def predict_and_correct(
        state: Array | tuple, *args
    ) -> tuple[Array | tuple, Array, Array]:
        """Full predictor-corrector time step (Euler predict + one CN correct).

        Computes the RHS at the current velocity, applies the Euler
        predictor, recomputes the RHS at the predicted velocity, and
        applies one Crank-Nicolson corrector.  Additional corrector
        iterations (if the error exceeds tolerance) are handled by
        ``iterate_correction``.

        **Functional Purity Exception:** This JIT-compiled function overrides
        the XLA buffers (via `donate_argnums=0`) of the inputs to save memory
        at runtime. While typical JAX code implies functional purity, treating
        the `state` outside of this return block will yield corrupted values
        due to buffer reuse.
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
    @jit(donate_argnums=(0, 1))
    def iterate_correction(
        state_prev: Array | tuple,
        prediction_state: Array | tuple,
        rhs_prev: Array,
        *args
    ) -> tuple[Array | tuple, Array, Array]:
        """One corrector iteration: recompute RHS, apply CN correction.

        **Functional Purity Exception:** The input buffers
        *prediction_state* and *rhs_prev* are donated (via
        `donate_argnums=(0, 1)`), meaning their memory is safely destroyed
        and reused for the outputs within XLA. Their references outside this
        function call become invalidated.
        """
        rhs_next = get_rhs_fn(prediction_state, *args)
        prediction_state, correction = correct_fn(
            state_prev, prediction_state, rhs_prev, rhs_next, *args
        )

        error = norm_fn(correction, *args)

        return prediction_state, rhs_next, error

    return predict_and_correct, iterate_correction
