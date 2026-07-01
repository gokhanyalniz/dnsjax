"""Predictor-corrector time integration factory.

Provides :func:`make_stepper`, which builds JIT-compiled stepping
functions from flow-specific callables -- ``predict_and_correct``,
``iterate_correction``, ``predict_and_fully_correct`` (the fused
corrector loop and primary path), and an optional measured variant.
The overall iteration structure (Euler
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

from .parameters import params


def make_stepper(
    get_rhs_fn: Callable[..., Array],
    predict_fn: Callable[..., Array],
    correct_fn: Callable[..., tuple[Array, Array]],
    norm_fn: Callable[..., Array],
    get_rhs_measured_fn: Callable[..., tuple[Array, dict[str, Array]]]
    | None = None,
) -> tuple[
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array, dict[str, Array]]] | None,
    Callable[..., tuple[Array, Array]],
    Callable[..., tuple[Array, Array, dict[str, Array]]] | None,
]:
    """Build the JIT-compiled predictor-corrector stepping functions.

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
    get_rhs_measured_fn:
        Optional ``state -> (rhs_no_lapl, measurements)`` variant
        of *get_rhs_fn* that also returns a dict of physical-space
        measurements (see :mod:`dnsjax.measurements`).

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
        ``state -> (prediction_state, error, num_c)``.
    predict_and_fully_correct_measured:
        As ``predict_and_fully_correct``, but the step's first
        RHS evaluation (at the accepted state `$u^n$`) also
        computes the physical-space measurements.  Signature:
        ``state -> (prediction_state, error, num_c,
        measurements)``.  No buffers are donated, so a warm-up
        call may safely discard its outputs.  ``None`` when
        *get_rhs_measured_fn* is not given.
    step_cnab2:
        One CN/AB2 step (Crank-Nicolson viscous + explicit 2nd-order
        Adams-Bashforth nonlinear), selected by ``step.scheme ==
        "cnab2"``.  One RHS/FFT evaluation, no corrector loop.
        Signature: ``(state, rhs_prev) -> (state_next, rhs_n)``; the
        caller carries ``rhs_n`` back as the next *rhs_prev*.
    step_cnab2_measured:
        As ``step_cnab2``, but its single RHS evaluation (at `$u^n$`)
        also returns the physical-space measurements.  Signature:
        ``(state, rhs_prev) -> (state_next, rhs_n, measurements)``.
        ``None`` when *get_rhs_measured_fn* is not given.
    """

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

    def _step_core(
        state: Array, rhs_prev: Array, *args
    ) -> tuple[Array, Array, Array]:
        """Predictor + corrector loop, given the RHS at `$u^n$`.

        Uses ``lax.while_loop`` so that the corrector convergence
        check stays on-device, eliminating per-iteration
        GPU-to-CPU synchronisation.
        """
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
            return pred, rhs_p, norm_fn(corr, *args), c + 1

        init = (prediction, rhs_prev, error, jnp.int32(0))
        prediction, _, error, num_c = jax.lax.while_loop(
            cond_fn, body_fn, init
        )
        return prediction, error, num_c

    @jit
    def predict_and_fully_correct(
        state: Array, *args
    ) -> tuple[Array, Array, Array]:
        """Predict + all corrector iterations in one JIT scope."""
        rhs_prev = get_rhs_fn(state, *args)
        return _step_core(state, rhs_prev, *args)

    if get_rhs_measured_fn is None:
        predict_and_fully_correct_measured = None
    else:

        @jit
        def predict_and_fully_correct_measured(
            state: Array, *args
        ) -> tuple[Array, Array, Array, dict[str, Array]]:
            """Fused step that also measures physical-space data.

            The measurements come from the step's first RHS
            evaluation, i.e. from the accepted state `$u^n$`
            (outside the corrector loop).  No buffers are
            donated.
            """
            rhs_prev, measurements = get_rhs_measured_fn(state, *args)
            prediction, error, num_c = _step_core(state, rhs_prev, *args)
            return prediction, error, num_c, measurements

    @jit
    def step_cnab2(
        state: Array, rhs_prev: Array, *args
    ) -> tuple[Array, Array]:
        r"""One CN/AB2 step (``step.scheme == "cnab2"``).

        Crank-Nicolson viscous + 2nd-order Adams-Bashforth explicit
        nonlinear.  Reuses the predictor solve (implicit viscous + IMM
        pressure) with the AB2 forcing `$\tfrac{3}{2} N^n
        - \tfrac{1}{2} N^{n-1}$` in place of the plain `$N^n$` -- the
        predictor is exactly ``_imm_iteration(u, u, F, F)``, so this is
        one solve and **one** RHS/FFT evaluation, no corrector loop.

        *rhs_prev* is the previous step's nonlinear RHS `$N^{n-1}$`,
        carried by the caller; seed it with ``get_rhs_fn(state_0)`` so
        the first step (``F = N^0``) is a forward-Euler self-start.
        Returns ``(state_next, rhs_n)``; feed ``rhs_n`` back as the next
        *rhs_prev*.
        """
        rhs_n = get_rhs_fn(state, *args)
        forcing = 1.5 * rhs_n - 0.5 * rhs_prev
        return predict_fn(state, forcing, *args), rhs_n

    if get_rhs_measured_fn is None:
        step_cnab2_measured = None
    else:

        @jit
        def step_cnab2_measured(
            state: Array, rhs_prev: Array, *args
        ) -> tuple[Array, Array, dict[str, Array]]:
            """CN/AB2 step that also returns physical-space measurements
            from its single RHS evaluation (at `$u^n$`)."""
            rhs_n, measurements = get_rhs_measured_fn(state, *args)
            forcing = 1.5 * rhs_n - 0.5 * rhs_prev
            return predict_fn(state, forcing, *args), rhs_n, measurements

    return (
        predict_and_correct,
        iterate_correction,
        predict_and_fully_correct,
        predict_and_fully_correct_measured,
        step_cnab2,
        step_cnab2_measured,
    )
