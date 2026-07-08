"""Predictor-corrector time integration factory.

Provides :func:`make_stepper`, which builds JIT-compiled stepping
functions from flow-specific callables -- ``predict_and_correct``,
``iterate_correction``, ``predict_and_fully_correct`` (the fused
corrector loop and primary path), and an optional measured variant.
The overall iteration structure (Euler
predictor + iterative Crank-Nicolson corrector, Willis 2017) is shared
across all flow types; only the RHS evaluation, Helmholtz solve, and
norm computation differ.  With *l_bf_fn* provided and
``step.split_corrector`` enabled (an opt-in, default off), the fused
corrector runs in split form (``_split_core``): the linear coupling
iterates FFT-free between full-RHS refreshes.

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
    l_bf_fn: Callable[..., Array] | None = None,
    finalize_fn: Callable[..., Array] | None = None,
) -> tuple[
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array, dict[str, Array]]] | None,
    Callable[..., tuple[Array, Array, Array, Array]],
    Callable[..., tuple[Array, Array, Array, Array, dict[str, Array]]] | None,
]:
    r"""Build the JIT-compiled predictor-corrector stepping functions.

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
    l_bf_fn:
        Optional ``state -> l_bf`` giving the *linear* base-flow
        coupling term (``u' x curl(U) + U x omega'``) evaluated
        **without** any FFT (spectral/matrix-free), used by the
        CN/AB2 scheme and by the opt-in split ``iterative-cn``
        corrector (``step.split_corrector``, default off).  When
        provided,
        ``step_cnab2`` advances the
        pure self-advection ``u' x omega' = get_rhs_fn - l_bf_fn``
        explicitly (AB2) while treating ``l_bf_fn`` implicitly
        (Crank-Nicolson) via an FFT-free corrector -- required for
        wall-bounded flows, where the base-flow coupling carries a
        stiff wall-normal derivative on the wall-clustered grid (see
        ``step_cnab2`` below) -- and
        ``predict_and_fully_correct(_measured)`` likewise split their
        corrector around it (``_split_core``;
        ``step.split_corrector = False`` restores the unsplit
        corrector exactly).  When ``None`` (triply-periodic, whose
        Fourier ``y`` makes the coupling non-stiff) ``step_cnab2`` is
        the plain explicit-AB2 step and the corrector is always
        unsplit.
    finalize_fn:
        Optional ``(state, *args) -> state`` applied **once** to the
        accepted state of every completed step -- at the end of
        ``predict_and_fully_correct(_measured)`` and
        ``step_cnab2(_measured)`` (after the fallback ``lax.cond``),
        inside the step's jit scope.  Triply-periodic flows pass their
        post-step divergence projection + mean-mode zeroing here so it
        fuses with the step (no separate dispatch, one fewer
        state-sized read/write pass); wall-bounded flows pass ``None``
        (the IMM already enforces continuity exactly), which leaves
        their traces unchanged.  The legacy manual-iteration pair
        ``predict_and_correct`` / ``iterate_correction`` does **not**
        apply it (mid-iteration states are not accepted steps); a
        caller driving those directly must finalize itself.

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
        With *l_bf_fn* and ``step.split_corrector`` enabled (opt-in,
        default off) the corrector runs in split form -- the linear
        coupling iterates FFT-free between full-RHS refreshes, same
        converged CN fixed point and ``error``/``num_c`` semantics (see
        ``_split_core``).
        **Donates** *state* (the main-loop rebind pattern
        ``state, ... = step(state, ...)``); a caller that reuses
        its state afterwards -- e.g. a warm-up call -- must pass
        ``jnp.copy(state)``.
    predict_and_fully_correct_measured:
        As ``predict_and_fully_correct``, but the step's first
        RHS evaluation (at the accepted state `$u^n$`) also
        computes the physical-space measurements.  Signature:
        ``state -> (prediction_state, error, num_c,
        measurements)``.  Donates *state* like
        ``predict_and_fully_correct``.  ``None`` when
        *get_rhs_measured_fn* is not given.
    step_cnab2:
        One CN/AB2 step (Crank-Nicolson viscous + explicit 2nd-order
        Adams-Bashforth nonlinear), selected by ``step.scheme ==
        "cnab2"``.  Signature: ``(state, carry) -> (state_next, carry,
        error, num_c)``; the caller carries ``carry`` back unchanged.
        **Donates** *state* and *carry* (both are rebound by the main
        loop); callers that reuse either afterwards must pass copies.
        **One FFT/step** either way (the single expensive nonlinear
        transform).  Without *l_bf_fn* (triply-periodic): the plain
        explicit-AB2 step, ``carry`` is the previous full nonlinear
        RHS `$N^{n-1}$`, ``error = 0``, ``num_c = 0`` (no corrector).
        With *l_bf_fn* (wall-bounded): ``carry`` is the previous
        **self-advection** RHS `$N_{nl}^{n-1} = (u' \times
        \omega')^{n-1}$`; the step forms the AB2 forcing
        `$\tfrac{3}{2} N_{nl}^n - \tfrac{1}{2} N_{nl}^{n-1}$` (fixed,
        one FFT) and makes the base-flow coupling implicit
        (Crank-Nicolson) via an **FFT-free** corrector loop (a
        converged linear corrector is the exact CN-implicit `$L_{bf}$`
        solve), reporting its iteration count ``num_c`` and final
        ``error``.
    step_cnab2_measured:
        As ``step_cnab2``, but its first RHS evaluation (at `$u^n$`)
        also returns the physical-space measurements.  Signature:
        ``(state, carry) -> (state_next, carry, error, num_c,
        measurements)``.  ``None`` when *get_rhs_measured_fn* is not
        given.
    """

    def _finalized(state: Array, *args) -> Array:
        """Apply *finalize_fn* to an accepted step output.

        Identity (trace-time branch, zero ops) when no finalizer is
        configured, so wall-bounded traces are unchanged.
        """
        if finalize_fn is None:
            return state
        return finalize_fn(state, *args)

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

    def _split_core(
        state: Array, rhs_prev: Array, *args
    ) -> tuple[Array, Array, Array]:
        r"""Split corrector: FFT-free coupling tail + full refreshes.

        Same CN fixed point as ``_step_core``, reorganised so the
        corrector iterations driven by the *linear* coupling
        ``l_bf_fn`` (base-flow coupling + frame term + -- per
        ``step.implicit_mean_coupling`` -- the instantaneous
        mean-flow coupling) cost no Fourier transform.  Each outer
        iteration first converges the coupling for the frozen pure
        self-advection `$N_{nl} = \text{get\_rhs} - \text{l\_bf}$`
        (the ``_cnab2_lbf_core`` iteration with `$N_{nl}$` in place
        of the AB2 forcing), then refreshes the full RHS once and
        corrects.

        The coupling tail's entry test is *cheap*: an implicit solve
        is launched only while the coupling estimate still moves the
        state -- `$c\,\Delta t\,\|l_{bf}(u_j) - l_{bf}(u_{j-1})\| >
        \text{tol}$` (the CN correction weights the implicit RHS
        estimate by *c* and its Helmholtz inverse is bounded by
        `$\Delta t$`, so this bounds the correction the new coupling
        estimate could induce).  A fluctuation-driven outer iteration
        therefore adds one ``l_bf_fn`` evaluation and one norm,
        **not** an extra Helmholtz/IMM solve.  The tail test only
        routes work (tail solve vs outer refresh): acceptance is
        always the outer fresh-RHS correction, so ``error`` /
        ``num_c`` keep their unsplit meaning (last full-refresh
        correction norm / extra FFT evaluations), and a step whose
        first correction already meets tolerance never enters the
        loop -- 2 FFT evaluations, exactly the unsplit path.

        A split corrector that fails to reach tolerance redoes the
        step with the unsplit ``_step_core`` (``lax.cond``, stdout
        diagnostic), pinning the worst case to the unsplit
        corrector's behaviour.
        """
        prediction = predict_fn(state, rhs_prev, *args)

        # First correction, exactly as the unsplit corrector; also
        # form the frozen self-advection remainder and keep the
        # coupling at the same iterate.  ``l_bf_fn`` here re-derives
        # the spectral curl ``get_rhs_fn`` already built -- XLA CSE
        # merges the identical subgraphs (see the ``_cnab2_lbf_core``
        # note).
        rhs_next = get_rhs_fn(prediction, *args)
        l_prev = l_bf_fn(prediction, *args)
        nnl = rhs_next - l_prev
        prediction, correction = correct_fn(
            state, prediction, rhs_prev, rhs_next, *args
        )
        error = norm_fn(correction, *args)

        tol = params.step.corrector_tolerance
        max_c = params.step.max_corrector_iterations
        # Gain of one correction w.r.t. its implicit RHS estimate.
        cdt = params.step.implicitness * params.step.dt

        def inner_cond(icarry):
            _, _, delta, ic = icarry
            return jnp.logical_and(delta > tol, ic < max_c)

        def outer_cond(carry):
            _, _, _, err, c = carry
            return jnp.logical_and(err > tol, c < max_c)

        def outer_body(carry):
            # ``l_prev_k`` is the coupling used by the last correction
            # (invariant kept below), so ``delta`` measures how much
            # the coupling estimate has moved since that correction.
            pred, nnl_k, l_prev_k, _err, c = carry

            def inner_body(icarry):
                ipred, l_i, _, ic = icarry
                ipred, _ = correct_fn(
                    state, ipred, rhs_prev, nnl_k + l_i, *args
                )
                l_next = l_bf_fn(ipred, *args)
                delta = cdt * norm_fn(l_next - l_i, *args)
                return ipred, l_next, delta, ic + 1

            # FFT-free coupling tail: converge l_bf for the frozen
            # self-advection, solving only while the coupling still
            # moves the state.
            l_new = l_bf_fn(pred, *args)
            delta0 = cdt * norm_fn(l_new - l_prev_k, *args)
            pred, l_last, _, _ = jax.lax.while_loop(
                inner_cond, inner_body, (pred, l_new, delta0, jnp.int32(0))
            )
            # One full refresh + correction: the outer convergence
            # check (and the reported error) always sees a fresh RHS.
            # ``l_last`` is the coupling at the refresh state, so the
            # remainder costs no extra ``l_bf_fn`` evaluation.
            rhs_k = get_rhs_fn(pred, *args)
            nnl_k = rhs_k - l_last
            pred, corr = correct_fn(state, pred, rhs_prev, rhs_k, *args)
            return pred, nnl_k, l_last, norm_fn(corr, *args), c + 1

        init = (prediction, nnl, l_prev, error, jnp.int32(0))
        prediction, _, _, error, num_c = jax.lax.while_loop(
            outer_cond, outer_body, init
        )

        def _fallback(_):
            jax.debug.print(
                "iterative-cn: split corrector did not converge "
                "(err={e:.2e} after {c} it); redoing the step with "
                "the unsplit corrector.",
                e=error,
                c=num_c,
            )
            return _step_core(state, rhs_prev, *args)

        def _keep(_):
            return prediction, error, num_c

        return jax.lax.cond(error > tol, _fallback, _keep, None)

    # Corrector core for predict_and_fully_correct(_measured): the
    # split corrector needs the FFT-free coupling (wall-bounded) and
    # is gated by ``step.split_corrector`` -- an A/B knob, see the
    # ``TimeStepping`` docstring.  Resolved at construction time (the
    # geometry builders run after the configuration is final).
    if l_bf_fn is not None and params.step.split_corrector:
        _fully_correct_core = _split_core
    else:
        _fully_correct_core = _step_core

    @jit(donate_argnums=0)
    def predict_and_fully_correct(
        state: Array, *args
    ) -> tuple[Array, Array, Array]:
        """Predict + all corrector iterations in one JIT scope.

        Wall-bounded flows run the split corrector (``_split_core``)
        when the opt-in ``step.split_corrector`` is enabled (default
        off), else the unsplit corrector.
        *state* is donated: the output state may reuse its buffer
        (one field-sized allocation saved per step in the main
        loop).  Callers that keep using their input must pass a
        copy (see the ``__main__`` warm-up calls).
        """
        rhs_prev = get_rhs_fn(state, *args)
        prediction, error, num_c = _fully_correct_core(state, rhs_prev, *args)
        return _finalized(prediction, *args), error, num_c

    if get_rhs_measured_fn is None:
        predict_and_fully_correct_measured = None
    else:

        @jit(donate_argnums=0)
        def predict_and_fully_correct_measured(
            state: Array, *args
        ) -> tuple[Array, Array, Array, dict[str, Array]]:
            """Fused step that also measures physical-space data.

            The measurements come from the step's first RHS
            evaluation, i.e. from the accepted state `$u^n$`
            (outside the corrector loop).  *state* is donated
            (warm-up callers pass a copy).
            """
            rhs_prev, measurements = get_rhs_measured_fn(state, *args)
            prediction, error, num_c = _fully_correct_core(
                state, rhs_prev, *args
            )
            return _finalized(prediction, *args), error, num_c, measurements

    def _cnab2_lbf_core(
        state: Array, nnl_prev: Array, full_rhs: Array, *args
    ) -> tuple[Array, Array, Array, Array]:
        r"""CN/AB2 body with implicit base-flow coupling.

        Splits the nonlinear RHS ``full_rhs = get_rhs_fn(u^n)`` into
        the FFT-free base-flow coupling ``L_bf`` and the pure
        self-advection ``N_nl = full_rhs - L_bf``.  ``N_nl`` is
        advanced explicitly (AB2 forcing `$\tfrac{3}{2} N_{nl}^n -
        \tfrac{1}{2} N_{nl}^{n-1}$`, fixed across the loop); the
        linear ``L_bf`` is made implicit (Crank-Nicolson) by an
        **FFT-free** corrector that re-evaluates only ``l_bf_fn`` each
        iteration -- a converged linear corrector is the exact
        CN-implicit ``L_bf`` solve.  Returns
        ``(state_next, N_nl^n, error, num_c)``.
        """
        # ``l_bf_fn(state)`` re-derives the spectral curl (and, in the
        # cylindrical/annular geometries, the u_+/- conversion) that
        # ``get_rhs_fn(state)`` already built for ``full_rhs``.  This
        # costs nothing: both live in one jit scope on the same input,
        # and XLA CSE merges the identical subgraphs -- verified on the
        # optimized HLO (Cartesian and annular: the pair compiles to
        # ONE curl D1 GEMM, not two), so no fused get_rhs+l_bf contract
        # is needed.
        l_n = l_bf_fn(state, *args)
        nnl_n = full_rhs - l_n
        f_ab2 = 1.5 * nnl_n - 0.5 * nnl_prev
        rhs_prev = f_ab2 + l_n  # effective RHS R(u) = F_ab2 + L_bf(u), at u^n

        tol = params.step.corrector_tolerance
        max_c = params.step.max_corrector_iterations

        prediction = predict_fn(state, rhs_prev, *args)
        rhs_next = f_ab2 + l_bf_fn(prediction, *args)
        prediction, correction = correct_fn(
            state, prediction, rhs_prev, rhs_next, *args
        )
        error = norm_fn(correction, *args)

        def cond_fn(carry):
            _, err, c = carry
            return jnp.logical_and(err > tol, c < max_c)

        def body_fn(carry):
            pred, _, c = carry
            rhs_n = f_ab2 + l_bf_fn(pred, *args)
            pred, corr = correct_fn(state, pred, rhs_prev, rhs_n, *args)
            return pred, norm_fn(corr, *args), c + 1

        prediction, error, num_c = jax.lax.while_loop(
            cond_fn, body_fn, (prediction, error, jnp.int32(0))
        )

        # Hybrid auto-fallback for a genuinely divergent corrector.  The
        # FFT-free base-flow-coupling corrector is a Picard iteration whose
        # contraction rate can reach 1 at large ``dt`` (the ``L_bf`` solve
        # is only stiff enough to diverge once ``dt`` is well past the
        # advective limit -- e.g. plane-Couette at ``dt`` >~ 0.2): it then
        # fails to reach ``corrector_tolerance`` within
        # ``max_corrector_iterations``.  When that happens, redo *this*
        # step with the robust full iterative-CN corrector (``_step_core``,
        # reusing the RHS already evaluated at `$u^n$`; deliberately the
        # *unsplit* corrector -- the split one's coupling tail is the same
        # Picard iteration that just failed); ``lax.cond`` runs
        # that branch -- and its extra FFTs -- only on the hard steps, so
        # the cheap 1-FFT path is unchanged elsewhere.  (This does *not*
        # cover the explicit-``N_nl`` advective-stability limit shared by
        # all explicit-nonlinear schemes, which is what bounds ``dt`` for a
        # strongly non-normal base flow such as counter-rotating
        # Taylor-Couette; there the corrector converges cleanly and the
        # remedy is a smaller ``dt`` or ``iterative-cn``.  See the
        # ``TimeStepping`` docstring in ``parameters.py``.)
        def _fallback(_):
            jax.debug.print(
                "cnab2: base-flow-coupling corrector did not converge "
                "(err={e:.2e} after {c} it); using iterative-cn this step.",
                e=error,
                c=num_c,
            )
            return _step_core(state, full_rhs, *args)

        def _keep(_):
            return prediction, error, num_c

        prediction, error, num_c = jax.lax.cond(
            error > tol, _fallback, _keep, None
        )
        return _finalized(prediction, *args), nnl_n, error, num_c

    _zero_err = jnp.zeros(())
    _zero_c = jnp.int32(0)

    @jit(donate_argnums=(0, 1))
    def step_cnab2(
        state: Array, carry: Array, *args
    ) -> tuple[Array, Array, Array, Array]:
        r"""One CN/AB2 step (``step.scheme == "cnab2"``).

        Crank-Nicolson viscous + 2nd-order Adams-Bashforth explicit
        nonlinear, **one FFT/step**.  ``carry`` is threaded by the
        caller (feed the returned ``carry`` back unchanged).  The
        returned ``carry`` is independent of the ``carry`` argument
        (it is `$N_{nl}(u^n)$`, recomputed from *state* alone), so the
        caller can prime it with a discarded
        ``step_cnab2(state_0, zeros)`` call and take the *first*
        integration step with ``iterative-cn`` (the ``__main__``
        bootstrap) -- no forward-Euler start is involved.  Returns
        ``(state_next, carry, error, num_c)``.  *state* and *carry*
        are both donated (callers reusing either pass copies).

        With *l_bf_fn* (wall-bounded) the base-flow coupling is made
        implicit via an FFT-free corrector (see ``_cnab2_lbf_core``);
        ``carry`` is the self-advection RHS `$N_{nl}^{n-1}$`.  Without
        it (triply-periodic) this is the plain explicit-AB2 step whose
        predictor is exactly ``_imm_iteration(u, u, F, F)`` with
        `$F = \tfrac{3}{2} N^n - \tfrac{1}{2} N^{n-1}$`; ``carry`` is
        `$N^{n-1}$` and there is no corrector (``error = 0``,
        ``num_c = 0``).

        **Memory**: cnab2 is a *throughput* optimisation (1 vs
        `$2 + c$` FFT evaluations per step), not a peak-memory one --
        it carries one extra state-sized array (``carry``) across
        steps, and for wall-bounded flows the compiled step still
        reserves buffers for the ``lax.cond`` iterative-cn fallback
        branch (XLA allocates the max over branches), so its allocated
        peak is about the iterative-cn step's.
        """
        full_rhs = get_rhs_fn(state, *args)
        if l_bf_fn is None:
            forcing = 1.5 * full_rhs - 0.5 * carry
            state_next = predict_fn(state, forcing, *args)
            return _finalized(state_next, *args), full_rhs, _zero_err, _zero_c
        return _cnab2_lbf_core(state, carry, full_rhs, *args)

    if get_rhs_measured_fn is None:
        step_cnab2_measured = None
    else:

        @jit(donate_argnums=(0, 1))
        def step_cnab2_measured(
            state: Array, carry: Array, *args
        ) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
            """CN/AB2 step that also returns physical-space measurements
            from its first RHS evaluation (at `$u^n$`).  *state* and
            *carry* are donated (warm-up callers pass copies)."""
            full_rhs, measurements = get_rhs_measured_fn(state, *args)
            if l_bf_fn is None:
                forcing = 1.5 * full_rhs - 0.5 * carry
                state_next = predict_fn(state, forcing, *args)
                return (
                    _finalized(state_next, *args),
                    full_rhs,
                    _zero_err,
                    _zero_c,
                    measurements,
                )
            out = _cnab2_lbf_core(state, carry, full_rhs, *args)
            return (*out, measurements)

    return (
        predict_and_correct,
        iterate_correction,
        predict_and_fully_correct,
        predict_and_fully_correct_measured,
        step_cnab2,
        step_cnab2_measured,
    )
