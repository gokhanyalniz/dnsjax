"""Iterative Euler predictor / Crank-Nicolson corrector time integration.

Implements the scheme from Willis (2017, Openpipeflow) for the
incompressible Navier-Stokes equations.  The viscous term is treated
implicitly with adjustable implicitness parameter *c*:

    Predictor (Euler):
        u_p = (ldt_1 * u^n + f^n) * ildt_2

    Corrector (Crank-Nicolson):
        u^{n+1} = u_p + c * (f(u_p) - f^n) * ildt_2

where ``ldt_1 = 1/dt + (1-c) * lapl / Re`` and
``ildt_2 = 1 / (1/dt - c * lapl / Re)`` encode the implicit/explicit
split of the Helmholtz operator, and ``f`` is the divergence-free RHS
(nonlinear term minus pressure gradient, without the Laplacian).

The ``vmap`` in ``get_prediction`` / ``get_correction`` maps over the
three velocity components (axis 0), since the Helmholtz operator is
identical for all components in the triply-periodic case.

Both ``predict_and_correct`` and ``iterate_correction`` use
``donate_argnums`` to allow JAX to reuse the input buffer for the output,
avoiding an extra allocation on GPU.
"""

from functools import partial

from jax import Array, jit, vmap

from .bench import timer
from .parameters import params
from .rhs import get_rhs_no_lapl
from .velocity import get_norm


@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(
    velocity_spec: Array,
    rhs_no_lapl: Array,
    ldt_1: Array,
    ildt_2: Array,
) -> Array:
    """Euler predictor step (vmapped over velocity components).

    Computes ``u_p = (u^n * ldt_1 + f^n) / (1/dt - c*nu*lapl)``
    as a pointwise operation in spectral space, where the Helmholtz
    inversion is algebraic (multiply by ``ildt_2``).
    """

    prediction = (velocity_spec * ldt_1 + rhs_no_lapl) * ildt_2

    return prediction


@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(
    prediction: Array,
    rhs_no_lapl_prev: Array,
    rhs_no_lapl_next: Array,
    ildt_2: Array,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector step (vmapped over velocity components).

    Computes the correction ``delta = c * (f_next - f_prev) * ildt_2``
    and returns the updated prediction and the correction itself (for
    convergence monitoring).
    """

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_new = prediction + correction

    return prediction_new, correction


@timer("timestep/iterate_correction")
@jit(donate_argnums=(0, 1))
def iterate_correction(
    prediction: Array,
    rhs_no_lapl_prev: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    inv_lapl: Array,
    k_metric: Array,
    ys: Array | None,
    ildt_2: Array,
    base_flow: Array,
    curl_base_flow: Array,
    nonlin_base_flow: Array,
) -> tuple[Array, Array, Array]:
    """One corrector iteration: recompute RHS from the latest prediction,
    apply the Crank-Nicolson correction, and return the convergence error.

    The input buffers *prediction* and *rhs_no_lapl_prev* are donated
    (their memory is reused for the outputs).

    Returns
    -------
    prediction_next:
        Updated velocity prediction, shape ``(3, *spec_shape)``.
    rhs_no_lapl_next:
        RHS evaluated at the updated prediction (carried forward as the
        "previous" RHS for the next iteration).
    error:
        L2 norm of the correction (used for convergence check).
    """
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )

    prediction_next, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, k_metric, ys)

    return prediction_next, rhs_no_lapl_next, error


@timer("timestep/predict_and_correct")
@jit(donate_argnums=0)
def predict_and_correct(
    velocity_spec: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    inv_lapl: Array,
    k_metric: Array,
    ys: Array | None,
    ldt_1: Array,
    ildt_2: Array,
    base_flow: Array,
    curl_base_flow: Array,
    nonlin_base_flow: Array,
) -> tuple[Array, Array, Array]:
    """Full predictor-corrector time step (Euler predict + one CN correct).

    Computes the RHS at the current velocity, applies the Euler predictor,
    recomputes the RHS at the predicted velocity, and applies one
    Crank-Nicolson corrector.  Additional corrector iterations (if the
    error exceeds tolerance) are handled by :func:`iterate_correction`.

    The input buffer *velocity_spec* is donated.

    Returns
    -------
    prediction:
        Corrected velocity at ``t^{n+1}``, shape ``(3, *spec_shape)``.
    rhs_no_lapl_next:
        RHS evaluated at the corrected velocity (reused if further
        corrector iterations are needed).
    error:
        L2 norm of the correction.
    """

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt_1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, k_metric, ys)

    return prediction, rhs_no_lapl_next, error
