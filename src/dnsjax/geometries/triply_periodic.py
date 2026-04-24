"""Triply-periodic geometry: Fourier class, differential operators, norms,
base dataclass, solvers, and stepper factory.

Provides all geometry-general infrastructure for triply-periodic flows:
the ``Fourier`` wavenumber class, the ``TriplyPeriodicFlow`` base
dataclass (time-stepping coefficients), algebraic Helmholtz predict /
correct operations, divergence correction, state initialisation, and the
``build_triply_periodic_stepper`` factory.

Flow-specific modules (e.g. ``flows.monochromatic``) subclass
``TriplyPeriodicFlow`` to define the base flow, then call
``build_triply_periodic_stepper`` to obtain ready-to-use time-stepping
functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
from jax import Array, jit, vmap
from jax import numpy as jnp

from ..bench import timer
from ..operators import (
    complex_harmonics,
    phys_to_spec,
    real_harmonics,
    spec_to_phys,
)
from ..parameters import derived_params, params
from ..rhs import get_nonlin
from ..sharding import register_dataclass_pytree, sharding
from ..timestep import make_stepper


@register_dataclass_pytree
@dataclass
class Fourier:
    """Wavenumber grids for the triply-periodic geometry.

    Broadcasting shapes match the spectral layout ``(ky, kz, kx)``:
    - ``kx``: shape ``(1, 1, nx//2)``
    - ``kz``: shape ``(1, nz-1, 1)``
    - ``ky``: shape ``(ny-1, 1, 1)``

    ``k_metric`` equals 2 for `$k_x > 0$` and 1 for `$k_x = 0$`,
    accounting for the Hermitian symmetry of the real FFT.
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    ky: Array = field(init=False)
    k_metric: Array = field(init=False)
    lapl: Array = field(init=False)
    inv_lapl: Array = field(init=False)

    def __post_init__(self) -> None:
        self.kx = (
            jax.device_put(
                real_harmonics(params.res.nx).reshape([1, 1, -1]),
                sharding.spec_scalar_shard,
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )
        self.kz = (
            jax.device_put(
                complex_harmonics(params.res.nz).reshape([1, -1, 1]),
                sharding.no_shard,
            )
            * 2
            * jnp.pi
            / params.geo.lz
        )
        self.ky = (
            jax.device_put(
                complex_harmonics(params.res.ny).reshape([-1, 1, 1]),
                sharding.no_shard,
            )
            * 2
            * jnp.pi
            / derived_params.ly
        )

        self.k_metric = jax.device_put(
            jnp.where(self.kx == 0, 1, 2),
            sharding.spec_scalar_shard,
        )
        self.lapl = jax.device_put(
            -(self.kx**2 + self.ky**2 + self.kz**2),
            sharding.spec_scalar_shard,
        )
        self.inv_lapl = jax.device_put(
            jnp.where(self.lapl < 0, 1 / self.lapl, 0),
            sharding.spec_scalar_shard,
        )


fourier: Fourier = Fourier()


# ── Norms and differential operators ─────────────────────────────────────


def get_inprod(
    vector_spec_1: Array, vector_spec_2: Array, k_metric: Array
) -> Array:
    """Volume-averaged L2 inner product ``<u1, u2>`` in spectral space.

    A direct Parseval sum over all Fourier modes.
    """
    return jnp.sum(
        jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
        dtype=sharding.float_type,
    )


def get_norm2(vector_spec: Array, k_metric: Array) -> Array:
    """Squared L2 norm ``||u||^2 = <u, u>``."""
    return get_inprod(vector_spec, vector_spec, k_metric)


def get_norm(vector_spec: Array, k_metric: Array) -> Array:
    """L2 norm ``||u|| = sqrt(<u, u>)``."""
    return jnp.sqrt(get_norm2(vector_spec, k_metric))


def derivative(
    data_spec: Array, kx: Array, ky: Array, kz: Array, axis: int
) -> Array:
    """Spectral derivative: `$i k_{\\text{axis}} \\, \\text{data\\_spec}$`."""
    match axis:
        case 0:
            return 1j * kx * data_spec
        case 1:
            return 1j * ky * data_spec
        case 2:
            return 1j * kz * data_spec


def divergence(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral divergence: `$i k_x u + i k_y v + i k_z w$`."""
    return sum([derivative(velocity_spec[i], kx, ky, kz, i) for i in range(3)])


def curl(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral curl (vorticity):
    `$i \\mathbf{k} \\times \\mathbf{u}_{\\text{spec}}$`.
    """
    return 1j * jnp.array(
        [
            ky * velocity_spec[2] - kz * velocity_spec[1],
            kz * velocity_spec[0] - kx * velocity_spec[2],
            kx * velocity_spec[1] - ky * velocity_spec[0],
        ]
    )


def gradient(data_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral gradient: `$[i k_x, i k_y, i k_z] \\, \\text{data\\_spec}$`."""
    return jnp.array([derivative(data_spec, kx, ky, kz, i) for i in range(3)])


def laplacian(data_spec: Array, lapl_spec: Array) -> Array:
    """Apply the spectral Laplacian (pointwise multiply by `$-k^2$`)."""
    return lapl_spec * data_spec


def inverse_laplacian(data_spec: Array, inv_lapl_spec: Array) -> Array:
    """Apply the inverse spectral Laplacian
    (pointwise multiply by `$-1/k^2$`)."""
    return inv_lapl_spec * data_spec


# ── TriplyPeriodicFlow base dataclass ────────────────────────────────────


@register_dataclass_pytree
@dataclass
class TriplyPeriodicFlow:
    """Precomputed data for triply-periodic flows.

    Subclasses must set ``base_flow``, ``curl_base_flow``, and
    ``nonlin_base_flow`` *after* calling ``super().__post_init__()``,
    which builds the time-stepping coefficients ``ldt_1`` and ``ildt_2``.
    """

    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    nonlin_base_flow: Array = field(init=False)
    ldt_1: Array = field(init=False)
    ildt_2: Array = field(init=False)

    def __post_init__(self) -> None:
        """Build time-stepping coefficients.

        For the triply-periodic case the Helmholtz operator is diagonal
        in Fourier space, so the implicit solve reduces to pointwise
        operations:

            `$ldt_1 = \\frac{1}{\\Delta t}
            + (1-c) \\frac{\\nabla^2}{\\mathrm{Re}}$`
            (explicit part)
            `$ildt_2 = \\left(
            \\frac{1}{\\Delta t}
            - c \\frac{\\nabla^2}{\\mathrm{Re}}
            \\right)^{-1}$`
            (inverse of implicit part)

        The mean mode `$(k_y, k_z, k_x) = (0, 0, 0)$` is zeroed out,
        since it is passive (constant shift) for periodic flows.
        """
        ldt_1 = (
            1 / params.step.dt
            + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
        )
        ildt_2 = 1 / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.re
        )

        # Zero the mean modes in timestepper matrices
        self.ldt_1 = ldt_1.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )
        self.ildt_2 = ildt_2.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )


# ── Initialization ────────────────────────────────────────────────────────


def init_state(snapshot: str | None, flow: TriplyPeriodicFlow) -> Array:
    """Initialise the flow state (velocity_spec)."""
    if params.init.start_from_laminar:
        return jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr,
            sharding.phys_vector_shard,
        )
        velocity_phys = velocity_phys.at[...].subtract(flow.base_flow)
        return phys_to_spec(velocity_phys)
    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)


# ── Algebraic Helmholtz operations (triply-periodic specific) ────────────


@partial(vmap, in_axes=(0, 0, None, None))
def _predict_component(
    state: Array,
    rhs_no_lapl: Array,
    ldt_1: Array,
    ildt_2: Array,
) -> Array:
    """Euler predictor step (vmapped over velocity components).

    Computes `$u_p = (u^n \\cdot ldt_1 + f^n) \\cdot ildt_2$`
    as a pointwise operation in spectral space, where the Helmholtz
    inversion is algebraic (multiply by ``ildt_2``).
    """
    return (state * ldt_1 + rhs_no_lapl) * ildt_2


@partial(vmap, in_axes=(0, 0, 0, None))
def _correct_component(
    prediction: Array,
    rhs_no_lapl_prev: Array,
    rhs_no_lapl_next: Array,
    ildt_2: Array,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector step (vmapped over velocity components).

    Computes the correction
    `$\\delta = c (f_{\\text{next}} - f_{\\text{prev}}) \\cdot ildt_2$`
    and returns the updated prediction and the correction itself (for
    convergence monitoring).
    """
    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )
    return prediction + correction, correction


# ── Geometry-general callables for the stepper factory ───────────────────


def _curl_fn(state: Array, fourier_: Fourier) -> Array:
    """Spectral curl with wavenumbers bound from ``fourier``."""
    return curl(state, fourier_.kx, fourier_.ky, fourier_.kz)


def _get_rhs(
    state: Array, fourier_: Fourier, flow_: TriplyPeriodicFlow
) -> Array:
    """Divergence-free RHS: nonlinear term + algebraic pressure projection."""
    nonlin = get_nonlin(
        state,
        flow_.base_flow,
        flow_.curl_base_flow,
        flow_.nonlin_base_flow,
        spec_to_phys,
        phys_to_spec,
        lambda s: _curl_fn(s, fourier_),
    )
    # Pressure Poisson: `$\\nabla^2 p = \\nabla \\cdot \\mathbf{NL}$`
    lapl_pressure = divergence(nonlin, fourier_.kx, fourier_.ky, fourier_.kz)
    # Subtract pressure gradient to enforce incompressibility
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, fourier_.inv_lapl),
        fourier_.kx,
        fourier_.ky,
        fourier_.kz,
    )
    return rhs_no_lapl


def _predict(
    state: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: TriplyPeriodicFlow,
) -> Array:
    """Euler predictor with algebraic Helmholtz inversion."""
    return _predict_component(state, rhs_no_lapl, flow_.ldt_1, flow_.ildt_2)


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: TriplyPeriodicFlow,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector with algebraic Helmholtz inversion."""
    return _correct_component(prediction, rhs_prev, rhs_next, flow_.ildt_2)


def _norm(
    correction: Array, fourier_: Fourier, flow_: TriplyPeriodicFlow
) -> Array:
    """L2 convergence norm."""
    return get_norm(correction, fourier_.k_metric)


# ── Divergence correction ────────────────────────────────────────────────


def correct_divergence(state: Array, fourier_: Fourier) -> Array:
    """Project the velocity onto the divergence-free subspace."""
    correction = -gradient(
        inverse_laplacian(
            divergence(
                state,
                fourier_.kx,
                fourier_.ky,
                fourier_.kz,
            ),
            fourier_.inv_lapl,
        ),
        fourier_.kx,
        fourier_.ky,
        fourier_.kz,
    )

    velocity_corrected = state + correction
    return velocity_corrected


@jit(donate_argnums=0)
def _correct_velocity_jit(state: Array, fourier_: Fourier) -> Array:

    velocity_corrected = correct_divergence(state, fourier_)

    velocity_corrected = velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )

    return velocity_corrected


# ── Stepper factory ─────────────────────────────────────────────────────


def build_triply_periodic_stepper(
    flow: TriplyPeriodicFlow,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], Array],
]:
    """Build time-stepping functions for a triply-periodic flow.

    Returns ``(predict_and_correct, iterate_correction, init_state_bound,
    correct_velocity)`` with the ``fourier`` and *flow* singletons
    already bound.
    """
    _predict_and_correct_jit, _iterate_correction_jit = make_stepper(
        _get_rhs, _predict, _correct, _norm
    )

    def predict_and_correct(
        state: Array,
    ) -> tuple[Array, Array, Array]:
        """Predictor-corrector step with bound singletons."""
        return _predict_and_correct_jit(state, fourier, flow)

    def iterate_correction(
        state_prev: Array,
        prediction: Array,
        rhs_prev: Array,
    ) -> tuple[Array, Array, Array]:
        """One corrector iteration with bound singletons."""
        return _iterate_correction_jit(
            state_prev, prediction, rhs_prev, fourier, flow
        )

    def init_state_bound(snapshot: str | None) -> Array:
        """Initialize the flow state with bound flow singleton."""
        return init_state(snapshot, flow)

    @timer("velocity/correct_velocity")
    def correct_velocity(
        state: Array,
    ) -> Array:
        return _correct_velocity_jit(state, fourier)

    return (
        predict_and_correct,
        iterate_correction,
        init_state_bound,
        correct_velocity,
    )
