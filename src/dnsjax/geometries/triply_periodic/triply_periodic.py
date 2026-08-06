"""Triply-periodic geometry: Fourier class, differential operators, norms,
base dataclass, solvers, and stepper factory.

Provides all geometry-general infrastructure for triply-periodic flows:
the ``Fourier`` wavenumber class, the ``TriplyPeriodicFlow`` base
dataclass (time-stepping coefficients), algebraic Helmholtz predict /
correct operations, divergence correction, state initialisation, and the
``build_triply_periodic_stepper`` factory.

Flow-specific modules (e.g. ``flows.triply_periodic.monochromatic``)
subclass
``TriplyPeriodicFlow`` to define the base flow, then call
``build_triply_periodic_stepper`` to obtain ready-to-use time-stepping
functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from jax import Array, device_put, jit, vmap
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...measurements import get_cfl
from ...operators import (
    complex_harmonics,
    phys_to_spec,
    real_harmonics,
    spec_to_phys,
)
from ...parameters import padded_res, params
from ...rhs import get_nonlin
from ...sharding import register_dataclass_pytree, sharding
from ...timestep import make_stepper

ly = 4  # Shear-direction box length is the length reference and fixed


@register_dataclass_pytree
@dataclass
class Fourier:
    r"""Wavenumber grids for the triply-periodic geometry.

    Broadcasting shapes match the spectral layout
    ``(ky, kz, kx)``:

    - ``kx``: shape ``(1, 1, nx_spec)`` — sharded on ``np1``.
    - ``kz``: shape ``(1, nz_spec, 1)`` — sharded on ``np0``.
    - ``ky``: shape ``(ny-1, 1, 1)`` — fully local.

    ``k_metric`` equals 2 for `$k_x > 0$` and 1 for
    `$k_x = 0$`, accounting for the Hermitian symmetry of
    the real FFT.

    The wavenumber arrays are global multi-device arrays: host-side
    consumers recompute them from the JAX-free
    :mod:`dnsjax.harmonics` sequences (`$\times\,2\pi/L$`), never
    ``np.asarray`` on these fields.
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    ky: Array = field(init=False)
    k_metric: Array = field(init=False)
    lapl: Array = field(init=False)
    inv_lapl: Array = field(init=False)

    def __post_init__(self) -> None:
        kx_true = real_harmonics(params.res.nx)
        if sharding.nx_spec_pad:
            kx_true = jnp.concatenate(
                [kx_true, jnp.zeros(sharding.nx_spec_pad)]
            )
        self.kx = (
            device_put(
                kx_true.reshape([1, 1, -1]),
                P(None, None, sharding.a1),
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )

        kz_true = complex_harmonics(params.res.nz)
        if sharding.nz_spec_pad:
            kz_true = jnp.concatenate(
                [kz_true, jnp.zeros(sharding.nz_spec_pad)]
            )
        self.kz = (
            device_put(
                kz_true.reshape([1, -1, 1]),
                P(None, sharding.a0, None),
            )
            * 2
            * jnp.pi
            / params.geo.lz
        )

        self.ky = (
            device_put(
                complex_harmonics(params.res.ny).reshape([-1, 1, 1]),
                sharding.no_shard,
            )
            * 2
            * jnp.pi
            / ly
        )

        self.k_metric = jnp.where(self.kx == 0, 1, 2).astype(
            sharding.float_type
        )
        self.lapl = -(self.kx**2 + self.ky**2 + self.kz**2)
        self.inv_lapl = jnp.where(self.lapl < 0, 1 / self.lapl, 0)


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


def inverse_laplacian(data_spec: Array, inv_lapl_spec: Array) -> Array:
    """Apply the inverse spectral Laplacian
    (pointwise multiply by `$-1/k^2$`)."""
    return inv_lapl_spec * data_spec


# ── TriplyPeriodicFlow base dataclass ────────────────────────────────────


@register_dataclass_pytree
@dataclass
class TriplyPeriodicFlow:
    """Precomputed data for triply-periodic flows.

    Subclasses must set ``base_flow`` and ``curl_base_flow``
    *after* calling ``super().__post_init__()``, which builds
    the time-stepping coefficients ``ldt_1`` and ``ildt_2``.
    ``dt``/``ab2_kappa`` are the live time step and AB2 step
    ratio as 0-d array leaves (read by the steppers instead of
    ``params.step.dt``; rebuilt with ``ldt_1``/``ildt_2`` by the
    builder's ``set_dt``).
    """

    dt: Array = field(init=False)
    ab2_kappa: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    cfl_inv_spacing: Array = field(init=False)
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
        # Live-dt pytree leaves (class docstring; rebuilt by the
        # builder's ``set_dt`` with identical dtype/shape).
        self.dt = jnp.asarray(params.step.dt, dtype=sharding.float_type)
        self.ab2_kappa = jnp.ones((), dtype=sharding.float_type)

        # Strong-typed 1/dt (Python-float division, then cast: the
        # values are bit-identical to the plain expression) so these
        # leaves carry the same avals as the jitted
        # ``_build_dt_leaves`` rebuild -- a weak-typed leaf would
        # retrace every stepper on the first ``set_dt``.
        inv_dt = jnp.asarray(1 / params.step.dt, dtype=sharding.float_type)
        ldt_1 = (
            inv_dt
            + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
        )
        ildt_2 = 1 / (
            inv_dt - params.step.implicitness * fourier.lapl / params.phys.re
        )

        # Zero the mean modes in timestepper matrices
        self.ldt_1 = ldt_1.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )
        self.ildt_2 = ildt_2.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )
        # (``_build_dt_leaves`` mirrors the two coefficients with a
        # traced dt for the adaptive-dt rebuild; this eager build
        # keeps the historical Python-float arithmetic bit-exact.)

        # Inverse local advection length scales for the CFL
        # diagnostic (:func:`dnsjax.measurements.get_cfl`),
        # per component (u, v, w); all three directions are
        # uniform.  Uses the spectral-resolution spacing
        # `$\Delta = L/n$`; switch to ``padded_res.nx_padded``
        # / ``ny_padded`` / ``nz_padded`` for the
        # dealiased-grid convention.
        inv_vals = jnp.array(
            [
                params.res.nx / params.geo.lx,
                params.res.ny / ly,
                params.res.nz / params.geo.lz,
            ],
            dtype=sharding.float_type,
        )
        self.cfl_inv_spacing = device_put(
            jnp.broadcast_to(
                inv_vals[:, None, None, None],
                (3, padded_res.ny_padded, 1, 1),
            ),
            sharding.no_shard,
        )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: TriplyPeriodicFlow,
) -> dict[str, Array]:
    r"""Rebuild the ``dt``-dependent flow leaves at the traced *dt*.

    The pure counterpart of the ``__post_init__`` coefficient setup,
    jitted by the builder's ``set_dt``: the algebraic Helmholtz
    inverse `$1/(1/\Delta t - c\,\nabla^2/\mathrm{Re})$` is regular
    for every `$\Delta t > 0$`, so no stability check is involved and
    ``dt`` is fully continuous.  *flow_* is unused (uniform
    ``(dt, fourier, flow)`` rebuild signature across the geometry
    families).
    """
    c = params.step.implicitness
    ldt_1 = 1 / dt + (1 - c) * fourier_.lapl / params.phys.re
    ildt_2 = 1 / (1 / dt - c * fourier_.lapl / params.phys.re)
    return {
        "dt": dt,
        "ldt_1": ldt_1.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        ),
        "ildt_2": ildt_2.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        ),
    }


# ── Initialization ────────────────────────────────────────────────────────


def init_state() -> Array:
    """The ``start_from_laminar`` state: zero spectral perturbation.

    Snapshot resume and the in-process random / localized-rolls modes
    are dispatched in ``__main__``; this is called only for the
    laminar start (the perturbation about the base flow is zero).
    """
    return jnp.zeros(
        shape=(3, *sharding.spec_shape),
        dtype=sharding.complex_type,
        out_sharding=sharding.spec_vector_shard,
    )


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


# Per-direction CFL column names, matching the physical-space
# component order (u, v, w) = (x, y, z).
CFL_NAMES: tuple[str, str, str] = ("CFL_x", "CFL_y", "CFL_z")


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: TriplyPeriodicFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    """Divergence-free RHS: nonlinear term + algebraic pressure projection."""
    nonlin = get_nonlin(
        state,
        flow_.base_flow,
        flow_.curl_base_flow,
        spec_to_phys,
        phys_to_spec,
        lambda s: _curl_fn(s, fourier_),
        measure_fn,
    )
    if measure_fn is not None:
        nonlin, measurements = nonlin
    # Pressure Poisson: `$\\nabla^2 p = \\nabla \\cdot \\mathbf{NL}$`
    lapl_pressure = divergence(nonlin, fourier_.kx, fourier_.ky, fourier_.kz)
    # Subtract pressure gradient to enforce incompressibility
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, fourier_.inv_lapl),
        fourier_.kx,
        fourier_.ky,
        fourier_.kz,
    )
    if measure_fn is None:
        return rhs_no_lapl
    return rhs_no_lapl, measurements


def _get_rhs(
    state: Array, fourier_: Fourier, flow_: TriplyPeriodicFlow
) -> Array:
    """Divergence-free RHS: nonlinear term + algebraic pressure projection."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array, fourier_: Fourier, flow_: TriplyPeriodicFlow
) -> tuple[Array, dict[str, Array]]:
    """Divergence-free RHS + CFL measurements."""

    def _measure(u_phys: Array, omega_phys: Array) -> dict[str, Array]:
        return get_cfl(
            u_phys,
            flow_.base_flow,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
            flow_.dt,
        )

    return _get_rhs_core(state, fourier_, flow_, _measure)


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
    r"""Project velocity onto the divergence-free subspace.

    This is the **post-step** incompressibility projection.
    The first projection happens inside :func:`_get_rhs`,
    where the pressure Poisson solve removes the divergent
    part of the nonlinear term before the Helmholtz step.
    This second projection removes any residual divergence
    accumulated during the corrector iterations.  It runs
    *inside* the stepper's jit scope (:func:`_finalize_state`
    via ``make_stepper``'s *finalize_fn*), fused with the
    step's tail rather than as a separate per-step dispatch.
    """
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


def _finalize_state(
    state: Array, fourier_: Fourier, flow_: TriplyPeriodicFlow
) -> Array:
    """Divergence correction + mean-mode zeroing (post-step).

    Passed to ``make_stepper`` as *finalize_fn*: applied once to the
    accepted state at the end of every step, inside the stepper's jit
    scope (fused; no separate per-step dispatch or extra state-sized
    read/write pass).
    """
    velocity_corrected = correct_divergence(state, fourier_)

    return velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )


# ── Stepper factory ─────────────────────────────────────────────────────


def build_triply_periodic_stepper(
    flow: TriplyPeriodicFlow,
) -> tuple[
    Callable[[], Array],
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array], tuple[Array, Array, Array, dict[str, Array]]],
    Callable[[Array, Array], tuple[Array, Array, Array, Array]],
    Callable[
        [Array, Array], tuple[Array, Array, Array, Array, dict[str, Array]]
    ],
    Callable[[float], None],
    Callable[[], None],
]:
    """Build time-stepping functions for a triply-periodic flow.

    Returns ``(init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured, step_cnab2,
    step_cnab2_measured, set_dt, reset_ab2_kappa)`` with the
    ``fourier`` and *flow* singletons already bound -- the same
    7-tuple as the wall-bounded builder (``set_dt`` /
    ``reset_ab2_kappa`` are the adaptive-dt hooks backed by
    ``_build_dt_leaves``).  ``step_cnab2`` and its measured variant
    are the CN/AB2 scheme
    (``step.scheme == "cnab2"``).  The post-step divergence
    projection + mean-mode zeroing (:func:`_finalize_state`) is
    fused into every step via ``make_stepper``'s *finalize_fn*,
    so the accepted state is already projected on return (no
    separate per-step call).
    """

    def _step_scales(fourier_: Fourier, flow_: TriplyPeriodicFlow) -> tuple:
        """Live ``(dt, kappa)`` from the flow leaves (adaptive dt)."""
        return flow_.dt, flow_.ab2_kappa

    (
        _predict_and_fully_correct_jit,
        _predict_and_fully_correct_measured_jit,
        _step_cnab2_jit,
        _step_cnab2_measured_jit,
    ) = make_stepper(
        _get_rhs,
        _predict,
        _correct,
        _norm,
        _get_rhs_measured,
        finalize_fn=_finalize_state,
        step_scales_fn=_step_scales,
    )

    def predict_and_fully_correct(
        state: Array,
    ) -> tuple[Array, Array, Array]:
        """Fused predict + corrector loop with bound singletons."""
        return _predict_and_fully_correct_jit(state, fourier, flow)

    def predict_and_fully_correct_measured(
        state: Array,
    ) -> tuple[Array, Array, Array, dict[str, Array]]:
        """Fused step + physical-space measurements (at `$u^n$`)."""
        return _predict_and_fully_correct_measured_jit(state, fourier, flow)

    def step_cnab2(
        state: Array, carry: Array
    ) -> tuple[Array, Array, Array, Array]:
        """One CN/AB2 step with bound singletons.  Returns
        ``(state_next, carry, error, num_c)`` (``error``/``num_c`` are
        ``0`` -- triply-periodic needs no base-flow-coupling corrector,
        its Fourier ``y`` making that term non-stiff).  The divergence
        projection (:func:`_finalize_state`) is fused into the step,
        as for the corrector scheme."""
        return _step_cnab2_jit(state, carry, fourier, flow)

    def step_cnab2_measured(
        state: Array, carry: Array
    ) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
        """CN/AB2 step + physical-space measurements (at `$u^n$`)."""
        return _step_cnab2_measured_jit(state, carry, fourier, flow)

    def init_state_bound() -> Array:
        """The ``start_from_laminar`` state (zero perturbation)."""
        return init_state()

    _dt_leaves_jit = jit(_build_dt_leaves)
    _dt_box = [float(params.step.dt)]

    def set_dt(new_dt: float) -> None:
        """Switch the live time step to *new_dt* (adaptive dt).

        Jitted recompute of ``ldt_1``/``ildt_2`` + in-place
        assignment on the bound flow singleton; also sets
        ``ab2_kappa = new_dt / dt_prev`` for the next CN/AB2 step
        (see ``build_wall_bounded_stepper`` for the shared
        contract).  No stepper recompiles.
        """
        kappa = new_dt / _dt_box[0]
        leaves = _dt_leaves_jit(
            jnp.asarray(new_dt, dtype=sharding.float_type), fourier, flow
        )
        for name, val in leaves.items():
            setattr(flow, name, val)
        flow.ab2_kappa = jnp.asarray(kappa, dtype=sharding.float_type)
        _dt_box[0] = new_dt

    def reset_ab2_kappa() -> None:
        """Reset the AB2 step ratio to 1 (one step after ``set_dt``)."""
        flow.ab2_kappa = jnp.ones((), dtype=sharding.float_type)

    return (
        init_state_bound,
        predict_and_fully_correct,
        predict_and_fully_correct_measured,
        step_cnab2,
        step_cnab2_measured,
        set_dt,
        reset_ab2_kappa,
    )
