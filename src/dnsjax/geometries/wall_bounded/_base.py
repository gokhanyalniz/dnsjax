"""Shared infrastructure for wall-bounded geometries.

Functions that are identical (or near-identical) between the
Cartesian and cylindrical geometry modules live here to avoid
duplication.  Geometry-specific code (operator assembly, IMM
iteration, curl, etc.) stays in the respective geometry
modules.
"""

from collections.abc import Callable

import jax
from jax import Array, lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
from ...sharding import sharding
from ...timestep import make_stepper

# ── Spectral transform aliases ──────────────────────────────────


phys_to_spec = phys_to_spec_2d
spec_to_phys = spec_to_phys_2d


# ── Mean-mode extraction ───────────────────────────────────────


def extract_mean_mode(state: Array) -> Array:
    r"""Extract the mean Fourier mode from a spectral state.

    Given a wall-bounded spectral state of shape
    ``(C, N_y, N_{k_z}, N_{k_x})`` where `$k_z$` is sharded
    by ``np0`` and `$k_x$` by ``np1``, returns the
    `$k_z = k_x = 0$` mode of shape ``(C, N_y)`` in
    `$O(N_y)$` work per device via ``shard_map`` + ``psum``.

    Parameters
    ----------
    state:
        Shape ``(C, N_y, N_{k_z}, N_{k_x})``.

    Returns
    -------
    :
        Shape ``(C, N_y)``, replicated across devices.
    """

    def _local(shard: Array) -> Array:
        first = shard[:, :, 0, 0]
        is_source = (lax.axis_index("np0") == 0) & (lax.axis_index("np1") == 0)
        return lax.psum(
            jnp.where(is_source, first, jnp.zeros_like(first)),
            ("np0", "np1"),
        )

    return shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=sharding.spec_vector_shard,
        out_specs=P(None, None),
    )(state)


# ── Norms and integration ───────────────────────────────────────


def integrate_scalar(scalar_data: Array, y_weights: Array) -> Array:
    r"""Quadrature integral using precomputed weights.

    Returns `$\sum_j w_j\,f(y_j)$` where `$w_j$` are
    precomputed quadrature weights (Clenshaw-Curtis for CGL
    grids, or composite polynomial weights with radial
    Jacobian for cylindrical grids).

    Parameters
    ----------
    scalar_data:
        1-D array of function values at grid points,
        shape ``(N,)``.
    y_weights:
        Precomputed quadrature weights, shape ``(N,)``.
    """
    return jnp.dot(y_weights, scalar_data)


def get_inprod(
    vector_spec_1: Array,
    vector_spec_2: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""Volume-averaged L2 inner product in spectral space.

    Fourier modes on the two periodic axes are summed first,
    then the resulting wall-normal profile is integrated with
    precomputed quadrature weights.

    Parameters
    ----------
    vector_spec_1, vector_spec_2:
        Spectral velocity fields, shape
        ``(C, N_wall, N_mode1, N_mode2)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Quadrature weights for wall-normal integration.
    """
    return (
        integrate_scalar(
            jnp.sum(
                jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                axis=(0, 2, 3),
            ).real,
            y_weights,
        )
        / derived_params.volume_fac
    )


def get_norm2(
    vector_spec: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""Squared L2 norm `$\|u\|^2 = \langle u, u \rangle$`."""
    return get_inprod(vector_spec, vector_spec, k_metric, y_weights)


def get_norm(
    vector_spec: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""L2 norm `$\|u\| = \sqrt{\langle u, u \rangle}$`."""
    return jnp.sqrt(get_norm2(vector_spec, k_metric, y_weights))


def get_pert_enstrophy(
    state: Array,
    D1: Array,
    k2: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""Perturbation enstrophy for Cartesian wall-bounded flows.

    Uses the identity
    `$\Omega' = \langle |\nabla \mathbf{u}'|^2 \rangle$`,
    split into horizontal and wall-normal contributions:

    .. math::
        \Omega' = \langle (k_x^2 + k_z^2)\,|\mathbf{u}'|^2
        \rangle
        + \langle |\partial_y \mathbf{u}'|^2 \rangle

    Parameters
    ----------
    state:
        Spectral velocity, shape ``(3, Ny, Nkz, Nkx)``.
    D1:
        First-derivative FD matrix, shape ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, shape ``(1, Nkz, Nkx)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Quadrature weights for wall-normal integration.
    """
    horiz = get_norm2(state, k2 * k_metric, y_weights)
    dy_state = jnp.einsum("ij, cjzx -> cizx", D1, state)
    wall_normal = get_norm2(dy_state, k_metric, y_weights)
    return horiz + wall_normal


# ── Flow state initialisation ───────────────────────────────────


def init_state(snapshot: str | None) -> Array:
    """Initialise the flow state (velocity_spec)."""
    if params.init.start_from_laminar:
        velocity_spec = jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys_nonexpanded"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr, sharding.phys_vector_shard
        )
        velocity_spec = phys_to_spec_2d(velocity_phys)
    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)

    return velocity_spec


# ── Stepper factory ─────────────────────────────────────────────


def build_wall_bounded_stepper(
    get_rhs_fn: Callable,
    predict_fn: Callable,
    correct_fn: Callable,
    norm_fn: Callable,
    fourier: object,
    flow: object,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array]],
]:
    """Build time-stepping functions for a wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct)`` with the
    *fourier* and *flow* singletons already bound.

    Parameters
    ----------
    get_rhs_fn, predict_fn, correct_fn, norm_fn:
        Geometry-specific callables passed to
        :func:`~dnsjax.timestep.make_stepper`.
    fourier:
        Geometry-specific ``Fourier`` singleton.
    flow:
        Geometry-specific flow dataclass instance.
    """
    (
        _predict_and_correct_jit,
        _iterate_correction_jit,
        _predict_and_fully_correct_jit,
    ) = make_stepper(get_rhs_fn, predict_fn, correct_fn, norm_fn)

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

    def predict_and_fully_correct(
        state: Array,
    ) -> tuple[Array, Array, Array]:
        """Fused predict + corrector loop with bound singletons."""
        return _predict_and_fully_correct_jit(state, fourier, flow)

    def init_state_bound(snapshot: str | None) -> Array:
        """Initialize the flow state."""
        return init_state(snapshot)

    return (
        predict_and_correct,
        iterate_correction,
        init_state_bound,
        predict_and_fully_correct,
    )
