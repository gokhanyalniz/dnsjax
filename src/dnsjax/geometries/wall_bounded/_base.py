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


# ── Wall-normal matrix application ──────────────────────────────


def apply_y_matrix(mat: Array, field: Array, component_axis: int = 0) -> Array:
    r"""Left-multiply along the wall-normal axis with a real matrix.

    Computes ``einsum("ij, jzx -> izx", mat, field)`` for a 3-d
    *field*, or a component-batched contraction for a 4-d *field*.

    **Layout / transposes.**  The contraction runs as a cuBLAS GEMM
    over the wall-normal axis.  When that axis is **leading** (the 3-d
    case, or 4-d with ``component_axis == 1`` so *field* is
    `$(N_y, C, N_1, N_2)$`) it is already in GEMM contraction position
    and **no transpose is emitted**.  With ``component_axis == 0``
    (*field* `$(C, N_y, N_1, N_2)$`) the wall-normal axis is interior,
    so XLA transposes it into position and back -- two field-sized
    transposes per call.  The batched IMM/RHS matvecs therefore stack
    their inputs **y-leading** and pass ``component_axis=1``; this keeps
    the single batched GEMM (one per ``D1``/``D2``) yet emits zero
    transposes (confirmed: ``ij, jczx -> iczx`` lowers transpose-free
    for ``cuda``, vs one transpose for ``ij, cjzx -> cizx``).

    A complex *field* is first split into a real trailing re/im
    axis so the contraction runs as a real GEMM: half the FLOPs
    of the complex GEMM that dtype promotion would otherwise
    produce (a real-by-complex ``einsum`` upcasts and runs
    4-real-multiply complex arithmetic).  The split and merge
    are field-sized elementwise passes that fuse with
    neighbouring ops; the win grows with `$N_y$` (GEMM FLOPs
    scale as `$N_y$` per element, the passes do not).

    Parameters
    ----------
    mat:
        Real matrix, shape ``(M, N_y)``.  ``M = N_y`` for full
        FD matrices; fewer rows for partial-row corrections.
    field:
        Real or complex field of shape ``(N_y, N_1, N_2)`` (3-d), or
        4-d with the wall-normal axis at position ``component_axis + 1``
        (so ``(C, N_y, N_1, N_2)`` for ``component_axis == 0``, the
        default, or the transpose-free ``(N_y, C, N_1, N_2)`` for
        ``component_axis == 1``).
    component_axis:
        Position of the leading batch (component) axis of a 4-d
        *field*: ``0`` (wall-normal interior) or ``1`` (wall-normal
        leading, transpose-free).  Ignored for a 3-d *field*.  The
        output preserves the input layout.
    """
    if jnp.iscomplexobj(field) and not jnp.iscomplexobj(mat):
        f = jnp.stack([field.real, field.imag], axis=-1)
        if field.ndim == 3:
            out = jnp.einsum("ij, jzxr -> izxr", mat, f)
        elif component_axis == 1:
            out = jnp.einsum("ij, jczxr -> iczxr", mat, f)
        else:
            out = jnp.einsum("ij, cjzxr -> cizxr", mat, f)
        return lax.complex(out[..., 0], out[..., 1])
    if field.ndim == 3:
        return jnp.einsum("ij, jzx -> izx", mat, field)
    if component_axis == 1:
        return jnp.einsum("ij, jczx -> iczx", mat, field)
    return jnp.einsum("ij, cjzx -> cizx", mat, field)


# ── Base-flow coupling (FFT-free, for the CN/AB2 scheme) ─────────


def base_flow_coupling(
    u: Array, omega: Array, base_flow: Array, curl_base_flow: Array
) -> Array:
    r"""Linear base-flow coupling `$\mathbf{u}' \times \nabla\times
    \mathbf{U} + \mathbf{U} \times \boldsymbol{\omega}'$`.

    The two base-flow cross-product terms of the rotational nonlinear
    form (:mod:`dnsjax.rhs`), as a component-wise expression in a local
    orthonormal basis -- Cartesian `$(x, y, z)$` or the cylindrical
    `$(z, r, \theta)$` triad (both right-handed, so the standard
    cross-product formula applies).  All inputs are in the **same**
    representation; *base_flow* / *curl_base_flow* are the wall-normal
    (or radial) profiles `$(3, N, 1, 1)$`, broadcast over the Fourier
    axes.  Evaluated with `$\boldsymbol{\omega}'$` already in hand
    (spectral curl), this needs **no Fourier transform** -- used by the
    CN/AB2 scheme to make the (stiff) base-flow coupling implicit; see
    ``step_cnab2`` in :mod:`dnsjax.timestep` and each geometry's
    ``_l_bf``.
    """
    u0, u1, u2 = u[0], u[1], u[2]
    w0, w1, w2 = omega[0], omega[1], omega[2]
    U0, U1, U2 = base_flow[0], base_flow[1], base_flow[2]
    c0, c1, c2 = curl_base_flow[0], curl_base_flow[1], curl_base_flow[2]
    return jnp.array(
        [
            (u1 * c2 - u2 * c1) + (U1 * w2 - U2 * w1),
            (u2 * c0 - u0 * c2) + (U2 * w0 - U0 * w2),
            (u0 * c1 - u1 * c0) + (U0 * w1 - U1 * w0),
        ]
    )


# ── Base-flow padding ───────────────────────────────────────────


def pad_base_flow(flow: object) -> None:
    """Precompute the y-padded base flow used by the RHS path.

    Sets ``flow.base_flow_padded`` and
    ``flow.curl_base_flow_padded``: zero-padded along the
    wall-normal axis by ``sharding.ny_y_pad`` rows (matching the
    physical-space fields of :mod:`dnsjax.fft`), so the RHS path
    needs no per-call ``jnp.pad``.  When no padding is needed
    the fields alias the originals (no extra memory; the padded
    profiles are tiny ``(3, ny + pad, 1, 1)`` arrays otherwise).

    Called by each flow subclass at the end of its
    ``__post_init__``, after ``base_flow`` is set.
    """
    if sharding.ny_y_pad:
        ypad = ((0, 0), (0, sharding.ny_y_pad), (0, 0), (0, 0))
        flow.base_flow_padded = jnp.pad(flow.base_flow, ypad)
        flow.curl_base_flow_padded = jnp.pad(flow.curl_base_flow, ypad)
    else:
        flow.base_flow_padded = flow.base_flow
        flow.curl_base_flow_padded = flow.curl_base_flow


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
    dy_state = apply_y_matrix(D1, state)
    wall_normal = get_norm2(dy_state, k_metric, y_weights)
    return horiz + wall_normal


# ── Flow state initialisation ───────────────────────────────────


def init_state(snapshot: str | None) -> Array:
    """Initialise the flow state (velocity_spec).

    A provided snapshot path (legacy ``.npz``) takes precedence over
    ``start_from_laminar`` so a supplied snapshot always wins; zarr3
    snapshot resume is handled in ``__main__`` before this is called.
    """
    if snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys_nonexpanded"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr, sharding.phys_vector_shard
        )
        velocity_spec = phys_to_spec_2d(velocity_phys)
    elif params.init.start_from_laminar:
        velocity_spec = jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
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
    get_rhs_measured_fn: Callable,
    l_bf_fn: Callable | None = None,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array], tuple[Array, Array, Array, dict[str, Array]]],
    Callable[[Array, Array], tuple[Array, Array, Array, Array]],
    Callable[
        [Array, Array], tuple[Array, Array, Array, Array, dict[str, Array]]
    ],
]:
    """Build time-stepping functions for a wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured, step_cnab2,
    step_cnab2_measured)`` with the *fourier* and *flow*
    singletons already bound.  The last two are the CN/AB2
    scheme (``step.scheme == "cnab2"``).

    Parameters
    ----------
    get_rhs_fn, predict_fn, correct_fn, norm_fn:
        Geometry-specific callables passed to
        :func:`~dnsjax.timestep.make_stepper`.
    fourier:
        Geometry-specific ``Fourier`` singleton.
    flow:
        Geometry-specific flow dataclass instance.
    get_rhs_measured_fn:
        Measured RHS variant (returns the RHS plus a dict of
        physical-space measurements; see
        :mod:`dnsjax.measurements`).
    l_bf_fn:
        FFT-free linear base-flow coupling ``state -> L_bf`` (see the
        geometry ``_l_bf``), treated implicitly by the CN/AB2 scheme
        so its stiff wall-normal derivative does not force a tiny
        time step.  Passed through to :func:`~dnsjax.timestep.make_stepper`.
    """
    (
        _predict_and_correct_jit,
        _iterate_correction_jit,
        _predict_and_fully_correct_jit,
        _predict_and_fully_correct_measured_jit,
        _step_cnab2_jit,
        _step_cnab2_measured_jit,
    ) = make_stepper(
        get_rhs_fn,
        predict_fn,
        correct_fn,
        norm_fn,
        get_rhs_measured_fn,
        l_bf_fn,
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
        ``(state_next, carry, error, num_c)``; feed ``carry`` back
        unchanged.  ``error``/``num_c`` are the FFT-free base-flow
        coupling corrector's (see ``step_cnab2`` in ``timestep.py``)."""
        return _step_cnab2_jit(state, carry, fourier, flow)

    def step_cnab2_measured(
        state: Array, carry: Array
    ) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
        """CN/AB2 step + physical-space measurements (at `$u^n$`)."""
        return _step_cnab2_measured_jit(state, carry, fourier, flow)

    def init_state_bound(snapshot: str | None) -> Array:
        """Initialize the flow state."""
        return init_state(snapshot)

    return (
        predict_and_correct,
        iterate_correction,
        init_state_bound,
        predict_and_fully_correct,
        predict_and_fully_correct_measured,
        step_cnab2,
        step_cnab2_measured,
    )
