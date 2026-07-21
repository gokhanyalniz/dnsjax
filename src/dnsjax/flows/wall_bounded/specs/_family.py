r"""Shared spec fragments and derive math for the wall-bounded flows.

Field fragments (tuples of :class:`~dnsjax.flow_spec.FieldSpec`) are
composed by the per-flow spec modules in this package; the derive
helpers are the family-shared parameter math (the former per-system
branches of ``parameters.update_parameters``).  Everything here is
JAX-free and never imports :mod:`dnsjax.parameters` (hooks receive the
live ``params`` / ``derived_params`` objects as arguments).
"""

from math import pi
from typing import Any

from ....flow_spec import UNSET, DeferredSpec, FieldSpec

# ── Field fragments ──────────────────────────────────────────────

#: Grid choices per family (the cylindrical names make the rigged /
#: half character explicit; see the ``Geometry`` docstring).
CARTESIAN_ANNULAR_GRIDS: tuple[str, ...] = ("cgl", "tanh")
CYLINDRICAL_GRIDS: tuple[str, ...] = ("half-cgl", "rigged-cgl", "half-tanh")


def wall_fields(
    u_grid_default: float, grid_choices: tuple[str, ...]
) -> tuple[FieldSpec, ...]:
    """Fields shared by every wall-bounded flow.

    ``u_grid_default`` is the flow's laminar bulk velocity in the grid
    direction (materialized when no layer sets ``phys.u_grid``);
    ``grid_choices`` the family's valid ``geo.grid_type`` values (the
    first entry is only the *scheme-independent* default -- the
    resolved default comes from ``FlowSpec.grid_type_default``).

    ``res.consistent_imm`` is omitted for the cylindrical family here
    (its axis operators need the ``x = r^2`` construction, not the
    Cartesian/annular ``D2 := D1 D1``); the pipe adds the field back on
    its own spec (``specs/pipe.py``).
    """
    cylindrical = grid_choices == CYLINDRICAL_GRIDS
    if cylindrical:
        grid_desc = (
            "Radial grid: 'half-cgl' (iterative-cn default), "
            "'rigged-cgl' (cnab2 default; twice the innermost radius), "
            "or 'half-tanh' (outer-wall clustered)."
        )
    else:
        grid_desc = (
            "Wall-normal grid: 'cgl' (Chebyshev-Gauss-Lobatto, the "
            "default) or 'tanh' (grid_stretch-controlled wall "
            "clustering)."
        )
    return (
        FieldSpec("phys", "u_grid", default=u_grid_default),
        FieldSpec("res", "fd_order"),
        *(() if cylindrical else (FieldSpec("res", "consistent_imm"),)),
        FieldSpec("geo", "wall_grid"),
        FieldSpec(
            "geo",
            "grid_type",
            description=grid_desc,
            choices=grid_choices,
        ),
        FieldSpec("geo", "grid_stretch"),
        FieldSpec("init", "random_mean_flow"),
        FieldSpec("init", "localized_rolls"),
        FieldSpec("init", "localized_rolls_amplitude"),
        FieldSpec("init", "localized_rolls_width"),
        FieldSpec("init", "localized_rolls_wavelength"),
        FieldSpec("step", "implicit_mean_coupling"),
        FieldSpec("step", "split_corrector"),
        FieldSpec("solver", "backend"),
        FieldSpec("solver", "pallas_block_m0"),
        FieldSpec("solver", "pallas_block_m1"),
        FieldSpec("solver", "pallas_stability_tol"),
    )


def cartesian_fields() -> tuple[FieldSpec, ...]:
    """Cartesian (plane-channel) domain and physics fields."""
    return (
        FieldSpec("geo", "lx"),
        FieldSpec("geo", "lz"),
        FieldSpec("geo", "tilt_degree"),
        FieldSpec("res", "nx"),
        FieldSpec("res", "ny"),
        FieldSpec("res", "nz"),
        FieldSpec("phys", "re"),
        FieldSpec("phys", "block_mean_spanwise_velocity"),
    )


def cyl_annular_fields(axial_default: Any = UNSET) -> tuple[FieldSpec, ...]:
    """Cylindrical/annular coordinate aliases + the azimuthal wedge.

    The user-facing names follow the geometry -- ``lz`` (axial length,
    internally ``geo.lx``), ``nz`` (axial modes, internally
    ``res.nx``), ``nr`` (radial points, internally ``res.ny``),
    ``ntheta`` (azimuthal modes, internally ``res.nz``) -- while the
    code keeps its internal layout.  The azimuthal *extent* is not a
    free length: it is the wedge ``2*pi/m0`` (internal ``geo.lz``,
    derived).  ``axial_default`` overrides the axial-period default
    (the viscoelastic annulus defaults to ``2*pi``).
    """
    return (
        FieldSpec(
            "geo",
            "lx",
            public="lz",
            description="Axial period of the domain.",
            default=axial_default,
        ),
        FieldSpec(
            "res",
            "nx",
            public="nz",
            description=(
                "Axial Fourier modes (= physical grid points before "
                "dealiasing)."
            ),
        ),
        FieldSpec(
            "res",
            "ny",
            public="nr",
            description="Radial finite-difference grid points.",
        ),
        FieldSpec(
            "res",
            "nz",
            public="ntheta",
            description=(
                "Azimuthal Fourier modes over the wedge (= physical "
                "grid points before dealiasing)."
            ),
        ),
        FieldSpec("geo", "m0"),
    )


#: Deferred: tilting is Cartesian/periodic-only for now (the
#: cylindrical/annular geometries never read the tilt).
DEFERRED_TILT = DeferredSpec(
    "geo",
    "tilt_degree",
    "geo.tilt_degree (tilted driving) is not implemented yet for the "
    "cylindrical/annular geometries.",
)

# ── Shared derive math ───────────────────────────────────────────


def cartesian_derive(params, derived, user_set) -> None:
    """Cartesian channel: int_{-1}^{1} dy area normalisation."""
    derived.volume_fac = 2


def annular_base_derive(params, derived) -> float:
    """Annular geometry shared by TC / quasi-Keplerian / Dean.

    Validates the radius ratio ``eta``, derives the non-dim radii on
    the unit gap, forces the azimuthal wedge extent ``lz = 2*pi/m0``,
    and sets the area normalisation.  Returns ``eta``.
    """
    eta = params.geo.eta
    if eta is None:
        raise ValueError(
            f"{params.phys.system} requires geo.eta (radius ratio r1/r2)"
        )
    r1 = eta / (1 - eta)
    r2 = 1 / (1 - eta)
    derived.r_inner = r1
    derived.r_outer = r2
    # The azimuthal modes are the integer multiples m = m0 * harmonic
    # over the wedge (m0 = 1 is the full circle); see ``geo.m0``.
    params.geo.lz = 2 * pi / params.geo.m0
    derived.volume_fac = (r2**2 - r1**2) / 2  # int_{r1}^{r2} r dr
    return eta


def circular_couette_derive(params, derived, eta: float) -> None:
    r"""Circular-Couette base flow `$U_\theta = A_0 r + B_0/r$`.

    Validates the ``(re1, re2)`` control parameters, sets the
    reference Reynolds number ``params.phys.re = Re_ref`` (so every
    downstream ``1/re`` viscous/IMM/stats path is reused unchanged),
    and derives the gap-scaled coefficients ``ccf_A``/``ccf_B``.
    """
    system = params.phys.system
    re1, re2 = params.phys.re1, params.phys.re2
    if re1 is None or re2 is None:
        raise ValueError(f"{system} requires phys.re1 and phys.re2")
    if re1 < 0:
        raise ValueError(f"{system}: re1 must be >= 0 (sign convention)")
    if re1 > 0:
        re_ref = re1  # Case 1: inner-driven
    elif re2 > 0:
        re_ref = re2  # Case 2: outer-driven (re1 == 0)
    else:
        raise ValueError(
            f"{system} needs re1 > 0, or re1 == 0 and re2 > 0 "
            f"(got re1={re1}, re2={re2})"
        )
    params.phys.re = re_ref
    #   A0 = (re2 - eta re1) / [(1+eta) Re_ref]
    #   B0 = eta (re1 - eta re2) / [(1+eta)(1-eta)^2 Re_ref]
    derived.ccf_A = (re2 - eta * re1) / ((1 + eta) * re_ref)
    derived.ccf_B = (
        eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
    )


def wedge_rehydrate(sections: dict) -> None:
    """Fill the internal azimuthal extent ``geo.lz = 2*pi/m0``."""
    geo = sections.setdefault("geo", {})
    geo["lz"] = 2 * pi / geo.get("m0", 1)


def circular_couette_rehydrate(sections: dict) -> None:
    """Fill the derived reference ``phys.re`` from ``(re1, re2)``."""
    phys = sections.setdefault("phys", {})
    re1, re2 = phys.get("re1"), phys.get("re2")
    if re1 is not None and re1 > 0:
        phys["re"] = re1
    elif re2 is not None:
        phys["re"] = re2
