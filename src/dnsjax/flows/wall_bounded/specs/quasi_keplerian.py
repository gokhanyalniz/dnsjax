r"""Parameter spec for quasi-Keplerian (annular) flow."""

from ....flow_spec import FieldSpec, FlowSpec
from ._family import (
    CARTESIAN_ANNULAR_GRIDS,
    DEFERRED_TILT,
    annular_base_derive,
    circular_couette_derive,
    circular_couette_rehydrate,
    cyl_annular_fields,
    wall_fields,
    wedge_rehydrate,
)


def _derive_re2(re1: float, r_omega: float, eta: float) -> float:
    # Invert R_Omega = (1-eta)(Re_i+Re_o)/(eta Re_o - Re_i) for Re_o
    # (the denominator eta r_omega - (1-eta) < 0 on the half-line, so
    # this is finite and Re_o > 0 throughout).
    return re1 * (1 - eta + r_omega) / (eta * r_omega - (1 - eta))


def _derive(params, derived, user_set) -> None:
    eta = annular_base_derive(params, derived)
    re1, r_omega = params.phys.re1, params.phys.r_omega
    if re1 is None or re1 <= 0:
        raise ValueError("quasi-keplerian requires phys.re1 > 0 (= Re_i)")
    if r_omega is None:
        raise ValueError(
            "quasi-keplerian requires phys.r_omega (rotation number R_Omega)"
        )
    if r_omega >= -1:
        raise ValueError(
            "quasi-keplerian requires R_Omega < -1 (the open half-line "
            "-inf < R_Omega < -1 between the Rayleigh line "
            "R_Omega = -1 and the solid-body limit R_Omega -> -inf); "
            f"got r_omega={r_omega}"
        )
    # ``re2`` is derived, not a quasi-Keplerian parameter: any directly
    # assigned value is simply overwritten (the derived value also
    # replays through resumed-snapshot layers).
    params.phys.re2 = _derive_re2(re1, r_omega, eta)
    circular_couette_derive(params, derived, eta)


def _validate(params, derived) -> None:
    # Startup summary: derived Re_o, shear Reynolds number Re_s,
    # rotation ratio mu = Omega_o/Omega_i, and the local exponent
    # q(r) = -d ln Omega / d ln r = 2 B0 / (A0 r^2 + B0) at the walls
    # (q in (0, 2) on the quasi-Keplerian half-line, bracketing the
    # Keplerian value 3/2).
    eta = params.geo.eta
    re_i, re_o = params.phys.re1, params.phys.re2
    re_s = 2 * abs(eta * re_o - re_i) / (1 + eta)
    mu = eta * re_o / re_i
    A0, B0 = derived.ccf_A, derived.ccf_B
    r1, r2 = derived.r_inner, derived.r_outer
    q1 = 2 * B0 / (A0 * r1**2 + B0)
    q2 = 2 * B0 / (A0 * r2**2 + B0)
    print(
        f"[quasi-keplerian] derived: Re_o={re_o:.6g} "
        f"Re_s={re_s:.6g} mu={mu:.6g} "
        f"q(r) in [{q2:.4g}, {q1:.4g}]"
    )


def _rehydrate(sections: dict) -> None:
    wedge_rehydrate(sections)
    phys = sections.setdefault("phys", {})
    re1 = phys.get("re1")
    r_omega = phys.get("r_omega")
    eta = sections.get("geo", {}).get("eta")
    if re1 is not None and r_omega is not None and eta is not None:
        phys["re2"] = _derive_re2(re1, r_omega, eta)
    circular_couette_rehydrate(sections)


SPEC = FlowSpec(
    system="quasi-keplerian",
    family="annular",
    geometry_label="annular",
    summary="co-rotating Rayleigh-stable annular flow",
    flow_module="dnsjax.flows.wall_bounded.quasi_keplerian",
    fields=(
        *wall_fields(0.0, CARTESIAN_ANNULAR_GRIDS),
        *cyl_annular_fields(),
        FieldSpec("geo", "eta"),
        FieldSpec("phys", "re1"),
        FieldSpec("phys", "r_omega"),
        FieldSpec("phys", "block_mean_spanwise_velocity"),
    ),
    deferred=(DEFERRED_TILT,),
    grid_type_default="cgl",
    derive=_derive,
    validate=_validate,
    rehydrate=_rehydrate,
)
