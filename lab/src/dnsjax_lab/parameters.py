r"""The lab flows' fields in the core parameter sections (JAX-free).

``FIELDS`` is the object of the ``dnsjax.parameters`` entry point in
``lab/pyproject.toml``: :mod:`dnsjax.parameters` adds these fields to
its section models, after the core's own, and a lab flow reaches them
through its spec's ``FieldSpec`` entries, exactly as a core field.  The
fields are declared here as they would be in the core models, one
namespace per section, and collected by :func:`_fields`.  This module
imports neither JAX nor :mod:`dnsjax.parameters`.
"""

import inspect

from pydantic import Field
from pydantic.fields import FieldInfo


class _Physics:
    """``[phys]``: the quasi-Keplerian, sPTT and curved-pipe controls."""

    # Rotation number R_Omega (system == "quasi-keplerian"), following
    # Dubrulle et al. 2005:
    #   R_Omega = (1 - eta) (Re_i + Re_o) / (eta Re_o - Re_i),
    # with Re_i = re1 and Re_o the derived outer Reynolds number.  It is
    # constant along half-lines through the origin of (Re_o, Re_i) space.
    # The quasi-Keplerian regime (co-rotating, angular momentum
    # increasing / angular velocity decreasing outward -- linearly stable
    # by Rayleigh's criterion) is the open half-line -inf < R_Omega < -1,
    # bounded by the Rayleigh line R_Omega = -1 (Re_o = eta Re_i) and the
    # solid-body limit R_Omega -> -inf (Re_o = Re_i / eta).  The
    # quasi-Keplerian spec's ``derive`` requires re1 > 0 and r_omega < -1,
    # then derives re2 = re1 (1 - eta + R_Omega) / (eta R_Omega -
    # (1 - eta)) and stores it, so every downstream circular-Couette
    # path is shared with Taylor-Couette (``re1`` is the inner Reynolds
    # number Re_i).
    r_omega: float | None = Field(
        default=None,
        description=(
            "Rotation number R_Omega < -1 selecting a quasi-Keplerian "
            "regime; the outer Reynolds number is derived from "
            "(re1, r_omega, eta)."
        ),
    )
    # Viscoelastic (sPTT) control parameters, shared by both
    # viscoelastic systems ("viscoelastic-pipe" / "viscoelastic-dean";
    # see ``flows.wall_bounded.viscoelastic_{pipe,dean}`` and
    # ``geometries.wall_bounded.{cylindrical,annular}_viscoelastic``).
    # All ``None`` for other systems; unset values fall back to each
    # flow's own FieldSpec defaults.  beta 0.8 / epsilon 0.001 /
    # kappa 5e-5 are shared -- the values of Lellep, Linkmann &
    # Morozov, PNAS 121, e2318851121 (2024), whose sPTT system both
    # flows integrate (``_viscoelastic_common``) -- but el and wi are
    # not: Re := Wi/El is derived, so those two *are* the Reynolds
    # number, and each flow picks the regime it is about -- el 80,
    # wi 105 (Re ~ 1.3, inertialess and strongly elastic) for the
    # annulus; el 0.02, wi 20 (Re = 1000, elasto-inertial) for the
    # pipe.
    el: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Elasticity number El; the Reynolds number is derived as "
            "Re = Wi / El."
        ),
    )
    wi: float | None = Field(
        default=None, gt=0, description="Weissenberg number Wi."
    )
    beta: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Solvent-to-total viscosity ratio in (0, 1]: the solvent "
            "carries nu = beta/Re, the polymer stress (1-beta)/(Re Wi)."
        ),
    )
    # Deliberately unbounded above: the relaxation factor
    # f = 1 + epsilon (tr c - 3) is positive for every admissible c only
    # while epsilon <= 1/3 (above it, f flips sign once tr c < 3 - 1/eps,
    # turning relaxation into growth), but the literature range extends
    # past that and no realizable state reaches the threshold at the
    # shipped defaults.  The cnab2 split stays exact either way -- its
    # implicit half carries (1 - 3 epsilon)/Wi, which merely turns
    # anti-dissipative there and costs corrector contraction, not
    # correctness.
    epsilon: float | None = Field(
        default=None,
        ge=0,
        description=(
            "sPTT extensibility parameter (>= 0); f = 1 + epsilon "
            "(tr c - 3) is sign-definite only up to 1/3."
        ),
    )
    kappa: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Artificial conformation stress diffusivity; 0 makes the "
            "conformation transport purely hyperbolic (no wall BC on "
            "c)."
        ),
    )
    # Off in the model, on in the curved-pipe spec: a snapshot written
    # before the field existed ran without it, which is what a resume's
    # trajectory comparison reads for an absent key.
    curvature_padding: bool = Field(
        default=False,
        description=(
            "Curved pipe: raise the azimuthal dealiasing grid from the "
            "3/2 rule's 3M + 3 points to 3M + J(kappa), M the largest "
            "stored azimuthal wavenumber.  The nonlinear term carries "
            "1/h^2, h = 1 + kappa r cos(theta), whose harmonics never "
            "end, so no oversampling factor dealiases it exactly; J is "
            "the smallest pad that bounds the aliasing below the "
            "working precision's roundoff (see "
            "dnsjax_lab.harmonics.curvature_dealiasing_pad).  "
            "Trajectory-defining; needs oversampling_factor >= 3."
        ),
    )


class _Geometry:
    """``[geo]``: the curved pipe's curvature, the sPTT annulus radius."""

    curvature: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Dimensionless centreline curvature kappa = a/R_c of a "
            "curved (toroidal, zero-torsion) pipe; 0 is the straight "
            "pipe.  The complete torus is a streamwise period of "
            "2*pi/kappa, but any period is geometrically consistent "
            "because no metric coefficient depends on the streamwise "
            "coordinate."
        ),
    )
    # Default 11 (applied in the viscoelastic ``derive`` hook).
    delta: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Inner radius of the viscoelastic annulus in half-gap "
            "units (gap fixed at 2: r1 = delta, r2 = delta + 2)."
        ),
    )


class _Initiation:
    """``[init]``: the sPTT random IC's tensor amplitude."""

    # Radially windowed to zero at both walls, which keep their laminar
    # conformation; shares the velocity draw's ``random_smoothness`` /
    # ``random_wall_smoothness`` / ``random_wall_confinement`` shaping.
    random_conformation_amplitude: float = Field(
        default=700.0,
        description=(
            "Amplitude of the random symmetric-tensor perturbation "
            "added to the laminar conformation in the viscoelastic "
            "random IC."
        ),
    )


def _fields(namespace: type) -> dict[str, tuple[object, FieldInfo]]:
    """``{name: (annotation, FieldInfo)}`` of *namespace*, in order."""
    annotations = inspect.get_annotations(namespace, eval_str=True)
    return {
        name: (annotation, getattr(namespace, name))
        for name, annotation in annotations.items()
    }


FIELDS: dict[str, dict[str, tuple[object, FieldInfo]]] = {
    "phys": _fields(_Physics),
    "geo": _fields(_Geometry),
    "init": _fields(_Initiation),
}
