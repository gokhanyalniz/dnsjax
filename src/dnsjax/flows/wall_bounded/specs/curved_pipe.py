r"""Parameter spec for the curved (toroidal) pipe.

Lengths are in pipe radii and the velocity unit is the straight pipe's
laminar centreline velocity, so ``phys.re`` is the same number
``pipe`` carries and `$\kappa \to 0$` reproduces it exactly.  Under
``phys.driving = "constant_bulk_velocity"`` the mass flux is held at
that laminar value, which makes ``phys.re`` the literature Reynolds
number `$\mathrm{Re} = U_b d/\nu$` of the toroidal-pipe work, with
Dean number `$\mathrm{De} = \mathrm{Re}\sqrt{\kappa}$`; under the
default constant pressure gradient it is instead the *nominal*
Reynolds number of the same driving, and the realised bulk is lower by
the Dean friction increase.

Defaults: `$\kappa = 0.037$` (`$R_c/a = 27.03$`, `$R_c/D = 13.51$`),
`$\mathrm{Re} = 5480$` (`$\mathrm{De} = 1054$` here) and a streamwise
box of one measured travelling-wave wavelength -- 19 degrees of
toroidal angle at `$R_c/a = 18.2$`, i.e. 6.04 radii.  They are a
starting point, not a validated configuration: that Reynolds number
was reported at `$\kappa = 0.055$`, and with a midplane reflection
symmetry imposed, which this solver does not impose.  The complete
torus is `$L_s = 2\pi/\kappa \approx 169.8$`.
"""

from math import pi

from ....flow_spec import DeferredSpec, FieldSpec, FlowSpec
from ._family import (
    CYLINDRICAL_GRIDS,
    DEFERRED_MEAN_FLOW,
    DEFERRED_TILT,
    PIPE_CARRY_FIELDS,
    cyl_annular_fields,
    wall_fields,
)

#: One measured travelling-wave wavelength in pipe radii: 19 degrees
#: of toroidal angle at `$R_c/a = 18.2$` (Webster & Humphrey 1997,
#: Table I, Re = 5480).
WH_BOX: float = 6.04

#: The azimuthal wedge is unavailable: `$h = 1 + \kappa r\cos\theta$`
#: breaks the discrete `$\theta \to \theta + 2\pi/m_0$` symmetry the
#: wedge assumes, and the metric couples `$m \to m \pm 1$`, which is
#: not a sublattice of `$m_0\mathbb{Z}$` for `$m_0 > 1$`.
_ALIAS_FIELDS = tuple(
    f
    for f in cyl_annular_fields(axial_default=WH_BOX)
    if f.key != ("geo", "m0")
)

#: Both opt-in corrector variants are driven by ``_l_bf``, which the
#: curved pipe cannot form without an FFT-free `$1/h$` (see
#: ``_validate``), so neither is offered and ``_l_bf`` is never called.
_DEFERRED_STEP = {
    ("step", "split_corrector"),
    ("step", "implicit_mean_coupling"),
}
_WALL_FIELDS = tuple(
    f
    for f in wall_fields(0.5, CYLINDRICAL_GRIDS)
    if f.key not in _DEFERRED_STEP
)

DEFERRED_SPLIT = DeferredSpec(
    "step",
    "split_corrector",
    "step.split_corrector is not implemented yet for curved-pipe: it "
    "iterates the FFT-free linear coupling _l_bf, which this flow "
    "cannot form without dividing by the metric in spectral space.",
)
DEFERRED_MEAN_COUPLING = DeferredSpec(
    "step",
    "implicit_mean_coupling",
    "step.implicit_mean_coupling is not implemented yet for "
    "curved-pipe: it is read only through _l_bf, which this flow does "
    "not form (see step.split_corrector).",
)


def _derive(params, derived, user_set) -> None:
    # Full circle: the wedge is unavailable (``_ALIAS_FIELDS``).
    params.geo.lz = 2 * pi
    derived.volume_fac = 0.5  # int_0^1 r dr


def _validate(params, derived) -> None:
    if params.step.scheme != "iterative-cn":
        raise ValueError(
            "curved-pipe requires step.scheme='iterative-cn' (got "
            f"{params.step.scheme!r}): the curvature terms in the "
            "continuity rows, like the pipe's two free wall values, "
            "must be read on the corrector iterate, and cnab2's "
            "FFT-free corrector has nothing to iterate here -- with "
            "no base flow its coupling term is identically zero, so "
            "both would be lagged across the time step (the "
            "instability recorded in _cylindrical_stepping._imm_iteration_vw)."
        )
    if not params.res.consistent_imm:
        raise ValueError(
            "curved-pipe requires res.consistent_imm=True: the legacy "
            "primitive (v, p) influence-matrix path carries no "
            "curvature terms, and would step the straight-pipe "
            "equations without saying so."
        )


def _rehydrate(sections: dict) -> None:
    sections.setdefault("geo", {})["lz"] = 2 * pi


SPEC = FlowSpec(
    system="curved-pipe",
    family="cylindrical",
    geometry_label="cylindrical",
    summary="pressure/bulk-driven toroidal (curved) pipe",
    flow_module="dnsjax.flows.wall_bounded.curved_pipe",
    fields=(
        *_WALL_FIELDS,
        *_ALIAS_FIELDS,
        *PIPE_CARRY_FIELDS,
        FieldSpec(
            "geo",
            "curvature",
            description=(
                "Dimensionless centreline curvature kappa = a/R_c "
                "(0 is the straight pipe)."
            ),
            default=0.037,
        ),
        FieldSpec("phys", "re", default=5480.0),
        FieldSpec("phys", "driving"),
    ),
    deferred=(
        DEFERRED_TILT,
        DEFERRED_MEAN_FLOW,
        DEFERRED_SPLIT,
        DEFERRED_MEAN_COUPLING,
    ),
    grid_type_default="half-cgl",
    derive=_derive,
    validate=_validate,
    rehydrate=_rehydrate,
)
