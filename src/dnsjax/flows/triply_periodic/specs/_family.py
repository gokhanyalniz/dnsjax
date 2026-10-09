r"""Shared spec fragments for the triply-periodic flows.

One surface for the family's flows (kolmogorov): the periodic box
lengths and tilt, the identity-named resolution and the Reynolds
number.  The moving frame, the mean-mode perturbation and the reduced
snapshots' pressure are deferred here; the wall-bounded-only fields
(grids, probes, forcing, ...) are not part of the surface.
"""

from ....flow_spec import DeferredSpec, FieldSpec


def periodic_fields() -> tuple[FieldSpec, ...]:
    """The relevant fields of a triply-periodic flow's surface."""
    return (
        FieldSpec("geo", "lx"),
        FieldSpec("geo", "lz"),
        FieldSpec("geo", "tilt_degree"),
        FieldSpec("res", "nx"),
        FieldSpec(
            "res",
            "ny",
            description=(
                "Shear-direction Fourier modes (= physical grid "
                "points before dealiasing)."
            ),
        ),
        FieldSpec("res", "nz"),
        FieldSpec("phys", "re"),
        FieldSpec("lowres", "nx"),
        FieldSpec(
            "lowres",
            "ny",
            description=(
                "Shear-direction Fourier modes the reduced snapshots "
                "keep (the highest dropped); unset = res.ny."
            ),
        ),
        FieldSpec("lowres", "nz"),
    )


def periodic_deferred() -> tuple[DeferredSpec, ...]:
    """The fields a triply-periodic flow declares but refuses."""
    return (
        DeferredSpec(
            "phys",
            "u_grid",
            "phys.u_grid (moving frame of reference) is not "
            "implemented yet for the triply-periodic systems.",
        ),
        DeferredSpec(
            "init",
            "random_mean_flow",
            "init.random_mean_flow (perturbing the kx = kz = 0 mean "
            "profile) is not implemented yet for the triply-periodic "
            "systems; their mean mode is a passive Galilean shift the "
            "solver re-zeroes every step anyway.",
        ),
        DeferredSpec(
            "lowres",
            "pressure",
            "lowres.pressure (the static pressure in reduced-resolution "
            "snapshots) is not implemented yet for the triply-periodic "
            "systems: plane Couette and plane Poiseuille have it, and this "
            "flow writes the velocity alone.",
        ),
    )


def periodic_derive(params, derived, user_set) -> None:
    """Set the volume factor to 1: a mode sum is already an average."""
    derived.volume_fac = 1
