r"""Shared spec fragments for the triply-periodic flows.

One surface for the family (currently kolmogorov alone): the periodic
box lengths/tilt, the identity-named resolution, and the Reynolds
number.  The moving frame and the localized-rolls IC are deferred
features here; the wall-bounded-only fields (grids, probes,
forcing, ...) are simply not part of the surface.
"""

from ....flow_spec import DeferredSpec, FieldSpec


def periodic_fields() -> tuple[FieldSpec, ...]:
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
        # The periodic random-IC generator honours the mean-mode
        # perturbation too (the kx = kz = 0 shear profile).
        FieldSpec("init", "random_mean_flow"),
    )


def periodic_deferred() -> tuple[DeferredSpec, ...]:
    rolls_msg = (
        "init.localized_rolls (localized-spot IC) is not implemented "
        "yet for the triply-periodic systems."
    )
    return (
        DeferredSpec(
            "phys",
            "u_grid",
            "phys.u_grid (moving frame of reference) is not "
            "implemented yet for the triply-periodic systems.",
        ),
        DeferredSpec("init", "localized_rolls", rolls_msg),
        DeferredSpec("init", "localized_rolls_amplitude", rolls_msg),
        DeferredSpec("init", "localized_rolls_width", rolls_msg),
        DeferredSpec("init", "localized_rolls_wavelength", rolls_msg),
    )


def periodic_derive(params, derived, user_set) -> None:
    """Periodic box: the ky sum already comes as a density."""
    derived.volume_fac = 1
