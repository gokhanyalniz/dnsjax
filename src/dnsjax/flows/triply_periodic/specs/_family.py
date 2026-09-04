r"""Shared spec fragments for the triply-periodic flows.

One surface for the family (currently kolmogorov alone): the periodic
box lengths/tilt, the identity-named resolution, the Reynolds number
and the localized-rolls IC.  The moving frame and the mean-mode
perturbation are deferred features here; the wall-bounded-only fields
(grids, probes, forcing, ...) are simply not part of the surface.
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
        # The localized spot needs no wall-normal grid: its `y` factor
        # is the same fixed-physical localization the homogeneous
        # directions take (``ic/localized_rolls.py``).
        FieldSpec("init", "localized_rolls"),
        FieldSpec("init", "localized_rolls_amplitude"),
        FieldSpec("init", "localized_rolls_width"),
        FieldSpec("init", "localized_rolls_wavelength"),
    )


def periodic_deferred() -> tuple[DeferredSpec, ...]:
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
    )


def periodic_derive(params, derived, user_set) -> None:
    """Periodic box: the ky sum already comes as a density."""
    derived.volume_fac = 1
