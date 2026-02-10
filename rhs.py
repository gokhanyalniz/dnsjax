import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import FORCING, IC_F, RE
from sharding import MESH
from transform import FORCE, INV_LAPL, KVEC, LAPL, phys_to_spec_vector

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
ISYM = jnp.array([0, 0, 0, 1, 1, 2], dtype=int)
JSYM = jnp.array([0, 1, 2, 1, 2, 2], dtype=int)
NSYM = jnp.zeros((3, 3), dtype=int)

for n in range(6):
    i = ISYM[n]
    j = JSYM[n]
    NSYM = NSYM.at[i, j].set(n)
    NSYM = NSYM.at[j, i].set(n)


@jit
def get_nonlin(velocity_phys):

    # No Basdevant version:
    # return jax.lax.with_sharding_constraint(
    #     phys_to_spec_vector(
    #         jnp.stack(
    #             [velocity_phys[ISYM[n]] * velocity_phys[JSYM[n]] for n in range(6)]
    #         )
    #     ),
    #     NamedSharding(MESH, P(None, "Z", "X", None)),
    # )

    # Basdevant version:
    uu = jnp.stack([velocity_phys[ISYM[n]] * velocity_phys[JSYM[n]] for n in range(6)])

    trace = jnp.sum(uu[tuple(NSYM[i, i] for i in range(3)),], axis=0)

    # No need to update (2,2), it's not used
    uu = uu.at[(NSYM[0, 0], NSYM[1, 1]),].subtract(trace / 3)

    nonlin = phys_to_spec_vector(uu[:5])
    nonlin = jnp.concatenate(
        (nonlin, -(nonlin[NSYM[0, 0]] + nonlin[None, NSYM[1, 1]])), axis=0
    )

    return jax.lax.with_sharding_constraint(
        nonlin, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit
def get_rhs_no_lapl(velocity_phys):

    nonlin = get_nonlin(velocity_phys)

    advect = jnp.stack(
        [
            -jnp.sum(1j * KVEC * nonlin[tuple(NSYM[n, m] for n in range(3)),], axis=0)
            for m in range(3)
        ]
    )
    div = jnp.sum(KVEC * advect, axis=0)

    rhs_no_lapl = advect + div * INV_LAPL * KVEC
    if FORCING != 0:
        rhs_no_lapl = rhs_no_lapl.at[IC_F].add(FORCE)

    return jax.lax.with_sharding_constraint(
        rhs_no_lapl, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit
def get_rhs(velocity_spec, velocity_phys):
    rhs = get_rhs_no_lapl(velocity_phys) + (LAPL / RE) * velocity_spec

    return jax.lax.with_sharding_constraint(
        rhs, NamedSharding(MESH, P(None, "Z", "X", None))
    )
