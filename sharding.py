import jax
from jax.sharding import AxisType

jax.config.update("jax_enable_x64", True)  # use 64 bit floating point
jax.config.update("jax_platforms", "cpu")  # stick to CPUs for now
jax.distributed.initialize()

# Parallelization
# Use [1, N] for slab decomposition
PDIMS = [2, 4]
NP = PDIMS[0] * PDIMS[1]
if len(jax.devices()) != NP:
    jax.distributed.shutdown()
    exit("# of devices not equal to NP.")

RANK = jax.process_index()

# With cuDecomp, a transposed mesh will be required
MESH = jax.make_mesh(
    PDIMS, axis_names=("Z", "X"), axis_types=(AxisType.Auto, AxisType.Auto)
)
