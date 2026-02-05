import jax
from jax.sharding import AxisType

# Parallelization
# Use [1, N] for slab decomposition
PDIMS = [2, 2]  # TODO: read manually later

RANK = jax.process_index()

# TODO: With cuDecomp, transposed mesh is required
MESH = jax.make_mesh(
    PDIMS, axis_names=("Z", "X"), axis_types=(AxisType.Auto, AxisType.Auto)
)
# TODO: jaxdecomp.autotune for optimal meshing
# - Check if I can choose the axes to shard

# X shape: (ny, nz, nx)
# Y shape: (nz, nx, ny)

# dnsbox
# vfieldk(nx_perproc, ny_half, nz, 3)
# vfieldxx(nyy, nzz_perproc, nxx, 3)
