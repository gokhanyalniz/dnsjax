import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Parallelization
# Use [1, N] for slab decomposition
PDIMS = [2, 2]  # TODO: read manually later

# TODO: Check if I can choose the axes to shard
# - jaxdecomp.autotune for optimal meshing

# jax.config.update("jax_enable_x64", True)
# jax.distributed.initialize()
RANK = jax.process_index()

MESH = jax.make_mesh(PDIMS, axis_names=("Z", "X"))
SHARDING = NamedSharding(MESH, P("Z", "X"))

# X shape: (ny, nz, nx)
# Y shape: (nz, nx, ny)

# dnsbox
# vfieldk(nx_perproc, ny_half, nz, 3)
# vfieldxx(nyy, nzz_perproc, nxx, 3)
