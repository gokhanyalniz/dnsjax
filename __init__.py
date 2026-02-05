import os

import jax

# use strictly the cpu (for now)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
# use 64 bit floating point
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
# TODO: Check if I can choose the axes to shard
# - jaxdecomp.autotune for optimal meshing
