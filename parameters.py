from jax import numpy as jnp

# TODO: Read these from a text file

# Geometry and discretization
NX = 64
NY = 12
NZ = 48

# Domain lengths
LX = 4.0
LZ = 4.0
LY = 4.0  # fixed by non-dimensionalization

# Physics
RE = 630.0  # Reynolds number
FORCING = 1  # none = 0, sine = 1, cosine = 2

# Initiation
IC = 0
I_START = 0
T_START = 0

# Outputs
I_PRINT_STATS = 20
I_PRINT_STEPS = 20
I_SAVE_FIELDS = 2000

# Time stepping
DT = 0.025
IMPLICITNESS = 0.51
STEPTOL = 1.0e-5
DTMAX = 0.1
NCORR = 10

# Termination
WALL_CLOCK_LIMIT = -1.0
I_FINISH = -1

# Physics
AMP = jnp.pi**2 / (4 * RE)

IC_F = 0  # Forced component
QF = 1  # Forcing harmonic
KF = 2 * jnp.pi * QF / LY

SUBSAMP_FAC = 2  # SUBSAMP_FAC / 2 dealiasing
