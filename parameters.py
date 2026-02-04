from jax import numpy as jnp

# Geometry and discretization
NX = 64
NY = 12
NZ = 48

SUBSAMP_FAC = 2  # SUBSAMP_FAC / 2 dealiasing
# TODO: Make sure it works with =3

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
IMPLICITNESS = 0.5
STEPTOL = 1.0e-9
DTMAX = 0.1
NCORR = 10

# Termination
WALL_CLOCK_LIMIT = -1.0
I_FINISH = -1

# Physics
AMP = jnp.pi**2

QF = 1  # Forcing harmonic
ICF = 0  # Forced component
KF = 2 * jnp.pi * QF / LY

EKIN_LAM = 1 / 4

# others
INVTDT = 1 / DT

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
ISYM = jnp.array([0, 0, 0, 1, 1, 2], dtype=int)
JSYM = jnp.array([0, 1, 2, 1, 2, 2], dtype=int)
NSYM = jnp.zeros((3, 3), dtype=int)

for n in range(6):
    i = ISYM[n]
    j = JSYM[n]
    NSYM = NSYM.at[i, j].set(n)
    NSYM = NSYM.at[j, i].set(n)

# Further definitions
NX_HALF = NX // 2
NY_HALF = NY // 2
NZ_HALF = NZ // 2

NXX = SUBSAMP_FAC * NX_HALF
NYY = SUBSAMP_FAC * NY_HALF
NZZ = SUBSAMP_FAC * NZ_HALF

NYY_HALF_PAD1 = NYY // 2 + 1
NY_HALF_PAD1 = NY // 2 + 1

DX = LX / NXX
DY = LY / NYY
DZ = LZ / NZZ
