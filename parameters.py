import numpy as np
import jaxdecomp

# Geometry and discretization
NX = 64
NY = 48
NZ = 12

SUBSAMP_FAC = 2 # SUBSAMP_FAC / 2 dealiasing

# Domain lengths
LX = 4.0
LZ = 4.0
LY = 4.0 # fixed by non-dimensionalization

# Physics
RE = 630.0 # Reynolds number
FORCING = 1 # none = 0, sine = 1, cosine = 2
QF = 1 # Forcing wave number

# Initiation
IC = 0
I_START = 0
T_START = 0

# Outputs
I_PRINT_STATS = 20
I_PRINT_STEPS = 20
I_SAVE_FIELDS = 2000

# Time stepping
dt = 0.025
IMPLICITNESS = 0.5
STEPTOL = 1.0e-9
DTMAX = 0.1
NCORR = 10

# Termination
WALL_CLOCK_LIMIT = -1.0
I_FINISH = -1

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
ISYM = np.array([1, 1, 1, 2, 2, 3],dtype=int)
JSYM = np.array([1, 2, 3, 2, 3, 3],dtype=int)
NSYM = np.zeros((6,6),dtype=int)

for n in range(6):
    i = ISYM[n]
    j = JSYM[n]
    NSYM[i,j] = n
    NSYM[j,i] = n

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

# Parallelization
PDIMS = [2, 2] # to be read manually later

GLOBAL_SHAPE = (NXX, NZZ, NYY)
LOCAL_SHAPE = (NXX // PDIMS[0], NZZ // PDIMS[1], NYY)


# Physics
AMP = np.pi**2

KF = 2 * np.pi * QF / LY

EKIN_LAM = 1/4
