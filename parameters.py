# Number of grid points = (before oversampling) # of Fourier modes
NX = 48
NY = 48
NZ = 24

# Domain lengths
LX = 4.0
LZ = 2.0
# LY is fixed by non-dimensionalization,
# but still kept here for future generality.
# Do not change for Kolmogorov/Waleffe flow!
LY = 4.0

# Physics
RE = 628.3185307179584  # Reynolds number
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
DT = 0.02
IMPLICITNESS = 0.5
STEPTOL = 1.0e-5
DTMAX = 0.1
NCORR = 10

# Termination
WALL_CLOCK_LIMIT = -1.0
T_STOP = -1

# Debugging
TIME_FUNCTIONS = False

# OVERSAMP_FAC / 2 dealiasing
# (n + 1) / 2 dealiasing for n'th order nonlinearity
OVERSAMP_FAC = 3
