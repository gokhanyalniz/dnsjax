# Kolmogorov flow — a laptop-sized transition to chaos

A sinusoidal body force drives a triply-periodic box, and a deterministic
localized roll seeded on the shear maximum grows by orders of magnitude
before saturating into a chaotic state.

Triply-periodic flow has no walls, so there is no wall-normal matrix
solve here at all: the implicit step is diagonal in spectral space, and
there is no influence matrix and no banded factorization to go wrong.
That makes this the *simplest* of the four examples — everything that
happens is the $\tfrac{3}{2}$-dealiased rotational nonlinear term and
the corrector — and the one to reach for when the question is whether
the core machinery works rather than whether a geometry does.

```bash
mkdir -p /tmp/kolmogorov && cd /tmp/kolmogorov
cp /path/to/dnsjax/examples/kolmogorov/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

Any value can be overridden on the command line (`--phys.re 60`), and
`--dist.platform cuda` runs it on a GPU; what the four examples share,
from the scratch directory and the GPU set-up to the laminarization
stop and their sizes, is in [`examples/README.md`](../README.md).

**What to watch — and here it is `E'`.** `stats.dat` gets one row per
100 steps. The perturbation energy `E'` climbs by orders of magnitude,
overshoots, and then fluctuates about a plateau. That fluctuation is the
point: a chaotic state keeps moving, where a steady secondary flow would
settle.

The wall-bounded examples say to watch the dissipation instead, because
`E'` there carries the mean profile's drift. This flow is the opposite
case: the laminar state is a **sinusoidal** shear profile, sheared
everywhere and therefore strongly dissipative, so breaking it up
*lowers* the dissipation rather than raising it. `D` falls across the
transition here, and the total energy `E` with it. Watch which way a
quantity moves before trusting its direction.

**Resolution.** $32^3$ keeps the example on a laptop core. To check it
for your purpose, rerun with `--res.nx 48 --res.ny 48 --res.nz 48` and
compare the saturated statistics.

**It may stop early.** `stop.check_laminarization`, on by default, ends
a run whose perturbation energy collapses, as it does at a low enough
`re`.
