# Kolmogorov flow — a laptop-sized transition to chaos

A sinusoidal body force drives a triply-periodic box, and a deterministic
localized roll seeded on the shear maximum grows through nearly three
decades before saturating into a chaotic state.

Triply-periodic flow has no walls, so there is no wall-normal matrix
solve here at all: the implicit step is diagonal in spectral space, and
there is no influence matrix and no banded factorisation to go wrong.
That makes this the *simplest* of the four examples — everything that
happens is the $\tfrac{3}{2}$-dealiased rotational nonlinear term and
the corrector — and the one to reach for when the question is whether
the core machinery works rather than whether a geometry does.

```bash
mkdir -p /tmp/kolmogorov && cd /tmp/kolmogorov
cp /path/to/dnsjax/examples/kolmogorov/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

A run reads `parameters.toml` from its own working directory, so copy it
into a scratch directory and launch there. Any value can be overridden on
the command line — `--phys.re 60`, `--res.nx 48 --res.ny 48 --res.nz 48`.

**What to watch — and here it really is `E'`.** `stats.dat` gets one
row per 100 steps, with a `#`-commented header, so `numpy.loadtxt` reads
it directly. The perturbation energy `E'` climbs by a **factor of about
770** through $t \approx 28$, overshoots, and then fluctuates at 10–20 %
about a plateau. That fluctuation is the point: it is what distinguishes
a chaotic state from the steady secondary flows just below it.

The wall-bounded examples say to watch the dissipation instead, because
`E'` there carries the mean profile's drift. This flow is the opposite
case, and instructively so: the laminar state is a **sinusoidal** shear
profile, sheared everywhere and therefore strongly dissipative, so
breaking it up *lowers* the dissipation rather than raising it. `D` falls
by about a factor of four across the transition here, and the total
energy `E` with it. Watch which way a quantity moves before trusting its
direction.

**Resolution.** $32^3$ is converged for the saturated statistics: the
same case at $48^3$ agrees on the mean perturbation energy to 0.1 % and
on the mean dissipation to 0.26 %, with an identical peak at an
identical time, for twelve times the cost.

**It may stop early.** `stop.check_laminarization` is on by default, so
a run whose perturbation energy collapses ends cleanly and says so rather
than integrating a laminar flow to the horizon. At this Reynolds number
and box that does not happen — the state is still chaotic at the horizon
— but a lower `re` will do it.

About three minutes on one core of an AMD Ryzen 7 PRO 7840U laptop CPU.
All four examples' measured times are collected in
[`examples/README.md`](../README.md).
