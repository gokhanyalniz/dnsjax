# Plane-Poiseuille flow in a small channel box

A channel in a small periodic box, $\pi$ long and $\pi/2$ wide in
half-height units. Even at the laminar friction Reynolds number,
$Re_\tau = \sqrt{2Re} = 100$, that is about $314 \times 157$ wall units,
and turbulence only raises $Re_\tau$: small, but larger than a minimal
channel. Channel flow is **linearly stable** at $Re = 5000$: the
critical Reynolds number for the laminar parabola is 5772 in this
normalization (Orszag 1971), so nothing here grows from an infinitesimal
disturbance. The transition is subcritical, driven entirely by the
finite-amplitude spot the initial condition puts in.

```bash
mkdir -p /tmp/plane-poiseuille && cd /tmp/plane-poiseuille
cp /path/to/dnsjax/examples/plane-poiseuille-re5000/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

A run reads `parameters.toml` from its own working directory, so copy it
into a scratch directory and launch there. Any value can be overridden on
the command line — `--phys.driving constant_pressure_gradient`,
`--init.localized_rolls_amplitude 0.5`.

**What to watch.** `stats.dat` gets one row per 50 steps, with a
`#`-commented header that `numpy.loadtxt` reads directly. Follow the
**dissipation `D`** against its laminar value, $4/(3Re)$: it rises
steeply as the spot breaks down, then settles onto a turbulent level
several times laminar.

**The forcing moves, not the bulk.** `phys.driving` is set to
`constant_bulk_velocity` here, so the flow rate is held fixed and the
streamwise pressure gradient is whatever it takes to maintain it. That
is the right way round for an example: the Reynolds number keeps meaning
what it says once the flow goes turbulent, and the applied forcing
appears as a **last column in `stats.dat`** (`-dPds'`, positive when
accelerating) which rises above its laminar value exactly when the flow
does — a second, independent reading of the same event as the
dissipation.

Under the default `constant_pressure_gradient` the roles swap: the
forcing is fixed and a turbulent channel, being far more dissipative,
settles at a *lower* flow rate than the laminar one. That is also why
`E'` is a poor turbulence indicator in general; it carries the mean
profile's deviation as well as the fluctuations.

**It may relaminarize.** Turbulence in a box this small need not last
forever: a longer run, or another perturbation, may decay back to
laminar. `stop.check_laminarization` is on by default and ends the run
once the *perturbation energy* falls below its threshold — a later event
than the dissipation returning to laminar, since the mean profile relaxes
on the viscous timescale.

All four examples' sizes are collected in
[`examples/README.md`](../README.md).
