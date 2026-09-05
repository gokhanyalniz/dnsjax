# Plane-Poiseuille flow in a minimal channel

A channel in a minimal box — $\pi$ long and $\pi/2$ wide in half-height
units, roughly one near-wall streak pair across the span at this
Reynolds number — which is about as small a domain as sustains near-wall
turbulence. Channel flow is **linearly stable** at $Re = 5000$: the
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
**dissipation `D`** against its value in the first row, which is
effectively the laminar reference: the spot breaks down by
$t \approx 15$ with dissipation overshooting to about **14 times
laminar**, then settles onto a turbulent state around **3.3 times
laminar** and stays there — fluctuating by a few per cent — for the whole
of the run.

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
settles at a *lower* flow rate — in this configuration about 20 % below
laminar. That is also why `E'` is a poor turbulence indicator in general;
it carries the mean profile's deviation as well as the fluctuations.

**It may relaminarize.** A minimal channel is a chaotic saddle like the
minimal Couette cell: turbulence in it has a finite, stochastic lifetime
rather than living forever. This configuration is still turbulent at the
end of its horizon, but a longer run, or another perturbation, will
eventually decay. `stop.check_laminarization` is on by default and ends
the run once the *perturbation energy* falls below its threshold — a
later event than the dissipation returning to laminar, since the mean
profile relaxes on the viscous timescale.

Just under two minutes on one core of an AMD Ryzen 7 PRO 7840U laptop CPU.
All four examples' measured times are collected in
[`examples/README.md`](../README.md).
