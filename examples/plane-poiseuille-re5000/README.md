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

Any value can be overridden on the command line
(`--phys.driving constant_pressure_gradient`,
`--init.localized_rolls_amplitude 0.5`); what the four examples share —
the scratch directory, why `D` and not `E'`, the laminarization stop,
their sizes — is in [`examples/README.md`](../README.md).

**What to watch.** `stats.dat` gets one row per 50 steps. Follow the
**dissipation `D`** against its laminar value, $4/(3Re)$: it rises
steeply as the spot breaks down, then settles onto a turbulent level
several times laminar.

**The forcing moves, not the bulk.** `phys.driving` is set to
`constant_bulk_velocity` here, so the streamwise pressure gradient
responds and the applied forcing is a **last column in `stats.dat`**
(`-dPds'`); [`examples/README.md`](../README.md) says why. Under the
default `constant_pressure_gradient` a turbulent channel, being far more
dissipative, settles at a *lower* flow rate than the laminar one. That
is also why `E'` is a poor turbulence indicator in general; it carries
the mean profile's deviation as well as the fluctuations.

**It may relaminarize.** Turbulence in a box this small need not last
forever: a longer run, or another perturbation, may decay back to
laminar.

**Reference.** S. A. Orszag, *Accurate solution of the Orr–Sommerfeld
stability equation*, J. Fluid Mech. **50**, 689–703 (1971).
