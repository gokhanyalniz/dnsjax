# Pipe flow in a short periodic cell

Four diameters of pipe at $Re = 2500$, a little above the Reynolds
number where pipe turbulence becomes sustainable at all, in a cell short
enough to run on a laptop. Pipe flow is **linearly stable**: the
parabolic profile has no known instability at any Reynolds number, so
nothing happens here unless something finite pushes it. The localized
spot in the initial condition is that push.

The full circle is simulated. `geo.m0` would restrict the domain to a
$2\pi/m_0$ wedge and cut the azimuthal cost by $m_0$, but it forbids
every azimuthal wavenumber that is not a multiple of $m_0$ — and near
onset the structures that matter are exactly the low ones. The length
comes down instead.

```bash
mkdir -p /tmp/pipe && cd /tmp/pipe
cp /path/to/dnsjax/examples/pipe-re2500/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

A run reads `parameters.toml` from its own working directory, so copy it
into a scratch directory and launch there. Any value can be overridden on
the command line — `--phys.re 3000`, `--geo.lz 12 --res.nz 36`.

**The forcing moves, not the flow rate.** `phys.driving` is set to
`constant_bulk_velocity`, so the flow rate is held fixed and the axial
pressure gradient is whatever it takes to maintain it. That keeps the
Reynolds number meaning what it says once the flow is turbulent, and it
puts the applied forcing in a **last column of `stats.dat`** (`-dPdz'`,
positive when accelerating), which rises above its laminar value exactly
when the flow does. Under the default `constant_pressure_gradient` the
roles swap and it is the flow rate that drops instead.

**What to watch.** `stats.dat` gets one row per 50 steps, with a
`#`-commented header that `numpy.loadtxt` reads directly. Follow the
**dissipation `D`** against its value in the first row, and the applied
forcing beside it: the spot breaks down over the first few tens of
advective time units, dissipation peaks around **four times laminar** at
$t \approx 30$, and the flow stays above 1.5 times laminar until
$t \approx 75$ before decaying back. `tau'_z` — the perturbation wall
shear — tracks the same event from the wall.

**It will relaminarize, and that is the physics.** At this Reynolds
number pipe turbulence is transient: it has a finite, memoryless
lifetime and eventually decays, which is why the onset of *sustained*
pipe turbulence is a question about lifetimes rather than a stability
threshold. This run spends its second half returning to laminar, which
is worth watching rather than skipping. A longer pipe or a higher `re`
lengthens the episode; `stop.check_laminarization`, on by default, ends
a run once the perturbation energy finally falls below its threshold.

Just under five minutes on one core of an AMD Ryzen 7 PRO 7840U laptop
CPU — the most expensive of the four, on two counts: it carries about
twice the modes of the plane channels, and a cylindrical geometry needs
four banded operators per Fourier mode where a plane channel needs two.
All four examples' measured times are collected in
[`examples/README.md`](../README.md).
