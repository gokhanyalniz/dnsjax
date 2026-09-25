# Pipe flow in a short periodic cell

Four diameters of pipe at $Re = 2500$, a transitional Reynolds number
for pipe flow, in a cell short enough to run on a laptop. Pipe flow is
**linearly stable**: the parabolic profile has no known instability at
any Reynolds number, so nothing happens here unless something finite
pushes it. The localized spot in the initial condition is that push.

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
**dissipation `D`** against its laminar value, $2/Re$, and the applied
forcing beside it: as the spot breaks down the dissipation climbs to
several times laminar, and it falls back as the flow relaminarizes.
`tau'_z` — the perturbation wall shear — tracks the same event from the
wall.

**Expect it to relaminarize.** A cell four diameters long gives
turbulence no room to spread or split, so here it is transient: it has a
finite, memoryless lifetime and eventually decays. A longer pipe or a
higher `re` tends to lengthen the episode; `stop.check_laminarization`,
on by default, ends a run once the perturbation energy finally falls
below its threshold.

It is the most expensive of the three wall-bounded examples, on two
counts: it carries two to three times the grid points of the plane
channels, and a cylindrical geometry needs three banded operators per
Fourier mode where a plane channel needs two. All four examples' sizes
are collected in [`examples/README.md`](../README.md).
