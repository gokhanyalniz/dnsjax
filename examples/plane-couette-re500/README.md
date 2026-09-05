# Plane-Couette flow in the minimal flow unit

Two walls sliding past each other, in the smallest box that will hold
turbulence between them. The domain is the classic minimal flow unit —
$1.75\pi$ long by $1.2\pi$ wide, in half-gap units — which fits exactly
one pair of streamwise rolls across the span: the self-sustaining cycle
of rolls, streaks and streak instability, and nothing else. That is what
makes it cheap enough to run on a laptop and, at the same time, an
honest turbulence calculation rather than a cartoon of one.

```bash
mkdir -p /tmp/plane-couette && cd /tmp/plane-couette
cp /path/to/dnsjax/examples/plane-couette-re500/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

A run reads `parameters.toml` from its own working directory, so copy it
into a scratch directory and launch there. Any value can be overridden on
the command line — `--phys.re 400`, `--step.scheme cnab2`.

**What to watch.** `stats.dat` gets one row per 50 steps, with a
`#`-commented header that `numpy.loadtxt` reads directly. The column to
follow is the **dissipation `D`**, against its value in the first row —
that first row is the laminar state plus a small spot, so it is
effectively the laminar reference. The localized spot breaks down by
$t \approx 25$, dissipation rises to about **4.7 times laminar**, and the
flow stays several times laminar for roughly a hundred advective time
units. The perturbation wall shear (`tau'_s,b`, `tau'_s,t`) tells the same
story from the walls.

Do *not* read the perturbation energy `E'` as the turbulence indicator on
its own: it also contains the mean profile's deviation from laminar, so
it can fall smoothly while the flow is still fully turbulent.

**It will relaminarize, and that is the point.** A minimal flow unit at
this Reynolds number is a chaotic saddle, not an attractor: turbulence
here has a finite, stochastic lifetime and the flow falls back to laminar
Couette flow — in this configuration a little after $t \approx 100$. That
number is not a property of the flow: lifetimes here are distributed, and
a different perturbation, or the same one at a different amplitude, gives
a different one. Raising `re` lengthens the episode; a larger box
does too.

`stop.check_laminarization` is on by default and ends the run once the
*perturbation energy* drops below its threshold. That is a later event
than the dissipation returning to laminar: the mean profile relaxes back
on the viscous timescale, so a run like this one finishes its horizon
with the flow already laminar but `E'` still decaying.

About a minute on one core of an AMD Ryzen 7 PRO 7840U laptop CPU. All
four examples' measured times are collected in
[`examples/README.md`](../README.md).
