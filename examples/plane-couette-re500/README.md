# Plane-Couette flow in the minimal flow unit

Two walls sliding past each other, in the smallest box that will hold
turbulence between them. The domain is the minimal flow unit of
Hamilton, Kim & Waleffe (1995) — $1.75\pi$ long by $1.2\pi$ wide, in
half-gap units — which fits exactly one pair of streamwise rolls across
the span: the self-sustaining cycle of rolls, streaks and streak
instability, and nothing else. That is what makes it cheap enough to
run on a laptop.

```bash
mkdir -p /tmp/plane-couette && cd /tmp/plane-couette
cp /path/to/dnsjax/examples/plane-couette-re500/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

Any value can be overridden on the command line (`--phys.re 400`,
`--step.scheme cnab2`), and `--dist.platform cuda` runs it on a GPU;
what the four examples share — the scratch directory, the GPU set-up,
why `D` and not `E'`, the laminarization stop, their sizes — is in
[`examples/README.md`](../README.md).

**What to watch.** `stats.dat` gets one row per 50 steps. The column to
follow is the **dissipation `D`**, against its laminar value $1/Re$.
As the localized spot breaks down, the dissipation rises to several
times laminar, and it stays well above laminar for as long as the flow
is turbulent. The perturbation wall shear (`tau'_s,b`, `tau'_s,t`) tells
the same story from the walls.

**It relaminarizes eventually, and that is the point.** A minimal flow
unit at this Reynolds number is a chaotic saddle, not an attractor:
turbulence here has a finite, stochastic lifetime and the flow falls
back to laminar Couette flow. When it does is not a property of the
flow: lifetimes here are distributed, and a different perturbation, or
the same one at a different amplitude, gives a different one. Raising
`re` tends to lengthen the episode; a larger box does too.

**Reference.** J. M. Hamilton, J. Kim and F. Waleffe, *Regeneration
mechanisms of near-wall turbulence structures*, J. Fluid Mech. **287**,
317–348 (1995).
