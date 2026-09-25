# Examples

Four worked configurations across three of the solver's four
geometries. Each is a single annotated `parameters.toml` plus a short
README, sized to reach turbulence on **one laptop core in minutes**, and
each starts from a deterministic localized-roll perturbation — so two
runs of the same file take the same trajectory, with no seed to record.
(The annular geometry — Taylor–Couette, quasi-Keplerian, Dean — has
no example here: in the linearly stable regimes its transition
threshold is amplitude- and geometry-specific, with no standard
minimal unit to copy.)

| example | geometry | what it shows |
|---|---|---|
| [`kolmogorov/`](kolmogorov/) | triply-periodic | the simplest full nonlinear path in the repository: no walls, so no influence matrix and no banded solve between the initial condition and the answer |
| [`plane-couette-re500/`](plane-couette-re500/) | Cartesian | the minimal flow unit — the smallest box that holds turbulence between two shearing walls, and a turbulent state with a finite lifetime |
| [`plane-poiseuille-re5000/`](plane-poiseuille-re5000/) | Cartesian | a small channel box: subcritical transition below the linear-stability limit, under constant-bulk driving |
| [`pipe-re2500/`](pipe-re2500/) | cylindrical | a short periodic pipe at a transitional Reynolds number, where turbulence in a cell this short is transient |

Run one by copying its configuration into a scratch directory — output
files land in the working directory, so it wants one of its own:

```bash
mkdir -p /tmp/run && cd /tmp/run
cp /path/to/dnsjax/examples/kolmogorov/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

Any value in the file can be overridden on the command line
(`--phys.re 200`), and `dnsjax --help <system>` lists everything a given
flow accepts.

**Reading the result.** Each example writes `stats.dat`, one row per
diagnostic interval with a `#`-commented header, so `numpy.loadtxt`
reads it directly. For the three wall-bounded examples the column that
says whether the flow is turbulent is the **dissipation `D`**, measured
against its laminar value — not the perturbation energy `E'`, which also
carries the mean profile's deviation from laminar and can move for
reasons that have nothing to do with fluctuations. Kolmogorov flow is
the exception, and its README explains why: its laminar state is a
sinusoidal shear profile that is *more* dissipative than the turbulence
that replaces it, so there `E'` is the reliable indicator and `D` runs
the other way. Each README says what to expect.

**They may stop early.** In wall-bounded boxes this small, turbulence
can be transient: it has a finite, stochastic lifetime and eventually
decays back to the laminar state, sometimes before the horizon.
`stop.check_laminarization` is on by default, so a run whose
perturbation energy collapses ends cleanly and says so instead of
integrating a laminar flow to the horizon.

**How big they are.** Each runs in double precision (the default) on one
laptop core, in minutes:

| example | resolution | horizon |
|---|---|---|
| `kolmogorov` | $32^3$ | 200 |
| `plane-couette-re500` | $16 \times 33 \times 16$ ($n_x, n_y, n_z$) | 200 |
| `plane-poiseuille-re5000` | $16 \times 49 \times 16$ ($n_x, n_y, n_z$) | 250 |
| `pipe-re2500` | $24 \times 32 \times 32$ ($n_z, n_r, n_\theta$) | 200 |

These are not GPU-sized problems. Cases this small do not come close to
filling one, so there is no reason to expect a GPU to help here and some
reason to expect it to hurt; they are sized for a laptop on purpose. What
the solver does on hardware it was built for is a different question, and
one this repository does not yet answer with numbers.

The production-scale run these are the small counterpart to — a
100-diameter pipe over 500 advective time units — is walked through flag
by flag in [`docs/running.md`](../docs/running.md).

## Notebook

[The notebook](notebooks/read_snapshot_numpy_only.ipynb) reads a
snapshot with NumPy and nothing else. It is not a fifth flow
example: it runs the solver as a *subprocess* and post-processes what
that wrote, so the notebook kernel never imports JAX or the solver at
all — which is the claim it exists to check, and its last cell does.
