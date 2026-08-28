# Twin runs: lockstep perturbation growth

`dnsjax-twin` steps **two** states of the same flow in lockstep — a
reference trajectory loaded from a snapshot and a perturbed partner
$\mathbf{u}^{(2)} = \mathbf{u}^{(1)} + \delta$ — and streams online
diagnostics of the difference field
$\Delta\mathbf{u} = \mathbf{u}^{(2)} - \mathbf{u}^{(1)}$: how fast two
initially-close realisations of the same turbulence separate, which
components carry the separation, which terms feed it, and which scales
have decorrelated by when.

Both states share every singleton — grid, operators, jitted steppers,
`dt` — so their difference is purely dynamical.

## Scope

**Cartesian wall-bounded flows only** (plane-Couette,
plane-Poiseuille) at a **fixed time step**. `step.adaptive` and the
`[force]` section are rejected: the streams assume uniform sampling,
and a kick would have to be applied identically to both states.
`[probes]`, `stats.dat`, `steps.dat` and `corrector.dat` all record
the **reference** state; the corrector-convergence and non-finite
guards watch **both**.

## Running one

Launch exactly like the production solver, from a scratch directory:
directly when it fits in one process, under `mpirun -np N` when it does
not. `python -m dnsjax.twin` is the equivalent module form.

```bash
.venv/bin/dnsjax-twin \
  --init.snapshot parent.tar \
  --twin.e0 1e-6 --twin.seed 3 \
  --twin.it_budget 100 --twin.it_spectra 100 \
  --stop.max_sim_time 5000
```

The parameter surface is the flow's own plus the `[twin]` section,
which `dnsjax-twin` registers and the solver does not.

| Knob | Default | Meaning |
|---|---|---|
| `twin.e0` | unset | Initial perturbation energy $E'(\delta)$ in the solver measure; setting it enables the section. `0` requests an exact zero perturbation |
| `twin.seed` | unset | Perturbation RNG seed; vary per ensemble member. Unset draws one from the system entropy pool on a fresh start and takes the recorded one from `twin.json` on a resume |
| `twin.smoothness` | `0.4` | Spectral envelope of the random perturbation (`init.random_smoothness` convention) |
| `twin.bins` | `false` | Also record the $\Delta U$ / $\Delta u_1$ / $\Delta u_2$ three-bin energies in `twin.dat`; required by `it_budget` |
| `twin.it_energy` | `1` | Steps between `twin.dat` rows |
| `twin.it_budget` | unset | Steps between `twin_budget.dat` rows; unset disables the stream |
| `twin.it_spectra` | unset | Steps between `twin_spectra.bin` records; unset disables the stream |
| `twin.it_yspectra` | unset | Steps between `twin_yspectra.bin` records (wall-normal-resolved componentwise spectra). Needs an even `res.nz` |
| `twin.it_ybudget` | unset | Steps between `twin_ybudget.bin` records (the same bins' budget). Needs an even `res.nz` |
| `twin.spectra_ref` | `true` | Also compute and store the reference spectrum with each `it_spectra` / `it_yspectra` sample; off, it is never traced |

`twin.bins` is off by default. The three-bin split is a three-bin
partition of the $(k_x, k_z)$ plane, and the reference paper restricts
it to minimal flow units; above that, `twin.it_yspectra` resolves the
same information in $k$ and $y$, and the three bin energies remain
exactly recoverable from it (`analysis.twin.bin_energies`). Turning
the bins off also drops a few percent of every step — the two masked
full-state copies the split forces.

Three of these are not priced by cadence alone. `it_budget` sets the
**run's peak memory**, not just its per-sample cost: the budget is a
separate compiled program whose transient is the driver's global
high-water mark, and the device allocator's pool grows to the maximum
over every program. `it_ybudget` costs memory in two places — a
resident second factored banded operator with its homogeneous columns
(`twin/pressure.py`'s "Cost" section), and a per-sample transient of 15
padded physical fields that is the easy one to miss. `spectra_ref`
gates the reference half of **both** spectra streams in compute as well
as on disk: it is a static flag on the two jitted samplers, so with it
off the reference reduction is never traced — saving a field pass and a
collective on the $(k_z, k_x)$ stream, and about half the sample on the
$y$-resolved one. What it costs is the decorrelation ratio.

An even `res.nz` is a hard requirement of the two $y$-resolved streams,
checked at parse: they store $k_z$ folded onto $|k_z|$, and at odd `nz`
the stored band is asymmetric, leaving the outermost negative mode with
no positive partner to fold onto.

## The initial perturbation

$\delta$ is the divergence-free random field of
`dnsjax.ic.random_field` — device-count independent, seeded per global
mode — rescaled so that its solver-measure perturbation energy is
*exactly* `twin.e0`:

```math
E'(\delta) = \tfrac{1}{2}\lVert \delta \rVert^2 = e_0 ,
```

the same convention as `snapshot_perturb --perturb.amplitude_energy`.
It is applied once, at the fresh start; a resume can never re-perturb.

Its $(k_x, k_z) = (0, 0)$ content follows `twin.mean_flow`, **on by
default** (the driver's own knob, not the shared
`init.random_mean_flow`, which every flow defaults off), and is
conditioned on that mode's conservation laws. So the partner drives
the flow at the reference's own mean pressure gradient — whatever the
driving — and under a held bulk velocity carries its bulk exactly:
the pair is one flow, driven identically, differing only in its
initial state.

`twin.e0 = 0` makes the partner an exact copy stepped by the same
jitted stepper, so every difference energy must be exactly zero. That
is the bit-identity determinism guard for the whole lockstep loop, and
the test suite pins it.

## Diagnostic streams

### `twin.dat` — component energies

Fields are split by their wall-parallel Fourier support into three
components: the **mean** $\Delta U$ (the $(k_z, k_x) = (0,0)$ mode),
the **streaks** $\Delta u_1$ ($k_x = 0$, $k_z \neq 0$ — the
streamwise-averaged fluctuation), and the **streamwise-varying**
$\Delta u_2$ ($k_x \neq 0$). The three masks partition the whole mode
grid, so

```math
E_{\Delta U} + E_{\Delta u_1} + E_{\Delta u_2} = E_\Delta
```

holds to rounding — a deliberate redundancy, and a consistency guard.
Columns are `E_d` and `E_ref` (the reference state's own energy)
always; under `twin.bins`, additionally `E_dU`, `E_du1`, `E_du2` and
the per-velocity-component split `E_du1_x` / `E_du1_y` / `E_du1_z`.
There is no driving column: the difference of a uniform mean-mode
force does no work on any $y$-integrated budget term (the reference's
own applied forcing is in `stats.dat`, and the $y$-*resolved* budget
carries the density it does have — see `twin_ybudget.bin` below).
Format, buffering, `fsync`, the non-finite guard and the flush sites are
those of `stats.dat`, with a `t0` row at setup and a final row after the
last step.

At the default `it_energy = 1` this is one extra jitted call per step —
the intended sampling rate for a growth-rate fit.

### `twin_budget.dat` — the decomposed energy budget

The volume-averaged budget of each component energy: **24 advective
(production and transport) terms** plus **3 dissipations**, plus
consistency sums that close against the stepped states to spatial
truncation. The advective terms group by which field occupies the
mean slot — 7 with a carrier profile, 2 advected by the mean
difference, 6 at the $(0,0)$ mode, and 9 triple-fluctuating — and the
transport terms cancel pairwise by parts, so their total is a further
check rather than a free parameter.

### `twin_yspectra.bin` / `twin_ybudget.bin` — wall-normal-resolved

The scale-resolved replacement for the three-bin split, and what to
reach for above a minimal flow unit. Per sample, the componentwise
difference energy as a density in $y$, marginalised each way,

```math
E_\Delta^x[u,v,w](y, k_z), \qquad E_\Delta^z[u,v,w](y, k_x)
```

— energy first, then the sum over the other wavenumber, which under
the forward-norm convention *is* the average over that direction —
plus the $k_x = 0$ plane, which is the spectrum of the
streamwise-averaged field and recovers $E_{\Delta U}$,
$E_{\Delta u_1}$, $E_{\Delta u_2}$ exactly. `twin_ybudget.bin`
carries the matching budget on the same bins: production against the
reference mean profile (the lift-up term) and against its
fluctuations, transfer by the reference and by the difference field,
the viscous term in both forms, and the pressure work.

Both wavenumber axes are one-sided, with $|k_z|$ folded — a
requirement, not a convenience: the stored half-plane's entries are
conjugate-*pair* energies, whose partners sit at $-k_z$.

Summing either marginal over its axis and integrating in $y$ returns
`twin.dat`'s `E_d`; summing the budget's returns
`twin_budget.dat`'s `P_tot` / `eps_tot`. Both sidecars ship the
wall-normal grid and its quadrature weights, so a reader integrates
without rebuilding the grid.

### `twin_spectra.bin` — $(k_z, k_x)$ energy spectra

The per-mode difference energy $E_\Delta(k_z, k_x)$ and, by default,
the reference state's own spectrum. Their ratio
$E_\Delta / 2E^{(1)}$ is the scale-by-scale decorrelation measure —
which scales have decorrelated at each time, rather than a single
scalar. FFT-free (a masked reduction of data already in spectral
space), so any cadence is cheap; the reason it is a binary stream is
volume: $O(N_{k_z} N_{k_x})$ values per sample is far beyond the
scalar `.dat` streams, and three orders of magnitude below a snapshot.

```python
numpy.dtype(
    [("t", "<f8"), ("e_delta", VAL, (N2, N3))]
    + ([("e_ref", VAL, (N2, N3))] if includes_ref else [])
)
```

with `N2 = nz - 1` true complex modes and `N3 = nx // 2` true real-FFT
modes — spectral padding is never stored — and `VAL` following
`res.double_precision`. A `twin_spectra.json` sidecar carries the
schema: mode counts, the integer harmonic lists of both axes, the
domain lengths, the cadence, and the resolved parameter dump. Its
`format_version` is enforced against the reader's floor, as for every
other dnsjax stream.

## Trajectory bookkeeping

A twin trajectory lives in its run directory: the streams above, a
`twin.json` member record (seed, `e0`, parent snapshot and clock, git
hash, the resolved parameter dump), and **paired snapshots** —
`state{isnap}.tar` for the reference and `state{isnap}_twin.tar` for
the partner, written back-to-back with identical `t`/`it`. The
reference keeps the standard name, so every existing tool works on the
reference trajectory unchanged.

Two files decide the start mode: the partner of `init.snapshot`, and
the run directory's `twin.json`.

| Partner | `twin.json` | Outcome |
|---|---|---|
| exists | matches | **Paired resume** — both states load, the reference clock is inherited, no re-perturbation, streams append |
| exists | absent | Error: either a resume in a directory missing its member record, or a fresh start aimed at a twin run's own output |
| absent | exists | Error: this directory already holds a twin trajectory |
| absent | absent | **Fresh start** — perturb, write `twin.json`, save the IC pair |

"Matches" is the `twin.json` configuration record — `e0`, the seed,
the smoothness, the cadences, `dt`, the precision — and a resume that
differs in any of them is refused rather than appending to streams it
does not belong to. `twin.seed` is *matched*, not re-read, which is why
a resume with the seed left unset adopts the recorded one first: the
pair is not re-perturbed, so what it needs is the seed it was born
with, and an un-pinned member would otherwise fail its own first resume.

On a paired resume a trajectory-defining parameter change is a hard
error: switching mid-trajectory would disconnect the pair from its own
streams. Start a fresh member instead. A fresh start inherits the
parent's clock, so offline analysis reads the perturbation time from
`twin.json`.

`stop.max_sim_time` is a horizon counted from whichever clock the run
starts on, not an absolute end time, so a member restarted mid-horizon
asks for what is left of it rather than for the original value.

## Ensembles

One member = one run directory = one `dnsjax-twin` invocation, varying
`twin.seed` and the parent snapshot — or leaving the seed unset, in
which case each member draws its own and records it, so members
launched by hand are independent without being told to be.
`scripts/ensemble_setup.py` harvests statistically independent parents
and builds the member tree:

```bash
uv run python scripts/ensemble_setup.py harvest \
  --run-dir prod/ --t-min 200 --spacing 5 --n 300 --out manifest.json

uv run python scripts/ensemble_setup.py build-twin \
  --manifest manifest.json --tree twins/ --e0 1e-5 \
  --horizon 5000 --it-budget 100 --it-spectra 100
```

`build-twin` needs no seeding subprocesses — the driver perturbs
in-process at start — so each member directory holds only a generated
`parameters.toml`, carrying `--horizon` straight through as
`stop.max_sim_time`: every member integrates the same span however far
apart their parents were harvested, which is what the shared time grid
means. Seeds run `--seed-base + k` over the flat member
index, and `--members-per-snapshot` fans several seeds out of one
parent. An unset `--seed-base` is drawn from the system entropy pool
and printed, so two ensembles built from the same parents are
independent rather than both running seeds `1..N`; either way each
member's concrete seed is written into its `parameters.toml`, which is
what keeps the tree reproducible and resumable. `check_laminarization` is forced off so every member runs the
full horizon and the streams aggregate on one shared time grid; a
relaminarised member stays visible offline in its `E_ref` column. The
script emits `run_commands.txt` and a `members.json` index, and never
runs the solver itself.

## Offline analysis

`dnsjax.analysis.twin` reads and aggregates all of it, and is
importable **without JAX** — a guarantee the test suite pins.

Two kinds of `twin.dat` row sit off the `it_energy` sampling grid, and
at `it_energy > 1` both are routine: the unconditional final row, so
the end-of-run state is never lost, and a resumed segment's `t0` row
when `outs.it_snapshot` is not a multiple of `it_energy`. They are real
samples and are kept, so a consumer that needs a *uniform* grid — a
centred difference, an across-member stack — selects one with
`series.uniform_grid` instead of assuming the raw stream is one.

| Module | Role |
|---|---|
| `series` | Readers for `twin.dat` / `twin_budget.dat` / `stats.dat` and the `twin.json` member record; per-component budget sums; `uniform_grid` |
| `ensemble` | Member-tree aggregation on aligned relative time, plus the growth-rate fits (exponential-phase $\lambda$, algebraic-phase linear rate) |
| `spectra` | Reader for `twin_spectra.bin` and the decorrelation ratio |
| `yspectra` | Readers for `twin_yspectra.bin` / `twin_ybudget.bin`, the wall-normal quadrature contraction, and the three-bin energies recovered from them |
| `lengths` | Integral length scales of the difference field from a paired snapshot |

Aggregation is also a command:

```bash
python -m dnsjax.analysis.twin.ensemble --tree twins/ --out twin_ens.npz
```

## See also

- [Runtime diagnostic streams](../extensions/README.md) — the
  `[probes]` and `[force]` sections (`[force]` is rejected here).
- [Response analysis](../analysis/response/README.md) — the other use
  of `scripts/ensemble_setup.py`, on ordinary solver runs.
- The root [README](../../../README.md) for the solver itself.

Per-function behaviour, the difference-field derivations, the budget
term list, the $\pm k_z$ fold, the pressure's wall closure and the
frame-invariance notes live in the `twin/driver.py`,
`twin/diagnostics.py`, `twin/pressure.py`, `twin/spectra.py` and
`twin/yspectra.py` module docstrings.
