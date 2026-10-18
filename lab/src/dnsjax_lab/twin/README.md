# Twin runs: lockstep perturbation growth

`dnsjax-twin` steps **two** states of the same flow in lockstep — a
reference trajectory loaded from a snapshot and a perturbed partner
$\mathbf{u}^{(2)} = \mathbf{u}^{(1)} + \delta$ — and streams online
diagnostics of the difference field
$\Delta\mathbf{u} = \mathbf{u}^{(2)} - \mathbf{u}^{(1)}$: how fast two
initially-close realizations of the same turbulence separate, which
components carry the separation, which terms feed it, and which scales
have decorrelated by when. The three-component energies and the
volume-averaged budget follow Egerique-de-la-Concha & Hwang, *J. Fluid
Mech.* **1036**, A52 (2026)
([doi:10.1017/jfm.2026.11608](https://doi.org/10.1017/jfm.2026.11608));
the wall-normal-resolved spectra and budget refine them.

Both states share every singleton — grid, operators, jitted steppers,
`dt` — so their difference is purely dynamical (the `twin/driver.py`
and `twin/diagnostics.py` module docstrings).

## Scope

**Cartesian wall-bounded flows only** (plane-Couette,
plane-Poiseuille) at a **fixed time step**. `step.adaptive` and the
`[force]` section are rejected: the streams assume uniform sampling,
and a kick would have to be applied identically to both states. The
per-state solver streams are recorded for **both** states (below);
`[probes]` records the reference alone; the corrector-convergence and
non-finite guards watch both.

## Running one

Launch exactly like the production solver, from a scratch directory:
directly when it fits in one process, under `mpirun -np N` when it does
not. `python -m dnsjax_lab.twin` is the equivalent module form.

```bash
.venv/bin/dnsjax-twin \
  --init.snapshot parent.tar \
  --twin.e0 1e-6 --twin.seed 3 \
  --twin.bins True --twin.it_budget 100 --twin.it_spectra 100 \
  --stop.max_sim_time 5000
```

The parameter surface is the flow's own plus the `[twin]` section,
which `dnsjax-twin` registers and the solver does not;
`dnsjax-twin --help` lists it. A member keeps the solver's run status
files and stops the same way (`rm RUNNING`, or SIGUSR1), ending on a
final snapshot pair; see
[Stopping a run](../../../../docs/running.md#stopping-a-run).

| Knob | Default | Meaning |
|---|---|---|
| `twin.e0` | unset | Initial perturbation energy $E'(\delta)$ in the solver measure; setting it enables the section. `0` requests an exact zero perturbation |
| `twin.seed` | unset | Perturbation RNG seed; vary per ensemble member. Unset draws one from the system entropy pool on a fresh start and takes the recorded one from `twin.json` on a resume |
| `twin.smoothness` | `0.022` | Spectral envelope of the random perturbation over the periodic directions (`init.random_smoothness` convention). The default seeds below the minimal flow unit of plane Poiseuille at $Re_\tau = 178.6$ (below) |
| `twin.wall_smoothness` | `0.4` | Its wall-normal envelope (`init.random_wall_smoothness` convention) |
| `twin.wall_confinement` | `0.14` | Scale-dependent narrowing of its wall window (`init.random_wall_confinement` convention); `0` gives every mode the same window. A member whose `twin.json` has no `wall_confinement` key ran with `0`, back-fills to it, and resumes only with `--twin.wall_confinement 0` |
| `twin.mean_flow` | `true` | Perturb the $(k_x, k_z) = (0, 0)$ profile as well, conditioned on its conservation laws (below); `false` gives a mean-free partner |
| `twin.bins` | `false` | Also record the $\Delta U$ / $\Delta u_1$ / $\Delta u_2$ three-bin energies in `twin.dat`; required by `it_budget` |
| `twin.x0_planes` | `false` | Also store the $k_x = 0$ plane (the `_x0` fields) in both wall-normal-resolved streams: $E_{\Delta u_1}$, and with it $E_{\Delta u_2}$, resolved in $k_z$. The three-bin split does not need it; off, it is never traced |
| `twin.it_energy` | `1` | Steps between `twin.dat` rows |
| `twin.it_budget` | unset | Steps between `twin_budget.dat` rows; unset or `0` disables the stream |
| `twin.it_spectra` | unset | Steps between `twin_spectra.bin` records; unset or `0` disables the stream |
| `twin.it_yspectra` | unset | Steps between `twin_yspectra.bin` records (wall-normal-resolved componentwise spectra) |
| `twin.it_ybudget` | unset | Steps between `twin_ybudget.bin` records (the same bins' budget) |
| `twin.rotational_ybudget` | `false` | Write that budget with the rotational nonlinear term instead of the convective one |
| `twin.spectra_ref` | `true` | Also record the reference state's spectra, in streams of their own (below); off, they are never traced |
| `twin.it_spectra_ref` | `it_spectra` | Steps between `twin_spectra_ref.bin` records; `0` switches this stream alone off |
| `twin.it_yspectra_ref` | `it_yspectra` | Steps between `twin_yspectra_ref.bin` records; `0` switches this stream alone off |
| `twin.it_spectra3d` | unset | Steps between `twin_spectra3d/` files: the difference energy as a subsampled $(y, k_z, k_x)$ cube, one file per sample |
| `twin.it_spectra3d_ref` | `it_spectra3d` | Steps between `twin_spectra3d_ref/` files, the reference state's cube; `0` switches them alone off |
| `twin.it_budget3d` | unset | Steps between `twin_budget3d/` files: the convective `twin_ybudget` terms on the same points |
| `twin.n_y3d`, `n_kz3d`, `n_kx3d` | unset | Points the cubes keep per axis, spaced uniformly in the logarithm of the wall distance, $\lvert k_z \rvert$ and $k_x$; unset keeps every one |
| `twin.it_lowres_delta` | unset | Steps between `lowres_delta/` reduced-resolution snapshots of the difference field (and, under `lowres.pressure`, its pressure), at the `[lowres]` resolution |

Every optional cadence here is off when unset **or `0`** (why both: the
root [configuration guide](../../../../docs/configuration.md)), except
the three reference cadences: unset, each takes its difference
cadence, so only `0` switches one off. An off cadence is recorded in
`twin.json` as unset. `it_energy` has no off state.

`twin.bins` and `twin.x0_planes` are both off by default, for one
reason: the three-bin split is a partition of the $(k_x, k_z)$ plane
that Egerique-de-la-Concha & Hwang restrict to minimal flow units, and
the default $y$-resolved streams already return all three bins
(`analysis.twin.bin_energies`), the plane adding only the $k_z$
resolution of $E_{\Delta u_1}$ and $E_{\Delta u_2}$
(`twin/diagnostics.py`, "Wall-normal-resolved spectra"). Off, the bins
save a few percent of every step (two masked full-state copies) and the
plane a third of each wall-normal-resolved record and of its
collective.

Some knobs are not priced by cadence alone (`twin/driver.py`, Design
notes, "What the [twin] cadences cost"). `it_budget` and `it_ybudget`
set the **run's peak memory**: each sample is a separate program whose
transient, measured on CPU, lies above the time step's, and
`it_ybudget` also holds a second factored banded operator for the run
(`twin/pressure.py`, "Cost"). `it_budget3d` is the y-budget's program
with the same costs, and a sample due for both shares one density
pass. `it_lowres_delta` under `lowres.pressure` holds the same resident
operator, but its sample peaks below the step (the figures:
`twin/diagnostics.py`, Design notes, "Memory"). A GPU schedules its
own: size a job against the driver's closing `Peak device memory` line
(`Peak host memory` on CPU). `spectra_ref` off skips the reference half
of **every** spectrum in compute as well as on disk — a field pass and
a collective on the $(k_z, k_x)$ stream, about half the sample on the
$y$-resolved one and the cubes — at the cost of the decorrelation
ratio.

Both wavenumber axes of the two $y$-resolved streams and the cubes are
one-sided, $k_z$ folded onto $|k_z|$ as a requirement, not a
convenience (`_fold_kz` in `twin/diagnostics.py` says why); the fold
needs the even `res.nz` that every run has anyway: the solver refuses
an odd Fourier count at parse.

The solver's own `[lowres]` section (the root
[running guide](../../../../docs/running.md#reduced-resolution-snapshots))
applies to the **reference** state: `lowres.it_lowres`, counted from
the perturbation step like every cadence here, writes `lowres/`, and
`lowres.nx` / `ny` / `nz` / `pressure` also shape the difference
snapshots of `twin.it_lowres_delta`.

## The initial perturbation

$\delta$ is the divergence-free random field of
`dnsjax.ic.random_field`, rescaled to a solver-measure perturbation
energy $E'(\delta) = \tfrac{1}{2}\lVert \delta \rVert^2$ of *exactly*
`twin.e0` and applied once, at the fresh start: a resume can never
re-perturb (`twin/driver.py`, "Initial perturbation").

Its scale is `twin.smoothness`, $s$: the premultiplied spectra peak at
$\lambda^* = 4\pi |\ln(1 - s)| h$ (`dnsjax.ic.random_field` derives
it), and the default puts that peak at $\lambda^{*+} = 50$, half the
span of the minimal flow unit, so the seed sits mostly below the
smallest perturbation of the laminar state that would trigger
turbulence and cannot carry a self-sustaining structure of its own.
Inverted exactly,

```math
s = 1 - \exp\left(-\frac{\lambda^{*+}}{4\pi\, Re_\tau}\right),
```

which is 0.022 for plane Poiseuille at $Re = 4200$ in the
$4\pi \times 2 \times 2\pi$ box ($Re_\tau = 178.6$) and 0.0079 at
$Re_\tau = 500$ (`twin/driver.py`, Design notes, "The smoothness
default"). A minimal flow unit has no room below itself, so a run in
one (the plane-Couette box of Hamilton, Kim & Waleffe, *J. Fluid
Mech.* **287**, 317–348, 1995) sets `--twin.smoothness 0.4`
(`ensemble_setup.py build-twin --smoothness 0.4` for a member tree);
members recorded with $s = 0.4$ resume only with that flag.

Its $(k_x, k_z) = (0, 0)$ content follows `twin.mean_flow`, **on by
default** (the driver's own knob, not the shared
`init.random_mean_flow`, which every flow defaults off), conditioned
on that mode's conservation laws (`dnsjax.ic.mean_mode`, which says
what the pair then shares).

`twin.e0 = 0` makes the partner an exact copy stepped by the same
jitted stepper, so every difference energy must be exactly zero and
every partner stream byte-identical to the reference's: the
bit-identity determinism guard for the whole lockstep loop, which the
test suite pins.

## Diagnostic streams

### `twin.dat` — component energies

Fields are split by their wall-parallel Fourier support into the
**mean** $\Delta U$ (the $(k_z, k_x) = (0,0)$ mode), the **streaks**
$\Delta u_1$ ($k_x = 0$, $k_z \neq 0$ — the streamwise-averaged
fluctuation) and the **streamwise-varying** $\Delta u_2$
($k_x \neq 0$), whose masks partition the mode grid, so
$E_{\Delta U} + E_{\Delta u_1} + E_{\Delta u_2} = E_\Delta$ holds to
rounding — a deliberate redundancy, and a consistency guard
(`twin/diagnostics.py`, "Component decomposition"). Columns are `E_d`
and `E_ref` (the reference state's own energy) always; under
`twin.bins`, additionally `E_dU`, `E_du1`, `E_du2` and the
per-velocity-component split `E_du1_x` / `E_du1_y` / `E_du1_z`.
There is no driving column: the difference of a uniform mean-mode
force does no work on any $y$-integrated budget term
(`twin/diagnostics.py`, "Mean-mode driving"). Each state's own applied
forcing is in its `stats*.dat`, so the difference stays recoverable
offline, and the $y$-*resolved* budget carries the density it does
have. Format, buffering, `fsync`, the non-finite guard and the flush
sites are those of `stats.dat`, with a `t0` row at setup and a final
row after the last step. At the default `it_energy = 1` this is one
extra jitted call per step, the intended sampling rate for a
growth-rate fit.

### `twin_budget.dat` — the decomposed energy budget

The volume-averaged budget of each component energy: **24 advective
(production and transport) terms** plus **3 dissipations**, plus
consistency sums that close against the stepped states to spatial
truncation. The transport terms cancel pairwise by parts, so their
total is a further check rather than a free parameter (the term list
and its evaluation: `twin/diagnostics.py`, "Budget terms").

### `twin_yspectra.bin` / `twin_ybudget.bin` — wall-normal-resolved

The scale-resolved replacement for the three-bin split, and what to
reach for above a minimal flow unit. Per sample, the componentwise
difference energy as a density in $y$, marginalized each way,

```math
E_\Delta^x[u,v,w](y, k_z), \qquad E_\Delta^z[u,v,w](y, k_x)
```

— energy first, then the sum over the other wavenumber — plus the
$(0,0)$ mode, $E_\Delta^{xz00}(y)$ per component, and under
`twin.x0_planes` the whole $k_x = 0$ plane it comes from, the spectrum
of the streamwise-averaged field (`twin/diagnostics.py`,
"Wall-normal-resolved spectra"; the record layout: `twin/yspectra.py`).

Which fields a record holds is the sidecar's `suffixes`, and the
readers follow it rather than assume: a member written before `_xz00`
existed carries `x, z, x0` and no such key, and still opens
(`analysis.twin.stored_fields`).

`twin_ybudget.bin` carries the matching budget on the same bins:
production against the reference mean profile (`P_U`, the lift-up
term) and against its fluctuation gradients (`P_r`), transfer by the
reference fluctuation (`T_ref`) and by the difference field's own
advection (`T_self`), the viscous term in both forms (`V`, `eps`),
and the work of the pressure gradient (`Wp`).
`lab/scripts/twin_spectral_maps.py` draws them by default, regrouped as
the terms of the difference-energy balance
(`analysis.twin.balance_term`), and under `--moment-budget` splits the
rates of the difference energy's log-coordinate moments over the same
terms (`analysis.twin.moments`); its docstring says what each figure
shows.

`twin.rotational_ybudget` (default **off**) writes the same budget
with the *rotational* nonlinear term the solver integrates: the two
transfer terms become exactly redistributive ($\sum_k T = 0$ at every
$y$, whatever the resolution) and the nonlinear term is checkable
against the solver's own right-hand side. That is a different
decomposition, not a repaired one: the two forms differ by a gradient,
so volume totals agree while the $y$-densities move between
production, transfer and `Wp`. An eighth column, `P_lift`, carries the
convective `P_U` unchanged and outside the sum: the classical
production density a channel-budget reader looks for, otherwise
unrecoverable from what the streams hold (`twin/diagnostics.py`,
"Spectral budget" and "Two budget forms").

Summing either marginal over its axis and integrating in $y$ returns
`twin.dat`'s `E_d`; summing the budget's returns `twin_budget.dat`'s
`P_tot` / `eps_tot` (under `twin.rotational_ybudget`, `eps_tot` and —
from `P_lift` — the three mean-gradient production columns). Both
sidecars ship the wall-normal grid and its quadrature weights, so a
reader integrates without rebuilding the grid.

### `twin_spectra.bin` — $(k_z, k_x)$ energy spectra

The per-mode difference energy $E_\Delta(k_z, k_x)$, and in
`twin_spectra_ref.bin` the reference state's own spectrum. Their ratio
$E_\Delta / 2E^{(1)}$ is the scale-by-scale decorrelation measure —
which scales have decorrelated at each time, rather than a single
scalar. FFT-free, so any cadence is cheap. Why a binary stream, the
record layout (true modes only, spectral padding never stored), the
`twin_spectra.json` sidecar and its enforced `format_version`: the
`twin/spectra.py` module docstring.

### `twin_spectra_ref.bin` / `twin_yspectra_ref.bin` — the reference

The reference halves of the two spectra (`e_ref`, and the `r_*` set of
the $y$-resolved stream), each a stream of its own on its own cadence;
at the default cadences one program samples them with the difference
streams. A member recorded before they had streams of their own stored
them inside the difference streams (`includes_ref: true` in the
sidecar) and keeps writing that layout when resumed. The readers return
the reference the same way from either layout — `read_twin_spectra` /
`read_twin_yspectra` on the difference stream's sample times, `nan`
where the reference has none, and the `*_ref` readers on its own — and
so does `lab/scripts/twin_spectral_maps.py`, for mixed ensembles too.

### `twin_spectra3d/`, `twin_spectra3d_ref/`, `twin_budget3d/` — cubes

The densities of the $y$-resolved streams without the sum over the
other wavenumber: the componentwise difference energy (`e_u`, `e_v`,
`e_w`), the reference's (`r_*`) and every convective budget term
(`P_U`, `P_r`, `T_ref`, `T_self`, `V`, `eps`, `Wp`; no sum of them) as
$(y, k_z, k_x)$ cubes, one dnsjax tar per sample named by step,
computed at full resolution, then folded in $y$ and $k_z$ and thinned
log-uniformly to the `n_y3d` / `n_kz3d` / `n_kx3d` points. The folds,
the thinning, the y-major container, what a file carries to be read
alone, and the resume check against the newest file: the
`twin/cubes.py` module docstring. `analysis.twin.read_cube` reads one
file, optionally only some wall distances; `read_twin_spectra3d`,
`read_twin_spectra3d_ref` and `read_twin_budget3d` stack a directory.

### `lowres/`, `lowres_delta/` — reduced-resolution snapshots

`lowres/` holds the reference state at `[lowres]`'s cadence, exactly
as a plain solver run writes it, and `lowres_delta/` the difference
field $\Delta\mathbf{u}$ at `twin.it_lowres_delta`. Under
`lowres.pressure` (off by default) each file also holds its static
pressure; the difference's, $\Delta p$, is the pressure the budget's
`Wp` is computed from, solved from $\Delta\mathbf{u}$ directly
(`_difference_pressure_jit` in `twin/diagnostics.py`). Both are
computed at the run's resolution and then reduced, both are snapshot
files read by `dnsjax.analysis.read_state` / `read_pressure`, and
neither is a checkpoint.

### `stats*.dat`, `steps*.dat`, `corrector*.dat` — per state

The three streams a plain solver run writes are written here for
**each** state of the pair, at the same `[outs]` cadences and in the
same format: the reference's under the standard names, the partner's
with the `_twin` suffix.

| stream | cadence | contents |
|---|---|---|
| `stats.dat` / `stats_twin.dat` | `outs.it_stats` | the flow's `get_stats` — $E'$, $I$, $D$, $E$, the wall shears, the bulk velocities — plus the applied mean-mode driving as the last column(s) under an active driving constraint |
| `steps.dat` / `steps_twin.dat` | `outs.it_steps` | the CFL diagnostic and `dt` |
| `corrector.dat` / `corrector_twin.dat` | `outs.it_corrector` | the corrector iteration count `c` and its final error |

Each stream is off unless its cadence is set, and a pair is always
written together: same columns, same sample times, same buffering and
`fsync`ing, so the two load side by side with
`analysis.twin.series.read_dat` and subtract column-wise (and are
**byte-identical** at `twin.e0 = 0`, above).

The partner's driving column is its own applied force, not a
difference: under `constant_bulk_velocity` (or a blocked spanwise
mean) the two states hold the same bulk with different forces, and
that difference is the interesting number (why `twin.dat` has no
driving column: above).

Recording `steps.dat` for both states is also what makes the lockstep
loop run the *identical* jitted program on the pair in every
iteration, the measured stepper variant being selected by the cadence
(`twin/driver.py`, "Diagnostic streams").

## Trajectory bookkeeping

A twin trajectory lives in its run directory: the streams above, a
`twin.json` member record, and **paired snapshots** —
`state{isnap}.tar` for the reference and `state{isnap}_twin.tar` for
the partner, with identical `t`/`it` (`twin/driver.py`, "Trajectory
home, fresh start vs paired resume"). The reference keeps the standard
names, the `_twin` suffix marking the partner throughout, so every
existing tool works on the reference trajectory unchanged.

Two files decide the start mode: the partner of `init.snapshot`, and
the run directory's `twin.json`.

| Partner | `twin.json` | Outcome |
|---|---|---|
| exists | matches | **Paired resume** — both states load, the reference clock is inherited, no re-perturbation, streams append |
| exists | absent | Error: either a resume in a directory missing its member record, or a fresh start aimed at a twin run's own output |
| absent | exists | Error: this directory already holds a twin trajectory |
| absent | absent | **Fresh start** — perturb, write `twin.json`, save the IC pair |

"Matches" is the `twin.json` configuration record, the driver's
`_TWIN_MATCH_KEYS` (the format version, the system, `e0`, the seed,
the perturbation shape, `bins`, the stream and cube cadences, `dt`,
the precision): a resume that differs in any of them is refused rather
than appending to streams it does not belong to. The refusal names the
recorded value beside the configured one: when a default has moved
since the member was recorded, the recorded value is what to pass.
`twin.seed` is *matched*, not re-read, so a resume with the seed left
unset adopts the recorded one first, or an unpinned member would fail
its own first resume.

A trajectory-defining parameter change is a hard error on a paired
resume (start a fresh member instead). A fresh start inherits the
parent's clock, so offline analysis reads the perturbation time from
`twin.json`. `stop.max_sim_time` is a horizon counted from whichever
clock the run starts on: to end a member restarted mid-horizon where
it was meant to, resume it with `--stop.max_sim_time` set to what is
left.

## Ensembles

One member = one run directory = one `dnsjax-twin` invocation, varying
`twin.seed` and the parent snapshot — or leaving the seed unset, in
which case each member draws its own and records it, so members
launched by hand are independent without being told to be.
`lab/scripts/ensemble_setup.py` harvests statistically independent
parents into a manifest (`harvest`) and builds the member tree from it;
its module docstring is the manual for both:

```bash
uv run python lab/scripts/ensemble_setup.py build-twin \
  --manifest manifest.json --tree twins/ --e0 1e-5 \
  --horizon 5000 --it-budget 100 --it-spectra 100
```

Every cadence a member records at is counted from its own perturbation
step (`parent_it` in its `twin.json`, inherited across a paired
resume), so its samples sit at $t_\mathrm{parent} + n c \Delta t$
whatever step the parent snapshot was harvested at: the members of an
ensemble share one grid on the relative clock $t - t_\mathrm{parent}$
(`twin/driver.py`, "Diagnostic streams"), which the readers count in
whole steps (`series.relative_time`; the `series` docstring says why).

`build-twin` writes each member a `parameters.toml` carrying
`--horizon` as `stop.max_sim_time` (every member integrates the same
span) and the member's seed and perturbation shape, written out whether
given or not: that keeps the tree reproducible and resumable whichever
code version launches or resumes a member. `check_laminarization` is
forced off so every member runs the full horizon; a relaminarized
member stays visible offline in its `E_ref` column. The script emits
`run_commands.txt` and a `members.json` index, and never runs the
solver itself.

## Offline analysis

`dnsjax_lab.analysis.twin` reads and aggregates all of it, and is
importable **without JAX** — a guarantee the test suite pins. Its
modules are in the table below, and in full in its package docstring.

At `it_energy > 1` a `twin.dat` stream routinely holds real samples
off its sampling grid (the final row; a resumed segment's `t0` row), so
a consumer that needs a *uniform* grid — a centred difference, an
across-member stack — selects one with `series.uniform_grid` (the
`series` docstring, "Off-grid rows").

Members recorded under an absolute-step cadence, whose parents sat at
different `it` residues, carry grids displaced against one another
that share no relative sample time but $t = 0$. `aggregate_members`
and `lab/scripts/twin_spectral_maps.py` both refuse them at the
default tolerance, naming the `align_atol` (`--align-atol`) that
accepts them: it pairs each frame with every member's nearest sample
instead, and reports how far apart the widest such pairing is.

| Module | Role |
|---|---|
| `series` | Readers for `twin.dat` / `twin_budget.dat` and the `twin.json` member record, plus the core's `read_dat` re-exported, which loads any `.dat` stream (`stats.dat`, `stats_twin.dat`, …); per-component budget sums; `uniform_grid` |
| `ensemble` | Member-tree aggregation on aligned relative time, plus the growth-rate fits (exponential-phase $\lambda$, algebraic-phase linear rate) |
| `growth` | Growth-law diagnostics of a decorrelating pair: the logarithmic, logistic and decorrelation rates and the algebraic exponent |
| `moments` | The log-coordinate moments of a $(y, k)$ stream and their rates |
| `spectra` | Readers for `twin_spectra.bin` and its reference stream (either layout), and the decorrelation ratio |
| `yspectra` | Readers for `twin_yspectra.bin` / `twin_ybudget.bin` and the reference stream, the wall-normal quadrature contraction, the three-bin energies recovered from them, and the budget regrouped as the difference-energy balance |
| `cubes` | Readers for the 3-D cube directories, whole or by wall distance |
| `lengths` | Integral length scales of the difference field from a paired snapshot |

Aggregation is also a command:

```bash
uv run python -m dnsjax_lab.analysis.twin.ensemble \
  --tree twins/ --out twin_ens.npz
```

## See also

- [Runtime diagnostic streams](../extensions/README.md) — the
  `[probes]` and `[force]` sections (`[force]` is rejected here).
- [Response analysis](../analysis/response/README.md) — the other use
  of `lab/scripts/ensemble_setup.py`, on ordinary solver runs.
- The root [README](../../../../README.md) for the solver itself.

Per-function behaviour, the difference-field derivations, the budget
term list, the $\pm k_z$ fold, the pressure's wall closure and the
frame-invariance notes live in the `twin/driver.py`,
`twin/diagnostics.py`, `twin/pressure.py`, `twin/spectra.py`,
`twin/yspectra.py` and `twin/cubes.py` module docstrings; the reduced
snapshots in `dnsjax/lowres.py`'s and the static pressure in
`geometries/wall_bounded/_cartesian_pressure.py`'s.
