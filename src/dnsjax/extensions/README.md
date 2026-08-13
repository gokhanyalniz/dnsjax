# Runtime diagnostic streams: `[probes]` and `[force]`

Two optional parameter sections add binary streams to a running
simulation, beside the always-available `stats.dat` / `steps.dat` /
`corrector.dat` text diagnostics. Both are **extensions** — sections
owned outside `parameters.py`, but parsed from the CLI and
`parameters.toml`, listed in `--help`, validated strictly per flow, and
recorded into snapshot metadata exactly like a core section (the
mechanism: the root [README](../../../README.md#parameter-layering)).

Together they are the runtime half of the input-output analysis in
[`analysis/response`](../analysis/response/README.md): `[probes]`
records what the flow does, `[force]` controls what is done to it.
Neither is needed for an ordinary run.

## `[probes]` — spectral-mode profile stream

Every `probes.it_probes` steps, the complex wall-normal profiles
$\hat{\mathbf{u}}(y, t)$ of a listed set of global spectral modes are
appended to a binary `probes.bin`. Wall-bounded systems only.

```bash
mpirun -np 1 .venv/bin/dnsjax \
  --phys.system plane-couette --phys.re 500 \
  --probes.modes "0,0;3,0" --probes.it_probes 10
```

`probes.modes` is an `"i2,i3;i2,i3;…"` list of **stored-layout** mode
indices — axis 2 is the complex slot, axis 3 the real-FFT slot, the
same convention the transient-growth CLI's `--tg.modes` uses. Indices
are bounds-checked against the true (unpadded) mode counts, so a
padding slot can never be probed. Unlike the transient-growth CLI, the
mean mode `0,0` **is** allowed: its record is the instantaneous mean
profile of the perturbation, and adding the closed-form laminar profile
recovers the total mean (the reader does this for you).

**Why a separate stream.** A mode time series wants $10^5$–$10^6$
samples. A snapshot per sample is some three orders of magnitude more
bytes, and the scalar `.dat` streams cannot hold a complex per-$y$
profile. The probe stream is the input for mode statistics —
covariances, spectra — and for every response route.

**On disk.** `probes.bin` is a flat sequence of fixed-size records,

```python
numpy.dtype([("t", "<f8"), ("u", VAL, (K, C, N_y, 2))])
```

for `K` probed modes and `C` state components, with the trailing axis
`(re, im)` and `VAL` following `res.double_precision` (`t` is always
float64). A `probes.json` sidecar, written once, carries the schema:
the mode list, the integer wavenumbers, component labels, the
wall-normal grid, the cadence, and the full resolved parameter dump.
The reader is `dnsjax.analysis.response.probes` — NumPy only, no JAX —
which also provides `mean_profile`, `re_tau`, and `write_profile_file`.

**Buffering matches the `.dat` streams.** Records accumulate in an
on-device `(nbuffer, K, C, N_y)` buffer and are flushed — appended and
`fsync`-ed — when it fills, at shutdown, before every snapshot write,
and on a termination signal. Flushed records are scanned for
non-finite values, and a hit aborts the run through the same
`FATAL` / exit-code-3 path as the other diagnostics.

**Resume.** An existing `probes.bin`/`probes.json` pair is appended to
only when the sidecar matches the current run — same modes, grid,
components, precision, system and cadence. Anything else is a hard
error asking you to move the old pair aside, rather than a stream whose
two halves mean different things. A clean continuation duplicates one
sample at the seam; the reader drops it, and flags genuinely
non-monotonic timestamps.

## `[force]` — white-in-time stochastic mode kicks

Every `force.it_force` steps, a random superposition of stored
wall-normal channel profiles is added to each listed spectral mode (and
its real-FFT conjugate partner): a sequence of independent state
increments — *kicks* — that realise white-in-time forcing localised at
those modes. The drawn coefficients stream to `forcing.bin`.

```bash
mpirun -np 1 .venv/bin/dnsjax \
  --phys.system plane-couette --phys.re 500 \
  --probes.modes "3,0" --probes.it_probes 5 \
  --force.modes "3,0" --force.profiles modes.npz \
  --force.amplitude 1e-3 --force.it_force 50
```

The section is **all-or-none**: `modes`, `profiles`, `amplitude` and
`it_force` are set together or not at all. Wall-bounded,
non-viscoelastic systems only. It is **trajectory-defining** — kicks
alter the dynamics exactly as a `phys` change does, so resuming with
changed forcing starts a new trajectory unless `init.force_resume`.

| Knob | Meaning |
|---|---|
| `force.modes` | Modes to kick, in the `probes.modes` convention; the mean mode `0,0` is rejected |
| `force.profiles` | `.npz` holding `profiles_{i2}_{i3}`, a `(m, C, Ny)` complex, unit-energy-norm channel set on **this run's** wall-normal grid — the bundle `operator_tools` writes |
| `force.n_channels` | Leading stored channels used per mode (default: all) |
| `force.amplitude` | Coefficient scale $\varepsilon$; expected injected energy is $\varepsilon^2$ per channel per kick |
| `force.it_force` | Steps between kicks; must be a multiple of the probe cadence when probing |
| `force.seed` | Kick-coefficient PRNG seed |

**Why kicks and not a body-force term.** A forcing term inside the
nonlinear right-hand side would be integrated by the scheme: `cnab2`
would Adams-Bashforth-extrapolate the random sequence
($1.5 f^n - 0.5 f^{n-1}$, which colours white noise), and the
`iterative-cn` corrector would iterate on it. A loop-level state
increment leaves both schemes untouched and makes the per-kick response
*exactly* the solver's own propagator — the object the transient-growth
export encodes — at the cost of one fused scatter-add every
`it_force` steps.

**Timing.** A kick fires at the top of the loop on every iteration with
`it % it_force == 0`: after the equal-$t$ probe sample and after any
snapshot write, so both record the **pre-kick** state. A probe sample
taken at a kick time therefore correlates with earlier kicks only,
which gives the identification a clean zero-lag causality check.
Snapshots are never post-kick, so a resumed continuation applies the
kick belonging to its own first iteration: none is lost or doubled.
The coefficient PRNG is host-side and rank-identical, and an
append-resume skips the already-recorded draws, continuing the stream
exactly as if the run had never stopped.

**Amplitude.** Each kick adds $\varepsilon \sum_j w_j \mathbf{p}_j$ per
mode with $w_j \sim \mathcal{CN}(0, 1)$ i.i.d. and unit-energy
profiles. Choose $\varepsilon$ inside the linear-response window:
halving it must leave the identified operator unchanged. The stationary
forced level follows from the operator's Lyapunov equation —
`predicted_forced_variance` in the `ssi` module computes it for
planning.

**On disk.** `forcing.bin` is a flat sequence of fixed-size records,

```python
numpy.dtype([("t", "<f8"), ("w", "<f8", (K, m, 2))])
```

for `K` forced modes and `m` channels, the trailing axis the
`(re, im)` of the coefficients exactly as applied — unscaled by
`amplitude`, which lives in the sidecar. Coefficients are host-generated
float64 whatever the state precision; the volume is negligible. The
`forcing.json` sidecar carries the modes, channel count, amplitude,
cadence, seed, the resolved parameters, and the profile bundle's path
and SHA-256 — an append-resume must match the latter, since changing
the injection basis mid-experiment invalidates the stream. The reader
is `dnsjax.analysis.response.ssi`.

**A kick's increment is generally not discretely solenoidal**, and what
becomes of that part depends on the scheme. Under the default
`res.consistent_imm` there is no pressure Poisson to absorb it: it lands
in the carried state intact and is discarded one step later, when the
reconstruction rebuilds the tangential pair from the wall-normal
velocity and vorticity alone. The legacy primitive influence-matrix
method (`res.consistent_imm = false`) instead feeds it into the pressure
Poisson right-hand side and damps it over the following steps. Either
way it reaches exactly one nonlinear evaluation (two under `cnab2`,
which carries that evaluation forward) and never a solve — a bounded,
per-event, truncation-class effect rather than an accumulating one. Keep
injected profiles solenoidal if the distinction matters for the response
being identified.

## Format versions

Both sidecars carry a `format_version` enforced against the reader's
`MIN_FORMAT_VERSION`, as the snapshot archive does. The record layouts
are fixed across schema versions, so a stale stream reads *cleanly* and
only its values mean something else — which makes the version the only
thing standing between an old file and a silent misread. Bump the
writer and the reader together whenever the stored *meaning* changes,
not merely the layout.

## What to do with the streams

The full workflow from a turbulent run to a data-driven linear
operator — and the three interchangeable routes to that operator, one
of which needs `[force]` and one of which needs only `[probes]` — is
[`analysis/response`](../analysis/response/README.md).
