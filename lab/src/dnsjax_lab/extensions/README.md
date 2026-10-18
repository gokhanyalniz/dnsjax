# Runtime diagnostic streams: `[probes]` and `[force]`

Two optional parameter sections add binary streams to a running
simulation, beside the always-available `stats.dat` / `steps.dat` /
`corrector.dat` text diagnostics. Both are **extensions**: sections
owned outside `parameters.py`, yet parsed from the CLI and
`parameters.toml`, listed in `--help`, validated strictly per flow and
recorded into snapshot metadata once configured (the mechanism:
[Parameter layering](../../../../docs/configuration.md)).

Together they are the runtime half of the input-output analysis in
[`analysis/response`](../analysis/response/README.md): `[probes]`
records what the flow does, `[force]` controls what is done to it.
Neither is needed for an ordinary run.

## `[probes]` — spectral-mode profile stream

Every `probes.it_probes` steps, the complex wall-normal profiles
$\hat{\mathbf{u}}(y, t)$ of a listed set of global spectral modes are
appended to a binary `probes.bin`. Wall-bounded systems only. The
section is inherited on resume; `--probes.it_probes 0` switches it off,
whatever modes it inherited
([`configuration.md`](../../../../docs/configuration.md)).

```bash
.venv/bin/dnsjax \
  --phys.system plane-couette --phys.re 500 \
  --probes.modes "0,0;3,0" --probes.it_probes 10
```

`probes.modes` is an `"i2,i3;i2,i3;…"` list of **stored-layout** mode
indices — axis 2 is the complex slot, axis 3 the real-FFT slot, counting
the component axis as axis 0 (the per-geometry layout:
[`docs/scaling.md`](../../../../docs/scaling.md#array-layout-by-geometry)),
the transient-growth CLI's `--tg.modes` convention — bounds-checked
against the true (unpadded) mode counts. Unlike that CLI, the mean
mode `0,0` **is** allowed: its record is the instantaneous mean profile
of the stored field (the `dnsjax_lab.extensions.probes` module
docstring). On `curved-pipe` the component-0 profile is the solver's
carried $w_s = h u_s$, not $u_s$, under the cylindrical label `u_z`
(why: `dnsjax_lab.extensions.ProbesParams`).

**Why a separate stream.** A mode time series wants $10^5$ to $10^6$
samples, more than snapshots or the scalar `.dat` streams can carry
(the module docstring); the probe stream is the input for mode
statistics — covariances, spectra — and for every response route.

**On disk.** `probes.bin` is a flat sequence of fixed-size records,
one complex profile per probed mode and state component, beside a
`probes.json` sidecar, written once, that carries the schema (the
module docstring's "File format"). The reader is
`dnsjax_lab.analysis.response.probes` — NumPy only, no JAX — which also
provides `mean_profile`, `re_tau`, and `write_profile_file`.

**Buffering matches the `.dat` streams.** Records accumulate on the
device and are flushed — appended and `fsync`-ed — when the buffer
fills and at every flush site of the run loop: after the first step,
before every snapshot write, at shutdown and on a termination signal.
A non-finite value in a flushed record aborts the run through the same
`FATAL` / exit-code-3 path as the other diagnostics.

**Resume.** An existing `probes.bin`/`probes.json` pair is appended to
only when the sidecar matches the current run — same modes *and the
wavenumbers they denote*, grid, components, precision, system, cadence
and `dt` (the module docstring's resume semantics) — and anything else
is a hard error asking you to move the old pair aside, rather than a
stream whose two halves mean different things. A clean continuation
duplicates one sample at the seam; the reader drops it, and flags
genuinely non-monotonic timestamps.

## `[force]` — white-in-time stochastic mode kicks

Every `force.it_force` steps, a random superposition of stored
wall-normal channel profiles is added to each listed spectral mode (and
its real-FFT conjugate partner): a sequence of independent state
increments — *kicks* — that realize white-in-time forcing localized at
those modes. The drawn coefficients stream to `forcing.bin`.

```bash
.venv/bin/dnsjax \
  --phys.system plane-couette --phys.re 500 \
  --probes.modes "3,0" --probes.it_probes 5 \
  --force.modes "3,0" --force.profiles modes.npz \
  --force.amplitude 1e-3 --force.it_force 50
```

The section is **all-or-none**: `modes`, `profiles`, `amplitude` and
`it_force` are set together or not at all. Wall-bounded,
non-viscoelastic systems only; on `curved-pipe` a component-0 profile
is added to the carried $w_s = h u_s$ (the column `[probes]` records
there), not to $u_s$. It is **trajectory-defining** — kicks
alter the dynamics exactly as a `phys` change does, so resuming with
changed forcing starts a new trajectory unless `init.force_resume`.
`--force.it_force 0` switches an inherited section off, the whole
quartet with it, which is such a change.

| Knob | Meaning |
|---|---|
| `force.modes` | Modes to kick, in the `probes.modes` convention; the mean mode `0,0` is rejected |
| `force.profiles` | `.npz` holding `profiles_{i2}_{i3}`, a `(m, C, Ny)` complex, unit-energy-norm channel set on **this run's** wall-normal grid — the bundle `operator_tools` writes |
| `force.n_channels` | Leading stored channels used per mode (default: all) |
| `force.amplitude` | Coefficient scale $\varepsilon$; expected injected energy is $\varepsilon^2$ per channel per kick |
| `force.it_force` | Steps between kicks; must be a multiple of the probe cadence when probing |
| `force.seed` | Kick-coefficient PRNG seed; unset draws one from the system entropy pool and records it, and an append-resume inherits that value rather than drawing again |

**Why kicks and not a body-force term.** A forcing term inside the
nonlinear right-hand side would be integrated by the scheme —
Adams–Bashforth-extrapolated under `cnab2`, which colours white noise,
and iterated on by the `iterative-cn` corrector. A loop-level state
increment leaves both schemes untouched and makes the per-kick response
*exactly* the solver's own propagator, the object the transient-growth
export encodes (the `dnsjax_lab.extensions.forcing` module docstring).

**Timing.** A kick fires at the top of the loop on every iteration with
`it % it_force == 0`, after the probe sample and any snapshot write at
the same $t$, so both record the **pre-kick** state: a probe sample at
a kick time correlates with earlier kicks only (a zero-lag causality
check), and a resumed continuation neither loses nor doubles a kick.
An append-resume continues the host-side, rank-identical coefficient
stream exactly as if the run had never stopped (the module docstring's
timing conventions).

**Amplitude.** Choose $\varepsilon$ inside the linear-response window:
halving it must leave the identified operator unchanged.
`predicted_forced_variance` in the `ssi` module computes the
stationary forced level for planning (the module docstring's amplitude
guidance).

**On disk.** `forcing.bin` is a flat sequence of fixed-size records of
the coefficients exactly as applied — unscaled by `amplitude`, which
lives in the sidecar, and float64 whatever the state precision. An
append-resume must match the `forcing.json` sidecar on the modes and
their wavenumbers, channel count, amplitude, cadence, seed, `dt`, the
system and the profile bundle's SHA-256, since changing the injection
basis, or what a mode index means, invalidates the stream
mid-experiment (the record layout and the sidecar: the module
docstring's "File format"). The reader is
`dnsjax_lab.analysis.response.ssi`.

**A kick's increment is generally not discretely solenoidal**, and what
becomes of that part depends on the scheme. Under the default
`res.consistent_imm` it lands in the carried state intact and is
discarded one step later, so it reaches exactly one nonlinear
evaluation (two under `cnab2`) and never a solve; the legacy primitive
influence-matrix method (`res.consistent_imm = false`) instead feeds it
into the pressure Poisson right-hand side and damps it over the
following steps. Either way it is a bounded, per-event,
truncation-class effect rather than an accumulating one (the
`forcing.py` module docstring). Keep injected profiles solenoidal if
the distinction matters for the response being identified.

## Format versions

Both sidecars carry a `format_version` enforced against the reader's
`MIN_FORMAT_VERSION`, as the snapshot archive does. The record layouts
are fixed across schema versions, so the version is the only guard
against a silent misread: bump the writer and the reader together
whenever the stored *meaning* changes, not merely the layout (the two
module docstrings).

## What to do with the streams

The full workflow from a turbulent run to a data-driven linear
operator — and the three interchangeable routes to that operator, one
of which needs `[force]` and one of which needs only `[probes]` — is
[`analysis/response`](../analysis/response/README.md).
