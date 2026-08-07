# Response analysis: from a turbulent run to a linear operator

Given a turbulent dnsjax run, what linear operator best describes how a
chosen Fourier mode responds to being pushed?

This package answers that end to end: record a mode's wall-normal
profile as the flow evolves, build the linear operator about the
measured turbulent mean, find the directions that operator is most
excitable in, excite them, and fit a generator to what comes back. It
also offers **three interchangeable ways** to identify that generator,
which trade experimental cost against how much you have to assume about
the turbulent background — and which share one basis, one fit and one
output convention, so their answers are directly comparable.

Nothing here is imported by `dnsjax.analysis`, so the package-level
"importing the analysis API never imports JAX" guarantee is untouched.
These modules **may** use JAX, and they do where it pays.

## Requirements

SciPy, declared as the optional `analysis` extra:

```bash
uv sync --extra analysis
```

A plain `uv sync` from a clone already provides it through the dev
group; the extra is what a non-development install needs. SciPy is
imported lazily, inside the functions that use it — `logm`, the
Lyapunov solve, non-symmetric `eig`: factorisations JAX has no GPU
kernels for. Without it the Lyapunov solve falls back to an
eigendecomposition closed form; the matrix logarithm does not.

The dense time sweeps (growth curves, input-response curves) run
batched `expm` + SVD on the JAX default device, are GPU-capable, and
need float64. The three identification CLIs take `--dist.platform` and
select the backend before any JAX import, exactly as the solver does;
the `operator_tools` controllability export needs neither, being NumPy
and SciPy only.

## The pipeline

Steps 1-3 stand alone — turbulent statistics, and optimal growth about
a measured mean. Steps 4-6 build on their outputs.

### 1. Probe the run

Add a probe stream to an ordinary DNS run: the mean mode, to get the
turbulent mean profile, and whichever mode you intend to study.

```bash
mpirun -np 1 .venv/bin/dnsjax \
  --phys.system plane-couette --phys.re 500 \
  --probes.modes "0,0;3,0" --probes.it_probes 10
```

Stream format, cadence guidance and resume rules:
[`extensions/README.md`](../../extensions/README.md).

### 2. The turbulent mean

`read_probes` loads the stream; `mean_profile` turns the `(0,0)` record
into a total mean profile (the closed-form laminar profile is added
back), and `write_profile_file` writes it in the two-column, top-wall-
first form the transient-growth CLI reads. Cut the initial transient
with `t_min`, and sanity-check `re_tau` before going further.

```python
from dnsjax.analysis.response.probes import (
    read_probes,
    mean_profile,
    re_tau,
    write_profile_file,
)

data = read_probes("run/")
print(re_tau(data, t_min=200.0))
write_profile_file("U_mean.txt", data, t_min=200.0)
```

### 3. The linear operator about that mean

The transient-growth CLI linearises about an arbitrary wall-normal
**total** profile — including one that is not a solution of the
equations, such as this measured mean — and reuses the solver's own
linear step per Fourier mode.

```bash
python -m dnsjax.analysis.transient_growth \
  --phys.system plane-couette --phys.re 500 \
  --tg.profile U_mean.txt --tg.modes "3,0" \
  --tg.save_operator True
```

Beside the usual `U_mean_tg_summary.txt` and `U_mean_tg.npz`,
`--tg.save_operator` writes `U_mean_tg_op.npz`: each mode's reduced
generator $\mathcal{A}$, restricted to the resolved eigenspace in an
orthonormal energy-coordinate basis, together with the bases and the
coordinate contract. That bundle is the input for everything below.

**$G_{\max}$ needs a converged wall-normal resolution** — at an
unconverged `ny`/`nr` the reported optimum is an artefact. The recipe
is in the transient-growth module docstring.

### 4. The injection basis

The leading **controllability modes** of a generator are the directions
it is most excitable in — the natural basis for a response experiment,
and far smaller than the full state.

```bash
python -m dnsjax.analysis.response.operator_tools \
  --operator U_mean_tg_op.npz --n-modes 30 --out U_mean_cont.npz
```

This writes `profiles_{i2}_{i3}` — `(m, C, Ny)` full-state profiles at
unit energy norm — and `gram_eigvals_{i2}_{i3}` per mode. Pick
`--n-modes` from the eigenvalue decay. The bundle is consumed directly
by `scripts/snapshot_perturb.py --perturb.modes_npz` and by
`--force.profiles`. NumPy and SciPy only: no JAX, no device.

### 5. Ensemble impulse responses

One perturbed run measures a perturbation *plus* the turbulence it
rides on. Averaging over statistically independent parents cancels the
incoherent part and leaves the coherent response; the residual decays
as $1/\sqrt{N}$ in the member count.

```bash
uv run python scripts/ensemble_setup.py harvest \
  --run-dir prod/ --t-min 200 --spacing 5 --n 300 --out manifest.json

uv run python scripts/ensemble_setup.py build \
  --manifest manifest.json --tree members/ --mode 3,0 \
  --tg-npz U_mean_tg.npz --which input --amplitude-energy 1e-6 \
  --horizon 30 --probe-modes "3,0" --it-probes 10
```

`harvest` selects snapshots past `--t-min` and thins them to a minimum
`--spacing` in simulation time (several eddy-turnover times), so the
members are statistically independent. `build` materialises one
directory per member — a perturbed seed snapshot from
`scripts/snapshot_perturb.py`, plus a generated `parameters.toml` — and
emits `run_commands.txt` (one scheduler-agnostic launch line per
member) and `members.json`. It never runs the solver itself;
`--dry-run` prints the whole tree and every command without writing.

Default `--pairing antithetic` seeds each parent twice, at $+\epsilon$
and $-\epsilon$. Because dnsjax runs are deterministic for a fixed
configuration and device layout, $(u_+ - u_-)/2$ cancels the shared
background **and** every even-order nonlinear contribution, at the same
cost as an unpaired pair. `baseline` pairs a perturbed member against
the unperturbed parent; `none` relies on the plain ensemble mean alone.

Run the members, then aggregate:

```bash
python -m dnsjax.analysis.response.ensemble aggregate \
  --tree members/ --out response_0.npz --operator U_mean_tg_op.npz
```

With `--operator` the aggregate also carries the linear prediction and
the $G(t)$ envelope beside the measured response energy.

### 6. Identification

Repeat step 5 once per basis index, then fit the generator from the
propagator samples $M(\tau) \approx e^{\tau L}$:

```bash
python -m dnsjax.analysis.response.ensemble identify \
  --responses response_0.npz response_1.npz response_2.npz \
  --operator U_mean_tg_op.npz --modes-npz U_mean_cont.npz \
  --horizons "1,2,4" --out identified.npz
```

The output holds the identified `L`, its spectrum and stability, the
per-lag residuals, and the growth curves `G_id` (of `L`) against
`G_ref` (of the reference operator restricted to the same basis).

## Three routes to the same operator

Steps 5-6 are the direct route. Two alternatives replace them on the
same basis, coordinates and output convention.

| Route | What it needs | What it assumes |
|---|---|---|
| `ensemble` | An ensemble per basis index — the most compute | Nothing about the background; the response is measured directly |
| `lim` | Only the step-1 probe stream of the plain, **unforced** run — no extra runs at all | That the turbulent forcing is white in time |
| `ssi` | One run re-run with `[force]` stochastic kicks — one experiment, not an ensemble | Nothing about the background: the kicks are known exactly |

```bash
# Linear inverse modeling: lagged covariances of an unforced stream.
python -m dnsjax.analysis.response.lim \
  --probes run1/ run2/ --mode 3,0 --operator U_mean_tg_op.npz \
  --modes-npz U_mean_cont.npz --n-modes 10 \
  --lags "0.5,1,2" --t-min 200 --out lim.npz

# Stochastic-forcing identification: kick/response cross-covariance.
python -m dnsjax.analysis.response.ssi \
  --runs run1/ run2/ --mode 3,0 --operator U_mean_tg_op.npz \
  --lags "0.5,1,2" --t-min 200 --out ssi.npz
```

`ssi` defaults its channel basis to the profile bundle recorded in each
run's `forcing.json`, so the run carries its own provenance;
`--modes-npz` overrides it. Beyond the common outputs it reports the
causality level and the measured against predicted variance. Setting
up the forced run — the kick timing, the amplitude window, why kicks
rather than a body-force term: the
[`[force]` section](../../extensions/README.md).

## Why the three are comparable

Every operator here lives in the export's **energy-orthonormal**
coordinates: a reduced state $a$ satisfies
$\lVert a \rVert_2^2 = q^H \mathrm{diag}(w) q$ for the full state
$q = T_{\mathrm{lift}} a$, with both maps precomputed. So the plain
matrix 2-norm *is* the energy norm. Three consequences make the whole
package simpler than it would otherwise be:

```math
G(t) = \lVert e^{tA} \rVert_2^2 ,
```

a Galerkin restriction onto orthonormal columns preserves the norm, and
the controllability Gramian with unit-covariance forcing in the energy
inner product is just the Lyapunov solution of $(A, I)$.

Because all three routes fit in those coordinates, on the same basis,
and report the same quantities, their operators can be compared
directly — against each other and against the reference operator
restricted to the same subspace.

## Module map

| Module | Role |
|---|---|
| `probes.py` | JAX-free reader for `probes.bin`/`probes.json`, plus `mean_profile`, `re_tau`, `write_profile_file` |
| `operator_tools.py` | Controllability Gramians and modes, growth curves of arbitrary operator matrices, Galerkin restriction, the full-state lift used for injection |
| `ensemble.py` | Member-tree aggregation (`aggregate`) and direct operator identification (`identify`) |
| `lim.py` | Linear inverse modeling from lagged covariances of an unforced probe stream |
| `ssi.py` | Reader for `forcing.bin` and the kick/response cross-covariance fit |

Orchestration lives in `scripts/ensemble_setup.py`; per-function
behaviour, storage layouts and knob guidance live in the module
docstrings. Guards: `tests/response/`.
