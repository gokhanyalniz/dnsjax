## Twin-run perturbation growth (`dnsjax-twin`)

`dnsjax-twin` steps a reference snapshot and a perturbed copy (a
random divergence-free field of exact energy `twin.e0`) in lockstep
and streams difference-field diagnostics. Cartesian wall-bounded
flows, fixed dt, launched like the solver (scratch dir; `mpirun -np N`
only when it is multi-process):

`.venv/bin/dnsjax-twin --init.snapshot parent.tar
--twin.e0 1e-6 --twin.seed 3 --stop.max_sim_time 10`

### Module layout

- `driver.py`: the `dnsjax-twin` console script (`__main__.py` is the
  `python -m dnsjax.twin` shim onto it) -- lockstep driver, the
  `[twin]` extension, paired snapshots/resume, stream wiring
- `diagnostics.py`: the difference-field quantities -- component
  masks, energies, the 27-term budget, `(kz, kx)` spectra, and the
  wall-normal-resolved `(y, k)` spectra and spectral budget that
  supersede the three bins
- `pressure.py`: the difference pressure on the IMM's own wall closure
  (the one term a `y`-resolved budget cannot omit)
- `_binstream.py`: `BinStream`, the buffered-binary state machine the
  three `.bin` writers share
- `spectra.py`: `TwinSpectraStream` -> `twin_spectra.bin`
- `yspectra.py`: `twin_yspectra.bin` / `twin_ybudget.bin`

Readers are `dnsjax.analysis.twin` (`analysis/CLAUDE.md`).

### Streams

Energies (`twin.dat`), the wall-normal-resolved spectra and matching
spectral budget (`twin_yspectra.bin` / `twin_ybudget.bin`,
`twin.it_yspectra` / `twin.it_ybudget`), `(kz, kx)` energy spectra
(`twin_spectra.bin`, `twin.it_spectra`), and the legacy three-bin
budget (`twin_budget.dat`, `twin.it_budget`, which needs `twin.bins`).
The three **per-state** solver streams are written for *both* states
at the ordinary `[outs]` cadences -- the partner's as
`stats_twin.dat` / `steps_twin.dat` / `corrector_twin.dat`, same
columns and sample times, byte-identical to the reference's at
`twin.e0 = 0`; only `[probes]` stays reference-only.

Start/resume rules (partner snapshot + `twin.json` decide; a resume
never re-perturbs), the fresh-start guard, stream formats, the ±k_z
fold the marginals require, the frame-invariance / dissipation-form
notes and the pressure's wall closure: the `driver.py`,
`diagnostics.py`, `pressure.py` and `yspectra.py` module docstrings;
the maths is Appendix A of the `perturbation_dynamics` document (not
in this repo), whose closing subsection keys the plotted panels to its
equations.

### The `[twin]` surface

Registered by `dnsjax-twin` only. The defaults that decide what a run
costs and what its streams can answer -- each field's own description
says why, the pointers carry the derivations:

- `bins` and `x0_planes` default **off**; `_xz00`, the `y`-resolved
  `(0,0)` mode, is stored instead, and `analysis.twin.bin_energies`
  refuses without `x0_planes` (`yspectra.py`, "Why these replace the
  three-bin diagnostics").
- `rotational_ybudget` defaults **off** = convective, matching
  `twin_budget.dat` term by term; the rotational form is a different
  decomposition, not a repair, which is why it also stores `P_lift`
  (`diagnostics.py`, "Two budget forms").
- `mean_flow` defaults **on** -- its own field, not the shared
  `init.random_mean_flow`, because `init.*` is snapshot-inherited in
  both directions (root CLAUDE.md, "Initial conditions").
- `smoothness` / `wall_smoothness` / `wall_confinement` mirror the
  three `init.random_*` shape knobs (2026-09-03). Only the last
  changed behaviour: 0.4 / 0.4 / **0.14**, where the wall window used
  to be the same for every mode. All three are `_TWIN_MATCH_KEYS`
  entries, so **a member recorded before the change needs
  `--twin.wall_confinement 0` to resume** -- no back-fill can
  reconcile a key whose old behaviour and new default differ (unlike
  `mean_flow`, whose did not). `wall_smoothness` does back-fill, and
  its `_TWIN_LEGACY_DEFAULTS` entry is a callable because the old
  behaviour was `= smoothness` rather than a constant. Why `s` did
  *not* move, and how to score a candidate: `ic/random_field.py` and
  `scripts/random_ic_calibrate.py`.
- `spectra_ref` defaults **on**: the reference spectrum every
  decorrelation divides by.

Three stream layouts exist and all three read (the sidecar's
`suffixes` names one), and `analysis.twin.stored_fields` /
`record_dtype` own the layout for the eager reader *and*
`scripts/twin_spectral_maps.py`'s memory map. The readers' floors are
deliberately **not** raised with the writers' versions -- the one
place in the repo where that lockstep is broken, and `yspectra.py`
says why (root CLAUDE.md, "Diagnostics", for the other five
writer/reader pairs).

### Offline tooling

Ensembles: `scripts/ensemble_setup.py build-twin` + `analysis.twin`.

- `scripts/twin_postprocess.py`: rebuilds `twin.dat`, the two `(y, k)`
  streams and `stats.dat` / `stats_twin.dat` from a member's snapshot
  pairs (`[recon]`), for members recorded before a stream existed.
  Bit-for-bit except the `stats*.dat` driving columns, and the
  `[recon]` stream-shaping flags are stated, not inherited: its
  module docstring.
- `scripts/twin_spectral_maps.py`: draws the `(y, k)` streams as
  premultiplied `(lambda, y)` maps and `k`-summed `(y, t)` spacetime
  maps over an ensemble, in inner units. Only the spectra marginals
  are drawn by default; the two decorrelations, the spacetime maps,
  the budget and the `k_x = 0` plane are each behind their own flag.
  What each is, and the premultiplier / `E_ref` / colour-scale
  conventions: its module docstring. Needs matplotlib -- `uv run
  --group plots python scripts/twin_spectral_maps.py` (the `plots`
  dependency group; `uv sync` alone does not install it, and
  `snapshot_figure.py` is the only other script in it).

### Tests

`test_twin_unit.py`, `test_twin_driver.py`, `test_twin_budget.py`,
`test_twin_analysis.py`, `test_twin_postprocess.py` and
`test_twin_spectral_maps.py`. One-liners in the root CLAUDE.md Tests
section; what each covers is in its own module docstring.
