---
paths:
  - "src/dnsjax/{__main__,measurements}.py"
  - "src/dnsjax/extensions/*.py"
  - "src/dnsjax/twin/*.py"
  - "src/dnsjax/analysis/{response,twin}/*.py"
  - "src/dnsjax/flows/**/*.py"
  - "src/dnsjax/geometries/**/*.py"
  - "tests/test_{driving,probes,forcing,energy_budget}.py"
---

# Diagnostic streams

- Streams: `stats.dat` (`get_stats`), `steps.dat` (the CFL, through
  the `rhs.py` `measure_fn` hook) and `corrector.dat`, buffered on
  device; `probes.bin` + `probes.json` (`[probes]`,
  `extensions/probes.py`, read by `analysis.response.probes`) and
  `forcing.bin` + `forcing.json` (`[force]`, `extensions/forcing.py`,
  read by `analysis.response.ssi`). Buffering, flushing and the
  non-finite guard: the `__main__.py` module docstring.
- Columns come in the sorted key order of the jitted dicts, never in
  insertion order.
- Header rows are `#`-commented (`__main__._write_dat_header`, shared
  with `twin/driver.py`): `np.loadtxt` reads a stream bare, and a
  reader must `lstrip("#")` the header line before splitting it.
- Driving columns (`-dPds'`, `-dPdn'`, `-dPdz'`, last, under
  `phys.driving = "constant_bulk_velocity"` or
  `phys.block_mean_spanwise_velocity`) hold the applied force `-Π`:
  `Π` is the `(0, 0)` mode of `∂p/∂s`, so every force on a
  Navier-Stokes RHS is `-Π` (`ic/mean_mode.py`).
  - One column set runs through the geometry corrector's `aux`, the
    flow's `get_driving` and `__main__`'s buffer width, so
    `validate_parameters` rejects a driving knob the flow's surface
    lacks.
  - Names come from `CylindricalFlow.driving_key` in the cylindrical
    family and from module constants in the Cartesian and annular
    ones.
  - `get_driving` takes the physical-basis state; the `t = t0` row is
    its inference, every later row the applied value (the read-site
    comment in `__main__.py`, `tests/test_driving.py`).
- Every stream with a JSON sidecar carries a `format_version` checked
  against its reader's floor; move writer and reader together when the
  stored meaning changes. The pairs:
  - `extensions/probes.py` -> `analysis/response/probes.py`;
  - `extensions/forcing.py` -> `analysis/response/ssi.py`;
  - `twin/spectra.py` -> `analysis/twin/spectra.py`;
  - `twin/driver.py` (`twin.json`) -> `analysis/twin/series.py`;
  - `twin/yspectra.py` (two streams) -> `analysis/twin/yspectra.py`,
    whose floors deliberately stay below the writers' versions: a
    layout change is named by the sidecar's `suffixes`.
- A non-finite flushed value prints one `FATAL: non-finite ...` line,
  skips the final snapshot and exits with code 3.
- `dnsjax-twin` writes each state's own driving (`stats.dat`,
  `stats_twin.dat`); `twin.dat` has no driving column.
