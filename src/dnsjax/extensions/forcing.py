r"""White-in-time stochastic mode forcing (state kicks) + coefficient log.

Adds, every ``force.it_force`` steps, a random superposition of
wall-normal channel profiles to each ``force.modes`` spectral mode
(the ``force`` extension section; :mod:`dnsjax.extensions`) --
a sequence of independent state increments ("kicks"), the
discrete-time realisation of white-in-time forcing localised at the
chosen modes.  The drawn coefficients stream to
``forcing.bin``/``forcing.json`` next to the ``.dat`` diagnostics,
keeping the run's full forcing history available to any offline
analysis.  E.g. cross-correlating the probe stream against it
identifies the mode's linear operator without any hypothesis on the
turbulent background (:mod:`dnsjax.analysis.response.ssi`, which
also holds the JAX-free reader).

Why kicks, not a body-force term
================================
A forcing term inside the nonlinear RHS would be traced into the
jitted steppers and integrated by the scheme: ``cnab2`` would
AB2-extrapolate the random sequence (``1.5 f^n - 0.5 f^{n-1}``,
colouring white noise), and the iterative-CN corrector would iterate
on it.  A loop-level state increment keeps both schemes untouched
and makes the per-kick response *exactly* the solver's own
propagator -- the quantity the transient-growth export encodes -- at
the price of one fused scatter-add per ``it_force`` steps.  (Under
``cnab2`` the carried nonlinear history is one kick stale for the
single step after a kick: an `$O(\varepsilon\,\Delta t)$`
perturbation of the same class as the scheme's local error.)

A kick's increment is generally **not** discretely solenoidal, and
what happens to that part depends on the scheme.  Under the default
``res.consistent_imm`` there is no pressure Poisson to absorb it: the
kick lands in the carried state intact and is **discarded** one step
later, when the reconstruction rebuilds the tangential pair from the
wall-normal velocity and vorticity alone (stage 7 in every geometry).
The legacy primitive IMM instead feeds it into the pressure Poisson
RHS and *damps* it over the following steps.  So it reaches exactly one step's
nonlinear evaluation -- two under ``cnab2``, which carries that
evaluation forward -- and never a solve.  Either way it is a bounded,
per-event, truncation-class effect, not an accumulating one; keep
injected profiles solenoidal if the distinction matters for the
response being identified.

Timing conventions (shared with the readers and resume)
=======================================================
- A kick fires at the **top** of the loop for every iteration with
  ``it % it_force == 0`` -- after the equal-``t`` probe sample and
  any snapshot write (both therefore record the *pre-kick* state; a
  probe sample at a kick time correlates with **earlier** kicks
  only, giving the identification a clean zero-lag causality check)
  and immediately before the step.
- Snapshots are never post-kick, so a resumed continuation applies
  the kick belonging to its first iteration itself: no kick is lost
  or doubled across a resume.
- The coefficient PRNG is host-side, rank-identical, and seeded by
  ``force.seed`` -- drawn from the system entropy pool when unset
  (:mod:`dnsjax.seeding`), and recorded here and in the snapshot so an
  append-resume inherits it; on an append-resume the already-recorded
  draws are skipped, so the coefficient stream continues exactly as if
  the run had never stopped.

Amplitude guidance: each kick adds `$\varepsilon\sum_j w_j\,
\mathbf{p}_j$` per mode with `$w_j \sim \mathcal{CN}(0,1)$` i.i.d.
and unit-energy profiles, so the expected injected energy is
`$\varepsilon^2$` per channel per kick and the stationary forced
level follows from the operator's Lyapunov equation
(``predicted_forced_variance`` in the ``ssi`` module).  Pick
`$\varepsilon$` in the linear-response window: halving it must leave
the identified operator unchanged.

File format
===========
``forcing.bin`` is a flat sequence of fixed-size records,

.. code-block:: python

    numpy.dtype([("t", "<f8"), ("w", "<f8", (K, m, 2))])

with ``K`` forced modes, ``m`` channels, and the trailing axis the
``(re, im)`` of the `$\mathcal{CN}(0,1)$` coefficients `$w$` exactly
as applied (unscaled by ``amplitude``, which is in the sidecar; the
kick was ``amplitude * sum_j w_j profile_j``).  Coefficients
are host-generated float64 regardless of the state precision (the
volume is tiny).  The ``forcing.json`` sidecar carries the schema:
modes (and the wavenumbers they denote -- a resume may change
``res.nx`` / ``res.nz``, which moves what a negative-block `$k_z$`
index means), channel count, amplitude, cadence, seed, the profile
bundle's path and SHA-256 (an append-resume must match all of these --
changing the basis mid-experiment invalidates the stream), and the full
resolved parameter dump.  No non-finite scan is needed: the coefficients are
finite by construction and the state itself is guarded by the
regular diagnostics.

The record layout is fixed across schema versions, so a stale stream
reads *cleanly* and only its values mean something else -- which makes
:data:`FORMAT_VERSION` the only thing between an old file and a silent
misread.  Bump it whenever the stored meaning changes (the injection
basis, the partner rule), and raise the reader's
``MIN_FORMAT_VERSION`` with it.

Sharded scatter
===============
:func:`build_mode_injector` is the scatter dual of
:func:`dnsjax.extensions.probes.build_mode_extractor`: inside a
``shard_map``, the device owning each static global mode index adds
the replicated column into its local shard (everyone else adds
zeros), so no global sharded axis is ever indexed.  The real-FFT
conjugate partner (``i3 = 0``) is just another scatter target; its
column -- the plain conjugate (every physical component is the
transform of a real field) -- is built on the host (the placement
rules of
``transient_growth.single_mode_state``, here in sharded form).  The
injector is generic runtime machinery: any future loop-level state
modification (feedback control, further forcing schemes) can reuse
it as is.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from jax import Array, lax, shard_map
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from ..flows.registry import cartesian_systems
from ..harmonics import complex_harmonics, parse_mode_pairs, real_harmonics
from ..param_surface import recorded_params_dump
from ..parameters import derived_params, params
from ..sharding import sharding
from ..snapshot_meta import git_hash, write_sidecar_json
from . import force_params
from .probes import _component_labels

#: Sidecar schema version.  Bumped whenever the stored *meaning*
#: changes at fixed layout -- the physical basis of the injected
#: profiles (and the real-FFT conjugate-partner rule that goes with
#: it), the physical (`$m = m_0 j$`) azimuthal wavenumber labels --
#: so a stale coefficient log cannot be silently replayed against
#: today's profile bundle.  The reader's floor is
#: ``analysis.response.ssi.MIN_FORMAT_VERSION``.
FORMAT_VERSION: int = 3

#: Sidecar keys that must match for an append (resume) to proceed.
_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "modes",
    # The wavenumbers, not the indices alone -- see the same entry in
    # ``probes.py``: a resolution-change resume moves what a stored
    # negative-block `$k_z$` index means.
    "wavenumbers",
    "n_channels",
    "amplitude",
    "it_force",
    "seed",
    "dt",
    "system",
    "profiles_sha256",
)


def build_mode_injector(
    mode_pairs: list[tuple[int, int]],
) -> Callable[[Array, Array], Array]:
    r"""Build a jitted scatter-add of columns into the probed layout.

    Returns a function ``(state, cols) -> state`` adding ``cols[k]``
    (``(K, C, N_y)`` complex, replicated) into the wall-bounded
    spectral ``state`` at the static global mode ``mode_pairs[k]``
    for every ``k`` -- the scatter dual of
    :func:`dnsjax.extensions.probes.build_mode_extractor` (same owner
    computation from the local shard shape inside a ``shard_map``;
    non-owners add zeros).  ``state`` is donated (the caller rebinds).

    *cols* are given in **physical** components -- the basis of the
    injection profiles and of the plain-conjugate partner rule -- and
    converted to the solver basis here, mapped over the mode axis so
    each ``(C, N_y)`` column goes through the ordinary
    component-axis-leading map.  Like the extractor's, this touches
    only the injected columns, never the field.  Cartesian carries
    physical components under both ``res.consistent_imm``
    formulations, so there the kick is a plain scatter-add.
    """
    pairs = tuple((int(i2), int(i3)) for i2, i3 in mode_pairs)
    if params.phys.system in cartesian_systems:
        to_solver = None
    else:  # cylindrical / annular (viscoelastic rejects [force])
        from ..geometries.wall_bounded._base import to_pm_basis

        to_solver = jax.vmap(to_pm_basis)
    # Owner masks only over mesh axes of size > 1: a size-1 axis has
    # owner 0 trivially, and querying its ``axis_index`` would stamp
    # the output as varying over an axis the (then axis-free)
    # ``out_specs`` declares replicated -- shard_map's vma check
    # rejects that.  (The extractor's ``psum`` re-replicates and
    # never hits this.)
    mesh_shape = dict(sharding.mesh.shape)

    def _local(shard: Array, cols: Array) -> Array:
        n2_loc, n3_loc = shard.shape[2], shard.shape[3]
        if to_solver is not None:
            cols = to_solver(cols)
        for k, (i2, i3) in enumerate(pairs):
            owner0, l2 = divmod(i2, n2_loc)
            owner1, l3 = divmod(i3, n3_loc)
            conds = []
            if mesh_shape["np0"] > 1:
                conds.append(lax.axis_index("np0") == owner0)
            if mesh_shape["np1"] > 1:
                conds.append(lax.axis_index("np1") == owner1)
            add = cols[k]
            if conds:
                is_owner = conds[0]
                for cond in conds[1:]:
                    is_owner = is_owner & cond
                add = jnp.where(is_owner, add, jnp.zeros_like(add))
            shard = shard.at[:, :, l2, l3].add(add)
        return shard

    injector = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=(sharding.spec_vector_shard, P(None, None, None)),
        out_specs=sharding.spec_vector_shard,
    )
    return jax.jit(injector, donate_argnums=0)


class StochasticForcer:
    r"""Kick generator + buffered coefficient writer (one per run).

    Construct once with the initial state (shape/dtype source) after
    the ``force`` extension section is validated; then let the main
    loop call
    :meth:`kick` at the ``it_force`` cadence and :meth:`flush` at the
    ``flush_all_buffers`` sites.  All randomness is host-side and
    rank-identical (every rank draws the same sequence); disk I/O is
    main-process only, with the buffer cleared on all ranks in
    lockstep (the ``.dat``-stream state machine).
    """

    def __init__(self, state: Array, directory: str | Path = ".") -> None:
        f = force_params
        self.modes = parse_mode_pairs(f.modes)
        self.amplitude: float = float(f.amplitude)
        self.nbuffer: int = params.outs.nbuffer
        n_components, ny = int(state.shape[0]), int(state.shape[1])
        self._profiles = self._load_profiles(f, n_components, ny)
        self.m: int = self._profiles[0].shape[0]

        # Scatter placements: per forced mode its target column and,
        # on the real-FFT plane (i3 = 0; validation excludes (0,0)),
        # the conjugate partner at the mirrored true index.  Every
        # physical component is the transform of a real field, so
        # the partner is the plain conjugate in every geometry
        # (``single_mode_state`` in dnsjax.analysis.transient_growth
        # is the single-device host-side form of the same rule).
        n2_true = params.res.nz - 1
        placements: list[tuple[int, int]] = []
        self._partner_slot: list[int | None] = []
        for i2, i3 in self.modes:
            placements.append((i2, i3))
            if i3 == 0:
                self._partner_slot.append(len(placements))
                placements.append((n2_true - i2, 0))
            else:
                self._partner_slot.append(None)
        self._n_slots = len(placements)
        self._inject = build_mode_injector(placements)
        # Compile outside the benchmark window (donated dummies).
        jax.block_until_ready(
            self._inject(jnp.zeros_like(state), self._device_cols())
        )

        self._rows: list[tuple[float, np.ndarray]] = []
        self.record_dtype = np.dtype(
            [("t", "<f8"), ("w", "<f8", (len(self.modes), self.m, 2))]
        )
        # Axis-2 harmonics are *physical* wavenumbers: on the
        # cylindrical/annular wedge the azimuthal mode of harmonic
        # `$j$` is `$m = m_0 j$` (``geo.m0 = 1`` is the full circle),
        # exactly as the geometry's ``Fourier.m`` builds it.  Cartesian
        # axis 2 is `$k_z$`, which has no wedge factor.
        m0 = 1 if params.phys.system in cartesian_systems else params.geo.m0
        q2 = complex_harmonics(params.res.nz) * m0
        q3 = real_harmonics(params.res.nx)
        self._sidecar = {
            "format_version": FORMAT_VERSION,
            "modes": [[i2, i3] for i2, i3 in self.modes],
            "wavenumbers": [
                [int(q2[i2]), int(q3[i3])] for i2, i3 in self.modes
            ],
            "n_channels": self.m,
            "amplitude": self.amplitude,
            "it_force": f.it_force,
            "seed": f.seed,
            "dt": params.step.dt,
            "system": params.phys.system,
            "n_components": n_components,
            "ny": ny,
            "component_labels": _component_labels(n_components),
            "wall_normal_grid": derived_params.wall_normal_grid,
            "profiles": str(f.profiles),
            "profiles_sha256": self._profiles_sha256(),
            "convention": (
                "kicks fire at loop top when it % it_force == 0, after "
                "the equal-t probe sample and any snapshot (both "
                "pre-kick) and before the step; each kick adds "
                "amplitude * sum_j w_j profile_j (+ the conjugate "
                "partner) with the recorded w ~ CN(0,1)"
            ),
            "git_hash": git_hash(),
            "params": recorded_params_dump(params),
        }
        self.bin_path = Path(directory) / "forcing.bin"
        self.json_path = Path(directory) / "forcing.json"
        n_existing = self._open_files()

        # Rank-identical coefficient stream; skip the draws already
        # taken so a resume continues the uninterrupted sequence (one
        # fixed-shape draw per kick, mirrored in ``kick``).
        #
        # On a *resume* the record count must equal the number of
        # kicks the state has actually been through, ``it0 //
        # it_force``.  The two agree only when resuming off the
        # snapshot the stream was written up to; resuming off an
        # earlier one (parent ran to it = 1000 snapshotting at 500 and
        # 1000, resume from the it = 500 one) would otherwise skip
        # past the whole file and append records duplicating times
        # already present, silently -- ``_MATCH_KEYS`` compares the
        # configuration, which is identical, so nothing else catches
        # it.  Deleting ``forcing.bin`` while keeping the sidecar is
        # the same failure with the skip count at zero.
        # A fresh start (``it0 == 0``) that finds records is the same
        # inconsistency with the count at zero: the kick times would
        # restart from t0 and duplicate every time already in the
        # file, so an existing stream there is a leftover, not a
        # parent.
        n_taken = params.init.it0 // f.it_force
        if n_existing != n_taken:
            where = (
                f"a resume at init.it0 = {params.init.it0}"
                if params.init.it0 > 0
                else "a fresh start (init.it0 = 0)"
            )
            raise ValueError(
                f"forcing.bin holds {n_existing} kick record(s) but "
                f"{where} with force.it_force = {f.it_force} has "
                f"taken {n_taken}: the coefficient stream does not "
                "belong to this state.  Resume from the snapshot the "
                "stream was written up to, or move the existing "
                "forcing.bin / forcing.json aside to start a new "
                "stream."
            )
        self._rng = np.random.default_rng(f.seed)
        for _ in range(n_existing):
            self._rng.standard_normal((len(self.modes), self.m, 2))

    def _load_profiles(
        self, f, n_components: int, ny: int
    ) -> list[np.ndarray]:
        """Per-mode ``(m, C, Ny)`` channel profiles, checked and cut."""
        path = Path(f.profiles)
        if not path.exists():
            raise SystemExit(f"[force] profiles npz {path} not found")
        profiles: list[np.ndarray] = []
        with np.load(path, allow_pickle=False) as npz:
            npz_system = str(np.asarray(npz["system"]))
            if npz_system != params.phys.system:
                raise SystemExit(
                    f"[force] {path} was computed for system "
                    f"{npz_system!r}; this run is "
                    f"{params.phys.system!r}"
                )
            # The bundle records the basis its rows are in; a stale
            # one written in the solver's decoupled `$u_\pm$` basis
            # has the right shape and grid, so nothing else here
            # would catch it -- it would simply be injected as if it
            # were physical.  ``FORMAT_VERSION`` guards readers of
            # ``forcing.bin``, not this input.
            want_labels = _component_labels(n_components)
            if "component_labels" in npz.files:
                got_labels = [
                    str(v) for v in np.asarray(npz["component_labels"])
                ]
                if got_labels != want_labels:
                    raise SystemExit(
                        f"[force] {path} holds profiles in components "
                        f"{got_labels}; this run injects "
                        f"{want_labels}.  Regenerate the bundle with "
                        "the current dnsjax."
                    )
            grid = np.asarray(npz["code_grid"], dtype=float)
            run_grid = np.asarray(derived_params.wall_normal_grid)
            if grid.shape != run_grid.shape or not np.allclose(
                grid, run_grid, rtol=0, atol=1e-12
            ):
                raise SystemExit(
                    f"[force] {path} profiles live on a different "
                    "wall-normal grid than this run; regenerate them "
                    "on this grid (or regrid offline via "
                    "scripts/snapshot_perturb.py's source machinery)."
                )
            for i2, i3 in self.modes:
                key = f"profiles_{i2}_{i3}"
                if key not in npz:
                    have = [k for k in npz.files if k.startswith("profiles")]
                    raise SystemExit(
                        f"[force] {path} has no {key!r} for forced "
                        f"mode ({i2},{i3}) (available: {have})"
                    )
                arr = np.asarray(npz[key], dtype=complex)
                if arr.shape[1:] != (n_components, ny):
                    raise SystemExit(
                        f"[force] {key} has shape {arr.shape}; expected "
                        f"(m, {n_components}, {ny})"
                    )
                if f.n_channels is not None:
                    if f.n_channels > arr.shape[0]:
                        raise SystemExit(
                            f"[force] n_channels = {f.n_channels} > the "
                            f"{arr.shape[0]} stored channels of {key}"
                        )
                    arr = arr[: f.n_channels]
                profiles.append(arr)
        counts = {p.shape[0] for p in profiles}
        if len(counts) != 1:
            raise SystemExit(
                f"[force] unequal channel counts across modes {counts}; "
                "set force.n_channels to their minimum"
            )
        return profiles

    def _profiles_sha256(self) -> str:
        h = hashlib.sha256()
        for arr in self._profiles:
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def _device_cols(self, cols: np.ndarray | None = None) -> Array:
        """Replicate the ``(n_slots, C, Ny)`` kick columns on devices."""
        if cols is None:
            n_c, ny = self._profiles[0].shape[1:]
            cols = np.zeros((self._n_slots, n_c, ny), dtype=complex)
        return jax.device_put(
            cols.astype(
                np.complex128
                if sharding.complex_type == jnp.complex128
                else np.complex64
            ),
            NamedSharding(sharding.mesh, P(None, None, None)),
        )

    def _open_files(self) -> int:
        """Validate/append or create the pair; return existing records.

        Same semantics as the probe stream: identical validation on
        every rank, sidecar written by the main process only, appends
        allowed only against a matching sidecar.
        """
        if self.bin_path.exists() and not self.json_path.exists():
            raise SystemExit(
                f"[force] {self.bin_path} exists without its "
                f"{self.json_path.name} sidecar; move it away."
            )
        n_existing = 0
        if self.json_path.exists():
            with open(self.json_path) as f:
                old = json.load(f)
            mismatch = [
                k for k in _MATCH_KEYS if old.get(k) != self._sidecar[k]
            ]
            if mismatch:
                raise SystemExit(
                    "[force] existing forcing.json does not match this "
                    f"run (differs in: {', '.join(mismatch)}); move the "
                    "old forcing.bin/forcing.json pair away to start a "
                    "fresh stream."
                )
            n_bytes = (
                self.bin_path.stat().st_size if self.bin_path.exists() else 0
            )
            if n_bytes % self.record_dtype.itemsize != 0:
                raise SystemExit(
                    f"[force] {self.bin_path} size ({n_bytes} B) is not "
                    f"a whole number of {self.record_dtype.itemsize}-B "
                    "records; the file is corrupt or from another "
                    "configuration."
                )
            n_existing = n_bytes // self.record_dtype.itemsize
            sharding.print(
                f"[force] appending to {self.bin_path} "
                f"({n_existing} kicks recorded; PRNG advanced past "
                "them)."
            )
        elif sharding.main_device:
            # Atomic, for the same reason as every other sidecar
            # (:func:`~dnsjax.snapshot_meta.write_sidecar_json`).
            write_sidecar_json(self.json_path, self._sidecar)
        return n_existing

    def kick(self, state: Array, t: float) -> Array:
        r"""Apply one kick at time *t*; buffer its coefficients.

        Draws `$w \sim \mathcal{CN}(0,1)$` per (mode, channel), adds
        `$\varepsilon \sum_j w_j \mathbf{p}_j$` (+ conjugate partner)
        to every forced mode in one fused scatter, and returns the
        new state (the input is donated).
        """
        draw = self._rng.standard_normal((len(self.modes), self.m, 2))
        # CN(0,1): real/imag ~ N(0, 1/2), so E|w|^2 = 1 per channel.
        # The *coefficients* (not the raw draws) are what the sidecar
        # documents and the record stores -- the SSI estimator's
        # E[w w^H] = I normalisation depends on it.
        coeff = (draw[..., 0] + 1j * draw[..., 1]) / np.sqrt(2.0)
        n_c, ny = self._profiles[0].shape[1:]
        cols = np.zeros((self._n_slots, n_c, ny), dtype=complex)
        slot = 0
        for k in range(len(self.modes)):
            prof = self.amplitude * (
                coeff[k] @ self._profiles[k].reshape(self.m, -1)
            ).reshape(n_c, ny)
            cols[slot] = prof
            slot += 1
            if self._partner_slot[k] is not None:
                cols[slot] = np.conj(prof)
                slot += 1
        state = self._inject(state, self._device_cols(cols))

        self._rows.append((t, coeff))
        if len(self._rows) >= self.nbuffer:
            self.flush()
        return state

    def flush(self) -> None:
        """Append the buffered coefficient records, durably.

        Main-process write + ``fsync``; all ranks clear the buffer.
        No non-finite scan (host-drawn Gaussians are always finite).
        """
        if not self._rows:
            return
        if sharding.main_device:
            rec = np.zeros(len(self._rows), dtype=self.record_dtype)
            for i, (t, coeff) in enumerate(self._rows):
                rec["t"][i] = t
                rec["w"][i, ..., 0] = coeff.real
                rec["w"][i, ..., 1] = coeff.imag
            with open(self.bin_path, "ab") as f:
                f.write(rec.tobytes())
                f.flush()
                os.fsync(f.fileno())
        self._rows.clear()
