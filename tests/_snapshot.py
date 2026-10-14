"""Shared machinery of the snapshot round-trip tests, core and lab.

Each script lists its cases and runs them through the ``run_*``
functions here, which run every save and load in a worker process of
this module (``python tests/_snapshot.py --worker ...``): the geometry
and sharding singletons are captured at import time, and several CPU
devices need ``--xla_force_host_platform_device_count`` before JAX
starts.  What the cases pin: the core script's docstring.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

NX, NY, NZ = 8, 8, 8  # nx // 2 = 4 (shardable by 1, 2, 4)
SEED = 1234
T_SAVE, IT_SAVE = 1.5, 7
ISNAP_SAVE = 7  # snapshot-lineage index round-tripped via metadata
STATS_SAVE = {"D": -3.5, "E": 1.25}  # embedded _dnsjax_stats.json values


# Periodic systems and stored component counts, from the registry.


def _periodic(system: str) -> bool:
    """Whether *system* is triply periodic (its ``ny`` a Fourier count)."""
    from dnsjax.flows.registry import periodic_systems

    return system in periodic_systems


def _n_comp(system: str) -> int:
    """Number of stacked state components for *system*.

    The flow's ``FlowSpec.n_components``, as ``snapshot._n_components``
    reads it back from the metadata.
    """
    from dnsjax.flows.registry import spec_for

    return spec_for(system).n_components


def parity_profiles(rs, parity):
    """Parity-definite mean-mode profiles, one per stored component.

    At `m = 0` the `(-1)^m` class is *even* in `r` and the
    `(-1)^{m+1}` class *odd*, so each profile is a low-degree polynomial
    of its component's own class (*parity*: the flow's stored
    component parity) -- which the spectral parity interpolation
    reproduces to machine precision on the new grid, but only if the
    component is regridded with the matrix its class calls for.  The
    amplitudes are all distinct (`2^{-c}`), so a swapped pair cannot
    cancel out.

    The velocity entries (the first three) vanish at `r = 1`, where the
    regrid re-imposes no-slip; any further entry deliberately does not:
    only the velocity has a Dirichlet wall condition, and a regrid that
    zeroed another component would show here.
    """
    import numpy as np

    rs = np.asarray(rs, dtype=np.float64)
    out = []
    for c, even in enumerate(parity):
        a = 2.0**-c
        if c < 3:
            wall = 1.0 - rs**2
            out.append(a * wall if even else a * rs * wall)
        else:
            out.append(1.0 + a * rs**2 if even else a * rs)
    return out


# ── worker (runs in its own process) ─────────────────────────────────


def _make_reference(np_mod, shape):
    """Deterministic complex128 field of *shape* from ``SEED``."""
    rng = np_mod.random.default_rng(SEED)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np_mod.complex128)


def _mode_value(np_mod, c, a, qz, qx):
    """Deterministic complex value keyed on **wavenumbers**, not indices.

    A resolution change renumbers the mode axes, so a reference keyed on
    indices could not tell a correct regrid from a shifted one.  Keyed
    on `(component, leading coordinate, k_z, k_x)` it can: a surviving
    mode must come back bit-identical *and* land where its wavenumber
    says.  The coefficients are exact binary fractions of well-separated
    magnitude, so the value is reproduced bit-for-bit in the loading
    process and no two distinct coordinates collide in both parts.
    """
    real = 1.0 * (c + 1) + 0.5 * a + 0.03125 * qz + 0.001953125 * qx
    imag = -0.25 * (c + 1) + 0.015625 * a - 0.5 * qz + 2.0 * qx
    return (real + 1j * imag).astype(np_mod.complex128)


def _mode_reference(np_mod, system: str, nx: int, ny: int, nz: int):
    """True-mode array of :func:`_mode_value` at resolution (nx, ny, nz).

    The leading coordinate is the wavenumber `k_y` for the periodic
    family (whose leading axis is a Fourier axis, regridded inside
    ``load_snapshot``) and the plain grid index otherwise (the FD
    wall-normal axis, which ``load_snapshot`` leaves at the stored
    length).
    """
    from dnsjax.harmonics import complex_harmonics, real_harmonics

    qz = complex_harmonics(nz)
    qx = real_harmonics(nx)
    a = complex_harmonics(ny) if _periodic(system) else np_mod.arange(ny)
    c = np_mod.arange(_n_comp(system))
    return _mode_value(
        np_mod,
        c[:, None, None, None],
        a[None, :, None, None],
        qz[None, None, :, None],
        qx[None, None, None, :],
    )


def _check_mode_regrid(np_mod, snapshot, sharding, system: str, d: str):
    """Assert a resolution-change load kept, added and dropped modes.

    Every mode the snapshot carried and this resolution still resolves
    must come back **bit-identical at its own wavenumber**; every mode
    this resolution adds must be exactly zero; every mode it no longer
    resolves must be gone.  Also checks that the mesh-divisibility
    padding slots stay zero, which is what keeps ``mean_mask`` a
    one-hot and the per-mode operators regular.
    """
    from dnsjax.harmonics import complex_harmonics, real_harmonics
    from dnsjax.parameters import params

    meta = snapshot.read_metadata(d)
    _, a_src, kz_src, kx_src = meta["native_shape"]
    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    periodic = _periodic(system)

    snapshot.validate_snapshot_params(d)
    state, t, it = snapshot.load_snapshot(d)
    got = np_mod.asarray(state)

    a_out = ny - 1 if periodic else a_src
    want_shape = (
        _n_comp(system),
        a_out,
        sharding.nz_spec,
        sharding.nx_spec,
    )
    assert got.shape == want_shape, (got.shape, want_shape)

    # The wall-bounded leading axis is untouched here (the wall-normal
    # interpolation is a separate step), so its reference keeps the
    # stored length.
    ref = _mode_reference(np_mod, system, nx, ny if periodic else a_src, nz)
    keep_z = np_mod.isin(complex_harmonics(nz), complex_harmonics(kz_src + 1))
    keep_x = real_harmonics(nx) < kx_src
    keep_a = (
        np_mod.isin(complex_harmonics(ny), complex_harmonics(a_src + 1))
        if periodic
        else np_mod.ones(a_src, dtype=bool)
    )
    mask = (
        keep_a[None, :, None, None]
        & keep_z[None, None, :, None]
        & keep_x[None, None, None, :]
    )
    expected = np_mod.where(mask, ref, 0.0)
    true = got[:, :, : nz - 1, : nx // 2]
    assert np_mod.array_equal(true, expected), (
        "mode regrid mismatch: "
        f"max|got - want| = {np_mod.max(np_mod.abs(true - expected)):.3e}, "
        f"{int(np_mod.sum(~mask))} of {mask.size} slots should be zero"
    )
    assert not np_mod.any(got[:, :, nz - 1 :, :]), "kz padding not zero"
    assert not np_mod.any(got[:, :, :, nx // 2 :]), "kx padding not zero"
    assert t == T_SAVE, (t, T_SAVE)
    assert it == IT_SAVE, (it, IT_SAVE)


def _true_shape(system: str, ny: int) -> tuple[int, ...]:
    """True (unpadded) spectral vector shape."""
    kz, kx = NZ - 1, NX // 2
    nc = _n_comp(system)
    if _periodic(system):
        return (nc, ny - 1, kz, kx)
    return (nc, ny, kz, kx)


def _embed_true_into_padded(true_arr, padded_shape, system):
    """Embed a true-shaped array into a zero-padded array."""
    import numpy as np

    out = np.zeros(padded_shape, dtype=true_arr.dtype)
    kz, kx = NZ - 1, NX // 2
    if _periodic(system):
        out[:, :, :kz, :kx] = true_arr
    else:
        out[:, :, :kz, :kx] = true_arr
    return out


def _read_state_with_tensorstore(state_dir):
    """Read the extracted zarr3 ``state/`` store without dnsjax."""
    import numpy as np
    import tensorstore as ts

    arr = (
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": state_dir},
            }
        )
        .result()
        .read()
        .result()
    )
    return np.asarray(arr)


def _check_standard_tools(d, ref_true, system):
    """The snapshot is a single tar readable with stdlib + zarr3.

    Validates the user-facing guarantee: ``_dnsjax_meta.json`` parses
    with the standard library alone, and ``tar xf`` + a zarr3 reader
    (TensorStore) recovers exactly the stored component data.
    """
    import json
    import os
    import tarfile
    import tempfile

    import numpy as np

    assert os.path.isfile(d), f"snapshot must be a single file: {d}"
    assert tarfile.is_tarfile(d), "snapshot must be a valid tar archive"

    with tarfile.open(d, "r") as tf:
        names = set(tf.getnames())
        expected = {"_dnsjax_meta.json", "state/zarr.json"} | {
            f"state/c/{i}/0/0/0" for i in range(_n_comp(system))
        }
        assert expected <= names, (sorted(names), sorted(expected))
        # stdlib-only metadata read (no dnsjax).
        meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
        assert meta["format_version"] == 6, meta["format_version"]
        # Provenance key present and non-empty (value depends on the
        # checkout, so only its presence is pinned).
        assert meta.get("git_hash"), meta

    with tempfile.TemporaryDirectory() as ex:
        with tarfile.open(d, "r") as tf:
            tf.extractall(ex, filter="data")
        on_disk = _read_state_with_tensorstore(os.path.join(ex, "state"))

    assert list(on_disk.shape) == meta["native_shape"], (
        on_disk.shape,
        meta["native_shape"],
    )
    # The on-disk bytes ARE the native (solver) spectral layout at
    # true mode counts -- no transpose between memory and disk.
    assert np.array_equal(on_disk, ref_true), (
        "TensorStore read of the extracted zarr3 store does not match "
        "the native-layout reference"
    )


def _check_slab_placement(state, snapshot, sharding) -> None:
    """The I/O layout must place the slab this module's math assumes.

    Every byte range a device writes comes from
    ``_a_ranges(_shard_device_index(shard), ...)`` -- the device's
    *flat* mesh position -- while the placement is chosen by
    ``P(None, (a0, a1), None, None)``.  The two agree only because a
    tuple of mesh axes splits the array axis in row-major (a0-major)
    order, which is a JAX convention this module reads but cannot
    enforce.  A silent change of it would hand every device another
    one's slab, so pin it against the real sharding.

    The mode axes are pinned here too.  Each writer sends
    ``vec[comp][:na]`` in a single call, which is that device's file
    range only if the buffer carries the **true** mode counts; one
    that kept the solver layout's divisibility padding would satisfy
    the placement check below and write the padding modes into the
    file.  On a single-device mesh this also pins the short-circuit in
    ``_to_io_layout``, which returns the solver array untouched on the
    grounds that ``np0 = np1 = 1`` pads nothing.
    """
    import numpy as np

    io = snapshot._to_io_layout(state)
    a_true, ndev = state.shape[1], snapshot._n_devices()
    a_local = snapshot._a_local(a_true, ndev)
    kz_true, kx_true = snapshot._kz_true(), snapshot._kx_true()
    assert io.shape[2:] == (kz_true, kx_true), (
        f"I/O layout carries mode axes {io.shape[2:]}, not the true "
        f"counts {(kz_true, kx_true)}: the writers would send padding "
        "modes, or fragment"
    )
    io_local = snapshot._io_local_shape(a_true, kz_true, kx_true)
    assert io_local[2:] == (kz_true, kx_true), (io_local, (kz_true, kx_true))
    for shard in io.addressable_shards:
        flat = snapshot._shard_device_index(shard)
        want = (flat * a_local, (flat + 1) * a_local)
        got = shard.index[1]
        assert (got.start or 0, got.stop or io.shape[1]) == want, (
            f"device at flat mesh position {flat} holds A rows {got}, "
            f"but the writer sends it {want}"
        )
        # And the payload: the reshard must have moved the rows, not
        # merely relabelled them.
        a_start, na = snapshot._a_ranges(flat, a_true, ndev)
        local = np.asarray(shard.data)
        ref = np.asarray(state)[:, a_start : a_start + na, :kz_true, :kx_true]
        assert np.array_equal(local[:, :na], ref), (
            f"slab content mismatch at flat mesh position {flat}"
        )


def regrid_profile(y):
    """The regrid cases' wall-normal profile: degree 6, zero at the walls.

    Chebyshev interpolation from 7 or more CGL nodes reproduces it
    exactly, and so does the local stencil on that few nodes (its
    window spans them all); what fails the ``1e-10`` comparison is a
    regrid built on a grid recorded in float32 arithmetic, off the
    exact nodes by ~1e-7.  The spectral route itself is pinned by the
    load worker's log line (``run_ny_mismatch_case``).
    """
    return (1 - y**2) * (y**4 + 0.5 * y**2 + 0.3)


def _worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    np0: int,
    d: str,
    ny_override: int | None = None,
    nx_override: int | None = None,
    nz_override: int | None = None,
):
    """Set up singletons for *npv* CPU devices, then save or load."""
    global NX, NZ
    if nx_override is not None:
        NX = nx_override
    if nz_override is not None:
        NZ = nz_override
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={npv}"

    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import padded_res, params

    params.phys.system = system
    params.res.nx = NX
    params.res.ny = ny_override if ny_override is not None else NY
    params.res.nz = NZ
    params.res.double_precision = True
    params.dist.np0 = np0
    params.dist.np1 = npv // np0
    params.outs.snapshot_write_mode = write_mode
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())

    import jax
    from jax.sharding import NamedSharding

    from dnsjax import snapshot
    from dnsjax.sharding import sharding

    padded_shape = (_n_comp(system), *sharding.spec_shape)
    ny = ny_override if ny_override is not None else NY
    tshape = _true_shape(system, ny)
    vshard = NamedSharding(sharding.mesh, sharding.spec_vector_shard)

    if action in ("save", "roundtrip"):
        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        _check_slab_placement(state, snapshot, sharding)
        # The state must survive its own save: nothing in the write
        # path may donate or mutate the caller's array (``__main__``
        # keeps stepping the state it just snapshotted).
        assert np.array_equal(np.asarray(state), state_np), (
            "save_snapshot mutated the caller's state"
        )
        # A completed save renames its scratch file into place.
        assert not os.path.exists(d + ".partial"), (
            "a completed save left its .partial file behind"
        )
        if action == "save":
            return
        # "roundtrip": same-np case -- fall through to the load checks
        # in this same process (one JAX init instead of two; the
        # cold-process load path stays covered by the cross-np cases).

    if action == "save_stats":
        import jax.numpy as jnp

        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        # Embedded stats are (replicated) device scalars in production;
        # exercise the float() host conversion in ``save_snapshot``.
        stats = {k: jnp.asarray(v) for k, v in STATS_SAVE.items()}
        snapshot.save_snapshot(
            state, T_SAVE, IT_SAVE, d, stats=stats, isnap=ISNAP_SAVE
        )
        # A second snapshot with no stats: the member must be omitted and
        # ``isnap`` defaults to 0.
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d + ".nostats")
        print("worker-save-stats-ok", flush=True)
        return

    if action == "save_fail":
        # A save that dies mid-write must not take the previous
        # snapshot with it.  The archive is laid out full-length with
        # zero-filled component regions before anything is written, so
        # a half-finished save under the final name would be a valid
        # tar that loads without complaint and is blank where the
        # writes did not reach -- zeros being a legal state, nothing
        # downstream could tell.
        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        with open(d, "rb") as f:
            good = f.read()

        def _boom(*_args, **_kwargs):
            raise OSError("simulated mid-write failure")

        snapshot._write_chunks_host = _boom
        try:
            snapshot.save_snapshot(state, T_SAVE + 1.0, IT_SAVE + 1, d)
        except OSError:
            pass
        else:
            raise AssertionError("the broken writer did not raise")

        with open(d, "rb") as f:
            assert f.read() == good, (
                "a failed save replaced the previous snapshot"
            )
        leftovers = [
            n for n in os.listdir(os.path.dirname(d)) if n.endswith(".partial")
        ]
        assert leftovers, "a failed save left no .partial to diagnose"
        print("worker-save-fail-ok", flush=True)
        return

    if action in ("save_poly", "save_poly_f32"):
        from dnsjax.parameters import derived_params

        y = np.asarray(
            -np.cos(np.arange(ny, dtype=np.float64) * np.pi / (ny - 1))
        )
        # ``save_poly_f32`` records the grid as a single-precision run
        # does: the CGL nodes evaluated in float32 arithmetic
        # (``build_cartesian_grid`` under ``sharding.float_type``), off
        # the exact nodes by up to ~2e-7, an odd grid's centre node
        # near 4e-8 instead of 0 -- which is what fails the detectors'
        # 1e-12 (rounding exact nodes to float32 would not).
        recorded = y
        if action == "save_poly_f32":
            theta = np.arange(ny, dtype=np.float32) * np.float32(np.pi)
            recorded = (-np.cos(theta / np.float32(ny - 1))).astype(np.float64)
        derived_params.wall_normal_grid = recorded.tolist()
        poly = regrid_profile(y)
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        state_np[0, :, 0, 0] = poly
        state_np[1, :, 0, 0] = poly * 0.5
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "load_interp":
        from dnsjax.__main__ import _interpolate_if_needed
        from dnsjax.parameters import derived_params

        ny = params.res.ny
        y_curr = -np.cos(np.arange(ny, dtype=np.float64) * np.pi / (ny - 1))
        derived_params.wall_normal_grid = y_curr.tolist()

        snapshot.validate_snapshot_params(d)
        state, t, it = snapshot.load_snapshot(d)
        state = _interpolate_if_needed(
            state,
            os.path.join(d),
            snapshot.read_metadata,
            sharding,
            jax.numpy,
        )
        got = np.asarray(state)
        expected_poly = regrid_profile(y_curr)
        np.testing.assert_allclose(
            got[0, :, 0, 0].real, expected_poly, atol=1e-10
        )
        np.testing.assert_allclose(
            got[1, :, 0, 0].real, expected_poly * 0.5, atol=1e-10
        )
        assert got[0, 0, 0, 0].real == 0.0
        assert got[0, -1, 0, 0].real == 0.0
        assert t == T_SAVE
        print("worker-load-interp-ok", flush=True)
        return

    if action == "save_poly_parity":
        from dnsjax.parameters import derived_params
        from dnsjax.snapshot import stored_component_parity

        # Rigged-CGL radial grid (axis gap 1) at this worker's nr, with
        # parity-definite mean-mode profiles (``parity_profiles``).
        n_full = 2 * ny + 1
        s = -np.cos(np.arange(n_full, dtype=np.float64) * np.pi / (n_full - 1))
        rs = s[ny + 1 :]
        derived_params.wall_normal_grid = rs.tolist()
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        for c, prof in enumerate(
            parity_profiles(rs, stored_component_parity())
        ):
            state_np[c, :, 0, 0] = prof
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "load_interp_parity":
        from dnsjax.__main__ import _interpolate_if_needed
        from dnsjax.parameters import derived_params
        from dnsjax.snapshot import stored_component_parity

        # Half-CGL radial grid (axis gap 0) at this worker's nr: both
        # the point count and the grid family differ from the save.
        n_full = 2 * ny
        s = -np.cos(np.arange(n_full, dtype=np.float64) * np.pi / (n_full - 1))
        rs = s[ny:]
        derived_params.wall_normal_grid = rs.tolist()

        snapshot.validate_snapshot_params(d)
        state, t, it = snapshot.load_snapshot(d)
        state = _interpolate_if_needed(
            state,
            os.path.join(d),
            snapshot.read_metadata,
            sharding,
            jax.numpy,
        )
        got = np.asarray(state)
        # The metadata stores the radial count under its public name
        # ("nr"); a non-alias-aware lookup would skip the regrid and
        # leave the old-count state (the shape assert below).
        want = parity_profiles(rs, stored_component_parity())
        assert got.shape[:2] == (len(want), ny), (got.shape, ny)
        for c, prof in enumerate(want):
            ref = prof.copy()
            if c < 3:
                ref[-1] = 0.0  # the regrid re-imposes no-slip
            np.testing.assert_allclose(
                got[c, :, 0, 0].real, ref, atol=1e-10, err_msg=f"component {c}"
            )
        assert t == T_SAVE
        print("worker-load-interp-parity-ok", flush=True)
        return

    if action == "save_modes":
        # A field whose every true mode is a known function of its
        # wavenumbers (see ``_mode_value``); the padding slots stay zero.
        ref = _mode_reference(np, system, NX, params.res.ny, NZ)
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        state_np[:, :, : NZ - 1, : NX // 2] = ref
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        print("worker-save-modes-ok", flush=True)
        return

    if action == "load_modes":
        _check_mode_regrid(np, snapshot, sharding, system, d)
        print("worker-load-modes-ok", flush=True)
        return

    if action == "load_shape":
        # ny-mismatch load only: the assembled global state must carry
        # the flow's component count at the *snapshot's* radial count;
        # interpolation is a separate step.
        snapshot.validate_snapshot_params(d)
        state, t, it = snapshot.load_snapshot(d)
        assert state.shape[:2] == (_n_comp(system), NY), state.shape
        print("worker-load-shape-ok", flush=True)
        return

    # action == "load" (or the "roundtrip" fall-through)
    ref_true = _make_reference(np, tshape)
    reference = _embed_true_into_padded(ref_true, padded_shape, system)
    snapshot.validate_snapshot_params(d)
    state, t, it = snapshot.load_snapshot(d)
    got = np.asarray(state)

    assert got.shape == reference.shape, (got.shape, reference.shape)
    assert np.array_equal(got, reference), "loaded state mismatch"
    assert t == T_SAVE, (t, T_SAVE)
    assert it == IT_SAVE, (it, IT_SAVE)

    # Single uncompressed tar, readable with standard tools / no dnsjax.
    _check_standard_tools(d, ref_true, system)

    print("worker-load-ok", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def _run_worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    d: str,
    ny: int | None = None,
    np0: int = 1,
    nx: int | None = None,
    nz: int | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--action",
        action,
        "--system",
        system,
        "--layout",
        layout,
        "--write-mode",
        write_mode,
        "--np",
        str(npv),
        "--np0",
        str(np0),
        "--dir",
        d,
    ]
    if ny is not None:
        cmd.extend(["--ny", str(ny)])
    if nx is not None:
        cmd.extend(["--nx", str(nx)])
    if nz is not None:
        cmd.extend(["--nz", str(nz)])
    return run_live(cmd, timeout=300)


def _fail(name: str, stage: str, res: subprocess.CompletedProcess) -> str:
    """Print the failure detail; return the one-line summary reason."""
    reason = f"{stage} exit {res.returncode}"
    print(f"  FAIL  {name}: {reason}")
    print(res.stdout[-2000:] if res.stdout else "(no stdout)")
    print(res.stderr[-2000:] if res.stderr else "(no stderr)")
    return reason


# XLA's SPMD partitioner prints this when asked to relocate two mesh
# axes at once and gives up, replicating the whole array on every
# device instead of exchanging it -- ndev x the traffic and ndev x the
# peak memory.  It is a warning, not an error, so only a log check
# catches a reshard that stops being routed one mesh axis at a time.
_REMAT = "Involuntary full rematerialization"


def _check(
    name: str, stage: str, res: subprocess.CompletedProcess
) -> str | None:
    """``None`` when the worker is clean, else its failure reason."""
    if res.returncode != 0:
        return _fail(name, stage, res)
    if _REMAT in (res.stdout or "") + (res.stderr or ""):
        reason = f"{stage} SPMD rematerialization"
        print(f"  FAIL  {name}: {reason}")
        return reason
    return None


def run_case(
    name: str,
    system: str,
    layout: str,
    write_mode: str,
    save_np: int,
    load_np: int,
    save_np0: int = 1,
    load_np0: int = 1,
    nx: int | None = None,
    ny: int | None = None,
) -> str | None:
    """Save then load a snapshot, across np configurations.

    One process when the np configuration is unchanged, separate
    save/load processes across configurations.
    """
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        if (save_np, save_np0) == (load_np, load_np0):
            r = _run_worker(
                "roundtrip",
                system,
                layout,
                write_mode,
                save_np,
                snap_path,
                np0=save_np0,
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "roundtrip", r):
                return reason
        else:
            r_save = _run_worker(
                "save",
                system,
                layout,
                write_mode,
                save_np,
                snap_path,
                np0=save_np0,
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "save", r_save):
                return reason
            r_load = _run_worker(
                "load",
                system,
                layout,
                write_mode,
                load_np,
                snap_path,
                np0=load_np0,
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "load", r_load):
                return reason
    print(f"  PASS  {name}")
    return None


def run_mode_case(
    name: str,
    system: str,
    save_res: tuple[int, int, int],
    load_res: tuple[int, int, int],
    save_np: int = 1,
    save_np0: int = 1,
    load_np: int = 1,
    load_np0: int = 1,
) -> str | None:
    """Save at *save_res*, load at *load_res*, check the mode regrid."""
    layout = "periodic" if _periodic(system) else "walled"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save_modes",
            system,
            layout,
            "concurrent",
            save_np,
            snap,
            np0=save_np0,
            nx=save_res[0],
            ny=save_res[1],
            nz=save_res[2],
        )
        if reason := _check(name, "save_modes", r):
            return reason
        r = _run_worker(
            "load_modes",
            system,
            layout,
            "concurrent",
            load_np,
            snap,
            np0=load_np0,
            nx=load_res[0],
            ny=load_res[1],
            nz=load_res[2],
        )
        if reason := _check(name, "load_modes", r):
            return reason
    print(f"  PASS  {name}")
    return None


def run_io_layout_case() -> str | None:
    """Check the I/O-layout slab arithmetic exhaustively, in-process.

    ``_a_ranges`` and ``_a_offset`` are pure, so the branch the
    round-trip cases cannot reach is cheapest to check directly: a
    device whose slab is **entirely padding** (``n_rows == 0``), which
    needs ``a_true <= (ndev - 1) * ceil(a_true / ndev)`` -- 5 rows over
    4 devices, never produced by the ``ny = 8`` grids used above.

    The invariants are the ones the format depends on: the slabs tile
    ``[0, a_true)`` exactly, and written at the offsets ``_a_offset``
    hands them they reassemble the component chunk byte for byte --
    each from a **single** C-contiguous transfer, which is the property
    the trimmed I/O layout exists to provide and the one the writers
    assume when they send ``vec[comp][:na]`` in one call.
    """
    import numpy as np

    from dnsjax.snapshot import _a_local, _a_offset, _a_ranges

    for a_true in (1, 5, 7, 8, 13, 193):
        for ndev in (1, 2, 4, 8):
            covered: list[int] = []
            for flat in range(ndev):
                a_start, na = _a_ranges(flat, a_true, ndev)
                covered.extend(range(a_start, a_start + na))
            if covered != list(range(a_true)):
                return (
                    f"slabs do not tile a_true={a_true} ndev={ndev}: "
                    f"{covered[:12]}..."
                )

    # The engines write ``off * itemsize`` bytes into the file, so
    # replay that: fill each device's local buffer with the chunk
    # values it should own, write its slab at its byte offset, and
    # require the result to be the reference chunk.  This is the layer
    # the element-index check above skips.
    dtype = np.dtype("complex128")
    for a_true, ndev, kz_true, kx_true in (
        (7, 4, 5, 3),  # a clipped last slab
        (5, 4, 15, 4),  # a device whose slab is entirely padding
        (13, 1, 255, 128),  # single device, production-shaped modes
    ):
        chunk = (
            np.arange(a_true * kz_true * kx_true, dtype=np.float64)
            .astype(dtype)
            .reshape(a_true, kz_true, kx_true)
        )
        file_bytes = bytearray(chunk.nbytes)
        for flat in range(ndev):
            a_start, na = _a_ranges(flat, a_true, ndev)
            local = np.zeros(
                (_a_local(a_true, ndev), kz_true, kx_true), dtype=dtype
            )
            local[:na] = chunk[a_start : a_start + na]
            span = local[:na]
            if not span.flags.c_contiguous:
                return f"slab {flat} is not contiguous ({local.shape})"
            start = _a_offset(a_start, kz_true, kx_true) * dtype.itemsize
            file_bytes[start : start + span.nbytes] = span.tobytes()
        if bytes(file_bytes) != chunk.tobytes():
            return (
                "replayed slab bytes differ from the reference chunk "
                f"for a_true={a_true} ndev={ndev} modes "
                f"({kz_true}, {kx_true})"
            )
    return None


def run_gds_detection_case() -> str | None:
    """``_gds_available`` must answer for the *shipped* kvikIO API.

    This check spent its life returning ``False`` on a cluster that
    had kvikIO installed, because ``kvikio.defaults`` is a submodule
    rather than an attribute of the package -- a failure invisible
    from the outside, since "no GDS" is also the correct answer on a
    node without the driver.  Stub kvikIO and drive every branch: the
    submodule bind, the cupy and nvidia-fs gates, each ``CompatMode``
    the enum can hold (a bare truth test misreads ``AUTO``), and a
    ``get`` that answers with something other than that enum -- which
    must take the "unreadable" path rather than raise out of the
    check.
    """
    import enum
    import sys
    import types

    from dnsjax import snapshot

    class CompatMode(enum.Enum):
        OFF = 0
        ON = 1
        AUTO = 2

    ABSENT = object()  # "kvikIO is not installed", not a compat mode

    def install(value) -> None:
        """Put a fake kvikIO answering ``get("compat_mode")`` on path.

        Deliberately faithful about the trap: the parent module gets
        **no** ``defaults`` attribute, exactly like the installed
        package before its submodule is imported, so reaching for
        ``kvikio.defaults`` through the package raises
        ``AttributeError`` here as it did with the real package.
        """
        kv = types.ModuleType("kvikio")
        kvd = types.ModuleType("kvikio.defaults")
        kvd.get = lambda name: {"compat_mode": value}[name]
        sys.modules["kvikio"], sys.modules["kvikio.defaults"] = kv, kvd

    watched = ("kvikio", "kvikio.defaults", "cupy")
    real = {k: sys.modules.get(k) for k in watched}
    nvfs = snapshot._NVFS_STATS
    with tempfile.TemporaryDirectory() as tmp:
        present = os.path.join(tmp, "stats")
        open(present, "w").close()
        absent = os.path.join(tmp, "no-such-driver")
        # (label, compat mode, cupy, driver, expected)
        cases = [
            ("kvikio absent", ABSENT, True, present, False),
            ("cupy absent", CompatMode.AUTO, False, present, False),
            ("driver absent", CompatMode.AUTO, True, absent, False),
            ("AUTO", CompatMode.AUTO, True, present, True),
            ("OFF", CompatMode.OFF, True, present, True),
            ("ON  -> compat", CompatMode.ON, True, present, False),
            # Not a CompatMode: ``type(mode).ON`` raises, and the
            # check must fall back to the host path.  This is why the
            # enum comparison sits *inside* the guarded block.
            ("non-enum mode", True, True, present, False),
        ]
        try:
            for label, mode, cupy, driver, expected in cases:
                if mode is ABSENT:
                    # ``None`` in sys.modules makes the import raise.
                    sys.modules["kvikio"] = None
                    sys.modules.pop("kvikio.defaults", None)
                else:
                    install(mode)
                # A real cupy (a GPU box) must not decide the outcome
                # either way, so stub both states explicitly.
                sys.modules["cupy"] = (
                    types.ModuleType("cupy") if cupy else None
                )
                snapshot._NVFS_STATS = __import__("pathlib").Path(driver)
                snapshot._gds_available.cache_clear()
                got = snapshot._gds_available()
                if got is not expected:
                    print(f"  FAIL  GDS detection: {label} -> {got}")
                    return f"GDS detection wrong for {label}"
        finally:
            snapshot._NVFS_STATS = nvfs
            snapshot._gds_available.cache_clear()
            for k, v in real.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    print("  PASS  GDS detection branches")
    return None


def run_truncation_case() -> str | None:
    """A truncated archive must fail loudly, not load silently.

    The component data regions are sparse-*reserved* (zero-filled) by
    the skeleton writer, so a short archive has zeros exactly where a
    state should be: the one failure mode that would resume a run
    from a partly blank field without a word.  Cut the last
    component's data in half and require the load to raise.
    """
    name = "truncated archive rejected"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save", "plane-couette", "walled", "concurrent", 1, snap
        )
        if reason := _check(name, "save", r):
            return reason
        # Halfway into the final component: every tar header lies
        # before it, so this is a data-region cut, not a header one.
        comp_nbytes = NY * (NZ - 1) * (NX // 2) * 16
        with open(snap, "r+b") as f:
            f.truncate(os.path.getsize(snap) - 1024 - comp_nbytes // 2)
        r_load = _run_worker(
            "load", "plane-couette", "walled", "concurrent", 1, snap
        )
        if r_load.returncode == 0:
            print(f"  FAIL  {name}: truncated archive loaded silently")
            return "truncated archive loaded silently"
        # ...and it must say so.  Untranslated this is a bare
        # ``ReadError: unexpected end of data`` naming neither the
        # file nor the reason, which on a resume reads as a dnsjax
        # bug rather than a damaged checkpoint.
        out = (r_load.stdout or "") + (r_load.stderr or "")
        if "truncated or corrupt" not in out:
            print(f"  FAIL  {name}: rejected, but not intelligibly")
            return "truncated archive rejected without naming the cause"
    print(f"  PASS  {name}")
    return None


def run_compressed_case() -> str | None:
    """A compressed copy of a snapshot is refused, never read.

    The data chunks are read at raw byte offsets, so a gzip'd archive
    (which ``tarfile`` would open transparently) must fail every
    reader's archive check rather than be read at offsets of the
    decompressed stream.
    """
    import gzip
    import shutil

    from dnsjax.snapshot_meta import (
        SnapshotArchiveError,
        is_snapshot_file,
        read_snapshot_meta,
    )

    name = "compressed archive rejected"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save", "plane-couette", "walled", "concurrent", 1, snap
        )
        if reason := _check(name, "save", r):
            return reason
        gz = snap + ".gz"
        with open(snap, "rb") as src, gzip.open(gz, "wb") as dst:
            shutil.copyfileobj(src, dst)
        if is_snapshot_file(gz):
            print(f"  FAIL  {name}: a gzip'd archive passes is_snapshot_file")
            return "compressed archive passed is_snapshot_file"
        try:
            read_snapshot_meta(gz)
        except SnapshotArchiveError as exc:
            if "compressed" not in str(exc):
                print(f"  FAIL  {name}: rejected without naming compression")
                return "compressed archive rejected without naming the cause"
        else:
            print(f"  FAIL  {name}: metadata read from a gzip'd archive")
            return "compressed archive read silently"
    print(f"  PASS  {name}")
    return None


def run_meta_chunk_consistency_case() -> str | None:
    """The chunks must match the ``native_shape`` describing them.

    Every read position is computed arithmetically from
    ``native_shape``; nothing consults the member it lands in.  So a
    snapshot whose metadata outgrew its chunks reads the *next*
    component's bytes as state -- well-formed complex numbers, no
    error.  Built by hand here (stdlib only, no JAX, no subprocess):
    a matching pair must be accepted, and each way of disagreeing
    must be refused.
    """
    import io
    import json
    import tarfile

    name = "chunks match the metadata"
    shape = [3, 8, 7, 4]  # (components, A, kz, kx)
    chunk = 8 * 7 * 4 * 16  # complex128

    def build(tar_path: str, declared, n_chunks: int, chunk_bytes: int):
        meta = json.dumps(
            {
                "format_version": 6,
                "native_shape": declared,
                "dtype": "complex128",
            }
        ).encode()
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("_dnsjax_meta.json")
            info.size = len(meta)
            tf.addfile(info, io.BytesIO(meta))
            for c in range(n_chunks):
                info = tarfile.TarInfo(f"state/c/{c}/0/0/0")
                info.size = chunk_bytes
                tf.addfile(info, io.BytesIO(b"\x00" * chunk_bytes))

    from dnsjax.snapshot_meta import (
        SnapshotArchiveError,
        snapshot_component_offsets,
    )

    # (label, declared native_shape, chunk count, chunk size, accept?)
    cases = [
        ("consistent", shape, 3, chunk, True),
        ("chunks too small", [3, 9, 7, 4], 3, chunk, False),
        ("chunks too large", [3, 7, 7, 4], 3, chunk, False),
        ("too few chunks", shape, 2, chunk, False),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for label, declared, n_chunks, chunk_bytes, accept in cases:
            p = os.path.join(tmp, "s.tar")
            build(p, declared, n_chunks, chunk_bytes)
            try:
                snapshot_component_offsets(p)
            except SnapshotArchiveError as exc:
                if accept:
                    print(f"  FAIL  {name}: {label} rejected -- {exc}")
                    return f"a consistent archive was rejected ({label})"
            else:
                if not accept:
                    print(f"  FAIL  {name}: {label} accepted")
                    return f"an inconsistent archive was accepted ({label})"
    print(f"  PASS  {name}")
    return None


def run_git_hash_case() -> str | None:
    """``git_hash`` must spawn git with ``posix_spawn`` and match it.

    CPython's other spawn path, ``vfork``, killed rank 0 under an
    exec-wrapping launcher tool (the ``git_hash`` docstring), and
    nothing else would notice a return to it: the provenance string
    comes out the same.  Spy on ``os.posix_spawn`` (CPython's
    ``posix_spawn`` path calls it through the module) and compare the
    result with git's own answer.
    """
    import shutil

    from dnsjax.snapshot_meta import git_hash

    name = "git_hash spawns git with posix_spawn"
    if shutil.which("git") is None:
        print(f"  SKIP  {name}: no git on PATH")
        return None
    spawned: list[str] = []
    real = os.posix_spawn

    def spy(path, *args, **kwargs):
        spawned.append(path)
        return real(path, *args, **kwargs)

    git_hash.cache_clear()
    os.posix_spawn = spy
    try:
        got = git_hash()
    finally:
        os.posix_spawn = real
        git_hash.cache_clear()
    pkg = os.path.join(os.path.dirname(__file__), "..", "src", "dnsjax")
    cli = subprocess.run(
        ["git", "-C", pkg, "describe", "--always", "--dirty", "--abbrev=12"],
        capture_output=True,
        text=True,
    )
    want = cli.stdout.strip() if cli.returncode == 0 else "unknown"
    if not spawned:
        print(f"  FAIL  {name}: git was spawned another way (vfork/fork)")
        return "git_hash did not spawn git with posix_spawn"
    if got != want:
        print(f"  FAIL  {name}: {got!r}, but git describes {want!r}")
        return f"git_hash returned {got!r}, git describe {want!r}"
    print(f"  PASS  {name} ({got})")
    return None


def run_atomic_write_case() -> str | None:
    """A save that fails mid-write must not destroy the good snapshot."""
    name = "failed save keeps the previous snapshot"
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_worker(
            "save_fail",
            "plane-couette",
            "walled",
            "concurrent",
            1,
            os.path.join(tmp, "snap.tar"),
        )
        if reason := _check(name, "save_fail", r):
            return reason
        if "worker-save-fail-ok" not in (r.stdout or ""):
            print(f"  FAIL  {name}: worker did not confirm")
            return "save_fail worker did not confirm"
    print(f"  PASS  {name}")
    return None


def run_write_mode_identity_case() -> str | None:
    """``serial`` and ``concurrent`` must store the same component data.

    The two modes differ only in *when* each process writes, so the
    stored field cannot depend on the choice.  Only the component
    members are compared: the metadata member records
    ``snapshot_write_mode`` itself (and "concurrent" and "serial" are
    not even the same length), so the archives are legitimately not
    byte-identical as files.
    """
    import tarfile

    name = "serial == concurrent component data"
    comps = [f"state/c/{i}/0/0/0" for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        data: dict[str, list[bytes]] = {}
        for mode in ("concurrent", "serial"):
            snap = os.path.join(tmp, f"{mode}.tar")
            r = _run_worker("save", "plane-couette", "walled", mode, 1, snap)
            if reason := _check(name, f"save {mode}", r):
                return reason
            with tarfile.open(snap, "r") as tf:
                data[mode] = [tf.extractfile(c).read() for c in comps]
        if data["concurrent"] != data["serial"]:
            print(f"  FAIL  {name}: component data differs")
            return "serial and concurrent component data differ"
    print(f"  PASS  {name}")
    return None


def run_stats_isnap_case() -> str | None:
    """``save_snapshot`` embeds ``_dnsjax_stats.json`` and ``isnap``.

    It omits the stats member (with ``isnap`` defaulting to 0) when no
    stats are supplied.  Verified with the standard library alone (no
    dnsjax).
    """
    import json
    import tarfile

    name = "isnap + embedded stats"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save_stats", "plane-couette", "walled", "concurrent", 1, snap
        )
        if reason := _check(name, "save_stats", r):
            return reason

        with tarfile.open(snap, "r") as tf:
            names = set(tf.getnames())
            assert "_dnsjax_stats.json" in names, sorted(names)
            meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
            stats = json.loads(tf.extractfile("_dnsjax_stats.json").read())
        assert meta["isnap"] == ISNAP_SAVE, meta["isnap"]
        assert stats == STATS_SAVE, stats

        with tarfile.open(snap + ".nostats", "r") as tf:
            names = set(tf.getnames())
            assert "_dnsjax_stats.json" not in names, sorted(names)
            meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
        assert meta["isnap"] == 0, meta["isnap"]
    print(f"  PASS  {name}")
    return None


def run_ny_mismatch_case(
    save_np: int = 1,
    save_np0: int = 1,
    load_np: int = 1,
    load_np0: int = 1,
    label: str = "",
    save_ny: int = 8,
    stored_float32: bool = False,
) -> str | None:
    """Save at *save_ny*, load at ny=16 with interpolation.

    A ``save_ny`` that the load mesh does not divide (7 over 4
    devices) is the one path where the leading axis is zero-padded
    to a divisible length *and* the assembly runs at the snapshot's
    row count rather than the current one -- the reader has to take
    ``a_true`` from the metadata, pad to the load mesh, and strip
    again after resharding back.

    *stored_float32* records the saved grid as a single-precision run
    does (float32 arithmetic; an odd *save_ny* puts its centre node off
    zero): the regrid must still take the spectral route
    (``regrid_profile``).
    """
    name = f"wb ny {save_ny}->16 interp{label}"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly_f32" if stored_float32 else "save_poly",
            "plane-couette",
            "walled",
            "concurrent",
            save_np,
            snap_path,
            ny=save_ny,
            np0=save_np0,
        )
        if reason := _check(name, "save_poly", r_save):
            return reason
        r_load = _run_worker(
            "load_interp",
            "plane-couette",
            "walled",
            "concurrent",
            load_np,
            snap_path,
            ny=16,
            np0=load_np0,
        )
        if reason := _check(name, "load_interp", r_load):
            return reason
        if stored_float32 and "Chebyshev coefficient map" not in r_load.stdout:
            print(
                f"  FAIL  {name}: the regrid did not take the spectral route"
            )
            return "stencil route"
    print(f"  PASS  {name}")
    return None


def run_parity_regrid_case(system: str) -> str | None:
    """A cylindrical flow's nr 8 (rigged-CGL) -> 10 (half-CGL) regrid.

    Pins the alias-aware stored-``nr`` lookup (public-named metadata)
    and the spectral parity interpolation across a radial grid-family
    change, per stored component and its own parity class
    (``parity_profiles``).
    """
    name = f"{system} nr 8->10 regrid (rigged->half)"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly_parity",
            system,
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=8,
        )
        if reason := _check(name, "save_poly_parity", r_save):
            return reason
        r_load = _run_worker(
            "load_interp_parity",
            system,
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=10,
        )
        if reason := _check(name, "load_interp_parity", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


def run_load_shape_case(system: str) -> str | None:
    """An nr-mismatch load keeps the flow's stored component layout.

    The assembled state carries the flow's component count at the
    snapshot's radial count.
    """
    name = f"{system} nr 8->16 load shape"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save", system, "walled", "concurrent", 1, snap_path
        )
        if reason := _check(name, "save", r_save):
            return reason
        r_load = _run_worker(
            "load_shape",
            system,
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=16,
        )
        if reason := _check(name, "load_shape", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


# ── worker entry point ───────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot test worker")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--action",
        choices=[
            "save",
            "load",
            "roundtrip",
            "save_poly",
            "save_poly_f32",
            "load_interp",
            "save_poly_parity",
            "load_interp_parity",
            "load_shape",
            "save_modes",
            "load_modes",
            "save_stats",
            "save_fail",
        ],
    )
    parser.add_argument("--system")
    parser.add_argument("--layout")
    parser.add_argument("--write-mode")
    parser.add_argument("--np", type=int)
    parser.add_argument("--np0", type=int, default=1)
    parser.add_argument("--dir")
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    args = parser.parse_args()

    if args.worker:
        _worker(
            args.action,
            args.system,
            args.layout,
            args.write_mode,
            args.np,
            args.np0,
            args.dir,
            ny_override=args.ny,
            nx_override=args.nx,
            nz_override=args.nz,
        )
        sys.exit(0)

    raise SystemExit("run a test script; this module only runs workers")
